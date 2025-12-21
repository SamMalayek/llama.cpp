#include "llama-embedding.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

// Embedding normalization implementation (same logic as common_embd_normalize)
static void llama_embedding_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0f / (float)sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

// Add a sequence of tokens to the batch
// Safety: Pre-checks capacity before loop starts (more efficient than checking each iteration)
static void batch_add_sequence(
    struct llama_batch & batch,
    const llama_token * tokens,
    size_t n_tokens,
    llama_seq_id seq_id,
    int32_t batch_capacity) {
    if (n_tokens == 0) {
        return;
    }
    
    // Pre-check: Verify we have space for all tokens before starting the loop
    // Valid indices are [0..batch_capacity-1], so we need:
    // batch.n_tokens + n_tokens <= batch_capacity
    GGML_ASSERT(batch.n_tokens + n_tokens <= (size_t)batch_capacity && "llama_batch size exceeded: insufficient capacity for sequence");
    
    for (size_t i = 0; i < n_tokens; i++) {
        // Set up the batch entry
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = (llama_pos)i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = seq_id;
        batch.logits[batch.n_tokens] = true;  // Request embeddings/logits for all tokens
        batch.n_tokens++;
    }
}

// Clean up resources on decode error and return nullptr
static struct llama_embedding_result * cleanup_on_decode_error(
    struct llama_batch & batch,
    struct llama_embedding_result * result) {
    llama_batch_free(batch);
    if (result != nullptr) {
        delete[] result->embeddings;
        delete result;
    }
    return nullptr;
}

// Extract embeddings from a decoded batch and write to output buffer
// Returns false if any required embedding could not be retrieved
static bool extract_embeddings_from_batch(
    struct llama_context * ctx,
    const struct llama_batch & batch,
    enum llama_pooling_type pooling_type,
    float * output_ptr,
    int32_t output_offset,
    int32_t sequences_in_batch,
    int32_t n_embd,
    int embd_normalize) {
    
    std::vector<bool> seq_processed(sequences_in_batch, false);
    
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int32_t embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = output_offset + i;
            if (embd == nullptr) {
                // Failed to get token embedding - this is an error
                return false;
            }
        } else {
            int32_t seq_id_in_batch = batch.seq_id[i][0];
            // Verify seq_id is within expected bounds (defensive check)
            GGML_ASSERT(seq_id_in_batch >= 0 && seq_id_in_batch < sequences_in_batch && "seq_id out of bounds");
            if (seq_processed[seq_id_in_batch]) {
                continue;  // Already processed this sequence
            }
            embd = llama_get_embeddings_seq(ctx, seq_id_in_batch);
            embd_pos = output_offset + seq_id_in_batch;
            seq_processed[seq_id_in_batch] = true;
            if (embd == nullptr) {
                // Failed to get sequence embedding - this is an error
                return false;
            }
        }

        float * out = output_ptr + embd_pos * n_embd;
        llama_embedding_normalize(embd, out, n_embd, embd_normalize);
    }
    
    return true;
}

struct llama_embedding_result * llama_embedding_compute(
    struct llama_context * ctx,
    const llama_token * const * token_arrays,
    const size_t * token_array_lengths,
    size_t n_sequences,
    int embd_normalize) {

    if (ctx == nullptr || token_arrays == nullptr || token_array_lengths == nullptr || n_sequences == 0) {
        return nullptr;
    }

    const struct llama_model * model = llama_get_model(ctx);
    if (model == nullptr) {
        return nullptr;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    if (n_embd <= 0) {
        return nullptr;
    }

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const uint32_t n_batch = llama_n_batch(ctx);
    const uint32_t n_seq_max = llama_n_seq_max(ctx);

    // Calculate total number of tokens and output embeddings
    size_t total_tokens = 0;
    for (size_t i = 0; i < n_sequences; i++) {
        total_tokens += token_array_lengths[i];
    }

    if (total_tokens == 0) {
        return nullptr;
    }

    int32_t n_outputs;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        n_outputs = (int32_t)total_tokens;
    } else {
        n_outputs = (int32_t)n_sequences;
    }

    // Allocate result
    struct llama_embedding_result * result = new llama_embedding_result;
    result->n_outputs = n_outputs;
    result->n_embd = n_embd;
    result->embeddings = new float[n_outputs * n_embd];
    std::memset(result->embeddings, 0, n_outputs * n_embd * sizeof(float));

    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // Initialize batch with capacity = min(n_batch, total_tokens)
    // We need to track the actual allocated capacity for safety checks
    // Keep capacity in size_t to avoid repeated casts and narrowing issues.
    const size_t batch_capacity = std::min((size_t)n_batch, total_tokens);
    struct llama_batch batch = llama_batch_init((int32_t)batch_capacity, 0, (int32_t)n_seq_max);
    if (batch.token == nullptr) {
        delete[] result->embeddings;
        delete result;
        return nullptr;
    }

    float * output_ptr = result->embeddings;
    int32_t output_offset = 0;  // Number of embeddings already written
    int32_t sequences_in_batch = 0;

    // Process sequences in batches
    for (size_t seq_idx = 0; seq_idx < n_sequences; seq_idx++) {
        const llama_token * tokens = token_arrays[seq_idx];
        const size_t n_tokens = token_array_lengths[seq_idx];

        if (n_tokens == 0) {
            continue;
        }

        // Hard guard: a single sequence must fit in the allocated batch capacity.
        // If n_tokens > batch_capacity, flushing can't help and batch_add_sequence()
        // will assert (or we'd overflow if assertions are disabled).
        if (n_tokens > batch_capacity) {
            return cleanup_on_decode_error(batch, result);
        }

        // Check if we need to flush the batch
        // Safety: Check against actual allocated capacity, not n_batch
        // Use size_t math against batch_capacity to avoid int32 truncation/negatives.
        if (batch.n_tokens + n_tokens > batch_capacity || sequences_in_batch >= (int32_t)n_seq_max) {
            // Process current batch
            if (batch.n_tokens > 0) {
                if (llama_decode(ctx, batch) < 0) {
                    return cleanup_on_decode_error(batch, result);
                }

                // Extract embeddings from batch
                if (!extract_embeddings_from_batch(
                        ctx, batch, pooling_type, output_ptr, output_offset,
                        sequences_in_batch, n_embd, embd_normalize)) {
                    return cleanup_on_decode_error(batch, result);
                }

                // Update output offset
                if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
                    output_offset += batch.n_tokens;
                } else {
                    output_offset += sequences_in_batch;
                }
            }

            // Reset batch and clear KV cache for next batch
            batch.n_tokens = 0;
            sequences_in_batch = 0;
            llama_memory_clear(llama_get_memory(ctx), true);
        }

        // Add sequence to batch
        batch_add_sequence(batch, tokens, n_tokens, sequences_in_batch, (int32_t)batch_capacity);
        sequences_in_batch++;
    }

    // Process final batch
    if (batch.n_tokens > 0) {
        if (llama_decode(ctx, batch) < 0) {
            return cleanup_on_decode_error(batch, result);
        }

        // Extract embeddings from final batch
        if (!extract_embeddings_from_batch(
                ctx, batch, pooling_type, output_ptr, output_offset,
                sequences_in_batch, n_embd, embd_normalize)) {
            return cleanup_on_decode_error(batch, result);
        }
    }

    llama_batch_free(batch);
    return result;
}

void llama_embedding_result_free(struct llama_embedding_result * result) {
    if (result != nullptr) {
        delete[] result->embeddings;
        delete result;
    }
}

