#pragma once

#include "llama.h"

#include <vector>
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Embedding normalization modes
 */
enum llama_embedding_normalize_mode {
    LLAMA_EMBEDDING_NORMALIZE_NONE      = -1, // no normalization
    LLAMA_EMBEDDING_NORMALIZE_MAX_ABS   =  0, // max absolute int16
    LLAMA_EMBEDDING_NORMALIZE_TAXICAB   =  1, // taxicab/L1 norm
    LLAMA_EMBEDDING_NORMALIZE_EUCLIDEAN =  2, // Euclidean/L2 norm (default)
    // > 2: p-norm with p = embd_normalize
};

/**
 * Embedding result structure
 * Allocates memory for embeddings on the heap.
 * Caller must free using llama_embedding_result_free()
 */
struct llama_embedding_result {
    float * embeddings;  // Flattened embeddings: [n_outputs * n_embd]
    int32_t n_outputs;   // Number of output embeddings (1 per sequence if pooling, n_tokens if no pooling)
    int32_t n_embd;      // Embedding dimension
};

/**
 * Compute embeddings for tokenized input sequences
 *
 * @param ctx                llama context (must have embeddings enabled via llama_context_params.embeddings)
 * @param token_arrays       Array of token arrays, one per input sequence
 * @param token_array_lengths Length of each token array
 * @param n_sequences        Number of sequences (size of token_arrays and token_array_lengths)
 * @param embd_normalize     Normalization mode (see llama_embedding_normalize_mode)
 * @return Result structure with embeddings, or nullptr on error. Caller must free with llama_embedding_result_free().
 *
 * Thread-safety: This function is NOT thread-safe if multiple threads use the same context.
 * For parallel usage, each thread must use its own context.
 *
 * The function handles:
 * - Batching multiple sequences efficiently
 * - Pooling (mean/cls/last/etc) based on context pooling_type
 * - Embedding normalization
 * - KV cache management (clears before processing)
 */
LLAMA_API struct llama_embedding_result * llama_embedding_compute(
    struct llama_context * ctx,
    const llama_token * const * token_arrays,
    const size_t * token_array_lengths,
    size_t n_sequences,
    int embd_normalize);

/**
 * Free an embedding result allocated by llama_embedding_compute()
 */
LLAMA_API void llama_embedding_result_free(struct llama_embedding_result * result);

#ifdef __cplusplus
}
#endif

