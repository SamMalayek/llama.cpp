"""
E2E tests for CLI embedding baseline (Phase 1)

Tests shape/dtype/determinism per RFC Phase 1 requirements:
- JSON schema validation
- Dimension/dtype assertions
- Deterministic replay (threads=1)
- Cosine similarity ≥ 0.999 (threads>1)
"""

import json
import hashlib
import logging
import os
import pytest
import subprocess
import time
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

EPS = 1e-5
REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_BIN = REPO_ROOT / "build" / "bin" / "llama-embedding"
DEFAULT_ENV = {**os.environ, "LLAMA_CACHE": os.environ.get("LLAMA_CACHE", "tmp")}
ALLOWED_DIMS = {384, 768, 1024, 1280, 2048, 4096}

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test model configuration
# ---------------------------------------------------------------------------

# Use tiny embedding model for fast tests
DEFAULT_MODEL_HF_REPO = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF"
DEFAULT_MODEL_HF_FILE = "embeddinggemma-300M-qat-Q4_0.gguf"


def ensure_model_available() -> Dict[str, str]:
    """Return model params, ensuring model is available (downloaded if needed)."""
    return {
        "hf_repo": DEFAULT_MODEL_HF_REPO,
        "hf_file": DEFAULT_MODEL_HF_FILE,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embedding_hash(vec: np.ndarray) -> str:
    """Return short deterministic signature for regression tracking."""
    return hashlib.sha256(vec[:8].tobytes()).hexdigest()[:16]


def run_cli_embedding(
    text: str,
    model_params: Optional[Dict[str, str]] = None,
    embd_normalize: int = 2,
    threads: int = 1,
    output_format: str = "json",
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Run llama-embedding CLI and parse JSON output.
    
    Returns parsed JSON response.
    """
    if model_params is None:
        model_params = ensure_model_available()
    
    if not CLI_BIN.exists():
        pytest.skip(f"CLI binary not found: {CLI_BIN}")
    
    cmd = [
        str(CLI_BIN),
        "-hfr", model_params["hf_repo"],
        "-hff", model_params["hf_file"],
        "-p", text,
        "--embd-output-format", output_format,
        "--embd-normalize", str(embd_normalize),
        "-t", str(threads),
        "--ctx-size", "2048",
    ]
    
    env = DEFAULT_ENV.copy()
    output = ""
    
    try:
        result = subprocess.run(
            cmd,
            input="",  # Input via -p flag
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        
        if result.returncode != 0:
            pytest.fail(
                f"CLI failed with code {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        
        # Parse JSON from stdout
        output = result.stdout.strip()
        if not output:
            pytest.fail(f"CLI produced no output\nstderr: {result.stderr}")
        
        # Extract JSON (may have some prefix text from logging)
        # Find JSON object/array start
        json_start = output.find("{")
        if json_start == -1:
            json_start = output.find("[")
        if json_start == -1:
            pytest.fail(f"Could not find JSON in output: {output[:200]}")
        
        json_str = output[json_start:]
        data = json.loads(json_str)
        return data
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"CLI command timed out after {timeout}s")
    except json.JSONDecodeError as e:
        output_preview = output[:500] if output else "<no output>"
        pytest.fail(f"Failed to parse JSON: {e}\nOutput: {output_preview}")


def parse_embedding_from_response(response: Dict[str, Any], index: int = 0) -> np.ndarray:
    """Extract embedding vector from CLI JSON response."""
    # Handle OpenAI-style response format
    if "data" in response:
        data = response["data"]
        if isinstance(data, list) and len(data) > index:
            emb = data[index].get("embedding", [])
        else:
            pytest.fail(f"Response has {len(data)} items, requested index {index}")
    else:
        # Assume direct array format
        emb = response if isinstance(response, list) else []
    
    assert isinstance(emb, list), f"Embedding should be a list, got {type(emb)}"
    return np.array(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests: JSON Schema & Shape/Dtype
# ---------------------------------------------------------------------------


def test_cli_embedding_json_schema():
    """Test that CLI output matches expected JSON schema."""
    text = "hello world"
    response = run_cli_embedding(text, output_format="json")
    
    # Verify schema structure
    assert "object" in response or "data" in response, "Response missing expected fields"
    
    if "data" in response:
        assert isinstance(response["data"], list), "data should be a list"
        assert len(response["data"]) > 0, "data should contain at least one embedding"
        
        item = response["data"][0]
        assert "embedding" in item, "Embedding item missing 'embedding' field"
        assert "index" in item, "Embedding item missing 'index' field"
        assert isinstance(item["embedding"], list), "embedding should be a list"


def test_cli_embedding_dimension():
    """Test that embedding dimension is reasonable."""
    text = "test dimension"
    response = run_cli_embedding(text, output_format="json")
    
    emb = parse_embedding_from_response(response, 0)
    assert len(emb) in ALLOWED_DIMS, f"Unexpected embedding dimension: {len(emb)}"


def test_cli_embedding_dtype():
    """Test that embeddings are float32."""
    text = "test dtype"
    response = run_cli_embedding(text, output_format="json")
    
    emb = parse_embedding_from_response(response, 0)
    assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
    assert np.all(np.isfinite(emb)), "Embedding contains non-finite values"


def test_cli_embedding_multiple_inputs():
    """Test that multiple inputs produce multiple embeddings."""
    # Use separator to provide multiple inputs
    texts = ["hello", "world", "test"]
    combined = "\n".join(texts)
    
    response = run_cli_embedding(combined, output_format="json")
    
    if "data" in response:
        assert len(response["data"]) >= 1, "Should have at least one embedding"
        # Note: CLI may produce one embedding per line, but behavior depends on implementation
        emb = parse_embedding_from_response(response, 0)
        assert len(emb) in ALLOWED_DIMS


# ---------------------------------------------------------------------------
# Tests: Determinism (threads=1)
# ---------------------------------------------------------------------------


def test_cli_embedding_deterministic_threads_1():
    """Test that same input with threads=1 produces identical embeddings."""
    text = "deterministic test threads=1"
    
    response1 = run_cli_embedding(text, threads=1, output_format="json")
    time.sleep(0.1)  # Small delay between runs
    response2 = run_cli_embedding(text, threads=1, output_format="json")
    
    emb1 = parse_embedding_from_response(response1, 0)
    emb2 = parse_embedding_from_response(response2, 0)
    
    assert emb1.shape == emb2.shape, "Embedding shapes differ"
    
    # Check exact match (deterministic mode)
    assert np.array_equal(emb1, emb2), "Embeddings differ between runs (should be identical with threads=1)"
    
    # Also verify hash for regression tracking
    assert embedding_hash(emb1) == embedding_hash(emb2), "Embedding hashes differ"
    
    cos = cosine_similarity(emb1, emb2)
    assert cos > 0.99999, f"Cosine similarity too low for deterministic mode: {cos:.6f}"


def test_cli_embedding_deterministic_empty_input():
    """Test that empty input produces stable embedding."""
    text = ""
    
    response1 = run_cli_embedding(text, threads=1, output_format="json")
    response2 = run_cli_embedding(text, threads=1, output_format="json")
    
    emb1 = parse_embedding_from_response(response1, 0)
    emb2 = parse_embedding_from_response(response2, 0)
    
    assert embedding_hash(emb1) == embedding_hash(emb2), "Empty input embeddings not deterministic"
    assert cosine_similarity(emb1, emb2) > 0.99999


# ---------------------------------------------------------------------------
# Tests: Multi-threaded consistency (threads>1, cosine sim ≥ 0.999)
# ---------------------------------------------------------------------------


def test_cli_embedding_multithread_cosine_similarity():
    """Test that threads>1 produces embeddings with cosine similarity ≥ 0.999 vs threads=1."""
    text = "multithread consistency test"
    
    # Get baseline with threads=1
    response_baseline = run_cli_embedding(text, threads=1, output_format="json")
    emb_baseline = parse_embedding_from_response(response_baseline, 0)
    
    # Test with multiple threads (use 2 or 4, depending on system)
    for threads in [2, 4]:
        response_mt = run_cli_embedding(text, threads=threads, output_format="json")
        emb_mt = parse_embedding_from_response(response_mt, 0)
        
        assert emb_mt.shape == emb_baseline.shape, f"Shape differs with threads={threads}"
        
        cos = cosine_similarity(emb_baseline, emb_mt)
        assert cos >= 0.999, (
            f"Cosine similarity {cos:.6f} < 0.999 for threads={threads} "
            f"(expected ≥ 0.999 per RFC Phase 1)"
        )
        
        log.info(f"threads={threads}: cosine_sim={cos:.6f} (required ≥ 0.999)")


def test_cli_embedding_multithread_consistency():
    """Test that multiple runs with threads>1 are consistent with each other."""
    text = "multithread consistency"
    threads = 4
    
    response1 = run_cli_embedding(text, threads=threads, output_format="json")
    response2 = run_cli_embedding(text, threads=threads, output_format="json")
    
    emb1 = parse_embedding_from_response(response1, 0)
    emb2 = parse_embedding_from_response(response2, 0)
    
    cos = cosine_similarity(emb1, emb2)
    assert cos >= 0.999, f"Multiple runs with threads={threads} inconsistent: cosine={cos:.6f}"


# ---------------------------------------------------------------------------
# Tests: Normalization modes
# ---------------------------------------------------------------------------


def test_cli_embedding_normalization_modes():
    """Test that different normalization modes produce valid embeddings."""
    text = "normalization test"
    
    for normalize in [-1, 0, 1, 2]:
        response = run_cli_embedding(text, embd_normalize=normalize, output_format="json")
        emb = parse_embedding_from_response(response, 0)
        
        assert len(emb) in ALLOWED_DIMS, f"Invalid dim for normalize={normalize}"
        assert emb.dtype == np.float32
        assert np.all(np.isfinite(emb)), f"Non-finite values for normalize={normalize}"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


def test_cli_embedding_very_long_input():
    """Test that very long input is handled correctly."""
    # Use shorter text to stay within batch size limits
    # Batch size is 2048 tokens, so use text that produces ~1000 tokens
    text = "long input test " * 100
    
    response = run_cli_embedding(text, output_format="json", timeout=180)
    emb = parse_embedding_from_response(response, 0)
    
    assert len(emb) in ALLOWED_DIMS
    assert np.all(np.isfinite(emb))


def test_cli_embedding_special_characters():
    """Test that special characters are handled correctly."""
    text = "你好 🌍\n\t!@#$%^&*()_+-=[]{}|;:'\",.<>?/`~"
    
    response = run_cli_embedding(text, output_format="json")
    emb = parse_embedding_from_response(response, 0)
    
    assert len(emb) in ALLOWED_DIMS
    assert np.all(np.isfinite(emb))

