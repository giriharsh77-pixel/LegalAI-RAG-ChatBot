"""
embeddings.py
-------------
Local sentence-transformers embedding wrapper.

Uses all-MiniLM-L6-v2 locally, so document/query embeddings
do not consume Gemini API embedding quota.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        print(f"Loading local embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def embed_texts(
    texts: List[str],
    client=None,
) -> np.ndarray:
    """
    Embed a list of texts locally.

    Returns an (N, 384) float32 numpy array.
    Vectors are L2-normalized so dot product = cosine similarity.
    """

    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    model = _get_model()

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return vectors.astype("float32")


def embed_query(
    text: str,
    client=None,
) -> np.ndarray:
    """Embed a single user query locally."""

    model = _get_model()

    vector = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return vector[0].astype("float32")


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    """
    Kept for compatibility with code that may import this function.
    """

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8

    return array / norms