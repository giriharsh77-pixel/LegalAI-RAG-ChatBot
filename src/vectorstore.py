"""
vectorstore.py
--------------
A small wrapper around a FAISS flat index plus the metadata (chunk
text, page numbers) needed to turn a similarity search result back
into something readable and citable.

Persists to disk as two files:
  - index.faiss   the FAISS index itself
  - chunks.json   the chunk text + page metadata, in index order
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import faiss
import numpy as np

from src.chunking import Chunk

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


def build_index(vectors: np.ndarray) -> "faiss.Index":
    """
    Build a flat, exact-search FAISS index using inner product.
    Vectors are expected to already be L2-normalized (see
    embeddings.py), which makes inner product equivalent to cosine
    similarity. Flat/exact search is intentional: for a single
    document (hundreds to a few thousand chunks) it's fast enough and
    needs no training step or tuning, unlike an approximate index.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array of shape (n_chunks, dim)")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    if vectors.shape[0] > 0:
        index.add(vectors)
    return index


def save_index(index: "faiss.Index", chunks: List[Chunk], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / INDEX_FILENAME))

    chunk_dicts = [asdict(c) for c in chunks]
    with open(out_dir / CHUNKS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)


def load_index(in_dir: str | Path) -> tuple["faiss.Index", List[Chunk]]:
    in_dir = Path(in_dir)
    index_path = in_dir / INDEX_FILENAME
    chunks_path = in_dir / CHUNKS_FILENAME

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No saved index found in {in_dir}. Run `python ingest.py` first, "
            "or use the 'Build / rebuild index' button in the app."
        )

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    chunks = [Chunk(**d) for d in chunk_dicts]
    return index, chunks


def index_exists(in_dir: str | Path) -> bool:
    in_dir = Path(in_dir)
    return (in_dir / INDEX_FILENAME).exists() and (in_dir / CHUNKS_FILENAME).exists()


def search(
    index: "faiss.Index",
    chunks: List[Chunk],
    query_vector: np.ndarray,
    top_k: int = 5,
) -> List[tuple[Chunk, float]]:
    """Return the top_k (chunk, similarity_score) pairs for a query vector."""
    if index.ntotal == 0:
        return []
    query_vector = query_vector.reshape(1, -1).astype("float32")
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))
    return results
