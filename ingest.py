"""
ingest.py
---------
Builds the FAISS vector index from the PDF in data/ and saves it to
vectorstore/. Run this once after cloning the repo (or whenever you
swap in a different PDF):

    python ingest.py
    python ingest.py --pdf data/my_other_document.pdf --out vectorstore

Requires GEMINI_API_KEY to be set (in a .env file or your shell
environment) since chunk embeddings are generated via the Gemini API.
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from google import genai

from src.chunking import chunk_pages
from src.embeddings import embed_texts
from src.pdf_loader import load_pdf_pages
from src.vectorstore import build_index, save_index

DEFAULT_PDF = "data/legal_reference.pdf"
DEFAULT_OUT = "vectorstore"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build the vector index for the legal RAG chatbot.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help=f"Path to the source PDF (default: {DEFAULT_PDF})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output directory for the index (default: {DEFAULT_OUT})")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Character overlap between chunks (default: 150)")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set.\n"
            "Create a .env file (see .env.example) or export it in your shell:\n"
            "  export GEMINI_API_KEY=...\n"
            "Get a free key at https://aistudio.google.com/apikey\n",
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"Loading PDF: {args.pdf}")
    pages = load_pdf_pages(args.pdf)
    non_empty = sum(1 for p in pages if p.text.strip())
    print(f"  {len(pages)} pages total, {non_empty} with extractable text.")
    if non_empty == 0:
        print(
            "WARNING: No extractable text was found. This PDF may be a scan/image-only "
            "document — this pipeline needs a text layer (consider OCR-ing it first).",
            file=sys.stderr,
        )

    print(f"Chunking (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"  Created {len(chunks)} chunks.")

    if not chunks:
        print("ERROR: No chunks to embed — nothing to index.", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding {len(chunks)} chunks via Gemini ({args.pdf})...")
    start = time.time()
    vectors = embed_texts([c.text for c in chunks], client)
    print(f"  Done in {time.time() - start:.1f}s.")

    print("Building FAISS index...")
    index = build_index(vectors)

    print(f"Saving index to: {args.out}/")
    save_index(index, chunks, args.out)

    print("\nDone! You can now run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
