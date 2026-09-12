"""
pdf_loader.py
-------------
Extracts text from a PDF file, page by page, so downstream chunking
can keep track of which page each piece of text came from (used later
for citing sources in the chatbot's answers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int  # 1-indexed, human friendly
    text: str


def load_pdf_pages(pdf_path: str | Path) -> List[PageText]:
    """
    Read a PDF and return a list of PageText objects, one per page.

    Pages with no extractable text (e.g. pure scanned images) are
    still included with an empty string so page numbering stays
    consistent; they simply won't contribute any chunks later.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: List[PageText] = []

    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_text(raw_text)
        pages.append(PageText(page_number=i, text=cleaned))

    return pages


def _clean_text(text: str) -> str:
    """Light cleanup: normalize whitespace without mangling content."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual check: `python -m src.pdf_loader data/legal_reference.pdf`
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/legal_reference.pdf"
    pages = load_pdf_pages(target)
    print(f"Loaded {len(pages)} pages.")
    non_empty = [p for p in pages if p.text.strip()]
    print(f"Pages with extractable text: {len(non_empty)}")
    if non_empty:
        print("\n--- Sample from page", non_empty[0].page_number, "---")
        print(non_empty[0].text[:500])
