"""
chunking.py
-----------
Splits page-level text into overlapping chunks small enough to embed
and feed to an LLM, while remembering which page(s) each chunk came
from so the chatbot can cite sources.
"""

from dataclasses import dataclass
from typing import List

from src.pdf_loader import PageText


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page_start: int
    page_end: int


def chunk_pages(
    pages: List[PageText],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    Concatenate all page text (with page markers) and split it into
    overlapping character-based chunks.

    A sliding window keeps context from being cut off mid-sentence at
    chunk boundaries (the overlap), and each chunk records which pages
    it spans so answers can be traced back to the source document.
    """
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    # Build one long string with (page_number, char_offset) breakpoints
    # so we can map any character position back to a page number.
    full_text = ""
    offset_to_page: List[tuple[int, int]] = []  # (start_offset, page_number)

    for page in pages:
        if not page.text.strip():
            continue
        offset_to_page.append((len(full_text), page.page_number))
        full_text += page.text + "\n\n"

    if not full_text.strip():
        return []

    def page_for_offset(offset: int) -> int:
        page_number = offset_to_page[0][1]
        for start_offset, pg in offset_to_page:
            if start_offset <= offset:
                page_number = pg
            else:
                break
        return page_number

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    text_len = len(full_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break on a paragraph/sentence boundary near `end`
        # instead of slicing mid-word, when there's room to look back.
        if end < text_len:
            search_window = full_text[max(start, end - 200):end]
            best_break = max(
                search_window.rfind("\n\n"),
                search_window.rfind(". "),
            )
            if best_break != -1:
                end = max(start, end - 200) + best_break + 1

        raw_chunk = full_text[start:end].strip()
        if raw_chunk:
            page_start = page_for_offset(start)
            page_end = page_for_offset(max(start, end - 1))
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=raw_chunk,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            chunk_id += 1

        if end >= text_len:
            break
        start = end - chunk_overlap

    return chunks


if __name__ == "__main__":
    from src.pdf_loader import load_pdf_pages

    pages = load_pdf_pages("data/legal_reference.pdf")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    if chunks:
        sample = chunks[len(chunks) // 2]
        print(f"\n--- Sample chunk #{sample.chunk_id} (pages {sample.page_start}-{sample.page_end}) ---")
        print(sample.text[:400])
