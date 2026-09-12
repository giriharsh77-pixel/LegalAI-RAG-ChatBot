"""
rag_chain.py
------------
Ties retrieval (FAISS similarity search) and generation (Gemini chat
completion) together. Builds a grounded prompt that instructs the
model to answer only from the retrieved context and to say so when
the document doesn't cover something, and returns both the answer
and the source chunks used, so the UI can show citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

from google import genai
from google.genai import types

from src.chunking import Chunk
from src.embeddings import embed_query
from src.vectorstore import search

CHAT_MODEL = "gemini-3.6-flash"
TOP_K = 5

SYSTEM_PROMPT = """You are a legal document assistant. You answer questions using ONLY the \
excerpts provided below, which come from a specific legal reference document.

Rules you must follow:
- Base your answer strictly on the provided excerpts. Do not use outside knowledge \
of the law, and do not guess.
- If the excerpts don't contain enough information to answer, say so clearly \
instead of making something up.
- When useful, mention which page(s) the information comes from (page numbers \
are given with each excerpt).
- Keep answers clear and well-organized (use short paragraphs or bullet points \
for lists of clauses/requirements).
- You are not a lawyer and this is not legal advice. If the question calls for \
advice on the user's specific situation, say the excerpts are for general/informational \
reference and a qualified lawyer should be consulted for advice.
"""


@dataclass
class RetrievedSource:
    page_start: int
    page_end: int
    text: str
    score: float


@dataclass
class RagResult:
    answer: str
    sources: List[RetrievedSource]


def retrieve(query: str, index, chunks: List[Chunk], client: genai.Client, top_k: int = TOP_K) -> List[RetrievedSource]:
    query_vector = embed_query(query, client)
    hits = search(index, chunks, query_vector, top_k=top_k)
    return [
        RetrievedSource(page_start=c.page_start, page_end=c.page_end, text=c.text, score=score)
        for c, score in hits
    ]


def build_context_block(sources: List[RetrievedSource]) -> str:
    parts = []
    for i, s in enumerate(sources, start=1):
        page_label = (
            f"page {s.page_start}" if s.page_start == s.page_end else f"pages {s.page_start}-{s.page_end}"
        )
        parts.append(f"[Excerpt {i} - {page_label}]\n{s.text}")
    return "\n\n".join(parts)


def build_contents(query: str, sources: List[RetrievedSource], history: List[dict]) -> List[types.Content]:
    """
    history: prior turns as [{"role": "user"/"assistant", "content": "..."}]
    (already excluding the current query), used only for conversational
    flow, not for re-retrieval. Gemini uses "model" instead of "assistant"
    for the bot's turns, so roles are translated here.
    """
    context_block = build_context_block(sources)
    user_turn = (
        f"Document excerpts:\n{context_block}\n\n"
        f"Question: {query}"
        if sources
        else f"No relevant excerpts were found in the document for this question.\n\nQuestion: {query}"
    )

    contents: List[types.Content] = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_turn)]))
    return contents


def answer_question(
    query: str,
    index,
    chunks: List[Chunk],
    client: genai.Client,
    history: List[dict] | None = None,
    top_k: int = TOP_K,
) -> RagResult:
    """Non-streaming version: returns the full answer at once."""
    history = history or []
    sources = retrieve(query, index, chunks, client, top_k=top_k)
    contents = build_contents(query, sources, history)

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return RagResult(answer=response.text, sources=sources)


def stream_answer(
    query: str,
    index,
    chunks: List[Chunk],
    client: genai.Client,
    history: List[dict] | None = None,
    top_k: int = TOP_K,
) -> tuple[Iterator[str], List[RetrievedSource]]:
    """
    Streaming version for a responsive chat UI. Returns a generator of
    text tokens plus the sources (retrieved up front, before streaming
    starts) so the caller can render citations alongside the answer.
    """
    history = history or []
    sources = retrieve(query, index, chunks, client, top_k=top_k)
    contents = build_contents(query, sources, history)

    stream = client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )

    def token_generator() -> Iterator[str]:
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    return token_generator(), sources
