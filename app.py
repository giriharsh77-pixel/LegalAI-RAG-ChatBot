"""
app.py
------
Streamlit front-end for the legal document RAG chatbot.

Run locally:    streamlit run app.py
Deploy:         push to GitHub, then deploy on share.streamlit.io
                 (see README.md for full instructions)
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

from src.chunking import chunk_pages
from src.embeddings import embed_texts
from src.pdf_loader import load_pdf_pages
from src.rag_chain import stream_answer
from src.vectorstore import build_index, index_exists, load_index, save_index

load_dotenv()

PDF_PATH = "data/legal_reference.pdf"
VECTORSTORE_DIR = "vectorstore"
TOP_K = 5

st.set_page_config(page_title="Legal Document Assistant", page_icon="⚖️", layout="centered")


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    """
    Resolution order:
      1. Streamlit secrets (st.secrets) — used on Streamlit Community Cloud
      2. Environment variable / .env — used for local runs
      3. Sidebar text input — fallback so anyone who clones the repo can
         paste in their own key without touching any config

    Accepts GEMINI_API_KEY or GOOGLE_API_KEY (Google's SDK recognizes both;
    we check GEMINI_API_KEY first since that's what the README/.env.example use).
    """
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    with st.sidebar:
        st.subheader("🔑 API Key")
        if key:
            st.success("Using API key from secrets/environment.")
            manual_override = st.text_input(
                "Override with a different key (optional)", type="password", key="key_override"
            )
            if manual_override:
                key = manual_override
        else:
            key = st.text_input(
                "Enter your Gemini API key",
                type="password",
                help="Get one free at aistudio.google.com/apikey. Only used for this "
                "session and never stored or logged.",
                key="key_input",
            )
    return key or None


# ---------------------------------------------------------------------------
# Index loading / building (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_index_and_chunks(_client: genai.Client, force_rebuild: bool = False):
    if not force_rebuild and index_exists(VECTORSTORE_DIR):
        return load_index(VECTORSTORE_DIR)

    pages = load_pdf_pages(PDF_PATH)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(
            "No text could be extracted from the PDF. If it's a scanned document, "
            "OCR it first (see README)."
        )
    vectors = embed_texts([c.text for c in chunks], _client)
    index = build_index(vectors)
    save_index(index, chunks, VECTORSTORE_DIR)
    return index, chunks


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚖️ Legal Document Assistant")
st.caption(
    "Ask questions about the reference document below. Answers are grounded in the "
    "document's own text — this is **not** a substitute for advice from a qualified lawyer."
)

api_key = get_api_key()

with st.sidebar:
    st.divider()
    st.subheader("📄 Source document")
    st.write(f"`{os.path.basename(PDF_PATH)}`")
    rebuild = st.button("🔄 Rebuild index", help="Re-process the PDF and regenerate embeddings.")
    st.divider()
    top_k = st.slider("Excerpts to retrieve per question", min_value=2, max_value=10, value=TOP_K)
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to get started.")
    st.stop()

client = genai.Client(api_key=api_key)

if rebuild:
    get_index_and_chunks.clear()

try:
    with st.spinner("Loading document index... (first run may take a minute)"):
        index, chunks = get_index_and_chunks(client, force_rebuild=rebuild)
except ClientError as exc:
    st.error(f"Gemini API error: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 - show any setup error directly to the user
    st.error(f"Couldn't build the document index: {exc}")
    st.stop()

st.success(f"Index ready — {len(chunks)} chunks loaded from the document.", icon="✅")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for i, s in enumerate(message["sources"], start=1):
                    label = (
                        f"page {s.page_start}" if s.page_start == s.page_end else f"pages {s.page_start}-{s.page_end}"
                    )
                    st.markdown(f"**Excerpt {i} ({label}, similarity {s.score:.2f})**")
                    st.text(s.text[:500] + ("..." if len(s.text) > 500 else ""))

user_query = st.chat_input("Ask a question about the document...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ][-8:]  # keep the last few turns so the prompt doesn't grow unbounded

    with st.chat_message("assistant"):
        try:
            token_stream, sources = stream_answer(
                user_query, index, chunks, client, history=history, top_k=top_k
            )
            full_answer = st.write_stream(token_stream)
        except ClientError as exc:
            st.error(f"Gemini API error: {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001 - surface any generation error
            st.error(f"Something went wrong generating a response: {exc}")
            st.stop()

        if sources:
            with st.expander("📚 Sources"):
                for i, s in enumerate(sources, start=1):
                    label = (
                        f"page {s.page_start}" if s.page_start == s.page_end else f"pages {s.page_start}-{s.page_end}"
                    )
                    st.markdown(f"**Excerpt {i} ({label}, similarity {s.score:.2f})**")
                    st.text(s.text[:500] + ("..." if len(s.text) > 500 else ""))
        else:
            st.caption("No closely matching excerpts were found for this question.")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources}
    )
