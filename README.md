# ⚖️ Legal Document RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a
specific legal reference PDF. Built with **Streamlit**, **Gemini**, and **FAISS**.
Clone it, drop in your own PDF, and deploy it in minutes.

> ⚠️ **Disclaimer:** This tool provides information sourced from a document for
> general reference only. It is **not** legal advice, and it does not replace
> consultation with a qualified lawyer.

---

## How it works

```
PDF  ─▶  Extract text (pypdf)  ─▶  Chunk text  ─▶  Local MiniLM  ─▶  FAISS index
                                                                                    │
User question ─▶ Embed question ─▶ Similarity search against FAISS index ◀─────────┘
                                            │
                                  Top-k relevant excerpts
                                            │
                          Prompt (question + excerpts) ─▶ Gemini chat model ─▶ Answer
```

1. **Ingestion** (`ingest.py`): the source PDF is split into pages, then into
   overlapping ~1000-character chunks (so context isn't lost at chunk
   boundaries). Each chunk is embedded with Local MiniLM
   and stored in a local FAISS index, along with the page numbers it came
   from.
2. **Retrieval** (`src/rag_chain.py`): when you ask a question, it's embedded
   with the same model, and FAISS returns the most semantically similar
   chunks from the document.
3. **Generation**: those chunks are inserted into a prompt that instructs the
   model to answer **only** from the provided excerpts, and to say so
   explicitly if the document doesn't cover the question — this keeps the
   bot grounded and reduces hallucination.
4. **UI** (`app.py`): a Streamlit chat interface streams the answer and shows
   which page(s) it came from in a "Sources" expander.

---

## Project structure

```
legal-rag-chatbot/
├── app.py                  # Streamlit app (run this)
├── ingest.py                # CLI script to (re)build the vector index
├── src/
│   ├── pdf_loader.py         # PDF → per-page text
│   ├── chunking.py           # Page text → overlapping chunks
│   ├── embeddings.py         # Local_mini LM
│   ├── vectorstore.py        # FAISS index build/save/load/search
│   └── rag_chain.py          # Retrieval + prompt + generation
├── data/
│   └── legal_reference.pdf   # Your source document (swap this out!)
├── vectorstore/              # Generated index (created by ingest.py)
├── requirements.txt
├── .env.example
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## Quickstart (run locally)
## Features

- PDF document ingestion
- Text chunking
- Local MiniLM embeddings
- FAISS vector search
- Gemini-powered answer generation
- Streamlit interface

## Architecture

PDF
 ↓
Chunking
 ↓
MiniLM Embeddings
 ↓
FAISS
 ↓
Relevant Context
 ↓
Gemini
 ↓
Answer

## Installation

git clone <repository-url>
cd legal-rag-chatbot

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

## API Key

Create a `.env` file:

GEMINI_API_KEY=your_api_key_here

## Build the index

python ingest.py

## Run

streamlit run app.py

> If you skip step 4, the app will build the index automatically on first
> load — it just makes the first page load slower.

---

## Using your own PDF

1. Replace `data/legal_reference.pdf` with your own file (keep the same
   filename, or update `PDF_PATH` in `app.py` and `DEFAULT_PDF` in
   `ingest.py`).
2. Re-run `python ingest.py`, or click **"🔄 Rebuild index"** in the app's
   sidebar.
3. Scanned/image-only PDFs won't work out of the box since there's no text
   layer to extract — OCR them first (e.g. with `ocrmypdf`) before ingesting.

---

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (include the `data/` PDF; the `vectorstore/`
   index files are gitignored by default — see below for options).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   select your repo, branch, and `app.py` as the entrypoint.
3. Before deploying, open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-gemini-key-here"
   ```
4. Click **Deploy**. On first load the app will embed the PDF and build the
   index automatically (takes about a minute for a ~100-page document, and
   fits well within Gemini's free tier for a document this size), then
   cache it for the rest of that session.

**Optional — ship a prebuilt index instead of building it on first load:**
run `python ingest.py` locally, then remove the `vectorstore/*.faiss` and
`vectorstore/*.json` lines from `.gitignore` and commit the generated files.
The app will detect and load them immediately instead of rebuilding.

---

## Configuration reference

| Setting | Where | Default | Notes |
|---|---|---|---|
| `GEMINI_API_KEY` | `.env` / Streamlit secrets / sidebar | — | Required (also accepts `GOOGLE_API_KEY`) |
| Chat model | `src/rag_chain.py` → `CHAT_MODEL` | `gemini-2.5-flash` | Swap for `gemini-3-flash-preview` or `gemini-3.1-pro-preview` for higher quality at higher cost |
| Embedding model | `src/embeddings.py` → `EMBEDDING_MODEL` | `text-embedding-004` | |
| Chunk size / overlap | `ingest.py` args or `src/chunking.py` defaults | 1000 / 150 chars | Larger chunks = more context per excerpt, fewer total chunks |
| Excerpts retrieved per question | app sidebar slider | 5 | Higher = more context, higher token cost |

---

## Tech stack

- **[Streamlit](https://streamlit.io)** — chat UI, deployable for free on Community Cloud
- **[Gemini API](https://ai.google.dev)** — chat completion (`gemini-3.6-flash`), via the [`google-genai`](https://github.com/googleapis/python-genai) SDK
- **[FAISS](https://github.com/facebookresearch/faiss)** — local vector similarity search (no external vector DB needed)
- **[pypdf](https://pypdf.readthedocs.io)** — PDF text extraction

No LangChain/LlamaIndex — the retrieval pipeline is implemented directly in
`src/` so it's easy to read, modify, and reason about end to end.

---

## Troubleshooting

- **"No text could be extracted from the PDF"** — the PDF is likely a scan
  with no text layer. Run OCR on it first.
- **Answers seem off-topic or too generic** — try increasing the number of
  retrieved excerpts in the sidebar, or reduce chunk size in `ingest.py` for
  more granular retrieval.
- **`ClientError` / "API key rejected" in the app** — the API key is invalid,
  expired, or has no remaining quota; check your key at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- **Slow first load on Streamlit Cloud** — that's the one-time index build;
  ship a prebuilt `vectorstore/` (see deployment section) to skip it.

---


