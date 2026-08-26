# Financial RAG V1

A strictly grounded Retrieval-Augmented Generation (RAG) system for financial and
stock-market analysis. Given a corpus of financial PDFs/TXT documents, it retrieves the
most relevant chunks for a query and generates a structured, source-attributed analysis —
**never** presenting information as fact unless it is supported by the retrieved
documents.

## 1. Project overview

The system answers queries like:

```
AAPL
Analyze AAPL based on the available research.
What is the outlook for AAPL?
What risks are mentioned for AAPL?
Does the available evidence support a bullish outlook?
```

It distinguishes between:

1. **Facts** directly supported by retrieved documents.
2. **Synthesis** — reasonable conclusions built from those facts.
3. **Unavailable information** — which is always reported as such, never guessed.

The system is explicitly designed for **grounding over guessing**. If evidence is
insufficient, it says so instead of inventing an answer.

## 2. Architecture

```
User Query
    ↓
Query preprocessing        (src/utils.py, src/retriever.py — ticker expansion)
    ↓
FAISS similarity retrieval  (src/retriever.py)
    ↓
Relevance filtering         (configurable threshold, real cosine similarity)
    ↓
Top-k evidence
    ↓
Context construction        (src/prompts.py — SOURCE_ID / FILE / PAGE / CHUNK_ID)
    ↓
Strict grounding prompt     (src/prompts.py — SYSTEM_PROMPT)
    ↓
LLM structured output       (src/rag_chain.py — ChatOpenAI.with_structured_output)
    ↓
Pydantic validation         (src/schemas.py)
    ↓
Citation validation         (src/rag_chain.py — validate_citations)
    ↓
Final FinancialAnalysis / safe fallback (src/schemas.py — RagResponse)
```

### Project structure

```
financial_rag/  (this repository root)
├── data/                 # your source PDFs/TXT (gitignored except README)
├── vectorstore/          # persisted FAISS index (gitignored except .gitkeep)
├── src/
│   ├── config.py         # environment-based settings, no hardcoded secrets
│   ├── schemas.py         # Pydantic v2 structured output models
│   ├── ingest.py          # load, chunk, embed, persist FAISS
│   ├── retriever.py       # semantic retrieval + relevance filtering
│   ├── prompts.py         # strict grounding system prompt + context formatting
│   ├── rag_chain.py       # orchestration + citation validation
│   └── utils.py           # logging, ticker detection, stable IDs
├── tests/                 # ingestion, retrieval, and grounding tests
├── main.py                # CLI: --ingest / --query / interactive
├── requirements.txt
└── .env.example
```

Each stage is a separate, independently testable module. FAISS can later be swapped for
Qdrant/Chroma by replacing `src/ingest.py` and `src/retriever.py` only — the rest of the
pipeline (schemas, prompts, orchestration) does not depend on FAISS internals.

## 3. Installation

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Environment variables

Copy `.env.example` to `.env` and fill in your API key:

```
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=5
RELEVANCE_THRESHOLD=0.25
DATA_DIR=data
VECTORSTORE_DIR=vectorstore
```

`OPENAI_API_KEY` is required — the app fails immediately with a clear error if it is
missing. All other variables have sensible defaults.

## 5. Ingestion

Place `.pdf` / `.txt` files (recursively) under `data/`, then run:

```powershell
python main.py --ingest
```

This scans `data/`, extracts text (preserving PDF page numbers), chunks it
(`chunk_size=1000`, `chunk_overlap=150` by default), embeds each chunk with the
configured OpenAI embedding model, and persists a FAISS index to `vectorstore/`.

If a vectorstore already exists, ingestion **reuses it** and skips re-embedding — pass
`--rebuild` to force a full rebuild:

```powershell
python main.py --ingest --rebuild
```

Ingestion failures on individual files are logged clearly and do not silently disappear;
if every document fails to load, ingestion raises an error.

## 6. Running queries

Interactive mode:

```powershell
python main.py
```

Single query:

```powershell
python main.py --query "What are the main risks for AAPL?"
```

## 7. Example queries

```
AAPL
Analyze AAPL based on the available research.
What is the outlook for AAPL?
What risks are mentioned for AAPL?
Does the available evidence support a bullish outlook?
```

A bare ticker like `AAPL` is automatically expanded into a retrieval query covering
financial outlook, risks, growth, valuation, earnings, competitive position, and
investment considerations — but the generated analysis still relies exclusively on
retrieved evidence.

## 8. Output schema

Defined in `src/schemas.py` using Pydantic v2:

```python
class FinancialAnalysis(BaseModel):
    ticker: str | None
    summary: str
    future_outlook: str
    positive_factors: list[str]
    risks: list[str]
    evidence: list[Evidence]              # claim + supporting source_ids
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    investment_view: Literal["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"]
    investment_reasoning: str
    sources: list[Source]                 # source_id, file_name, page, chunk_id, citation
    limitations: list[str]
    conflicting_evidence: list[str]        # explicit disagreement between sources
```

`RagResponse` wraps this with a `status` of `OK`, `INSUFFICIENT_EVIDENCE`, or
`VALIDATION_FAILED`, so callers always get a well-defined result even when the analysis
is rejected.

## 9. Grounding strategy

Strict grounding is enforced through multiple, independent safeguards (not a single
mechanism, and **not** a claim of mathematically guaranteed zero hallucination):

1. **Retrieval-only context** — the LLM only ever sees retrieved chunks, never asked to
   use general world knowledge.
2. **Strict system prompt** (`src/prompts.py`) — explicit rules against fabricating
   prices, metrics, dates, citations, or probabilities; requires reporting
   disagreement between sources; requires `INSUFFICIENT_EVIDENCE` when unsupported.
3. **Structured output** — `ChatOpenAI.with_structured_output(FinancialAnalysis)`
   constrains the response shape via the current LangChain API (not deprecated
   function-calling chains).
4. **Empty-context rejection** — if no chunk clears the relevance threshold, the LLM is
   **never called**; the system returns "Insufficient evidence..." directly.
5. **Minimum relevance threshold** — `RELEVANCE_THRESHOLD` filters out low-similarity
   chunks before they ever reach the LLM. The score is genuine cosine similarity (FAISS
   built with `normalize_L2=True` + `MAX_INNER_PRODUCT`), never a fabricated number.
6. **Citation validation** (`validate_citations` in `src/rag_chain.py`) — every
   `source_id` referenced in `evidence` or `sources` must match an actually retrieved
   chunk; any fabricated citation causes the whole response to be rejected.
7. **Post-generation validation failure fallback** — if validation fails for any reason,
   the system returns:
   ```
   The generated analysis could not be validated against the retrieved sources.
   Please refine the query or provide additional documents.
   ```

The system deliberately does **not** claim "zero hallucination." It claims strict,
verifiable grounding with automatic rejection of unverifiable output.

## 10. Limitations

- **No live market data.** V1 has no real-time stock prices, quotes, or market feeds.
  Every analysis is based solely on documents present in `data/` at ingestion time.
- Relevance scores are cosine similarities over embedding space; they are meaningful but
  approximate — tune `RELEVANCE_THRESHOLD` for your corpus.
- PDF text extraction quality depends on the source PDF (scanned/image-only PDFs may
  yield no extractable text and are logged as warnings).
- No reranking, hybrid (BM25 + vector) search, or query rewriting in V1.
- `FAISS.load_local` requires `allow_dangerous_deserialization=True`; only load a
  vectorstore this application built itself — never load an untrusted `vectorstore/`
  directory from an unknown source.

## 11. Future improvements (V2 extension points)

The architecture leaves room for, without implementing in V1:

- Qdrant / Chroma as a drop-in replacement for FAISS.
- Hybrid BM25 + vector retrieval, cross-encoder reranking.
- Real-time market data / financial APIs, time-aware and freshness-aware retrieval.
- Source reliability scoring, query rewriting, multi-document comparison.
- Earnings-call transcript and SEC filing ingestion, portfolio-level analysis.

## Testing

```powershell
pytest
```

Tests use stub retrievers, stub LLM generators, and a deterministic fake embeddings
class — no network access or API key is required to run the test suite. Covered:

- **Ingestion** — TXT/PDF loading preserves metadata; chunking preserves metadata and
  produces stable, unique chunk IDs.
- **Retrieval** — relevant chunks are returned; low-relevance chunks are filtered out.
- **Missing evidence** — an empty retrieval result returns `INSUFFICIENT_EVIDENCE`
  without ever calling the LLM.
- **Citation validation** — a fabricated `source_id` is rejected.
- **Hallucination resistance** — asked for a stock price absent from context, the
  pipeline does not fabricate one.
- **Conflicting evidence** — the schema can represent explicit source disagreement
  rather than silently resolving it (full conflict *detection* is enforced by the
  system prompt and requires a live model to fully exercise).

## Logging

The app logs document/chunk counts, vectorstore creation, queries, retrieval counts,
relevance filtering outcomes, and validation failures via Python's `logging` module. It
never logs API keys or other secrets.
