# hybrid-rag-pipeline

RAG pipeline with Hybrid Search for Internal Company Documentation. The intended
system combines dense (embedding-based) and sparse (keyword/BM25) retrieval over
internal documents, feeds the merged results through a reranker, and grounds LLM
answers with verifiable citations back to source documents.

## Status

**Early development.** The project foundation (packaging, configuration,
logging, tests) is in place, along with document ingestion/normalization and
configurable chunking:

- Loads `.txt`, `.md`/`.markdown`, `.html`/`.htm`, and text-based `.pdf` files
- Normalizes extracted content into provenance-tagged segments (Markdown/HTML
  preserve heading structure; PDFs are split one segment per page)
- Persists an untouched copy of the raw source plus a normalized, versioned
  JSON representation, keyed by a SHA-256 content hash
- Splits normalized documents into retrieval-oriented chunks using one of
  three switchable strategies — **fixed-size**, **recursive/structure-aware**,
  or **semantic (embedding-similarity)** — each producing chunks with
  deterministic IDs and full document/section/page provenance
- Embeds each chunking strategy's corpus once, then filters it through
  **exact- and near-duplicate detection** before anything is indexed:
  chunks with identical final text are dropped as exact duplicates, and any
  chunk whose cosine similarity to an already-kept chunk exceeds a
  configurable threshold (`DEDUP_SIMILARITY_THRESHOLD`, default **0.95**,
  strict `>`) is dropped as a near duplicate. Every skipped duplicate is
  recorded — never silently discarded — in a persisted, schema-versioned
  duplicate report alongside the snapshot it belongs to.
- Builds a **synchronized dense + sparse index** from one canonical, ordered,
  **post-deduplication** chunk corpus: dense embeddings
  (`text-embedding-3-small` by default) in a local persistent **ChromaDB**
  collection (explicit **cosine** vector space), and a **BM25** sparse index
  (`rank_bm25`) built with a shared, versioned technical-documentation
  tokenizer. Both sides are built from the exact same kept chunk IDs,
  verified to match before anything is activated, and recorded in a
  deterministic, schema-versioned, per-strategy **index snapshot manifest**
  (SHA-256 fingerprint over ordered chunk IDs + strategy + embedding model +
  tokenizer version + deduplication algorithm/threshold) — so
  fixed/recursive/semantic indexes coexist independently, and re-indexing an
  unchanged corpus under an unchanged configuration reuses the existing
  snapshot instead of rebuilding it (or re-embedding it).

**Chunking strategies, conceptually:**
- *Fixed*: slices raw character windows on a fixed stride, so overlap between
  consecutive chunks is exact and easy to reason about. Simplest and fastest;
  ignores document structure.
- *Recursive*: tries paragraph, then line, then sentence/space separators
  before falling back to a raw character cut — keeping natural structural
  boundaries intact wherever the text allows it.
- *Semantic*: embeds paragraph-level units and starts a new chunk wherever
  cosine similarity between neighboring units drops below a threshold (i.e.
  a topic change), with a hard size cap so one uniform section can't grow
  unbounded. Requires `OPENAI_API_KEY`.

**Not yet implemented:** OCR for scanned/image-only PDFs, dense query
retrieval, sparse (BM25) query retrieval, hybrid search / Reciprocal Rank
Fusion, reranking, grounded generation, citations, evaluation, an API, a
frontend, and containerization.

## Planned architecture

- **Ingestion** *(implemented)*: load and parse source documents
  (.txt/.md/.html/.pdf) into normalized text with provenance metadata,
  persisting both raw and processed representations
- **Chunking** *(implemented)*: split normalized documents into
  retrieval-oriented chunks using a configurable fixed/recursive/semantic
  strategy
- **Deduplication** *(implemented)*: filter exact- and near-duplicate chunks
  (cosine similarity above a configurable threshold) out of each chunking
  strategy's corpus before indexing, with every skipped duplicate recorded
  in a persisted, auditable report
- **Indexing** *(implemented)*: embed each corpus once, deduplicate it, and
  build a synchronized ChromaDB dense index + BM25 sparse index from the
  resulting canonical, post-dedup chunk ordering, tracked by a deterministic
  snapshot manifest
- **Hybrid retrieval**: combine dense vector search with sparse/BM25 keyword search
- **Reranking**: reorder merged candidates for relevance
- **Grounded generation**: LLM answers with citation verification against sources
- **Evaluation**: measure retrieval and answer quality
- **API / dashboard**: expose the pipeline for querying and inspection
- **Containerization**: package services for deployment

Ingestion, chunking, deduplication, and indexing are implemented as
described above; the remaining stages describe intent, not current
behavior. In particular, dense query retrieval, sparse query retrieval,
hybrid search / RRF, and reranking are not implemented yet — the index can
be built and inspected at a low level, but there is no query-side search
API.

## Sample corpus

`data/sample/` contains a small, **fictional/synthetic** internal-document
corpus for a made-up company ("Acme Cloud"), used for local development and
exercising ingestion, chunking, deduplication, and indexing end-to-end. It is not real
company data. It is intended for future hybrid-retrieval evaluation work,
but no retrieval or evaluation has been implemented or run against it yet.

`scripts/index_sample_corpus.py` indexes this corpus with a chosen chunking
strategy using the real OpenAI embedding provider (requires
`OPENAI_API_KEY`); local runtime index data is written under
`data/indexes/` (git-ignored, never `data/sample/`).

## Local development setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
cp .env.example .env
```

### Install the package and development dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Lint

```bash
ruff check .
```

### Format

```bash
ruff format .
```

### Type check

```bash
mypy
```

## Roadmap

- [x] Ingestion (multi-format loading and normalization)
- [x] Chunking (fixed, recursive, semantic)
- [x] Near-duplicate chunk deduplication (exact + cosine-similarity, pre-indexing)
- [x] Indexing (synchronized ChromaDB dense + BM25 sparse, per-strategy snapshots)
- [ ] Hybrid retrieval (dense + sparse + RRF + reranking)
- [ ] Grounded generation and citation verification
- [ ] Evaluation
- [ ] API / dashboard
- [ ] Containerization and portfolio polish
