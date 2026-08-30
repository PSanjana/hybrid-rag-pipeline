# hybrid-rag-pipeline

RAG pipeline with Hybrid Search for Internal Company Documentation. The intended
system combines dense (embedding-based) and sparse (keyword/BM25) retrieval over
internal documents, feeds the merged results through a reranker, and grounds LLM
answers with verifiable citations back to source documents.

## Status

**Early development.** The project foundation (packaging, configuration,
logging, tests) is in place, and Phase 1 Step 1 adds multi-format document
ingestion and normalization:

- Loads `.txt`, `.md`/`.markdown`, `.html`/`.htm`, and text-based `.pdf` files
- Normalizes extracted content into provenance-tagged segments (Markdown/HTML
  preserve heading structure; PDFs are split one segment per page)
- Persists an untouched copy of the raw source plus a normalized, versioned
  JSON representation, keyed by a SHA-256 content hash

**Not yet implemented:** chunking (segments above are *not* retrieval
chunks), OCR for scanned/image-only PDFs, embeddings, hybrid (dense + sparse)
retrieval, reranking, grounded generation, citations, evaluation, an API,
a frontend, and containerization.

## Planned architecture

- **Ingestion** *(implemented — Phase 1 Step 1)*: load and parse source
  documents (.txt/.md/.html/.pdf) into normalized text with provenance
  metadata, persisting both raw and processed representations
- **Chunking**: split normalized documents into retrieval-sized units
- **Hybrid retrieval**: combine dense vector search with sparse/BM25 keyword search
- **Reranking**: reorder merged candidates for relevance
- **Grounded generation**: LLM answers with citation verification against sources
- **Evaluation**: measure retrieval and answer quality
- **API / dashboard**: expose the pipeline for querying and inspection
- **Containerization**: package services for deployment

Ingestion is implemented as described above; the remaining stages describe
intent, not current behavior.

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
- [ ] Chunking
- [ ] Hybrid retrieval (dense + sparse)
- [ ] Grounded generation and citation verification
- [ ] Evaluation
- [ ] API / dashboard
- [ ] Containerization and portfolio polish
