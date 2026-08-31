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
- Answers questions with **dense retrieval**: a question is embedded with the
  same shared `EmbeddingProvider` used to build the index, then the active
  Chroma snapshot for the requested chunking strategy — resolved solely from
  its manifest, never guessed — is queried via cosine nearest-neighbor
  search. Results are returned ranked (default top-k **10**, configurable via
  `DENSE_TOP_K` or per-call override), each carrying its raw cosine distance,
  a derived similarity (`1.0 - distance`), and full source provenance
  (document, chunk index, source file, section heading/page number where
  present). Purely read-only: no index artifact is ever mutated by a query.
- Answers questions with **sparse BM25 retrieval**: a question is tokenized
  with the exact same shared, versioned technical tokenizer used to build
  the index, then the active BM25 sparse corpus for the requested chunking
  strategy — resolved solely from its manifest, never guessed — is scored
  via `rank_bm25.BM25Okapi.get_scores()`, unnormalized. Results are ranked
  by descending BM25 score (ties broken by ascending canonical corpus
  position, never randomly), returned ranked (default top-k **10**,
  configurable via `SPARSE_TOP_K` or per-call override), each carrying its
  raw BM25 score and full source provenance. Chunk text/metadata are
  hydrated from Chroma's synchronized record store using `Collection.get()`
  only — never vector search — and cross-checked against the sparse
  snapshot's own text for corruption. No embedding provider or API key is
  needed. Purely read-only: no index artifact is ever mutated by a query.
- Combines both channels with **weighted Reciprocal Rank Fusion (RRF)**:
  `retrieve_hybrid()` calls the existing dense and sparse retrievers
  unmodified, then fuses their independently-ranked results by *rank
  position* — never by comparing or normalizing the incompatible raw
  cosine/BM25 score scales. For a chunk at dense rank `r_d` and/or sparse
  rank `r_s` (a missing side contributes exactly 0):
  `rrf_score = dense_weight / (rank_constant + r_d) + sparse_weight / (rank_constant + r_s)`.
  Defaults: dense weight **0.7**, sparse weight **0.3**, rank constant
  **60**, hybrid top-k **10** (`RRF_DENSE_WEIGHT` / `RRF_SPARSE_WEIGHT` /
  `RRF_RANK_CONSTANT` / `HYBRID_TOP_K`). The candidate set is the *union*
  of dense and sparse chunk IDs — a chunk found by only one retriever
  stays eligible — and ties are broken by a fully deterministic rule (best
  available rank, then dense rank, then sparse rank, then chunk ID). Both
  retrieval channels must succeed; a failure in either is surfaced as a
  clear hybrid-retrieval error rather than silently falling back to a
  single channel. Native dense similarity/distance and BM25 score are
  retained on each result for diagnostics only and are never summed,
  normalized, or compared against each other.
- Reorders a wide hybrid candidate set with **reranking**: `retrieve_reranked()`
  calls the existing `retrieve_hybrid()` unmodified, requesting a wider
  candidate pool than ordinary hybrid retrieval uses (`rerank_candidate_k`,
  default **20** — deliberately independent of `hybrid_top_k`'s
  general-purpose default of 10), then scores each `(query, chunk)` pair
  with a `Reranker` (default: a local `sentence-transformers` cross-encoder,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) and returns the final top
  `rerank_top_k` (default **5**) chunks, sorted by `reranker_score` DESC.
  Fast dense/sparse/RRF retrieval maximizes recall over the whole corpus;
  reranking then improves precision over that narrowed candidate set by
  scoring each pair jointly instead of comparing independent vectors.
  Reranking is pure and provider-agnostic (`rerank_candidates()` depends
  only on a small `Reranker` protocol, never a specific hosted API or
  library), read-only, and never mutates a hybrid result or index
  artifact; the original `hybrid_rank` and `rrf_score` are retained
  alongside the new `reranker_score` for diagnostics, never discarded.
  `reranker_score` is a raw, provider-specific relevance score, never a
  probability/confidence value. The cross-encoder model is loaded lazily
  (never at import time) and requires the optional `sentence-transformers`
  extra (`pip install 'rag-pipeline[rerank]'`).
- Answers questions with **grounded generation and bracketed citations**:
  `retrieve_and_generate()` calls the existing `retrieve_reranked()`
  unmodified, numbers its final top-5 chunks as evidence `[1]`..`[5]` (the
  citation number is exactly the reranked rank — never the underlying
  SHA-256 chunk ID), and asks a `Generator` (default: OpenAI Chat
  Completions, `GENERATION_MODEL`, default `gpt-5.6-terra`) to answer using
  only that evidence, with every material claim followed by a bracket
  citation. Evidence blocks are rendered with fixed, content-independent
  delimiters (`[n]` / `Source:` / `Section:` / `Page:` / `Content:`) and are
  explicitly framed in the system prompt as **untrusted reference
  material, not instructions** — text inside a document such as "ignore
  previous instructions" is never treated as a directive. Retrieval-
  diagnostic scores (cosine similarity/distance, BM25 score, RRF score,
  reranker score) are never included in what the model sees. Generated
  answers are validated before being returned: every cited number must
  fall within the supplied evidence range (`[1, N]`), and a substantive
  answer with supplied evidence but zero citations is rejected rather than
  silently accepted — both raise a clear error instead of being repaired
  or hidden. If the evidence genuinely doesn't answer the question, the
  model is instructed to say so explicitly rather than guess; this is the
  generation instruction and response *form* only — recognizing it and
  deciding what to do with it is the job of the separate confidence and
  abstention layers below. A retrieval failure during
  `retrieve_and_generate()` is surfaced, never silently answered without
  evidence.
- Semantically verifies citations with an **LLM judge**:
  `verify_grounded_answer()` extracts every citation *occurrence* from a
  `GroundedAnswer` (deterministic regex, not an LLM — a repeated `[1]`
  is a separate occurrence each time it appears, judged independently),
  builds a judge-only annotated copy of the answer with each occurrence
  marked (`<occurrence id="N">[k]</occurrence>` — the user-facing answer
  is never altered), and asks a `CitationJudge` (default: OpenAI Chat
  Completions structured outputs, `CITATION_JUDGE_MODEL`) whether the
  cited evidence actually supports the associated claim. Each occurrence
  gets exactly one verdict — `SUPPORTED`, `PARTIALLY_SUPPORTED`,
  `UNSUPPORTED`, or `CONTRADICTED` — never a numeric/probability score.
  The judge's raw output is never trusted blindly: a
  `CitationVerificationReport` is only built after strictly validating
  that the judge returned exactly one well-formed result per expected
  occurrence (rejecting missing/duplicate/extra occurrences, a wrong
  citation number, an invalid verdict, or an empty rationale). The
  judge's system prompt explicitly frames the answer and evidence as
  untrusted data, not instructions. Verification is read-only — it
  never rewrites the answer, adds/removes citations, or reorders
  evidence — and its derived counts/`all_supported` are a **factual
  tally of verdicts, not a calibrated confidence score or an
  accept/reject policy** (that's a later phase). The fixed
  insufficient-evidence response has zero citations, so it gets an
  empty report without ever calling the judge.
- Produces a **deterministic confidence assessment**: `score_confidence()`
  turns a verified `GroundedAnswer` into an immutable, decomposable
  `ConfidenceAssessment` in pure Python — **no LLM call, no network, no
  retrieval, no mutation**. It combines two signals: (1) **semantic
  citation support** — the mean over all citation *occurrences* of a
  fixed verdict mapping (`SUPPORTED` → 1.0, `PARTIALLY_SUPPORTED` → 0.5,
  `UNSUPPORTED` → 0.0, `CONTRADICTED` → 0.0), the **dominant** component
  at default weight **0.9**; and (2) **retrieval-channel agreement** — a
  weaker corroborating signal at default weight **0.1**, the fraction of
  *unique cited* evidence chunks that were found by **both** the dense
  and sparse channels (`dense_rank is not None and sparse_rank is not
  None`), joined back to the final `RerankedRetrievalResult`s by stable
  `chunk_id`. Repeated citations to the same chunk count once for
  agreement; uncited retrieved evidence is ignored. The composite is
  `(w_c·citation + w_r·agreement) / (w_c + w_r)`, normalized by the
  configured weight sum so it stays in `[0, 1]` by construction for any
  non-negative weights (at least one positive). Raw BM25 / cosine
  distance / RRF / reranker scores are **deliberately excluded** — they
  are uncalibrated, scale-incompatible native diagnostics, so only rank
  *presence* enters, never any magnitude. The recognized
  insufficient-evidence answer scores `0.0` with
  `is_insufficient_evidence=True` (a vacuous `all_supported=True` over
  zero occurrences is never read as high confidence). `has_contradiction`
  is exposed as a first-class diagnostic but is **not** used to cap,
  override, or reject anything here. This score is a **heuristic quality
  signal, not a calibrated probability and not an accept/reject
  decision**; the confidence scorer is a trust boundary and raises a
  clear `ConfidenceInputError` rather than score a report/retrieval set
  that is inconsistent with the answer. `retrieve_generate_verify_and_score()`
  composes retrieval → generation → verification → scoring, reusing the
  same underlying stages (it calls `retrieve_reranked()` directly to keep
  the intermediate results the agreement component needs) and duplicating
  none of them.
- Makes a final **deterministic abstention decision** in a **separate
  policy layer**: `apply_abstention_policy()` consumes the already-computed
  `GroundedAnswer`, `CitationVerificationReport`, and `ConfidenceAssessment`
  (it recomputes **no** retrieval, generation, verification, or confidence)
  and returns a `FinalAnswer` carrying an `AnswerDecision` enum. The
  precedence is fixed and order-sensitive: **(1)** the generator emitted
  the canonical insufficient-evidence response → `ABSTAINED_INSUFFICIENT_EVIDENCE`;
  **(2)** any citation was verified `CONTRADICTED` → `ABSTAINED_CONTRADICTION`;
  **(3)** any citation was verified `UNSUPPORTED` → `ABSTAINED_UNSUPPORTED_CITATION`;
  **(4)** `confidence.score < confidence_threshold` (default **0.8**) →
  `ABSTAINED_LOW_CONFIDENCE`; **(5)** otherwise → `ANSWERED`. A
  *partially*-supported citation on its own never forces abstention — a
  partial verdict already lowers the Step 3 citation-support component, so
  rule 4's threshold decides whether such an answer still passes. On
  `ANSWERED` the grounded answer text is returned **verbatim** (no
  rewrite, no citation edits, no appended metadata); on any abstention
  the user-facing text is a single fixed sentence
  (`"I don't have enough reliable information in the supplied documents to
  answer that confidently."`) that never mentions scores, retrieval
  algorithms, or judge verdicts, and the rejected draft is retained only
  on `FinalAnswer.grounded_answer` for debugging/evaluation. The policy
  is a trust boundary too: it re-checks the handful of fields it relies
  on (score finite and in `[0, 1]`, citation counts vs the verification
  report, contradiction flag vs count, insufficiency flag vs the answer's
  canonical form, threshold valid) and raises `AbstentionPolicyInputError`
  rather than decide from a contradictory hand-built trio.
  `answer_question_with_policy()` composes
  `retrieve_generate_verify_and_score()` (called **once**) with the policy.
  The **0.8 threshold is an initial uncalibrated heuristic**; Phase 4
  evaluation is expected to tune it (and the Step 3 component weights) —
  it is a distinct policy setting and never reuses the `confidence_*`
  weights.

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

**Not yet implemented:** OCR for scanned/image-only PDFs, calibration of
the confidence score and the abstention threshold against real data
(both are deterministic but uncalibrated heuristics), retrieval
evaluation, an API, a frontend, and containerization.

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
- **Dense retrieval** *(implemented)*: embed a question with the same shared
  provider used at indexing time and return the top-k nearest chunks (cosine
  distance/similarity) from the active Chroma snapshot for a chunking
  strategy, with full source provenance
- **Sparse (BM25) retrieval** *(implemented)*: tokenize a question with the
  same shared technical tokenizer used at indexing time and return the top-k
  BM25 lexical matches from the active sparse snapshot, with full source
  provenance
- **Hybrid retrieval** *(implemented)*: fuse the dense and sparse rankings
  with weighted Reciprocal Rank Fusion (RRF), combining rank positions
  rather than the incompatible native score scales
- **Reranking** *(implemented)*: reorder a wide (default 20) hybrid
  candidate pool with a cross-encoder-style reranker, keeping the final
  top 5 chunks by `reranker_score`
- **Grounded generation** *(implemented)*: generate an answer from only the
  final reranked evidence, with bracketed `[n]` citations validated
  against the supplied evidence range
- **Citation verification** *(implemented)*: an LLM judge checks, per
  citation occurrence, whether the cited evidence actually supports the
  associated claim (`SUPPORTED`/`PARTIALLY_SUPPORTED`/`UNSUPPORTED`/
  `CONTRADICTED`)
- **Confidence scoring** *(implemented)*: combine semantic citation-support
  verdicts (dominant) with weak dense+sparse retrieval-channel agreement
  for the cited evidence into one deterministic, decomposable
  `ConfidenceAssessment` — a heuristic quality signal, not a calibrated
  probability
- **Abstention policy** *(implemented)*: a separate deterministic layer
  turns the confidence assessment (plus the insufficiency, contradiction,
  and unsupported-citation signals) into a final `ANSWERED` / graceful
  `"I don't know"` decision via a fixed precedence and one configurable
  `confidence_threshold` (default 0.8, uncalibrated); Phase 4 evaluation
  will tune the threshold and the Step 3 weights
- **Evaluation**: measure retrieval and answer quality
- **API / dashboard**: expose the pipeline for querying and inspection
- **Containerization**: package services for deployment

Ingestion, chunking, deduplication, indexing, dense retrieval, sparse
retrieval, hybrid RRF fusion, reranking, grounded generation with
bracketed citations, semantic citation verification, deterministic
confidence scoring, and the deterministic abstention policy are
implemented as described above; the remaining stages describe intent,
not current behavior. In particular, the confidence score and the
abstention `confidence_threshold` are deterministic but **uncalibrated**
heuristics — no evaluation has tuned them — and there is no query-side
search API (FastAPI) yet, only the `scripts/query_dense.py`,
`scripts/query_sparse.py`, `scripts/query_hybrid.py`,
`scripts/query_reranked.py`, `scripts/ask_grounded.py`,
`scripts/ask_verified.py`, `scripts/ask_with_confidence.py`, and
`scripts/ask_final.py` development scripts. No retrieval evaluation has
been run or is claimed.

## Sample corpus

`data/sample/` contains a small, **fictional/synthetic** internal-document
corpus for a made-up company ("Acme Cloud"), used for local development and
exercising ingestion, chunking, deduplication, indexing, and dense/sparse/
hybrid/reranked retrieval, grounded generation, citation verification,
confidence scoring, and the abstention policy end-to-end. It is not real
company data. It is intended for future
retrieval evaluation work, but no evaluation has been implemented or run
against it yet.

`scripts/index_sample_corpus.py` indexes this corpus with a chosen chunking
strategy using the real OpenAI embedding provider (requires
`OPENAI_API_KEY`); `scripts/query_dense.py`, `scripts/query_sparse.py`,
`scripts/query_hybrid.py`, `scripts/query_reranked.py`,
`scripts/ask_grounded.py`, `scripts/ask_verified.py`,
`scripts/ask_with_confidence.py`, and `scripts/ask_final.py` then run one
dense-, sparse-, hybrid-, reranked-, grounded-generation,
grounded-generation-plus-verification,
grounded-generation-plus-verification-plus-confidence, or
full-pipeline-plus-abstention-decision query against the resulting active
snapshot (`query_sparse.py` needs no API key — sparse retrieval never
touches embeddings; the rest need one, since they call the dense channel
too, and `ask_grounded.py`/`ask_verified.py`/`ask_with_confidence.py`/
`ask_final.py` additionally call the OpenAI generation model, with
`ask_verified.py`/`ask_with_confidence.py`/`ask_final.py` also calling the
OpenAI citation judge; `query_reranked.py`, `ask_grounded.py`,
`ask_verified.py`, `ask_with_confidence.py`, and `ask_final.py` also
require the optional `sentence-transformers` extra and download its
cross-encoder model on first use). `ask_with_confidence.py` labels its
output a *heuristic confidence score*, never a percentage chance of
correctness; `ask_final.py` prints only the final user-facing answer by
default (either the grounded answer or the fixed abstention sentence) and
shows the decision/score/counts only with `--debug`. Local runtime index
data is written under `data/indexes/` (git-ignored, never `data/sample/`).

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

To run `scripts/query_reranked.py` (or anything using `CrossEncoderReranker`
for real), also install the optional reranking extra:

```bash
pip install -e ".[rerank]"
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
- [x] Dense retrieval (cosine nearest-neighbor, top-k, source provenance)
- [x] Sparse (BM25) retrieval (shared tokenizer, top-k, source provenance)
- [x] Hybrid retrieval (weighted Reciprocal Rank Fusion of dense + sparse)
- [x] Reranking (cross-encoder reorders top 20 hybrid candidates to a final top 5)
- [x] Grounded generation (answer from reranked evidence only, bracketed `[n]` citations)
- [x] Citation semantic verification (per-occurrence LLM-judge support verdicts)
- [x] Deterministic confidence scoring (citation-support verdicts + retrieval-channel agreement; heuristic signal, not calibrated)
- [x] Deterministic abstention policy (fixed precedence + `confidence_threshold`; graceful "I don't know", uncalibrated threshold)
- [ ] Evaluation (and confidence-score / abstention-threshold calibration/tuning)
- [ ] API / dashboard
- [ ] Containerization and portfolio polish
