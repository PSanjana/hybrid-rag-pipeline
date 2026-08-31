# Golden Evaluation Dataset (Phase 4 Step 1)

`golden_qa.jsonl` is a **hand-authored, version-controlled** set of questions over the
committed fictional *Acme Cloud* sample corpus in [`data/sample/`](../sample/), each with
manually-grounded expected truth.

It exists so that later Phase 4 steps can measure retrieval relevance, answer correctness,
faithfulness, citation accuracy, abstention behaviour, and chunking-strategy performance
against a **stable reference**. **No metrics are implemented yet**, and nothing in this
dataset was produced by running the RAG pipeline.

## Purpose and ground rules

- Every expected fact, source file, and identifier was written by **reading the corpus
  documents directly**. The current retrieval/generation system was never consulted to
  decide expected truth.
- **Do not tune this dataset to system output.** If a future benchmark run disagrees with a
  golden case, the case is only changed if re-reading the corpus shows the case was wrong —
  never to make a score look better.
- The corpus is fictional. Treat it as a closed world: a question is *unanswerable* iff the
  answer is absent from **every** document in `data/sample/`.

## File format

One JSON object per line (JSONL, UTF-8, ASCII-escaped). Human-reviewable and diffable.
Blank lines are ignored by the loader. Optional fields are omitted from a line when empty.

### Schema

| field | type | notes |
|---|---|---|
| `id` | string | unique, kebab-case, stable |
| `question` | string | the question as asked |
| `answerability` | enum | `answerable` \| `unanswerable` |
| `question_type` | enum | see *Categories* below |
| `difficulty` | enum | `easy` \| `medium` \| `hard` — see *Difficulty* |
| `requires_multi_document_reasoning` | bool | see *Source-label semantics* below |
| `expected_answer` | string \| null | human-readable reference answer; `null` for unanswerable |
| `expected_facts` | string[] | **required for answerable cases** — atomic claims a correct answer must contain |
| `expected_source_files` | string[] | **required for answerable cases** — the ingestion basenames of the documents *required* to support the complete set of `expected_facts` |
| `expected_identifiers` | string[] | exact technical tokens whose retrieval matters for the case |
| `acceptable_source_files` | string[] | additional documents that are a legitimate supporting/citation source but are **not required** to satisfy the complete golden answer |
| `tags` | string[] | free-form labels for slicing results later |
| `notes` | string \| null | rationale: why these sources, what the ambiguity is, why a topic is absent |

`expected_source_files` and `acceptable_source_files` use the **basename only** (e.g.
`authentication-api.md`, `backup-recovery.html`, `employee-handbook.pdf`) because that is
exactly what the ingestion layer records as `source_file` (`path.name`). The loader rejects
a corpus containing two files that share a basename (e.g.
`domain_a/runbook.md` and `domain_b/runbook.md`) rather than silently picking one — that
would make source identity ambiguous.

### Source-label semantics (enforced)

- `expected_source_files` = documents **required** to support the complete set of golden
  `expected_facts`. `acceptable_source_files` = documents that would also be a legitimate
  citation for the same detail but are **not required** for the complete answer.
- `requires_multi_document_reasoning == false` ⇒ **exactly one** `expected_source_files`
  entry (an answerable non-multi case is, by definition, fully supported by one document;
  every corroborating document belongs in `acceptable_source_files`).
- `requires_multi_document_reasoning == true` ⇒ **at least two** `expected_source_files`,
  each contributing at least one fact no other listed document states, and
  `question_type` must be `multi_document_reasoning`.

### Exact-identifier semantics (enforced)

An answerable `exact_identifier` case must list ≥ 1 `expected_identifiers`, and at least
one of them must appear **verbatim (case-insensitively) in the question** — so the category
genuinely represents an exact lexical-query benchmark, not merely an answer that happens to
contain an identifier. (The separate corpus check that each listed identifier also occurs
in its `expected_source_files` is unchanged.)

### Why there are no chunk IDs

Golden truth is expressed as **source files + atomic facts + identifiers only**. Chunk IDs
are a function of the chunking strategy and its boundaries; Phase 4 later compares the
fixed / recursive / semantic strategies, so any chunk ID stored here would not be stable
across the very comparison this dataset supports. Validation rejects any 64-hex-character
(SHA-256-shaped) token appearing anywhere in a case, **including the `id` and `question`**.
Normal technical identifiers (`ERR_DB_1042`, `AUTH_TOKEN_TTL`, `DATABASE_POOL_TIMEOUT`, …)
are never mistaken for chunk IDs — they are not 64-character lowercase-hex runs.

## Categories (`question_type`)

| value | meaning | target count |
|---|---|---|
| `exact_identifier` | contains a rare exact token (`ERR_DB_1042`, `AUTH_TOKEN_TTL`, …); exercises BM25 | 8–12 |
| `semantic_paraphrase` | wording deliberately does **not** copy the docs; exercises dense retrieval | 8–12 |
| `direct_factual` | a straightforward single-fact policy/product/operations lookup | 8–12 |
| `multi_document_reasoning` | needs facts from ≥ 2 documents to answer completely | 8–12 |
| `overlap_ambiguity` | related info is in several docs but only one (or a subset) is authoritative for the detail asked; exercises reranking + citation correctness | 6–10 |
| `unanswerable_absent` | the information is genuinely absent from every document | ≥ 10 |

An `unanswerable` case always has `question_type = unanswerable_absent`, and vice versa.

## Difficulty

- **EASY** — a single explicit fact stated in one document.
- **MEDIUM** — a paraphrase, ambiguous terminology, or connecting several nearby facts
  within one document or a small set.
- **HARD** — multi-document reasoning, overlapping/competing evidence, or a subtle
  abstention judgment.

## Answerability semantics

- **answerable** — the corpus contains the facts needed; the pipeline is expected to
  produce a grounded substantive answer citing the `expected_source_files`. `expected_answer`
  and `expected_facts` describe *what a correct answer says*, not a required string match.
- **unanswerable** — the corpus genuinely does not contain the information; the pipeline is
  expected to abstain. These cases carry **no** `expected_answer`, `expected_facts`, or
  source files (an unanswerable case must not pretend to have authoritative sources). The
  `notes` field records *why* the topic is absent from every document.

## How expected sources are chosen

For an answerable case, `expected_source_files` lists **only the documents required** to
support the complete set of `expected_facts` — for a non-multi case that is exactly one
document. A document that merely mentions similar terminology, or that corroborates the
answer without being needed for it, goes in `acceptable_source_files` instead, and `notes`
explains the distinction. For a `multi_document_reasoning` case, each listed document
contributes at least one fact that no other listed document states, and the golden
`expected_facts` / `expected_answer` are written so they genuinely require every listed
document (if one document turns out to be sufficient, the case is re-scoped to a single
required source with the other moved to `acceptable_source_files`).

## Loading and validation

`rag_pipeline.evaluation.load_golden_dataset()` parses and per-record validates this file
(schema, enums, answerability invariants, the source-count rule, the exact-identifier rule);
`rag_pipeline.evaluation.validate_dataset()` adds dataset-level and corpus-grounding checks
(≥ 50 cases, ≥ 10 unanswerable, every category present, unique IDs, multi-doc consistency,
no ambiguous duplicate basenames in the corpus, every source basename present under
`data/sample/`, every `exact_identifier` case's identifiers actually occurring in its
sources, and no 64-hex chunk-id-shaped token anywhere — `id` and `question` included).
`scripts/summarize_golden_dataset.py` prints the distribution. None of this runs the RAG
pipeline or needs network / an API key.
