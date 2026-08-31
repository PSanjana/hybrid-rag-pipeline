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

## Evaluation metrics (Phase 4 Step 2)

`rag_pipeline.evaluation` now also defines the **measurement** functions that will
later consume this dataset. **Step 2 defines metrics only — no benchmark has been
run, no strategy has been compared, and no weight or threshold has been tuned.**
No scores are reported anywhere. The benchmark run itself (all cases × all
chunking strategies) is **Phase 4 Step 3**.

The five families are deliberately **orthogonal** — there is no single combined
"RAG score". A correct answer built on failed retrieval stays distinguishable
from a correct answer built on successful retrieval.

**`None` means not-applicable; it never means zero.** `0.0` is a real measurement
(the metric applied and the system scored zero). A metric that genuinely cannot
be computed reports `None` (retrieval source recall on an unanswerable case;
identifier recall when the case lists no identifiers; correctness/faithfulness/
citation metrics when the final policy abstained; an aggregate rate with a zero
denominator).

### 1. Retrieval relevance — `evaluate_retrieval(case, results, k)` (deterministic)

Operates only on each result's `source_file`, `text`, and list position — never
on a native score (cosine similarity/distance, BM25, RRF, reranker). The same
function scores dense, sparse, hybrid, and reranked lists. Let `R` =
`expected_source_files`, `S_k` = the set of distinct `source_file`s in the top `k`.

| metric | definition |
|---|---|
| `required_source_hit_at_k` | `1.0` if `R ∩ S_k ≠ ∅` else `0.0` (`None` for unanswerable) |
| `required_source_recall_at_k` | `|R ∩ S_k| / |R|` — distinct sources; repeats never inflate it |
| `complete_required_source_retrieval_at_k` | `True` iff `R ⊆ S_k` (all required docs present) |
| `reciprocal_rank` | `1 / rank` of the first result (1-based, over the **whole** sequence) whose `source_file ∈ R`; `0.0` if none |
| `identifier_recall_at_k` | distinct `expected_identifiers` occurring (case-insensitive substring, no tokenisation) in ≥ 1 top-`k` chunk's text `/ |expected_identifiers|`; `None` if the case has none |

Hit, Recall, and Complete are kept **separate on purpose**: for a multi-document
case Hit@k can be `1.0` while Complete@k is `False` (only one of two required
documents was found).

**There is no `chunk_precision_at_k`.** The dataset labels required/acceptable
*source documents*, not every relevant chunk, so "every chunk from an expected
document is relevant" would claim a stronger relevance labelling than the
benchmark contains. An unlisted document is likewise not assumed irrelevant.
Required-source recall is therefore **not** chunk precision.

### 2. Answer correctness — `evaluate_correctness(case, final_answer, judge)` (semantic → deterministic)

The golden facts are authoritative. The injected `CorrectnessJudge` is **never
shown retrieved evidence** — a retrieval error cannot contaminate correctness.
It only *classifies*; Python computes every number. It produces **two
orthogonal signals**:

**(a) `expected_fact_score` — required-fact coverage.** Per-fact verdict → score:
`CORRECT → 1.0`, `PARTIALLY_CORRECT → 0.5`, `MISSING → 0.0`, `CONTRADICTED → 0.0`.
Then `expected_fact_score = mean(mapped verdicts over all expected_facts)`.
Per-verdict counts (including `contradicted_count`) are exposed separately. This
is the **authoritative numeric coverage score**. `score` holds the same value
and is kept only for backwards clarity — *the float alone is not a complete
correctness decision.*

**(b) `has_golden_contradiction` — answer-level golden contradiction.** Separately,
the judge inspects the *complete* answer for one or more **material claims that
directly conflict with the supplied golden truth** (the numbered `expected_facts`
and the reference `expected_answer`). Per-fact scoring cannot catch this: an
answer can state every expected fact correctly (`expected_fact_score == 1.0`) and
still add a claim that contradicts the golden truth. The report exposes
`golden_contradiction_count`, `has_golden_contradiction`, and a tuple of
`GoldenContradiction` objects (`contradiction_id`, `claim_text`, `rationale`,
optional `conflicting_fact_ids`). Zero contradictions is the normal case.

**An extra claim that is merely absent from the golden facts is NOT a
contradiction** — the benchmark is not exhaustive, so a statement is only flagged
when it conflicts with truth that can actually be established from the supplied
golden material. This keeps correctness independent of faithfulness: catching an
extra contradicting claim must **not** rely on the faithfulness judge.

**No numeric penalty** for contradictions is applied here — `score` /
`expected_fact_score` stay the pure coverage mean. How to combine the coverage
score with `has_golden_contradiction` into an overall correctness decision is
deferred to later Phase 4 analysis.

**Applicable only** for an ANSWERABLE case whose final decision was `ANSWERED`.
An answerable case that abstained → `None` (the false abstention is the
abstention metric's job, not correctness 0.0). An unanswerable case → `None`
(no golden facts). On any non-applicable path the judge is not called, and both
score fields and the contradiction tuple are empty.

### 3. Faithfulness — `evaluate_faithfulness(question, final_answer, judge)` (semantic → deterministic)

Asks: are the answer's material claims supported by the evidence that was
**supplied to generation** (`GroundedAnswer.evidence`)? The golden expected
answer is **never** consulted. Correctness ≠ faithfulness: an answer can be
faithful to incorrect evidence yet wrong, or correct yet unfaithful.

Per-claim verdict → score: `SUPPORTED → 1.0`, `PARTIALLY_SUPPORTED → 0.5`,
`UNSUPPORTED → 0.0`, `CONTRADICTED → 0.0`. Then
`faithfulness_score = mean(mapped claim verdicts)`. Zero claims for a
substantive answer is **rejected**, never scored as perfect faithfulness.

**Applicable only** when the final decision was `ANSWERED` (the rejected draft
behind an abstention is not the user-facing result). Independent of golden
answerability — an erroneously-answered *unanswerable* case is still scored
against its supplied evidence.

### 4. Citation accuracy — `evaluate_citation_accuracy(case, final_answer)` (deterministic)

No second LLM. Maps the production `CitationVerificationReport` verdicts and
resolves cited evidence chunks to their `source_file`s. `T` = citation
occurrences, `verdict → score` as for faithfulness.

| metric | definition |
|---|---|
| `semantic_citation_support_score` | `mean(mapped verdict over occurrences)` — **independent of the Step 3 confidence score**, which is never read here |
| `fully_supported_citation_rate` | `supported_count / T` |
| `cited_source_golden_match_rate` | of the **distinct** `source_file`s among cited evidence, the fraction in `expected_source_files ∪ acceptable_source_files` — a **golden-source** match, not universal precision |
| `required_source_citation_recall` | distinct `expected_source_files` represented among cited evidence `/ |expected_source_files|`; `None` if the case lists no required sources. Repeated citations to one source never inflate it |

**Applicable only** when the final decision was `ANSWERED`.

### 5. Abstention behaviour — `evaluate_abstention` / `aggregate_abstention` (deterministic)

Grades the **actual policy result** (`FinalAnswer.abstained`) against golden
`Answerability` — **never inferred from the confidence score**. Per case:
`expected_abstain` (golden unanswerable), `actual_abstain`, `decision_correct`,
`false_abstention` (golden answerable but abstained), `false_answer` (golden
unanswerable but answered).

`aggregate_abstention()` rolls these up: `decision_accuracy`,
`answerable_coverage = answered answerable / total answerable`,
`false_abstention_rate = false abstentions / total answerable`,
`unanswerable_abstention_recall = correctly abstained unanswerable / total
unanswerable`, `false_answer_rate = answered unanswerable / total unanswerable`.
Every rate with a zero denominator is `None`.

### Semantic judge architecture

`CorrectnessJudge` / `FaithfulnessJudge` are small protocols (mirroring
`generation.CitationJudge`). `assess_correctness(system_prompt, user_prompt)`
returns a `RawCorrectnessAssessment` — `fact_verdicts` (OUTPUT A: one per
numbered golden fact) **and** `golden_contradictions` (OUTPUT B: material answer
claims that conflict with the golden truth; empty is normal).
`assess_faithfulness(...)` returns raw claim verdicts. The evaluators build the
prompts, call the judge, then **independently re-validate** every result:

- fact verdicts — exact id set `1..N`, enum verdict, non-empty rationale, `bool`
  ids rejected;
- contradictions — contiguous ids `1..M`, no duplicates, `bool` ids rejected,
  non-empty `claim_text` + `rationale`, and any `conflicting_fact_ids` must be
  real ints in `1..N` with no repeats; an empty list is accepted;
- faithfulness claims — contiguous `1..M`, ≥ 1 for a substantive answer.

Any mismatch raises `EvaluationJudgeOutputError` — a malformed result never
surfaces as a raw `KeyError`/`TypeError`. The correctness and faithfulness
**prompts are kept logically separate** and each spells out its scope
(authoritative golden facts, no evidence / evidence only, no golden truth,
untrusted data, classify-don't-score). The correctness prompt additionally
distinguishes OUTPUT A from OUTPUT B and states that a claim merely *absent* from
the golden facts is not a contradiction, and that contradiction detection is not
evidence-support judging (that is faithfulness).

The production `OpenAIEvaluationJudge` implements both protocols via OpenAI
structured outputs (its correctness schema returns both `verdicts` and
`golden_contradictions`; Python still computes every number), using its own
`EVALUATION_JUDGE_MODEL` setting so it can be tuned independently of generation /
citation judging. A missing `OPENAI_API_KEY` fails fast when it is instantiated.
**No automated test makes a network call** — every metric test uses an offline
fake judge.

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
