"""Shared BM25 tokenizer for technical documentation.

Used for both document-side indexing (this step) and, later, query-side
preprocessing (Phase 2) — the same function must be used on both sides for
lexical overlap to work at all.

Rule (versioned as `technical_v1`):

  1. A token is a maximal run of Unicode "word" characters (`\\w`: letters,
     digits, underscore), optionally extended by `.<digits>` groups.
     Underscores are never treated as separators, so identifiers like
     `ERR_AUTH_4017`, `AUTH_TOKEN_TTL`, and `DATABASE_POOL_TIMEOUT` survive
     as one token each — splitting them would make them unsearchable as the
     identifiers they are.
  2. The trailing `(?:\\.[0-9]+)*` extension keeps dotted version numbers
     like `v1.2.3` as one token, while a dot followed by non-digits (as in
     `example.com`, `errors.md`) simply isn't consumed by it, so those
     still split into separate word tokens at the dot — a deliberate,
     simple tradeoff rather than URL/domain-aware parsing.
  3. Tokens are lowercased for case-insensitive matching.
  4. No stemming, lemmatization, or stopword removal — technical
     identifiers and exact error codes must remain exact-match searchable,
     which stemming in particular would break.

This is intentionally a small stdlib `re` pattern, not a heavy NLP
dependency. Any future change to this rule must bump `TOKENIZER_VERSION`,
since index snapshot fingerprints depend on it (a tokenizer change can
silently make an old BM25 snapshot's tokenization inconsistent with newly
tokenized queries).
"""

from __future__ import annotations

import re

TOKENIZER_VERSION = "technical_v1"

_TOKEN_RE = re.compile(r"\w+(?:\.[0-9]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Deterministically tokenize `text` for BM25 indexing/querying."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
