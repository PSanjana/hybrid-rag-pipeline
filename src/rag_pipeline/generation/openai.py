"""Production generation provider: OpenAI Chat Completions.

Mirrors `embeddings.openai.OpenAIEmbeddingProvider`'s shape: the client
is constructed once in `__init__` (never at module import), missing
`OPENAI_API_KEY` fails fast there with a clear project-specific error,
and every OpenAI SDK failure is wrapped with the cause chained. Uses the
Chat Completions API (`client.chat.completions.create`) rather than the
newer Responses API for this simple single-turn system+user-prompt use
case -- Chat Completions remains fully supported and is the smaller,
more stable surface for what this provider needs.

Never logs the API key, the prompt, or the generated text -- logging
document content by default is exactly what section 15 of this phase
warns against, so this module logs nothing at all.
"""

from __future__ import annotations

from openai import OpenAI, OpenAIError

from ..config import Settings
from .exceptions import GenerationProviderError

_DEFAULT_TEMPERATURE = 0.0


class OpenAIGenerator:
    """Generator backed by the OpenAI Chat Completions API.

    Requires `settings.openai_api_key`; raises `GenerationProviderError`
    immediately if it's missing, rather than failing on first use.
    `temperature` defaults to 0.0 (deterministic-leaning), appropriate
    for grounded, citation-disciplined answers rather than creative text.
    """

    def __init__(self, settings: Settings, temperature: float = _DEFAULT_TEMPERATURE) -> None:
        if settings.openai_api_key is None:
            raise GenerationProviderError("Generation requires OPENAI_API_KEY to be configured.")
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.generation_model
        self._temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as exc:
            raise GenerationProviderError(
                f"Generation request failed for model {self._model!r}: {exc}"
            ) from exc

        if not response.choices:
            raise GenerationProviderError(
                f"Generation provider returned no choices for model {self._model!r}."
            )

        text = response.choices[0].message.content
        if text is None or not text.strip():
            raise GenerationProviderError(
                f"Generation provider returned an empty response for model {self._model!r}."
            )
        return text
