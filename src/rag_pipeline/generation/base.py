"""Generation-provider abstraction: turn (system prompt, user prompt) into text.

A `Generator` is the only thing `generation.service` depends on -- a
production OpenAI provider and offline test doubles are interchangeable
implementations of it. This keeps the generation service independently
testable and never coupled to a specific SDK.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Generator(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return generated text for the given system/user prompts.

        Must return the model's raw text response; empty-response and
        provider-failure handling is the provider implementation's
        responsibility (see `GenerationProviderError`).
        """
        ...
