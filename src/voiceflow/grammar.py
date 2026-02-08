"""Grammar polishing via Qwen LLM on Apple Silicon (MLX). Lazy-loaded."""

from __future__ import annotations

import logging

from voiceflow.config import GrammarConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Fix grammar and spelling errors in this dictation transcript. "
    "Preserve the original meaning, tone, and any technical terms. "
    "If the text is already correct, return it unchanged. "
    "Do not add explanations. "
    "For mixed Chinese-English text, fix grammar in both languages "
    "while preserving code-switching."
)


class GrammarPolisher:
    """Lazy-loaded LLM grammar corrector using MLX on Apple Silicon."""

    def __init__(self, config: GrammarConfig | None = None) -> None:
        self._config = config or GrammarConfig()
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        """Load the model on first use. Returns True if model is ready."""
        if self._loaded:
            return self._model is not None

        self._loaded = True
        if not self._config.enabled:
            logger.info("Grammar polishing disabled by config")
            return False

        try:
            from mlx_lm import load

            logger.info("Loading grammar model: %s", self._config.model_name)
            self._model, self._tokenizer = load(self._config.model_name)
            logger.info("Grammar model loaded successfully")
            return True
        except Exception:
            logger.exception("Failed to load grammar model: %s", self._config.model_name)
            self._model = None
            self._tokenizer = None
            return False

    def polish(self, text: str) -> str:
        """Apply grammar correction to text.

        Returns the corrected text, or the original text unchanged if:
        - Grammar polishing is disabled
        - The model fails to load or inference errors
        - The output is suspiciously longer than input (hallucination guard)
        """
        if not text:
            return text

        if not self._ensure_loaded():
            return text

        try:
            from mlx_lm import generate

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            result = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._config.max_tokens,
                temp=self._config.temperature,
            )

            # Hallucination guard: discard if output is suspiciously long
            if len(result) > len(text) * self._config.max_output_ratio:
                logger.warning(
                    "Grammar output too long (%d chars vs %d input), discarding",
                    len(result),
                    len(text),
                )
                return text

            return result.strip() if result.strip() else text
        except Exception:
            logger.exception("Grammar inference failed, returning original text")
            return text
