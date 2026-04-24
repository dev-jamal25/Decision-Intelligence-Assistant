"""LLM client: OpenAI direct primary → OpenRouter fallback → Gemini fallback."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# HTTP status codes that indicate a transient upstream failure worth retrying.
_TRANSIENT_STATUS_CODES = {429, 502, 503, 504}


class LLMResponse:
    """Response from LLM generation."""

    def __init__(
        self,
        text: str,
        model: str,
        usage: Optional[dict] = None,
        provider: str = "openai",
        fallback_used: bool = False,
    ):
        self.text = text
        self.model = model
        self.usage = usage or {}
        self.provider = provider
        self.fallback_used = fallback_used

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage,
            "provider": self.provider,
            "fallback_used": self.fallback_used,
        }


class LLMService:
    """
    OpenAI direct primary → OpenRouter fallback → Gemini fallback.

    Each step is only attempted if the previous step raised a transient error.
    Non-transient errors (auth, request shape) propagate immediately.
    """

    def __init__(
        self,
        openai_api_key: str,
        openai_model: str,
        openrouter_api_key: str,
        openrouter_fallback_model: str,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-3-flash-preview",
        gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ):
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_fallback_model = openrouter_fallback_model
        self.openrouter_base_url = openrouter_base_url
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.gemini_base_url = gemini_base_url
        logger.info(f"Initialized LLM service: primary=openai/{openai_model}")
        if openrouter_fallback_model:
            logger.info(f"OpenRouter fallback enabled: {openrouter_fallback_model}")
        if gemini_api_key:
            logger.info(f"Gemini fallback enabled: {gemini_model}")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        # 1. Primary: OpenAI Responses API
        try:
            return self._call_openai_responses(
                system_prompt, user_message, temperature, max_tokens
            )
        except Exception as primary_exc:
            if not self._is_transient(primary_exc):
                raise
            logger.warning(f"OpenAI primary transient error ({primary_exc}); trying OpenRouter fallback.")

        # 2. Secondary: OpenRouter fallback model
        if self.openrouter_fallback_model:
            try:
                import httpx
                result = self._call_openrouter(
                    self.openrouter_fallback_model,
                    system_prompt,
                    user_message,
                    temperature,
                    max_tokens,
                    httpx,
                )
                result.fallback_used = True
                return result
            except Exception as fallback_exc:
                if not self._is_transient(fallback_exc):
                    raise
                logger.warning(f"OpenRouter fallback transient error ({fallback_exc}); trying Gemini.")
        else:
            logger.warning("OpenRouter fallback model not configured; skipping to Gemini.")

        # 3. Tertiary: Gemini
        if not self.gemini_api_key:
            raise RuntimeError(
                "All configured LLM backends failed and Gemini is not configured."
            )
        logger.warning(f"Using Gemini fallback: {self.gemini_model}")
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for Gemini calls. Install with: uv add httpx")
        result = self._call_gemini(
            system_prompt, user_message, temperature, max_tokens, httpx
        )
        result.fallback_used = True
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _call_openai_responses(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required for OpenAI calls. Install with: uv add openai")

        client = OpenAI(api_key=self.openai_api_key)
        response = client.responses.create(
            model=self.openai_model,
            instructions=system_prompt,
            input=user_message,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        text = response.output_text
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        logger.debug(f"OpenAI response from {self.openai_model}")
        return LLMResponse(text=text, model=self.openai_model, usage=usage, provider="openai")

    def _call_openrouter(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        httpx,
    ) -> LLMResponse:
        url = f"{self.openrouter_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client() as client:
            response = client.post(url, json=payload, headers=headers, timeout=60.0)

        if response.status_code in _TRANSIENT_STATUS_CODES:
            raise _TransientHTTPError(response.status_code, response.text)

        response.raise_for_status()

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        logger.debug(f"OpenRouter response from {model}")
        return LLMResponse(text=text, model=model, usage=usage, provider="openrouter")

    def _call_gemini(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        httpx,
    ) -> LLMResponse:
        url = (
            f"{self.gemini_base_url}/models/{self.gemini_model}"
            f":generateContent?key={self.gemini_api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        with httpx.Client() as client:
            response = client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60.0,
            )
        response.raise_for_status()

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        logger.debug(f"Gemini response from {self.gemini_model}")
        return LLMResponse(text=text, model=self.gemini_model, usage=usage, provider="gemini")

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """Return True if the exception signals a transient upstream failure."""
        if isinstance(exc, _TransientHTTPError):
            return True
        try:
            import httpx
            if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                return True
        except ImportError:
            pass
        try:
            import openai
            if isinstance(
                exc,
                (
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ),
            ):
                return True
        except ImportError:
            pass
        return False


class _TransientHTTPError(Exception):
    """Raised when a provider returns a transient HTTP error code."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {body[:200]}")
