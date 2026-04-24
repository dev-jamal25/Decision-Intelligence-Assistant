"""LLM client for OpenRouter chat completions with Gemini + tertiary fallback."""

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
        provider: str = "openrouter",
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
    OpenRouter primary → OpenRouter fallback → Gemini tertiary.

    Each step is only attempted if the previous step raised a transient error.
    Non-transient errors (auth, request shape) propagate immediately.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-3-flash-preview",
        gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        fallback_model: str = "",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.gemini_base_url = gemini_base_url
        self.fallback_model = fallback_model
        logger.info(f"Initialized LLM service: primary={model}")
        if fallback_model:
            logger.info(f"OpenRouter fallback enabled: {fallback_model}")
        if gemini_api_key:
            logger.info(f"Gemini tertiary fallback enabled: {gemini_model}")

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for LLM calls. Install with: uv add httpx")

        # 1. Primary: OpenRouter main model
        try:
            return self._call_openrouter(
                self.model, system_prompt, user_message, temperature, max_tokens, httpx
            )
        except Exception as primary_exc:
            if not self._is_transient(primary_exc):
                raise
            logger.warning(f"OpenRouter primary transient error ({primary_exc}); trying OpenRouter fallback.")

        # 2. Secondary: OpenRouter fallback model
        if self.fallback_model:
            try:
                result = self._call_openrouter(
                    self.fallback_model, system_prompt, user_message, temperature, max_tokens, httpx
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
        logger.warning(f"Using Gemini tertiary fallback: {self.gemini_model}")
        result = self._call_gemini(
            system_prompt, user_message, temperature, max_tokens, httpx
        )
        result.fallback_used = True
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _call_openrouter(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        httpx,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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
        try:
            import httpx
            if isinstance(exc, _TransientHTTPError):
                return True
            if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                return True
        except ImportError:
            pass
        return False


class _TransientHTTPError(Exception):
    """Raised when a provider returns a transient HTTP error code."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {body[:200]}")
