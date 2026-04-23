"""LLM client for OpenRouter chat completions."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMResponse:
    """Response from LLM generation."""

    def __init__(self, text: str, model: str, usage: Optional[dict] = None):
        self.text = text
        self.model = model
        self.usage = usage or {}

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage,
        }


class LLMService:
    """Simple OpenRouter LLM client for chat completions."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize LLM service with OpenRouter credentials.

        Args:
            api_key: OpenRouter API key
            model: Model name (e.g. "openai/gpt-oss-120b:free")
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized LLM service: {model}")

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 500) -> LLMResponse:
        """
        Generate an answer using OpenRouter LLM.

        Args:
            system_prompt: System instruction for the model
            user_message: User query/message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated text
        """
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Install with: uv add httpx")
            raise ImportError("httpx is required for OpenRouter LLM calls")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, json=payload, headers=headers, timeout=60.0)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        logger.debug(f"Generated answer from {self.model}")
        return LLMResponse(text=text, model=self.model, usage=usage)
