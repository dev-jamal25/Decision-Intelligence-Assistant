import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class OpenRouterEmbedder:
    """Wrapper around OpenRouter embedding API."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized OpenRouter embedder: {model}")

    def embed(self, texts: list[str]) -> list[list[float]]:

        if not texts:
            return []

        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed. Install with: uv add httpx")
            raise ImportError("httpx is required for OpenRouter embeddings")

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, json=payload, headers=headers, timeout=30.0)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        logger.debug(f"Generated {len(embeddings)} embeddings (model: {self.model})")
        return embeddings


def get_embedder(api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1") -> OpenRouterEmbedder:
    """
    Create embedder instance (not cached to allow different credentials).

    Args:
        api_key: OpenRouter API key
        model: Model name
        base_url: API base URL

    Returns:
        OpenRouterEmbedder instance
    """
    if not api_key:
        raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY in .env")
    return OpenRouterEmbedder(api_key=api_key, model=model, base_url=base_url)
