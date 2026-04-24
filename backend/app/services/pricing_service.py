"""LLM pricing lookup for per-call cost estimation.

OpenRouter:  fetches /models once per hour, caches pricing per model-id.
             For :free variants, strips the suffix and looks up the paid price.
Gemini:      static table from published pricing page.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Gemini pricing: USD per 1 million tokens (input, output) ─────────────────
# Source: https://ai.google.dev/gemini-api/docs/pricing
_GEMINI_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-preview": (0.15, 0.60),
    "gemini-3-flash-preview":   (0.15, 0.60),   # same Flash tier estimate
    "gemini-2.0-flash":         (0.075, 0.30),
    "gemini-1.5-flash":         (0.075, 0.30),
    "gemini-1.5-pro":           (1.25,  5.00),
}

# ── Helper ────────────────────────────────────────────────────────────────────

def normalize_usage(
    raw: dict | None,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Normalise an OpenRouter or Gemini usage dict → (prompt, completion, total)."""
    if not raw:
        return None, None, None
    prompt = raw.get("prompt_tokens") or raw.get("promptTokenCount")
    completion = raw.get("completion_tokens") or raw.get("candidatesTokenCount")
    total = raw.get("total_tokens") or raw.get("totalTokenCount")
    if prompt is None and completion is None:
        return None, None, None
    p = int(prompt) if prompt is not None else None
    c = int(completion) if completion is not None else None
    t = int(total) if total is not None else ((p or 0) + (c or 0)) or None
    return p, c, t


class PricingService:
    """Thread-safe (read-only after first load) pricing lookup."""

    _CACHE_TTL = 3600  # seconds

    def __init__(self, openrouter_api_key: str = "", openrouter_base_url: str = "https://openrouter.ai/api/v1"):
        self._api_key = openrouter_api_key
        self._base_url = openrouter_base_url.rstrip("/")
        # model_id → (prompt_per_token, completion_per_token) in USD
        self._or_cache: dict[str, tuple[float, float]] = {}
        self._cache_loaded_at: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
    ) -> Optional[float]:
        """Return estimated USD cost, or None if pricing is unavailable."""
        if prompt_tokens is None or completion_tokens is None:
            return None
        if provider == "gemini":
            return self._gemini_cost(model, prompt_tokens, completion_tokens)
        return self._openrouter_cost(model, prompt_tokens, completion_tokens)

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _gemini_cost(self, model: str, prompt: int, completion: int) -> Optional[float]:
        prices = _GEMINI_PRICES.get(model.lower().strip())
        if prices is None:
            return None
        input_usd_per_m, output_usd_per_m = prices
        return (prompt * input_usd_per_m + completion * output_usd_per_m) / 1_000_000

    # ── OpenRouter ────────────────────────────────────────────────────────────

    def _openrouter_cost(self, model: str, prompt: int, completion: int) -> Optional[float]:
        # Strip :free suffix → look up paid-tier price
        lookup_key = model.replace(":free", "").strip("/").strip()
        prices = self._get_or_prices(lookup_key)
        if prices is None:
            return None
        prompt_per_token, completion_per_token = prices
        # Price 0 means truly free — still return 0.0 to distinguish from None (unknown)
        return prompt * prompt_per_token + completion * completion_per_token

    def _get_or_prices(self, model_id: str) -> Optional[tuple[float, float]]:
        self._maybe_refresh()
        return self._or_cache.get(model_id)

    def _maybe_refresh(self) -> None:
        if time.monotonic() - self._cache_loaded_at < self._CACHE_TTL:
            return
        self._load_or_prices()

    def _load_or_prices(self) -> None:
        if not self._api_key:
            # Without a key we can still call the public endpoint but it may be rate-limited.
            # Attempt it anyway; failure is handled gracefully below.
            pass
        try:
            import httpx
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            with httpx.Client() as client:
                resp = client.get(
                    f"{self._base_url}/models",
                    headers=headers,
                    timeout=10.0,
                )
            resp.raise_for_status()
            new_cache: dict[str, tuple[float, float]] = {}
            for m in resp.json().get("data", []):
                mid = m.get("id", "")
                pricing = m.get("pricing") or {}
                try:
                    new_cache[mid] = (
                        float(pricing.get("prompt") or 0),
                        float(pricing.get("completion") or 0),
                    )
                except (ValueError, TypeError):
                    pass
            self._or_cache = new_cache
            logger.info(f"OpenRouter pricing loaded: {len(new_cache)} models cached")
        except Exception as exc:
            logger.warning(f"OpenRouter pricing fetch failed: {exc}; cost will be null")
        finally:
            # Bump timestamp even on failure to avoid hammering the endpoint
            self._cache_loaded_at = time.monotonic()
