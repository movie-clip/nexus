"""Claude models for nexus agents, and what their tokens cost.

LangGraph and Deep Agents are free; the Claude API calls they make are billed
per token. ``cost_usd`` turns LangChain usage metadata into dollars so every
run can report what it spent.
"""

import logging
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.messages.ai import UsageMetadata

log = logging.getLogger("nexus.models")

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache writes cost 1.25x input


# Anthropic first-party API rates (claude-api reference, cached 2026-09-25).
# Cache reads are 0.1x input unless the reference lists a different rate.
# Order matters: longer IDs first, so "claude-opus-5" doesn't match "claude-opus-5-5".
PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(input=10.00, output=50.00, cache_read=0.25, cache_write=12.50),
    "claude-fable-5": Price(input=10.00, output=50.00, cache_read=1.00, cache_write=12.50),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-opus-5": Price(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
    "claude-opus-4-8": Price(input=5.00, output=25.00, cache_read=0.50, cache_write=6.25),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-sonnet-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_read=0.10, cache_write=1.25),
}
# An unknown model (e.g. one a refusal fallback chose) is priced at the most
# expensive known rate: the reported cost may overstate, never understate.
UNKNOWN_PRICE = max(PRICES.values(), key=lambda p: p.output)


def make_model(model: str = DEFAULT_MODEL, effort: str = "high") -> ChatAnthropic:
    """A ChatAnthropic client with nexus defaults.

    * ``effort`` is set explicitly: Opus 5.5 defaults to ``medium``, and its
      thinking can't be turned off, so effort is the only depth control.
    * Server-side refusal fallbacks are on: if a safety classifier declines a
      request, the API re-runs it on a fallback model instead of stopping.
    * The SDK already retries 429, 5xx and connection errors per API call.
    """
    return ChatAnthropic(
        model=model,
        max_tokens=16000,
        effort=effort,
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


def cost_usd(usage_by_model: dict[str, UsageMetadata]) -> float:
    """Dollar cost of LangChain usage metadata, keyed by model name.

    LangChain's ``input_tokens`` includes cache reads and writes, so those are
    subtracted before the base input rate is applied. Unknown models
    are priced at ``UNKNOWN_PRICE`` with a warning, never counted as free.
    """
    total = 0.0
    for model, usage in usage_by_model.items():
        price = _price_for(model)
        details = usage.get("input_token_details") or {}
        cache_read = details.get("cache_read", 0) or 0
        cache_write = details.get("cache_creation", 0) or 0
        uncached = usage.get("input_tokens", 0) - cache_read - cache_write
        total += (
            uncached * price.input
            + cache_read * price.cache_read
            + cache_write * price.cache_write
            + usage.get("output_tokens", 0) * price.output
        ) / 1_000_000
    return total


def _price_for(model: str) -> Price:
    # Responses may name a dated or fallback model; match on the known prefix.
    for name, price in PRICES.items():
        if model == name or model.startswith(name):
            return price
    log.warning("no price for model %r; using the highest known rate", model)
    return UNKNOWN_PRICE
