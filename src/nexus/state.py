"""Base state shared by every pipeline, plus helpers for engine-level fields.

Pipelines extend ``BaseState`` with their own keys. Keep state small: store
large data as files and put only their paths in ``artifacts``. Never put
secrets in state, because state is checkpointed and traced.

Two kinds of retry exist, and they are handled differently:
  * transient failures (an API timeout) raise an exception and are retried by
    LangGraph's ``RetryPolicy`` on the node, with backoff;
  * "try again" decisions (a verify step not satisfied) are a route in the
    graph, capped per node with ``retry_update``.
"""

from operator import add
from typing import Annotated, Any, TypedDict


def merge_dicts[K, V](left: dict[K, V], right: dict[K, V]) -> dict[K, V]:
    return {**left, **right}


class BaseState(TypedDict, total=False):
    # Reducers: nodes return only their delta, and LangGraph combines it.
    errors: Annotated[list[str], add]
    cost_usd: Annotated[float, add]
    artifacts: Annotated[dict[str, str], merge_dicts]
    retries: Annotated[dict[str, int], merge_dicts]  # per node
    # Plain field: the latest write wins. Set by a node, read by the route after it.
    verdict: str


RETRY = "retry"
GIVE_UP = "give_up"


def retry_count(state: BaseState, node: str) -> int:
    return state.get("retries", {}).get(node, 0)


def retry_update(state: BaseState, limit: int, node: str) -> dict[str, Any]:
    """State update for a node that wants another attempt.

    Returns ``verdict=retry`` and bumps the node's retry count while under
    ``limit``. Once the limit is reached it returns ``verdict=give_up`` and
    records an error instead, so the run ends cleanly rather than looping.
    """
    count = retry_count(state, node)
    if count >= limit:
        return {"verdict": GIVE_UP, "errors": [f"{node}: retry limit ({limit}) reached"]}
    return {"verdict": RETRY, "retries": {node: count + 1}}
