"""The two kinds of retry, and how each is handled.

* Transient failures (a timeout, a 5xx) raise an exception. LangGraph retries
  the node with backoff when it is added with ``retry_policy=DEFAULT_RETRY``.
* "Try again" decisions (a check not satisfied) are a route in the graph,
  capped per node in state with ``retry_update``.
"""

from typing import Any

from langgraph.types import RetryPolicy

from nexus.state import BaseState

# Retries ConnectionError, 5xx and unknown exceptions; never programming errors
# (ValueError, KeyError, TypeError, ...). Waits about 1s, then 2s, between attempts.
DEFAULT_RETRY = RetryPolicy(initial_interval=1.0, backoff_factor=2.0, max_attempts=3)

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
