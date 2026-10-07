"""Base state shared by every pipeline.

Pipelines extend ``BaseState`` with their own keys. Keep state small: store
large data as files and put only their paths in ``artifacts``. Never put
secrets in state, because state is checkpointed and traced.
"""

from operator import add
from typing import Annotated, TypedDict


def merge_dicts[K, V](left: dict[K, V], right: dict[K, V]) -> dict[K, V]:
    return {**left, **right}


class BaseState(TypedDict, total=False):
    # Reducers: nodes return only their delta, and LangGraph combines it.
    errors: Annotated[list[str], add]
    cost_usd: Annotated[float, add]
    artifacts: Annotated[dict[str, str], merge_dicts]
    retries: Annotated[dict[str, int], merge_dicts]  # per node, see nexus.retry
    # Plain field: the latest write wins. Set by a node, read by the route after it.
    verdict: str
