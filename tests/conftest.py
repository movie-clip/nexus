from pathlib import Path

import pytest

from nexus.pipelines import smoke
from nexus.retry import DEFAULT_RETRY

ROOT = Path(__file__).resolve().parents[1]

# Same attempts as the real default policy, without the backoff sleeps.
FAST_RETRY = DEFAULT_RETRY._replace(initial_interval=0.0, jitter=False)


@pytest.fixture
def smoke_graph():
    return smoke.build(retry_policy=FAST_RETRY).compile()


@pytest.fixture
def in_repo_root(monkeypatch):
    """The CLI reads ./langgraph.json, like the langgraph CLI does."""
    monkeypatch.chdir(ROOT)
