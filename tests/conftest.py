from pathlib import Path

import pytest

from nexus.pipelines import smoke
from nexus.retry import DEFAULT_RETRY

ROOT = Path(__file__).resolve().parents[1]

# Same attempts as the real default policy, without the backoff sleeps.
FAST_RETRY = DEFAULT_RETRY._replace(initial_interval=0.0, jitter=False)


@pytest.fixture(autouse=True)
def isolated_runs_db(tmp_path, monkeypatch):
    """Never touch the real runs/nexus.sqlite; subprocesses inherit this too."""
    path = tmp_path / "runs.sqlite"
    monkeypatch.setenv("NEXUS_RUNS_DB", str(path))
    # No tracing unless a test turns it on (test_tracing.py). Set to empty
    # rather than removed: the CLI tests' subprocesses load nexus's .env, which
    # doesn't override a variable that is already set.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    return path


@pytest.fixture
def smoke_graph():
    return smoke.build(retry_policy=FAST_RETRY).compile()


@pytest.fixture
def in_repo_root(monkeypatch):
    """The CLI reads ./langgraph.json, like the langgraph CLI does."""
    monkeypatch.chdir(ROOT)
