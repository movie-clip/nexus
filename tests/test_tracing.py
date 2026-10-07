"""Tracing to Langfuse: off without keys, the whole team run when on, secrets
masked, and a Langfuse that is down never fails a run."""

import functools
import time
import uuid

import pytest
from langfuse import get_client
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nexus import tracing
from nexus.run import open_runs_db, run_pipeline
from tests.fakes import ScriptedOrchestrator, decide, final, scripted, tool_call
from tests.test_team_tester import make_graph, project  # noqa: F401 (fixture)

SECRET = "sk-ant-test-0123456789abcdef"


@pytest.fixture
def langfuse_keys(monkeypatch):
    # The SDK keeps one client per public key for the whole process, so each
    # test gets its own key and shuts its client down afterwards.
    public_key = f"pk-lf-test-{uuid.uuid4().hex}"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    yield public_key
    get_client(public_key=public_key).shutdown()


@pytest.fixture
def spans(langfuse_keys, monkeypatch):
    """Capture exported spans in memory instead of sending them."""
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        tracing, "run_config", functools.partial(tracing.run_config, span_exporter=exporter)
    )
    return exporter


def scout_run(project, tmp_path, request):  # noqa: F811 (fixture)
    scout = scripted(tool_call("read_file", file_path="/Jump.cs"), final(f"Speed is 10f. {SECRET}"))
    orch = ScriptedOrchestrator(
        decide("scout", brief="Read /Jump.cs."), decide("finish", answer="Speed is 10f.")
    )
    graph = make_graph(tmp_path, orch, [scout])
    return run_pipeline(
        graph, "trace-1", {"request": request, "project": str(project)}, open_runs_db()
    )


def test_tracing_is_off_without_keys(project, tmp_path):  # noqa: F811
    assert tracing.run_config("t", "team", resumed=False) == {}
    assert scout_run(project, tmp_path, "How fast?").status == "completed"


def test_run_is_traced_down_to_the_specialists_tool_calls(project, tmp_path, spans):  # noqa: F811
    assert scout_run(project, tmp_path, "How fast?").status == "completed"

    finished = spans.get_finished_spans()
    names = {s.name for s in finished}
    assert {"team", "orchestrator", "scout", "read_file"} <= names
    root = next(s for s in finished if s.name == "team")
    assert "trace-1" in root.attributes.values()  # the session: the nexus thread id
    # One trace for the whole run.
    assert len({s.context.trace_id for s in finished}) == 1


def test_secrets_are_masked_before_sending(project, tmp_path, spans):  # noqa: F811
    scout_run(project, tmp_path, f"Is {SECRET} leaked?")
    sent = " ".join(str(v) for s in spans.get_finished_spans() for v in s.attributes.values())
    assert SECRET not in sent
    assert tracing.MASK in sent  # from the request and from the scout's report


def test_secret_values_come_from_env_names_and_anthropic_vars(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("MY_TOKEN=tok-123456789\nLANGFUSE_BASE_URL=http://localhost:3000\n")
    monkeypatch.setattr(tracing, "ENV_FILE", env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.delenv("MY_TOKEN", raising=False)
    assert tracing.secret_values() == [SECRET, "tok-123456789"]  # not the URL


def test_unreachable_langfuse_never_fails_a_run(project, tmp_path, langfuse_keys, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")  # nothing listens there
    started = time.monotonic()
    result = scout_run(project, tmp_path, "How fast?")
    assert result.status == "completed"
    assert time.monotonic() - started < 30
