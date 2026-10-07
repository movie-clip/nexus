"""Optional tracing of runs to a local Langfuse (see README, "Tracing").

Tracing is on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (nexus
reads them from its .env); LANGFUSE_BASE_URL says where Langfuse runs. Each
start or resume of a run becomes one trace: every node, model call and tool
call inside it, the specialists' included. A run and its resumes share a
Langfuse session named after the thread id.

Traces hold what the agents read and wrote, so they stay on this machine.
nexus's own secrets are masked before anything is sent, and a Langfuse that
is down or unreachable never fails a run: spans are exported in the
background, and failures only log a warning.
"""

import os
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD", re.IGNORECASE)
MASK = "[secret]"


def enabled() -> bool:
    return all(os.environ.get(k) for k in KEYS)


def secret_values() -> list[str]:
    """Values of nexus's secrets: names in its .env that look like a key,
    secret, token or password, and anything ANTHROPIC_*. Longest first, so a
    secret that contains another is masked whole."""
    names = {n for n in dotenv_values(ENV_FILE) if SECRET_NAME.search(n)}
    names |= {n for n in os.environ if n.upper().startswith("ANTHROPIC_")}
    values = {os.environ.get(n) or dotenv_values(ENV_FILE).get(n) or "" for n in names}
    return sorted((v for v in values if len(v) >= 8), key=len, reverse=True)


def masker(secrets: list[str]):
    """A Langfuse export hook that replaces each secret in every span attribute.

    It runs on the final, serialized attributes, so it also covers values that
    were objects (LangChain messages, LangGraph commands) when recorded. If it
    raised, Langfuse would drop the batch rather than send it unmasked.
    """
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, MASK)
            return value
        if isinstance(value, list | tuple):
            return type(value)(scrub(v) for v in value)
        return value

    def mask_spans(*, params: Any) -> MaskOtelSpansResult:
        patches = {}
        for identifier, span in params.spans.items():
            changed = {}
            for key, value in span.attributes.items():
                if (masked := scrub(value)) != value:
                    changed[key] = masked
            if changed:
                patches[identifier] = OtelSpanPatch(set_attributes=changed)
        return MaskOtelSpansResult(span_patches=patches)

    return mask_spans


def run_config(thread_id: str, pipeline: str, resumed: bool, **client_options: Any) -> dict:
    """Callbacks and metadata to add to a run's config; empty when tracing is off.

    ``client_options`` go to the Langfuse client (the tests pass an in-memory
    span exporter).
    """
    if not enabled():
        return {}
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    # The first client made for a key is the one the handler uses; it carries the mask.
    Langfuse(public_key=public_key, mask_otel_spans=masker(secret_values()), **client_options)
    return {
        "callbacks": [CallbackHandler(public_key=public_key)],
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_trace_name": f"{pipeline} (resume)" if resumed else pipeline,
            "langfuse_tags": [pipeline],
        },
    }


def flush() -> None:
    """Send what is buffered before the process exits (a short-lived CLI)."""
    if not enabled():
        return
    from langfuse import get_client

    get_client(public_key=os.environ["LANGFUSE_PUBLIC_KEY"]).flush()
