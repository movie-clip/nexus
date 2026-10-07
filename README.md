# nexus

A small engine for unattended, durable pipelines built on LangGraph.
Background and the reasons for this design: `research/langgraph-standalone-app.md`.
What "done" means: `DONE.md`.

## Use

```sh
uv sync
uv run pytest
uv run python -m nexus.run --pipeline smoke --set scenario=flaky
```

The run logs one line per finished node to stderr and prints the final state as
JSON to stdout. It exits 0 on success, 1 if the run recorded errors, and 2 on
bad arguments or inputs.

## Layout

```
src/nexus/
  state.py        BaseState (errors, cost_usd, artifacts, retries, verdict) + retry helpers
  registry.py     @register("name") for pipeline graph builders
  run.py          CLI and run_pipeline(); validates inputs, compiles, streams
  pipelines/      one module per pipeline; smoke.py is the engine's own test pipeline
tests/
```

## Adding a pipeline

1. Create `src/nexus/pipelines/<name>.py`.
2. Define `<Name>Input(TypedDict)` with the run inputs, and
   `<Name>State(BaseState, <Name>Input)` with the working fields.
3. Write a builder decorated with `@register("<name>")` that returns an
   **uncompiled** `StateGraph(<Name>State, input_schema=<Name>Input)`. The
   runner compiles it with the checkpointer.
4. Import the module in `pipelines/__init__.py`.

Rules:
- Use plain code for deterministic steps, and LLM nodes only for judgment or wording.
- Handle transient failures with `RetryPolicy` on the node. Model "try again"
  decisions as a route capped with `retry_update`.
- Keep state small: store large data as files and keep only their paths in
  `artifacts`. Never put secrets in state.
