# nexus

Pipelines built on LangGraph that you start by hand and that run to the end on
their own, with retries, clear failure reports and (soon) resume.
Background: `research/langgraph-standalone-app.md`. What "done" means: `DONE.md`.

The layout follows LangGraph's [application structure](https://docs.langchain.com/oss/python/langgraph/application-structure):
each pipeline is a module exporting a compiled `graph`, listed in `langgraph.json`.
The same graphs work in two ways:

| | Command | Use it for |
|---|---|---|
| **Run** | `uv run python -m nexus.run --pipeline <name>` | doing the work: one run, result as JSON |
| **Inspect** | `uv run langgraph dev` | developing: local API and Studio UI to watch and step through runs (in-memory, dev only) |

## Run

```sh
uv sync
uv run pytest
uv run python -m nexus.run --pipeline smoke --set scenario=flaky
uv run python -m nexus.run --pipeline smoke --set fail_node=step_b --set fail_times=1
```

`--set key=value` sets a run input. Unknown keys and wrong types are refused
before anything runs. Progress goes to stderr: one line per finished node or
retry, plus the traceback once if the run fails. The result (`thread_id`,
`status`, `state`, and `failed_node`/`error` on failure) goes to stdout as JSON.

| Exit | Meaning |
|---|---|
| 0 | completed |
| 1 | completed, but the run recorded errors in state |
| 2 | bad arguments or inputs; nothing ran |
| 3 | failed: a node raised and its retries ran out |

## Layout

```
langgraph.json    graph name -> ./src/nexus/pipelines/<module>.py:graph
src/nexus/
  state.py        BaseState: errors, cost_usd, artifacts, retries, verdict
  retry.py        DEFAULT_RETRY (transient errors) and retry_update (capped "try again" routes)
  run.py          the CLI: loads a graph from langgraph.json, validates inputs, runs, reports
  errors.py       InputError
  pipelines/
    smoke.py      the engine's own test pipeline: routing scenarios and failure drills
tests/
```

## Adding a pipeline

1. Create `src/nexus/pipelines/<name>.py`.
2. Define `<Name>Input(TypedDict)` with the run inputs, and
   `<Name>State(BaseState, <Name>Input)` with the working fields.
3. Build a plain `StateGraph(<Name>State, input_schema=<Name>Input)`, add each
   node with `retry_policy=DEFAULT_RETRY`, and export `graph = build().compile(name="<name>")`
   **without** a checkpointer. `langgraph dev` and `nexus.run` each add their own.
4. Add `"<name>": "./src/nexus/pipelines/<name>.py:graph"` to `langgraph.json`.

Rules:
- Use plain code for deterministic steps, and LLM nodes only for judgment or wording.
- Transient failures (timeouts, 5xx) are retried by `DEFAULT_RETRY`: 3 attempts,
  about 1s then 2s apart. Programming errors (`ValueError`, `KeyError`, ...) are
  not retried. Model "try again" decisions as a route capped with `retry_update`.
- Keep state small: store large data as files and keep only their paths in
  `artifacts`. Never put secrets in state.
- Put secrets in `.env` (git-ignored).
