# nexus

An engineering assistant built on LangGraph and Deep Agents. You send it a
request in plain text about a project; agents explore the project and answer.
You start every run by hand. Background: `research/langgraph-standalone-app.md`.
What "done" means: `DONE.md`.

## Ask

```sh
# once: put your key in nexus's .env (git-ignored)
echo ANTHROPIC_API_KEY=sk-ant-... > .env

uv run python -m nexus "how does the character controller handle jumping?" --project C:\projects\impact
```

The answer is printed as Markdown; progress, the thread id and the run's cost
in US dollars go to stderr. `--json` prints the full result instead.

Today this is one **read-only** Deep Agent (`pipelines/ask.py`, Claude Opus 5.5):
it can list, read, glob and grep inside `--project` and nothing else. Write
tools are not offered, writes are denied by permission rules anyway, paths
outside the project are blocked, `.env` files are unreadable, and there is no
shell. The project's own `AGENTS.md` / `CLAUDE.md` are loaded when present.

LangGraph and Deep Agents are free and open source; the Claude API calls are
billed per token to your key. Install [ripgrep](https://github.com/BurntSushi/ripgrep)
(`winget install BurntSushi.ripgrep.MSVC`) so searches are fast and skip
`.gitignore`d folders.

The layout follows LangGraph's [application structure](https://docs.langchain.com/oss/python/langgraph/application-structure):
each pipeline is a module exporting a compiled `graph`, listed in `langgraph.json`.
The same graphs work in two ways:

| | Command | Use it for |
|---|---|---|
| **Ask** | `uv run python -m nexus "<request>" --project <path>` | a request in plain text |
| **Run** | `uv run python -m nexus.run --pipeline <name>` | any pipeline, inputs via `--set`, result as JSON |
| **Inspect** | `uv run langgraph dev` | developing: local API and Studio UI to watch and step through runs (in-memory, dev only) |

## Run any pipeline

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
  models.py       Claude model defaults (effort, refusal fallbacks) and cost_usd from token usage
  agents.py       shared Deep Agent pieces: read-only and no-secrets permissions, HideTools
  __main__.py     `python -m nexus "<request>"`
  run.py          the CLI: loads a graph from langgraph.json, validates inputs, runs, reports
  errors.py       InputError
  pipelines/
    ask.py        read-only Deep Agent that answers a request about --project
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
