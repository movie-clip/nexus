# Definition of done

Agreed before building (see `research/langgraph-standalone-app.md` §6).
Nexus is an engine first, so the criteria are what the engine promises. Each
promise is proven by a test, not by a demo.

## Engine promises

| Promise | Proven by | Slice |
|---|---|---|
| Conditional routing goes where the state says | routing tests on the `smoke` pipeline | 1 |
| Retries are capped per node in state, and a run that hits the cap ends with the error recorded | `stuck` scenario test, per-node retry test | 1 |
| Bad inputs (unknown key, wrong type, engine field) are refused before the run starts | input validation tests | 1 |
| Unattended runs log progress per node, key names only | progress log test | 1 |
| Pipelines follow LangGraph's application structure: every graph in `langgraph.json` loads in `nexus.run` and in `langgraph dev` | loading test; manual `langgraph dev` check | 2 |
| A node that raises a transient error is retried with backoff, one log line per retry; programming errors are not retried | failure drills via the smoke pipeline's `fail_*` inputs | 2 |
| A failed run returns `status=failed`, the failed node and the error as JSON, keeps the work finished before the failure, logs the traceback once, and exits 3 | failure-report tests, subprocess test | 2 |
| `python -m nexus "<request>" --project P` answers about P, with cost in US dollars | ask pipeline test (scripted model); first live run | A |
| The ask agent cannot change the project: no write tools offered, writes denied, traversal blocked, `.env` unreadable, no shell | agent permission tests (exact error per case) | A |
| Cost counts every model call, including subagents and cache reads and writes; unknown models are never counted as free | cost tests | A |
| A run survives a crash: resume continues, finished nodes are not run again | kill-and-resume test, counting node calls | 3 |
| `run` refuses a thread id that already exists (no piling onto old state); only `resume` continues one | existing-thread test | 3 |
| A human gate pauses the run with no side effects, and `resume --decision` finishes it | approve and reject scenario tests | 4 |
| Every run reports its LLM cost in `cost_usd` | LLM-node test with a fake model | 5 |
| Secrets and account-like values never reach traces | redaction test | 6 |

## Pilot metrics (once a real pipeline exists)

- **Cost per run:** stays at cents per run for a mostly plain-code pipeline.
- **Human interventions:** share of runs that stop at a gate, plus how long they wait.
- **Correctness:** the output matches a known expected result or a manual check.
  Every disagreement is logged during the shadow run.

## Decision point

After slice 5, check whether the graph earns its keep. If the flows are linear
with one pause, plain Python plus a state file would do; switch then.
