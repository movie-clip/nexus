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
| A node that raises is retried with backoff (`RetryPolicy`); if it still fails, the CLI reports the thread and exits non-zero without a raw traceback | fault-injection tests | 2 |
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
