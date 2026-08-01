# BoundaryRepro v0.6.0

BoundaryRepro is a stateful repository issue repair agent built with LangGraph.
It accepts a constrained public task, reproduces the failure in a disposable
workspace, investigates the repository with generic tools, and makes a bounded
number of source-edit attempts before accepting a behaviorally verified result.

## 1. What the project is

The project demonstrates the engineering boundaries needed around a coding
agent:

- strict task and model-output schemas;
- concurrent read-only investigation with serialized writes;
- resumable SQLite checkpoints;
- provider and long-term-memory isolation;
- behavior-based verification before completion or memory writes;
- bounded iterative repair with deterministic rollback and no-progress guards;
- auditable tool, provider, and verifier traces.

The primary and only installed command is `boundary-repair`.

## 2. Thirteen-node bounded repair graph

```mermaid
flowchart LR
    A["load_task"] --> B["reproduce_failure"]
    B -->|baseline fails| C["retrieve_memory"]
    B -->|otherwise| K["finalize"]
    C --> D["plan"]
    D --> E["dispatch_read_workers"]
    E --> F["read_worker via Send"]
    F --> G["aggregate"]
    G --> H["patch"]
    H -->|applied with new diff| I["verify_candidate"]
    H -->|recoverable apply/no-progress failure| R["prepare_retry"]
    I -->|public + regression pass| V["verify_hidden"]
    I -->|recoverable failure| R
    R -->|budget and deadline remain| D
    R -->|exhausted / rollback failed| K
    V -->|accepted| J["commit_memory"]
    V -->|hidden failure, no retry| K
    J --> K
```

The compiled graph contains exactly these 13 nodes:

1. `load_task`
2. `reproduce_failure`
3. `retrieve_memory`
4. `plan`
5. `dispatch_read_workers`
6. `read_worker`
7. `aggregate`
8. `patch`
9. `verify_candidate`
10. `prepare_retry`
11. `verify_hidden`
12. `commit_memory`
13. `finalize`

`max_patch_attempts` (default 3, allowed 1-5) is separate from transient
Provider request retries. Recoverable patch-application, public-test, and
regression-test failures are recorded as structured feedback, rolled back to
the clean public template, and routed through an explicit graph edge to a new
plan. Provider failure, a passing baseline, deadline exhaustion, hidden-test
failure, unsafe rollback, or an exhausted attempt budget routes to a distinct
non-completed terminal state.

## 3. TaskSpec and repository tools

A task is a strict JSON object with no answer-bearing fields:

```json
{
  "task_id": "public-task-id",
  "issue_text": "Public issue report and expected behavior.",
  "repository": "repository",
  "test_command": ["{python}", "-m", "unittest", "tests.test_public", "-v"],
  "regression_command": ["{python}", "-m", "unittest", "discover", "-s", "tests", "-v"],
  "editable_paths": ["src/package"],
  "timeout": 30
}
```

The generic tool contract contains:

- `read_issue`
- `list_files`
- `search_code`
- `read_file`
- `run_tests`
- `apply_patch`
- `show_diff`
- `submit_solution`

There is no arbitrary shell tool and no task-specific tool. File access is
rooted in a copied workspace. A patch must be one exact replacement inside an
allowlisted source path; tests, traversal paths, symlinks, dependency
directories, and `.git` are not editable.

## 4. Send concurrency and workspace serialization

`dispatch_read_workers` creates independent `read_worker` branches with
LangGraph `Send`. Evidence and trace lists use append reducers, while a shared
`asyncio.Semaphore` enforces the configured read concurrency.

Every workspace mutation and test command acquires the same workspace lock.
`apply_patch`, public tests, regression tests, verifier-only tests, and
filesystem rollback therefore cannot race one another, even if callers
schedule them concurrently. Rollback uses no shell or VCS command: it restores
only the disposable workspace from the immutable public task template, records
pre/post hashes and changed paths, and requires an empty post-rollback diff.

## 5. Checkpoints, pause/resume, and memory

Runtime state is stored under `.boundary_state/`:

```text
.boundary_state/
  repair-checkpoints.sqlite
  repair-memory.sqlite
  repair-workspaces/<thread-id>/
  exports/
```

Checkpoints persist the provider/model identity, run configuration, task,
workspace and baseline hashes, current diff, evidence, tool trace, deadline,
patch attempt, structured feedback and history, prior diff hashes, rollback
count, pause timestamp, and next graph node. Resume validates provider identity,
configuration, task content, repository hashes, workspace location, and diff
before continuing. Completed reads and attempts are not rerun. Older checkpoint
configurations that omit `max_patch_attempts` receive its Pydantic default
without bypassing any identity or hash validation.

Wall-clock time spent intentionally paused does not consume the active run
deadline. Resume adds only the measured pause duration to the existing absolute
deadline.

Long-term memory is written only after all behavioral checks pass. A
non-scripted provider cannot retrieve memory written by the scripted provider,
and memory from the current `task_id` is never reinjected into that task.

## 6. Evaluation isolation and behavioral verification

Candidate verification checks only provider-safe observable evidence:

- the public test fails before the patch;
- the source diff is non-empty;
- every changed path is legal;
- the public test passes after the patch;
- regression tests pass;

Only then does `verify_hidden` run verifier-only tests and call
`submit_solution`. Hidden output is never placed in Provider context or the
public trace; the trace records only status, return code, duration, and output
hash. A hidden failure is terminal and cannot drive another repair attempt.
Long-term memory remains gated on full public, regression, and hidden success.

Every applied patch also receives a deterministic diff SHA-256. Repeating a
previous diff is classified as `no_progress`, skips duplicate behavioral
verification, and consumes only the remaining bounded attempt budget.

It does not compare against a stored answer or inspect diagnosis keywords.
Verifier-only tests are outside the agent tool boundary, but they are not
secret from readers once this GitHub repository is public.

`evaluation_eligible=true` requires a non-scripted provider and zero injected
memory records. A memory-assisted run may complete, but it is marked ineligible
for a clean model evaluation.

## 7. Install and run the scripted demonstration

Python 3.10 through 3.12 are supported:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the bundled clean-room task deterministically:

```powershell
boundary-repair run `
  --task benchmarks\blind-python-dotenv-207\task.json `
  --thread-id scripted-demo `
  --brain scripted `
  --max-patch-attempts 3 `
  --state-dir .boundary_state\scripted-demo `
  --json-out .boundary_state\exports\stateful_repair_trace.json
```

The scripted provider demonstrates orchestration, safety, persistence, and
verification. It is not a language model and its result is not an LLM score.
The checked-in scripted sample goes further: it is an actual offline run with
a deterministic test-only Provider that deliberately proposes a failing first
candidate, records public/regression feedback, rolls the workspace back to an
empty diff, replans, and succeeds on patch attempt two. This proves the bounded
retry plumbing deterministically; it is not evidence that a real model can
perform a multi-attempt repair.

Raw exports may contain machine-local absolute paths. `.boundary_state/` is
ignored by Git, so raw output belongs there. The checked-in
`examples/stateful_repair_trace.sample.json` is that recursively sanitized
two-attempt public sample and must not be overwritten directly by a run.

Pause and resume:

```powershell
boundary-repair run `
  --task benchmarks\blind-python-dotenv-207\task.json `
  --thread-id resume-demo `
  --brain scripted `
  --state-dir .boundary_state\resume-demo `
  --pause-after aggregate

boundary-repair resume `
  --thread-id resume-demo `
  --state-dir .boundary_state\resume-demo
```

## 8. Configure Groq for a real-provider run

Set the API key only in the environment:

```powershell
$env:GROQ_API_KEY="your-key"
boundary-repair run `
  --task path\to\task.json `
  --thread-id groq-run-001 `
  --brain groq `
  --model openai/gpt-oss-120b `
  --state-dir .boundary_state\groq-run-001 `
  --json-out .boundary_state\exports\groq-run-001.json
```

`openai/gpt-oss-20b` and `openai/gpt-oss-120b` use Groq Chat Completions
strict JSON Schema Structured Outputs. `aplan` supplies a strict
`RepairPlan` output contract whose `read_tasks` can express only
`list_files` with empty arguments, `search_code` with one string `query`, or
`read_file` with one string `path`. `apatch` supplies the checked strict
`PatchProposal` schema. Both responses are validated locally before they
cross into the runtime.

The provider produces only plan or patch data. It does not register or execute
Groq-native tools: no `tools=` value is sent, and planned `ReadTask` records
are executed later by the LangGraph Harness. This is not a Groq native
function-calling agent. Other Groq models retain an explicit JSON Object Mode
fallback with the output schema in the prompt and the same local validation;
they do not fall back to scripted behavior.

The default model is now `openai/gpt-oss-120b`;
`llama-3.3-70b-versatile` is no longer the default.

There is no silent fallback from Groq to scripted behavior. Provider or schema
failure cannot be marked completed and cannot write long-term memory.

### Real Groq v0.6.0 validation

One clean integration run used `openai/gpt-oss-120b` on the bundled benchmark
and finished with `status=completed`. The Harness dispatched five read workers
with a maximum observed concurrency of three. Four workers succeeded; one
`read_file` worker failed because `tests/.env` did not exist, and that failure
remains visible in the sanitized trace rather than being removed. Public,
regression, and hidden tests all passed. The run used one patch attempt,
performed no rollback, retained `repair_feedback=null`, had `memory_hits=0`
and `memory_assisted=false`, was marked `evaluation_eligible=true`, and
recorded an observed elapsed time of 19.923 seconds.

This real run demonstrates end-to-end v0.6.0 Provider compatibility and that
one failed read worker is isolated from the other workers and final repair. It
was not a multi-attempt Groq repair, and it is not evidence of a success rate
or generalization capability. The deterministic scripted sample above—not the
real Groq result—is the evidence for failure feedback, rollback, replanning,
and second-attempt success. The current behavioral verification gate does not
include lint or static analysis. The recursively sanitized real trace is
checked in as `examples/groq_repair_trace.sample.json`.

Trace events are envelopes around recorded audit data. For Provider and
Verifier events emitted after an operation completes, an outer
`duration_ms=0` can reflect only the sub-millisecond cost of creating that
envelope. Provider audit and hidden-test durations are retained in
`result.duration_ms`; the public/regression `run_tests` wrapper nests the
command result at `result.result.duration_ms`. The samples preserve these
fields exactly as observed rather than rewriting historical timings for
presentation.

## 9. Tests

Run the current repair-agent suite:

```powershell
python -m pytest
```

GitHub Actions runs the same command on Python 3.10, 3.11, and 3.12. Tests do
not require or access the Groq API.

## 10. Current limitations

- BoundaryRepro is not an arbitrary GitHub Issue solver; every run requires a
  strict `TaskSpec` and an already-local public repository template.
- The repository includes one small clean-room benchmark, so no generalization
  rate can be reported.
- Patch application supports one exact replacement, not complex multi-file
  edits.
- The scripted provider is a deterministic demonstration, not an LLM result.
- Real Groq quality, cost, and success rate are not established by offline
  tests.
- Memory retrieval is structured keyword matching rather than vector search.
- Verifier-only tests share the host Python environment and are visible to
  readers of the public repository.
- The runtime does not clone repositories, run Docker, expose arbitrary shell,
  or execute untrusted remote repositories.
