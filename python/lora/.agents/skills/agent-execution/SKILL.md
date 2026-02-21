---
name: agent-execution
description: Use for Agents when implementing a plan. Prioritize `just` tasks, allowlisted native commands, and end-to-end completion before final reporting.
---

# Agent Execution Discipline

Follow these rules for every implementation task.

## 1. Tool And Command Discipline

- Prefer native tools first.
- For shell commands, use allowlisted/native commands and keep them minimal.
- Prefer `just` tasks over direct command variants when a `justfile` target exists.
- For experiments, run and poll only through `just run-experiment`, `just poll`, `just poll-raw`, and `just status`.
- Do not run background experiments with direct `uv run python ...`.

## 2. Approval Avoidance

- Avoid command shapes that commonly trigger approvals when an equivalent allowlisted command exists.
    - Especially for edits, these kinds of approval requirements should **Not** be triggered.
        - `cat, heredoc (<<), redirection (>), uv, rm, uv`
- Prefer single, direct commands over complex chained shell constructs.
- Reuse existing project scripts/tasks instead of ad-hoc command composition.

## 3. Execution Persistence

- Start by using the ToDo tool and follow that until completion. 
- Continue until all plan steps are implemented, validated, and summarized.
- Do not pause for “check-in” messages.
- Unblock yourself using best practices and documentation.
- Note down any special actions, decisions etc required during implementation in `notes.md`.

## 4. Completion Contract

- End only after finishing all steps in the plan.
- Validate outcomes with the project’s standard commands (prefer `just test`, `just lint`, or other relevant `just` targets).
- Make sure to test out whatever feature was built end to end in a non-interactive manner.
- Provide the final response only after implementation and validation are done, including:
- What changed.
- What commands were run.
- What remains (only if genuinely blocked).

## 5. Implementation Standards

- **No Fallbacks (Fail Fast):** Never write defensive fallback logic (e.g., catching generic exceptions, guessing default values, or silently handling missing dependencies/configurations). 
- **Explicit Contracts:** If an expected state, key, or configuration is missing, fail immediately and explicitly with a descriptive error (e.g., `ValueError`, `KeyError`). Code must loudly reject invalid or ambiguous conditions instead of guessing.
