---
name: gemini-cli-execution
description: Enforce reliable Gemini CLI execution behavior for coding and repo tasks. Use this skill when the agent is drifting into approval-triggering shell patterns instead of native/allowlisted tools, or when the agent pauses mid-implementation to discuss before finishing the plan. Prioritize `just` tasks, allowlisted native commands, and end-to-end completion before final reporting.
---

# Gemini CLI Execution Discipline

Follow these rules for every implementation task.

## 1. Tool And Command Discipline

- Prefer Gemini-native tools first.
- For shell commands, use allowlisted/native commands and keep them minimal.
- Prefer `just` tasks over direct command variants when a `justfile` target exists.
- For experiments, run and poll only through `just run-experiment`, `just poll`, `just poll-raw`, and `just status`.
- Do not run background experiments with direct `uv run python ...`.

## 2. Approval Avoidance

- Avoid command shapes that commonly trigger approvals when an equivalent allowlisted command exists.
- Prefer single, direct commands over complex chained shell constructs.
- Reuse existing project scripts/tasks instead of ad-hoc command composition.
- If a required action still needs approval, defer that step until all non-blocked steps are complete, then request approval once with clear purpose.

## 3. Execution Persistence

- Start by defining an internal step plan, then execute it without stopping mid-way for discussion.
- Continue until all plan steps are implemented, validated, and summarized.
- Do not pause for “check-in” messages unless:
- A true blocker exists (missing requirement, unavailable dependency, approval-gated critical step, or conflicting instructions).
- A blocker is hit:
- State the blocker briefly.
- Complete any remaining unblocked work first.
- Return once with exactly what is blocked and what approval/input is required.

## 4. Completion Contract

- End only after finishing all feasible steps in the plan.
- Validate outcomes with the project’s standard commands (prefer `just test`, `just lint`, or other relevant `just` targets).
- Provide the final response only after implementation and validation are done, including:
- What changed.
- What commands were run.
- What remains (only if genuinely blocked).
