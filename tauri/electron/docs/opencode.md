# OpenCode In This App

```text
Launcher window
   |
   | click "OpenCode"
   v
createOpenCodeWindow(repoRoot)
   |
   +--> BaseWindow
   +--> one local WebContentsView
   +--> preload bridge exposes window.opencode
   |
   v
OpenCode renderer (React)
   |
   | loadState() / sendPrompt(prompt) / onStateChange(...)
   v
preload/opencode.ts
   |
   v
IPC handlers in register-ipc.ts
   |
   v
OpenCodeService
   |
   +--> spawn: opencode serve --hostname 127.0.0.1 --port <free-port>
   +--> cwd: repo root for this app
   +--> env: OPENCODE_CONFIG_CONTENT=<read-only config>
   +--> wait for GET /global/health
   +--> POST /session
   +--> POST /session/:id/message
   |
   v
local OpenCode server
   |
   +--> agent: plan
   +--> repo scope: tauri/ by default
   +--> allowed: read, glob, grep, list, lsp
   `--> denied: edit, bash, task, skill, webfetch, websearch, external_directory
```

## Summary

OpenCode is a separate Electron app window that provides read-only chat over the
local repo through a narrowly scoped preload bridge and a main-process-owned
OpenCode server lifecycle.

The renderer does not get direct shell access, direct filesystem access, or a
raw server handle. It can only load the current state, send a prompt, and
subscribe to state updates. The main process owns everything else: starting the
server, creating the session, enforcing the repo root, and publishing chat
state back to the UI.

## What Exists Today

- The launcher opens OpenCode as its own window, separate from the Browser +
  Notes workspace.
- The OpenCode window uses `BaseWindow` with a single local
  `WebContentsView`.
- The preload bridge exposes `window.opencode`.
- The main process starts `opencode serve` on localhost, creates a session, and
  submits prompts on the renderer's behalf.
- The OpenCode config is injected through `OPENCODE_CONFIG_CONTENT` with a
  deny-by-default permission map.
- The repo scope is the `tauri/` directory by default, but can be overridden by
  environment variable.

## Window And Process Architecture

The main app bootstrap lives in `src/app/main/index.ts`.

- On startup, Electron creates the launcher window.
- When the user opens OpenCode, `openOpenCode()` calls
  `createOpenCodeWindow(resolveOpenCodeRepoRoot())`.
- The repo root defaults to `path.resolve(__dirname, '../../..')`, which points
  at the `tauri/` repo in this project layout.
- If `ELECTRON_OPENCODE_REPO_ROOT` is set, that value replaces the default repo
  root.

`createOpenCodeWindow()` in `src/app/main/create-opencode-window.ts` builds the
OpenCode app surface.

- It creates a `BaseWindow` titled `OpenCode`.
- It creates one `WebContentsView` with:
  - `sandbox: true`
  - `contextIsolation: true`
  - `nodeIntegration: false`
  - preload script `opencode.js`
- It creates one `OpenCodeService` instance bound to the repo root.
- It subscribes to service state changes and forwards them to the renderer over
  IPC.
- After the renderer finishes loading, it sends the initial state and starts
  service initialization.

This means the renderer stays simple: it renders chat UI and asks the preload
bridge for state changes, but it does not own any server lifecycle behavior.

## Renderer Boundary

The public renderer contract is defined in `src/opencode-contract.ts`.

The renderer gets exactly three methods on `window.opencode`:

- `loadState()`
- `sendPrompt(prompt)`
- `onStateChange(listener)`

The preload implementation in `src/app/preload/opencode.ts` forwards those
calls through Electron IPC.

That boundary is intentionally narrow.

- The renderer cannot spawn processes.
- The renderer cannot call the OpenCode HTTP server directly.
- The renderer cannot access arbitrary files or directories.
- The renderer cannot invoke unrestricted shell commands.

## State Model

The renderer-facing data model lives in `src/opencode-model.ts`.

Current state shape:

- `status`: `idle`, `connecting`, `ready`, `responding`, or `error`
- `repoRoot`: current repo scope used by the service
- `sessionId`: current OpenCode session id, or `null`
- `messages`: chat messages shown in the UI
- `error`: latest error string, or `null`

The default state includes a system welcome message telling the user that the
chat is read-only and intended for questions about files, architecture, and
repo behavior.

## Configuration

The main-process OpenCode configuration is built in
`src/features/opencode/main/OpenCodeService.ts` by `buildOpenCodeConfig()`.

Current config behavior:

- `$schema`: `https://opencode.ai/config.json`
- `default_agent`: `plan`
- `share`: `disabled`
- permissions: deny everything by default, then explicitly allow a small set of
  read-only capabilities

Allowed tools today:

- `read`
- `glob`
- `grep`
- `list`
- `lsp`

Explicitly denied tools today:

- `edit`
- `bash`
- `task`
- `skill`
- `webfetch`
- `websearch`
- `todoread`
- `todowrite`
- `external_directory`

This keeps the Electron OpenCode app much narrower than the full CLI
experience. It is designed to inspect the repo, not to modify it.

## Server Lifecycle

`OpenCodeService` owns all server lifecycle and prompt orchestration.

Initialization flow:

1. `initialize()` moves the UI state to `connecting`.
2. If mock mode is enabled, it creates a fake session immediately.
3. Otherwise it finds a free localhost port.
4. It spawns `opencode serve --hostname 127.0.0.1 --port <port>`.
5. It sets `cwd` to the repo root so the server is scoped to that repo.
6. It injects `OPENCODE_CONFIG_CONTENT` into the environment.
7. It polls `GET /global/health` until the server is ready.
8. It creates a session with `POST /session`.
9. It stores the returned session id and moves the UI state to `ready`.

Prompt flow:

1. `sendPrompt(prompt)` trims and ignores empty input.
2. It ensures initialization has completed.
3. It appends the user message to state and moves to `responding`.
4. It posts to `POST /session/:id/message`.
5. The request body uses agent `plan` and sends the prompt as a text part.
6. The service extracts plain-text assistant output from the response parts.
7. It appends the assistant reply and moves state back to `ready`.

If the request fails, the service moves state to `error` and records the error
message.

## IPC Flow

The renderer talks only through IPC.

`src/app/main/register-ipc.ts` currently wires these handlers:

- `opencodeGetState`
  - returns a default state if the OpenCode window is not open yet
  - otherwise initializes the service if needed and returns the current state
- `opencodeSendPrompt`
  - requires the OpenCode window to be open
  - forwards the prompt to `OpenCodeService.sendPrompt()`

The main process also pushes live state changes back to the renderer through
the `opencodeState` event channel.

So the actual control flow is:

- renderer calls `window.opencode.loadState()`
- preload forwards to IPC
- main process initializes `OpenCodeService`
- service publishes state changes
- `createOpenCodeWindow()` forwards those changes back to the renderer

## Repo Scope

The repo scope matters because OpenCode is intentionally local to this app's
repository.

Current behavior:

- default scope: the `tauri/` repo in this project
- override: `ELECTRON_OPENCODE_REPO_ROOT`
- external directory access: denied by config

In practice, this means the app is built to answer questions about this repo as
it exists on disk, not the wider filesystem.

## Mock Mode

There is a test-friendly mock path in `OpenCodeService`.

- `ELECTRON_OPENCODE_MOCK=1` switches initialization and prompt handling to a
  fake in-memory implementation
- the service returns a fixed mock session id
- prompt replies are synthetic strings rather than real server responses

This is used by the Electron end-to-end tests so UI behavior can be exercised
without depending on a real local OpenCode server.

## Failure And Cleanup Behavior

Current failure handling:

- If the server never becomes healthy, initialization fails and state moves to
  `error`.
- If the child process exits unexpectedly after startup, the service clears its
  server handle and publishes an error state.
- If a prompt request fails, the service keeps the chat state but marks the app
  as `error`.

Current cleanup behavior:

- When the OpenCode window closes, the app unsubscribes from service listeners.
- The service kills the spawned child process.
- The `WebContentsView` web contents are closed explicitly.

## Relevant Files

These files are the current source of truth for how OpenCode works:

- `README.md`
- `docs/architecture.md`
- `src/app/main/index.ts`
- `src/app/main/create-opencode-window.ts`
- `src/app/main/register-ipc.ts`
- `src/app/preload/opencode.ts`
- `src/features/opencode/main/OpenCodeService.ts`
- `src/opencode-contract.ts`
- `src/opencode-model.ts`

## Notes About Current Gaps

The current implementation is intentionally narrow and already has a few known
follow-up areas noted elsewhere in the repo:

- richer message rendering for markdown and code blocks
- better scroll behavior
- improved prompt sizing and loading feedback
- stronger visual differentiation between message types

Those are UI and experience improvements. They do not change the current core
architecture described in this document.
