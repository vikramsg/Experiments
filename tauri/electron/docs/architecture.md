# Electron Architecture

This document describes the implemented feature-first, process-aware structure
for the Electron app, including the split Browser + Notes workspace and the
separate OpenCode app that chats with the repo through a read-only local
OpenCode server.

## Code Organization

- `src/app/` is the composition layer.
  - It may import feature entrypoints and root boundary files.
  - It should not absorb feature business logic.
- `src/features/` contains isolated business features.
  - Features may import their own files.
  - Features may import shallow root boundary files.
  - Features must not import other features directly.
- Root `src/*.ts` boundary files exist for cross-cutting contracts that are
  truly application-wide:
  - `src/ipc.ts`
  - `src/workspace-contract.ts`
  - `src/workspace-model.ts`
  - `src/test-setup.ts`
  - `src/opencode-contract.ts`
  - `src/opencode-model.ts`
- `src/types.d.ts` is ambient-only and exists to declare preload-exposed globals.
- `src/features/workspace/shared/` is still valid because it is scoped to the
  workspace feature and does not act as a project-wide runtime bucket.

## File Structure

```text
electron/
|-- README.md                               # Project overview, commands, and code-organization guide
|-- Justfile                                # Local command aliases for development and verification
|-- package.json                            # npm scripts and dependency definitions
|-- forge.config.ts                         # Electron Forge packaging and Vite plugin wiring
|-- vite.main.config.ts                     # Vite config for the main-process bundle
|-- vite.preload.config.ts                  # Vite config for preload bundles
|-- vite.renderer.config.ts                 # Vite config for renderer HTML entrypoints
|-- vitest.config.ts                        # Unit and renderer test configuration
|-- playwright.config.js                    # Playwright Electron E2E configuration
|-- docs/
|   |-- reference.md                        # Stable Electron guidance extracted from research
|   `-- architecture.md                     # Runtime diagrams and file responsibility guide
`-- src/
    |-- app/
    |   |-- main/
    |   |   |-- index.ts                    # App bootstrap, launcher creation, workspace ownership, OpenCode ownership
    |   |   |-- create-launcher-window.ts   # BrowserWindow creation and launcher entry loading
    |   |   |-- create-workspace-window.ts  # BaseWindow + four sibling WebContentsView composition
    |   |   |-- create-opencode-window.ts   # BaseWindow + local OpenCode WebContentsView composition
    |   |   `-- register-ipc.ts             # Central IPC handler registration
    |   |-- preload/
    |   |   |-- launcher.ts                 # Safe launcher preload bridge
    |   |   |-- workspace.ts                # Safe workspace preload bridge
    |   |   `-- opencode.ts                 # Safe OpenCode preload bridge
    |   `-- renderer/
    |       `-- entries/
    |           |-- launcher.html           # Launcher HTML entrypoint
    |           |-- notes.html              # Notes HTML entrypoint
    |           |-- splitter.html           # Splitter HTML entrypoint
    |           |-- browser-chrome.html     # Browser chrome HTML entrypoint
    |           `-- opencode.html           # OpenCode HTML entrypoint
    |-- features/
    |   |-- browser/
    |   |   |-- main/
    |   |   |   |-- browser-session.ts      # URL normalization, browser-state reading, security rules
    |   |   |   `-- browser-session.test.ts # Browser main-process tests
    |   |   `-- renderer/
    |   |       |-- App.tsx                 # Right-side local arrow controls, URL bar, and Go action
    |   |       |-- App.test.tsx            # Browser chrome renderer tests
    |   |       `-- main.tsx                # Browser chrome React entrypoint
    |   |-- launcher/
    |   |   `-- renderer/
    |   |       |-- App.tsx                 # Launcher UI
    |   |       |-- App.test.tsx            # Launcher renderer tests
    |   |       `-- main.tsx                # Launcher React entrypoint
    |   |-- notes/
    |   |   |-- main/
    |   |   |   |-- NoteStore.ts            # Persistent workspace snapshot storage
    |   |   |   `-- NoteStore.test.ts       # NoteStore tests
    |   |   `-- renderer/
    |   |       |-- App.tsx                 # Notes editor UI only
    |   |       |-- App.test.tsx            # Notes renderer tests
    |   |       `-- main.tsx                # Notes React entrypoint
    |   |-- splitter/
    |   |   `-- renderer/
    |   |       |-- SplitterHandle.tsx      # Splitter drag UI
    |   |       |-- SplitterHandle.test.tsx # Splitter renderer tests
    |   |       `-- main.tsx                # Splitter React entrypoint
    |   |-- opencode/
    |   |   |-- main/
    |   |   |   |-- OpenCodeService.ts      # Local OpenCode server lifecycle and prompt orchestration
    |   |   |   `-- OpenCodeService.test.ts # OpenCode service tests
    |   |   `-- renderer/
    |   |       |-- App.tsx                 # OpenCode chat UI
    |   |       |-- App.test.tsx            # OpenCode renderer tests
    |   |       `-- main.tsx                # OpenCode React entrypoint
    |   `-- workspace/
    |       |-- main/
    |       |   |-- WorkspaceController.ts  # Four-view bounds, resize handling, teardown, state publication
    |       |   `-- WorkspaceController.test.ts
    |       `-- shared/
    |           |-- split-layout.ts         # Pure width math for the split workspace
    |           `-- split-layout.test.ts    # Layout math tests
    |-- ipc.ts                              # Root IPC transport boundary
    |-- opencode-contract.ts                # Root renderer-facing OpenCode contract
    |-- opencode-model.ts                   # Root OpenCode state model
    |-- test-setup.ts                       # Root Vitest setup boundary
    |-- types.d.ts                          # Ambient window typings only
    |-- vite-env.d.ts                       # Vite environment types
    |-- workspace-contract.ts               # Root renderer-facing workspace contract
    `-- workspace-model.ts                  # Root workspace snapshot model and persistence helpers
|-- e2e/                                    # Playwright Electron end-to-end tests
```

## Boundary Rule

```text
Allowed
=======

src/app/main/create-workspace-window.ts
  -> ../../features/browser/...
  -> ../../features/notes/...
  -> ../../features/workspace/...
  -> ../../ipc
  -> ../../workspace-model

src/app/main/create-opencode-window.ts
  -> ../../features/opencode/...
  -> ../../ipc
  -> ../../opencode-model

src/features/browser/renderer/App.tsx
  -> ../../../workspace-contract
  -> ../../../workspace-model

src/features/opencode/renderer/App.tsx
  -> ../../../opencode-contract
  -> ../../../opencode-model
```

```text
Forbidden
=========

src/features/browser/renderer/App.tsx
  -> ../../../features/notes/...
  -> ../../../features/workspace/...

src/features/opencode/renderer/App.tsx
  -> ../../../features/browser/...
  -> ../../../features/notes/...
```

## Responsibility Map

- `src/ipc.ts` owns IPC channel names only.
- `src/workspace-contract.ts` owns the renderer-facing workspace API contract.
- `src/workspace-model.ts` owns the canonical workspace snapshot model and the durable-vs-live state split.
- `src/test-setup.ts` owns shared Vitest setup only.
- `src/opencode-contract.ts` owns the renderer-facing OpenCode API contract.
- `src/opencode-model.ts` owns the OpenCode chat state model.
- `src/app/main/create-workspace-window.ts` owns composing the four sibling views, loading local and remote surfaces, and wiring remote browser navigation back into workspace state.
- `src/app/main/create-opencode-window.ts` owns composing the OpenCode app window and wiring service-driven state publication into the local renderer.
- `src/features/workspace/main/WorkspaceController.ts` is the layout authority and publisher of workspace snapshots.
- `src/features/browser/main/browser-session.ts` owns URL normalization, browser navigation-state reading, and remote-browser security policy.
- `src/features/notes/main/NoteStore.ts` persists only durable workspace fields, while browser history availability remains live-only state derived from `webContents`.
- `src/features/opencode/main/OpenCodeService.ts` owns the local OpenCode server lifecycle, session creation, prompt submission, and read-only repo boundary.

## Launcher Overview

```text
Launcher BrowserWindow
   |
   +--> Browser + Notes button
   |      `--> createWorkspaceWindow()
   |
   `--> OpenCode button
          `--> createOpenCodeWindow()
```

## Runtime Overview

```text
+-------------------------------------------------------------------+
| Workspace BaseWindow                                              |
|                                                                   |
| +------------------+ +----------+ +----------------------------+  |
| | Notes view       | | Splitter | | Browser chrome view        |  |
| | local React UI   | | local UI | | local React UI            |  |
| +------------------+ +----------+ +----------------------------+  |
|                                  | Browser content view        |  |
|                                  | remote site in WebContents  |  |
|                                  +----------------------------+  |
|                                                                   |
| main process computes and applies bounds for all four siblings    |
+-------------------------------------------------------------------+
```

## Main Process Call Flow

```text
src/app/main/index.ts
   |
   +--> createLauncherWindow()
   |
   `--> registerIpc()
           |
           +--> openWorkspace()
           |       `--> createWorkspaceWindow()
           |
           `--> openOpenCode()
                   `--> createOpenCodeWindow()
```

## Startup Resilience

```text
createWorkspaceWindow()
   |
   +--> create window + views + controller
   +--> return bundle to index.ts early
   +--> workspaceBundle becomes available to IPC
   +--> renderer loadState() can resolve immediately
   `--> page loads continue asynchronously
```

```text
workspace:get-state
   |
   +--> returns live controller snapshot when workspace is ready
   `--> falls back to DEFAULT_WORKSPACE_SNAPSHOT during startup instead of throwing
```

## Navigation Flow

```text
Browser chrome renderer
   |
   | window.workspace.setBrowserUrl(url) / goBack() / goForward()
   v
src/workspace-contract.ts
   |
   v
src/app/preload/workspace.ts
   |
   v
src/app/main/register-ipc.ts
   |
   +--> setBrowserUrl(url) normalizes and loads the target URL
   +--> goBack()/goForward() delegate to browser history actions
   `--> browser navigation events publish the authoritative URL and history state
```

```text
browserView.webContents
   |
   +--> did-navigate / did-navigate-in-page
   |
   +--> read getURL(), canGoBack(), canGoForward()
   +--> WorkspaceController.setBrowserNavigationState(...)
   +--> publish workspace state to notes/browser chrome renderers
   `--> persist the latest durable workspace snapshot
```

## OpenCode Flow

```text
OpenCode renderer
   |
   | window.opencode.loadState() / sendPrompt(prompt)
   v
src/app/preload/opencode.ts
   |
   v
src/app/main/register-ipc.ts
   |
   +--> OpenCodeService.initialize()
   +--> OpenCodeService.sendPrompt(prompt)
   `--> publish opencode:state updates back to the renderer
```

```text
OpenCodeService
   |
   +--> spawn `opencode serve` in the tauri/ repo scope
   +--> apply a read-only config with the plan agent
   +--> create a session
   +--> submit prompt messages over HTTP
   `--> publish renderer-facing chat state
```

## OpenCode Permission Boundary

The OpenCode app is intentionally narrower than the full CLI experience.

- Allowed: `read`, `glob`, `grep`, `list`, and LSP-backed inspection
- Denied: `edit`, write-style operations, arbitrary `bash`, destructive git, and external directory access
- The renderer never receives raw shell access or unrestricted filesystem handles

## Persistence Flow

```text
notes textarea edit ------------------------------+
                                                   |
browser URL change ----------------------------+   |
browser history/link navigation -----------+ |   |
                                           | |   |
splitter drag ------------------------------+|   |
                                            ||   |
                                            vv   v
                                  WorkspaceController snapshot
                                            |
                                            +--> toPersistedWorkspaceSnapshot(...)
                                            |
                                            v
                                  NoteStore.save(snapshot)
                                            |
                                            v
                        app.getPath('userData')/workspace-state.json
```

## Cleanup Model

```text
workspace window closes
        |
        v
WorkspaceController closed listener
        |
        +--> notesView.webContents.close()
        +--> splitterView.webContents.close()
        +--> browserChromeView.webContents.close()
        `--> browserView.webContents.close()
```
