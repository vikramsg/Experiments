# Electron Architecture

This document describes the implemented feature-first, process-aware structure
for the Electron app, including the split Browser + Notes workspace and the
OpenCode + Browser app that chats with the repo through a read-only local
OpenCode server plus a main-process-owned browser MCP tool.

## Code Organization

- `src/app/` is the composition layer.
  - It may import feature entrypoints and root boundary files.
  - It should not absorb feature business logic.
- Services, adapters, and transports should be owned by the domain they expose.
  - Browser inspection and browser MCP live with the browser or app-main side.
  - Browser history and URL autocomplete live with the browser side.
  - OpenCode process lifecycle lives with OpenCode.
  - Cross-domain wiring belongs in `app/*`.
- `src/features/` contains isolated business features.
  - Features may import their own files.
  - Features may import shallow root boundary files.
  - Features must not import other features directly.
- Root `src/*.ts` boundary files exist for cross-cutting contracts that are
  truly application-wide:
  - `src/browser-contract.ts`
  - `src/browser-model.ts`
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
    |   |   |-- browser-host.ts             # Shared browser chrome/content composition for both apps
    |   |   |-- OpenCodeBrowserController.ts # OpenCode + Browser split layout authority
    |   |   |-- create-workspace-window.ts  # BaseWindow + four sibling WebContentsView composition
    |   |   |-- create-opencode-window.ts   # BaseWindow with OpenCode left and browser right
    |   |   `-- register-ipc.ts             # Central IPC handler registration
    |   |-- preload/
    |   |   |-- launcher.ts                 # Safe launcher preload bridge
    |   |   |-- browser.ts                  # Safe browser chrome preload bridge
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
    |   |   |   |-- BrowserMcpServer.ts      # Local MCP server that exposes browser inspection tools
    |   |   |   |-- BrowserMcpServer.test.ts # Browser MCP server tests
    |   |   |   |-- BrowserHistoryStore.ts   # Shared last-10 URL autocomplete history
    |   |   |   `-- BrowserHistoryStore.test.ts
    |   |   |   |-- browser-context.ts      # Current URL + screenshot capture for browser inspection
    |   |   |   `-- browser-context.test.ts # Browser context capture tests
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
    |-- browser-contract.ts                 # Root renderer-facing browser contract
    |-- browser-model.ts                    # Root browser navigation model
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
- `src/browser-contract.ts` owns the renderer-facing browser chrome API contract.
- `src/browser-model.ts` owns browser navigation state shared across browser-backed windows.
- `src/workspace-contract.ts` owns the renderer-facing workspace API contract.
- `src/workspace-model.ts` owns the canonical workspace snapshot model and the durable-vs-live state split.
- `src/test-setup.ts` owns shared Vitest setup only.
- `src/opencode-contract.ts` owns the renderer-facing OpenCode API contract.
- `src/opencode-model.ts` owns the OpenCode chat state model.
- `src/app/main/browser-host.ts` owns browser chrome/content view composition that can be reused by multiple app windows.
- `src/app/main/OpenCodeBrowserController.ts` owns the draggable left/right split for the OpenCode launcher.
- `src/app/main/create-workspace-window.ts` owns composing Browser + Notes and wiring browser navigation back into workspace state.
- `src/app/main/create-opencode-window.ts` owns composing the OpenCode left pane with the browser right pane and wiring service-driven state publication into the local renderer.
- `src/features/workspace/main/WorkspaceController.ts` is the layout authority and publisher of workspace snapshots.
- `src/features/browser/main/browser-session.ts` owns URL normalization, browser navigation-state reading, and remote-browser security policy.
- `src/features/browser/main/browser-context.ts` owns live browser URL inspection and `capturePage()`-based screenshot capture.
- `src/features/browser/main/BrowserHistoryStore.ts` owns shared browser URL autocomplete history and persistence.
- `src/features/browser/main/BrowserMcpServer.ts` owns the localhost MCP endpoint that exposes browser tools to OpenCode.
- `src/features/notes/main/NoteStore.ts` persists only durable workspace fields, while browser history availability remains live-only state derived from `webContents`.
- `src/features/opencode/main/OpenCodeService.ts` owns the local OpenCode server lifecycle, session creation, prompt submission, MCP configuration, and read-only repo boundary.

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

## Browser Flow

```text
Browser chrome renderer
   |
   | window.browser.setBrowserUrl(url) / goBack() / goForward()
   v
src/browser-contract.ts
   |
   v
src/app/preload/browser.ts
   |
   v
src/app/main/register-ipc.ts
   |
   `--> sender-specific BrowserHost selected by app/main composition
```

## Browser History Flow

```text
browser navigation or URL submit
   |
   v
BrowserHistoryStore.remember(url)
   |
   +--> dedupe
   +--> keep last 10
   +--> persist to userData
   `--> publish shared autocomplete suggestions to both browser surfaces
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
   +--> register the browser MCP server and allow `browser_*`
   +--> create a session
   +--> inject a no-reply instruction telling the model to call the browser tool when the user asks what it sees in the browser
   +--> verify browser MCP connection status before browser-aware mode is considered ready
   +--> submit prompt messages over HTTP
   `--> publish renderer-facing chat state
```

## Browser MCP Flow

```text
OpenCode model
   |
   | calls browser_browser_context_current
   v
OpenCode MCP client
   |
   v
BrowserMcpServer (Electron main)
   |
   +--> current browser host for the OpenCode window
   +--> browserView.webContents.getURL()
   +--> browserView.webContents.capturePage()
   `--> return text + image attachment
```

## OpenCode Permission Boundary

The OpenCode app is intentionally narrower than the full CLI experience.

- Allowed: `read`, `glob`, `grep`, `list`, and LSP-backed inspection
- Allowed: `browser_*` MCP browser inspection tools
- Denied: `edit`, write-style operations, arbitrary `bash`, destructive git, and external directory access
- The renderer never receives raw shell access or unrestricted filesystem handles
- Recent browser URL history is not part of MCP context; OpenCode receives only the current browser tool result on demand.

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
