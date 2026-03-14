# Electron Architecture

This document describes the implemented feature-first, process-aware structure for the Electron workspace app, including the browser-owned renderer surface in the right split.

## File Structure

```text
electron/
|-- README.md                               # Project overview, commands, and verification guide
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
    |   |   |-- index.ts                    # App bootstrap, lifecycle, launcher creation, workspace state ownership
    |   |   |-- create-launcher-window.ts   # BrowserWindow creation and launcher entry loading
    |   |   |-- create-workspace-window.ts  # BaseWindow + four sibling WebContentsView composition
    |   |   `-- register-ipc.ts             # Central IPC handler registration
    |   |-- preload/
    |   |   |-- launcher.ts                 # Safe launcher preload bridge
    |   |   `-- workspace.ts                # Safe workspace preload bridge
    |   `-- renderer/
    |       `-- entries/
    |           |-- launcher.html           # Launcher HTML entrypoint
    |           |-- notes.html              # Notes HTML entrypoint
    |           |-- splitter.html           # Splitter HTML entrypoint
    |           `-- browser-chrome.html     # Browser chrome HTML entrypoint
    |-- features/
    |   |-- browser/
    |   |   |-- main/
    |   |   |   |-- browser-session.ts      # URL normalization and remote-content security rules
    |   |   |   `-- browser-session.test.ts # Tests for browser security/session behavior
    |   |   `-- renderer/
    |   |       |-- App.tsx                 # Right-side local URL bar and Go action
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
    |   |       |-- App.test.tsx            # Notes renderer tests after URL removal
    |   |       `-- main.tsx                # Notes React entrypoint
    |   |-- splitter/
    |   |   `-- renderer/
    |   |       |-- SplitterHandle.tsx      # Splitter drag UI
    |   |       |-- SplitterHandle.test.tsx # Splitter renderer tests
    |   |       `-- main.tsx                # Splitter React entrypoint
    |   `-- workspace/
    |       |-- main/
    |       |   |-- WorkspaceController.ts  # Four-view bounds, resize handling, teardown, state publication
    |       |   `-- WorkspaceController.test.ts
    |       |                                # Workspace controller tests
    |       `-- shared/
    |           |-- split-layout.ts         # Pure width math for the split workspace
    |           `-- split-layout.test.ts    # Layout math tests
    |-- shared/
    |   |-- ipc/
    |   |   `-- channels.ts                 # Shared IPC channel names
    |   |-- test/
    |   |   `-- setup.ts                    # Vitest setup shared across renderer tests
    |   `-- types/
    |       `-- workspace.ts                # Shared workspace snapshot types and defaults
    |-- types.d.ts                          # Global window.launcher and window.workspace typings
    `-- vite-env.d.ts                       # Vite environment types
|-- e2e/                                    # Playwright Electron end-to-end tests
```

## Responsibility Map

- `src/app/main/create-workspace-window.ts` owns composing the four sibling views, registering the workspace bundle early, and then loading the local/remote surfaces asynchronously.
- `src/features/workspace/main/WorkspaceController.ts` is the layout authority for:
  - notes view
  - splitter view
  - browser chrome view
  - browser content view
- `src/features/browser/main/browser-session.ts` owns URL normalization and remote-browser security policy.
- `src/features/browser/renderer/` owns browser-specific local UI only.
- `src/features/notes/renderer/` owns note-taking UI only.
- `src/features/notes/main/NoteStore.ts` persists notes, browser URL, and splitter width into the workspace snapshot.
- `src/shared/types/workspace.ts` defines the shared state contract passed across process boundaries.

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
           `--> openWorkspace()
                   |
                   `--> createWorkspaceWindow()
                           |
                           +--> new BaseWindow(...)
                           +--> new WebContentsView(notes)
                           +--> new WebContentsView(splitter)
                           +--> new WebContentsView(browser chrome)
                           +--> new WebContentsView(browser content)
                           +--> load local notes/splitter/browser-chrome entries
                           +--> load remote browser URL
                           `--> new WorkspaceController(...)
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
   | window.workspace.setBrowserUrl(url)
   v
src/app/preload/workspace.ts
   |
   v
src/app/main/register-ipc.ts
   |
   +--> normalizeUrl(url)
   +--> browserView.webContents.loadURL(url)
   +--> persist snapshot
   `--> publish workspace state
```

## Persistence Flow

```text
notes textarea edit ------------------------------+
                                                   |
browser URL change ----------------------------+   |
                                               |   |
splitter drag ------------------------------+  |   |
                                            |  |   |
                                            v  v   v
                                 WorkspaceController snapshot
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
