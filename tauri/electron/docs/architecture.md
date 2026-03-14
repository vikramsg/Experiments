# Electron Architecture

This document describes the implemented feature-first, process-aware structure for the Electron workspace app.

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
    |   |   |-- create-workspace-window.ts  # BaseWindow + sibling WebContentsView composition
    |   |   `-- register-ipc.ts             # Central IPC handler registration
    |   |-- preload/
    |   |   |-- launcher.ts                 # Safe launcher preload bridge
    |   |   `-- workspace.ts                # Safe workspace preload bridge
    |   `-- renderer/
    |       `-- entries/
    |           |-- launcher.html           # Launcher HTML entrypoint
    |           |-- notes.html              # Notes HTML entrypoint
    |           `-- splitter.html           # Splitter HTML entrypoint
    |-- features/
    |   |-- browser/
    |   |   `-- main/
    |   |       |-- browser-session.ts      # URL normalization and remote-content security rules
    |   |       `-- browser-session.test.ts # Tests for browser security/session behavior
    |   |-- launcher/
    |   |   `-- renderer/
    |   |       |-- App.tsx                 # Launcher UI
    |   |       |-- App.test.tsx            # Launcher renderer tests
    |   |       `-- main.tsx                # Launcher React entrypoint
    |   |-- notes/
    |   |   |-- main/
    |   |   |   |-- NoteStore.ts           # Persistent workspace snapshot storage
    |   |   |   `-- NoteStore.test.ts      # NoteStore tests
    |   |   `-- renderer/
    |   |       |-- App.tsx                 # Notes editor and browser URL UI
    |   |       |-- App.test.tsx            # Notes renderer tests
    |   |       `-- main.tsx                # Notes React entrypoint
    |   |-- splitter/
    |   |   `-- renderer/
    |   |       |-- SplitterHandle.tsx      # Splitter drag UI
    |   |       |-- SplitterHandle.test.tsx # Splitter renderer tests
    |   |       `-- main.tsx                # Splitter React entrypoint
    |   `-- workspace/
    |       |-- main/
    |       |   |-- WorkspaceController.ts  # View bounds, resize handling, teardown, state publication
    |       |   `-- WorkspaceController.test.ts
    |       |                                # Workspace controller tests
    |       `-- shared/
    |           |-- split-layout.ts         # Pure layout math for the split workspace
    |           `-- split-layout.test.ts    # Layout math tests
    |-- shared/
    |   |-- ipc/
    |   |   `-- channels.ts                 # Shared IPC channel names
    |   |-- test/
    |   |   `-- setup.ts                    # Vitest setup shared across renderer tests
    |   `-- types/
    |       `-- workspace.ts                # Shared workspace snapshot types and defaults
    |-- types.d.ts                          # Global `window.launcher` and `window.workspace` typings
    `-- vite-env.d.ts                       # Vite environment types
|-- e2e/                                    # Playwright Electron end-to-end tests
```

## Responsibility Map

- `src/app/main/` owns startup and wiring. It should stay thin and orchestration-focused.
- `src/features/*/main/` owns feature-specific Node/Electron logic such as persistence, layout control, and browser security.
- `src/features/*/renderer/` owns feature-specific UI code only.
- `src/app/preload/` owns the narrow bridges between renderer and main.
- `src/shared/` only holds cross-feature code that is genuinely shared, not feature logic hiding in generic folders.

## Runtime Overview

```text
+-------------------------------------------------------------+
| Launcher BrowserWindow                                      |
|  local React renderer                                       |
|  preload: src/app/preload/launcher.ts                       |
|  action: open "Browser + Notes" workspace                  |
+----------------------------+--------------------------------+
                             |
                             | ipcMain.handle('launcher:open-workspace')
                             v
+-------------------------------------------------------------+
| Workspace BaseWindow                                         |
|                                                             |
|  sibling child views owned by the main process              |
|                                                             |
|  +----------------+ +----------+ +------------------------+ |
|  | Notes view     | | Splitter | | Browser view           | |
|  | local React UI | | local UI | | remote web content     | |
|  | preload bridge | | preload  | | sandboxed session      | |
|  +----------------+ +----------+ +------------------------+ |
|                                                             |
|  WorkspaceController computes bounds for all three views    |
+-------------------------------------------------------------+
```

## Main Process Call Flow

```text
src/app/main/index.ts
   |
   +--> createLauncherWindow()
   |
   +--> registerIpc()
   |      |
   |      `--> workspace-related handlers
   |
   `--> openWorkspace()
           |
           `--> createWorkspaceWindow(userDataPath)
                   |
                   +--> NoteStore.load()
                   +--> applyBrowserSecurityPolicy(...)
                   +--> new WorkspaceController(...)
                   `--> publish initial workspace state
```

## IPC Boundaries

```text
Launcher renderer
  window.launcher.openWorkspace()
        |
        v
src/app/preload/launcher.ts
        |
        v
ipcMain.handle('launcher:open-workspace')


Notes renderer
  window.workspace.loadState()
  window.workspace.saveNotes(notes)
  window.workspace.setBrowserUrl(url)
        |
        v
src/app/preload/workspace.ts
        |
        v
ipcMain.handle('workspace:get-state')
ipcMain.handle('workspace:save-notes')
ipcMain.handle('workspace:set-browser-url')


Splitter renderer
  window.workspace.adjustSplitter(delta)
        |
        v
src/app/preload/workspace.ts
        |
        v
ipcMain.handle('workspace:adjust-splitter')
```

## Persistence Flow

```text
notes textarea edit ------------------------------+
                                                  |
splitter drag --------------------------------+   |
                                              |   |
                                              v   v
                                    WorkspaceController snapshot
                                              |
                                              v
                                     NoteStore.save(snapshot)
                                              |
                                              v
                         app.getPath('userData')/workspace-state.json
                                              |
                                              v
                                     NoteStore.load() on reopen
```

## Resize And Drag Behavior

```text
window resize or splitter delta
           |
           v
WorkspaceController.applyLayout()
           |
           v
computeSplitLayout({
  windowWidth,
  windowHeight,
  notesWidth,
  splitterWidth,
  minNotesWidth,
  minBrowserWidth,
})
           |
           +--> clamp notes width
           +--> derive splitter x
           +--> derive browser width
           |
           v
setBounds() on notes, splitter, and browser sibling views
```

## Security Posture For Remote Content

```text
Remote browser feature rules
  - nodeIntegration: false
  - contextIsolation: true
  - sandbox: true
  - dedicated session partition
  - permission requests denied
  - popup windows denied
  - non-http/https navigation blocked
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
        +--> browserView.webContents.close()
```
