# Electron Workspace

Electron workspace app with a launcher window and a split workspace containing notes on the left, a draggable splitter in the middle, and a browser area on the right with a local URL bar above remote content.

## Requirements

- Node.js
- npm
- `just`

## Install

```bash
just install
```

## Run Locally

```bash
just dev
```

## Command Reference

- `just install` - install project dependencies
- `just dev` - start the Electron app in development mode
- `just lint` - run ESLint
- `just test` - run Vitest unit and renderer tests
- `just e2e` - package the app and run Playwright Electron tests
- `just build` - package the Electron app
- `just package` - run the Forge package step directly
- `just make` - build platform distributables
- `just verify` - run lint, tests, E2E, and build
- `just verify-no-e2e` - run lint, tests, and build without Playwright

## Structure

- `src/app/` holds bootstrap, preload wiring, and renderer HTML entries.
- `src/features/` holds business-owned code:
  - `launcher`
  - `workspace`
  - `notes`
  - `browser`
  - `splitter`
- `src/shared/` holds only true cross-feature code such as IPC channel names, shared types, and test setup.

## Architecture

- Launcher uses a local `BrowserWindow` renderer.
- Workspace uses a `BaseWindow` with four sibling `WebContentsView`s:
  - notes
  - splitter
  - browser chrome
  - browser content
- Main-process layout ownership lives in `src/features/workspace/main/WorkspaceController.ts`.
- Browser URL updates are triggered from `src/features/browser/renderer/App.tsx` and applied in the main process.
- Workspace startup now registers the window bundle before renderer page loads finish, which prevents transient `workspace:get-state` errors during initialization.
- Notes, browser URL, and splitter width are persisted in `app.getPath('userData')/workspace-state.json`.

For the detailed file tree, diagrams, and responsibilities, see `docs/architecture.md`.

## Browser Security

The remote browser pane uses:

- `nodeIntegration: false`
- `contextIsolation: true`
- `sandbox: true`
- denied permission requests
- denied popup windows
- http/https-only navigation through the app URL path

## Verification

```bash
just verify
```
