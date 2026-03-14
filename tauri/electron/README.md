# Electron Workspace

Electron workspace app with a launcher window and a split workspace containing notes on the left, a draggable splitter in the middle, and a browser area on the right with local back/forward controls plus a synchronized URL bar above remote content.

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
- `src/ipc.ts` owns IPC channel names only.
- `src/workspace-contract.ts` owns the public renderer-facing workspace API contract.
- `src/workspace-model.ts` owns the shared workspace snapshot model and persistence helpers.
- `src/test-setup.ts` owns shared Vitest setup only.
- `src/types.d.ts` is the ambient global bridge for `window.launcher` and `window.workspace`.

## Code Organization

- `app/*` may compose feature entrypoints and root boundary files.
- `features/*` may import only:
  - their own feature files
  - `src/ipc.ts`
  - `src/workspace-contract.ts`
  - `src/workspace-model.ts`
- Features must not import other features directly.
- Root `src/*.ts` boundary files exist to keep cross-cutting contracts shallow and obvious, instead of growing a generic root `src/shared/` bucket.
- `src/features/workspace/shared/` remains acceptable because it is scoped to the workspace feature rather than acting as a global runtime catch-all.

## Architecture

- Launcher uses a local `BrowserWindow` renderer.
- Workspace uses a `BaseWindow` with four sibling `WebContentsView`s:
  - notes
  - splitter
  - browser chrome
  - browser content
- Main-process layout ownership lives in `src/features/workspace/main/WorkspaceController.ts`.
- Browser chrome actions in `src/features/browser/renderer/App.tsx` can navigate directly, go back, and go forward.
- Browser URL and history availability stay synchronized from the remote browser `webContents`, so the local chrome reflects link clicks, redirects, and in-page navigation.
- Workspace startup now registers the window bundle before renderer page loads finish, which prevents transient `workspace:get-state` errors during initialization.
- Notes, browser URL, and splitter width are persisted in `app.getPath('userData')/workspace-state.json`, while browser history availability remains live-only state.

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
