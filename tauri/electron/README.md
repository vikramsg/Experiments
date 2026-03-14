# Electron Workspace

Electron workspace app with a launcher window and a split workspace containing notes, a draggable splitter, and a browser pane.

## Commands

```bash
npm run dev
npm run lint
npm test
npm run e2e
npm run build
npm run verify
```

If you use `just`:

```bash
just dev
just lint
just test
just e2e
just build
just verify
```

## Structure

- `src/app/` holds bootstrap, preload wiring, and renderer HTML entries.
- `src/features/` holds business-owned code:
  - `launcher`
  - `workspace`
  - `notes`
  - `browser`
  - `splitter`
- `src/shared/` holds only true cross-feature code such as IPC channel names, shared types, and test setup.

For the detailed tree and diagrams, see `docs/architecture.md`.

## Process Boundaries

- `app/main` and `features/*/main` are Electron/Node-side code.
- `app/preload` exposes narrow safe APIs to renderer code.
- `features/*/renderer` contains UI-only code.

## Persistence

Workspace state is stored under:

```text
app.getPath('userData')/workspace-state.json
```

Persisted values include:

- notes content
- notes pane width
- browser URL

## Browser Security

The remote browser pane uses:

- `nodeIntegration: false`
- `contextIsolation: true`
- `sandbox: true`
- denied permission requests
- denied popup windows
- http/https-only navigation through the app URL path

## Verification

Full verification:

```bash
npm run verify
```

This runs:

- lint
- unit and renderer tests
- Playwright Electron E2E tests
- build/package flow
