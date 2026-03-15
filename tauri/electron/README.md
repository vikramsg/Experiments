# Electron Workspace

Electron app with a launcher window that opens two local apps:

- `Browser + Notes` for the split notes/browser workspace
- `OpenCode` for read-only chat against this repo with a browser pane on the right

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

## Code Organization

- `src/app/` composes features, owns window creation, preload wiring, and HTML entrypoints.
- `src/features/` holds isolated business-owned code:
  - `launcher`
  - `workspace`
  - `notes`
  - `browser`
  - `splitter`
  - `opencode`
- `app/*` may compose feature entrypoints and root boundary files.
- Features should depend on themselves plus shallow root runtime boundaries under `src/`; they should not import other features directly.
- Services, adapters, and transports belong to the domain they expose.
  - Browser screenshot capture and browser MCP services belong to the browser or app-main side.
  - OpenCode process lifecycle belongs to OpenCode.
  - Cross-domain wiring belongs in `app/*`, not in feature-to-feature imports.
- Shallow root boundary files own cross-cutting runtime contracts and models:
  - `src/browser-contract.ts`
  - `src/browser-model.ts`
  - `src/ipc.ts`
  - `src/workspace-contract.ts`
  - `src/workspace-model.ts`
  - `src/test-setup.ts`
  - `src/opencode-contract.ts`
  - `src/opencode-model.ts`
- `src/types.d.ts` stays ambient-only so the global window bridge is declared in one place without becoming the primary owner of runtime API types.

## Architecture

- Launcher uses a local `BrowserWindow` renderer.
- Workspace uses a `BaseWindow` with four sibling `WebContentsView`s:
  - notes
  - splitter
  - browser chrome
  - browser content
- OpenCode uses its own `BaseWindow` with sibling `WebContentsView`s for:
  - OpenCode chat on the left
  - browser chrome on the right top
  - browser content on the right bottom
- Main-process layout ownership lives in `src/features/workspace/main/WorkspaceController.ts`.
- Browser chrome actions in `src/features/browser/renderer/App.tsx` use arrow buttons with accessible back/forward names, plus a synchronized URL field.
- Browser URL and history availability stay synchronized from the remote browser `webContents`, so the local chrome reflects link clicks, redirects, and in-page navigation.
- Workspace startup registers the window bundle before renderer page loads finish, which prevents transient `workspace:get-state` errors during initialization.
- Notes, browser URL, and splitter width are persisted in `app.getPath('userData')/workspace-state.json`, while browser history availability remains live-only state.
- The main-process `OpenCodeService` starts a local `opencode serve` process rooted at the `tauri/` repo scope, creates a chat session, and publishes renderer-facing state.
- Electron main also hosts a localhost browser MCP server in the browser domain, so OpenCode can call a browser tool instead of relying on prompt-time screenshot injection.
- The first browser MCP tool captures the current browser URL and a fresh screenshot of the browser pane when the model needs to explain what it sees.
- OpenCode verifies that the browser MCP server is connected before browser-aware prompts are treated as ready, and integration tests exercise the real MCP registration and invocation path without mock mode.
- The OpenCode bridge is intentionally read-only for the repo scope:
  - reads, globbing, listing, and search are allowed
  - edits, write-style tools, and destructive shell or git actions are denied
- The OpenCode config explicitly allows the `browser_*` MCP tool namespace while keeping the rest of the app's destructive tool surface denied.

For the detailed file tree, diagrams, and responsibilities, see `docs/architecture.md`.

## Browser Security

The remote browser pane uses:

- `nodeIntegration: false`
- `contextIsolation: true`
- `sandbox: true`
- denied permission requests
- denied popup windows
- http/https-only navigation through the app URL path

## OpenCode Boundary

The OpenCode app does not expose raw shell execution, arbitrary filesystem
access, or unrestricted server handles to the renderer. The renderer can only:

- load the current OpenCode state
- send a prompt
- subscribe to state updates

Browser inspection happens through a separate main-process-owned MCP tool. That
tool can:

- read the current browser URL
- capture a fresh screenshot of the browser pane
- return both to OpenCode on demand

## TODO / Follow-up

- Render markdown and code blocks in OpenCode replies.
- Improve auto-scroll behavior so it respects intentional user scroll position.
- Add auto-growing prompt height instead of relying on manual textarea resize.
- Polish loading feedback in the composer and messages.
- Improve message differentiation with stronger visual separation or avatars.
- Add quick-start prompts and a richer empty state for the first OpenCode session.

## Verification

```bash
just verify
```
