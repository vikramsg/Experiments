# Electron Workspace

Electron app with a launcher window that opens three local apps:

- `Browser + Notes` for the split notes/browser workspace
- `OpenCode` for read-only chat against this repo through a local OpenCode server
- `Terminal` for a full local shell rendered with `ghostty-web` and backed by a main-process PTY

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
  - `terminal`
- `app/*` may compose feature entrypoints and root boundary files.
- Features should depend on themselves plus shallow root runtime boundaries under `src/`; they should not import other features directly.
- Shallow root boundary files own cross-cutting runtime contracts and models:
  - `src/ipc.ts`
  - `src/workspace-contract.ts`
  - `src/workspace-model.ts`
  - `src/test-setup.ts`
  - `src/opencode-contract.ts`
  - `src/opencode-model.ts`
  - `src/terminal-contract.ts`
  - `src/terminal-model.ts`
- `src/types.d.ts` stays ambient-only so the global window bridge is declared in one place without becoming the primary owner of runtime API types.

## Architecture

- Launcher uses a local `BrowserWindow` renderer.
- Workspace uses a `BaseWindow` with four sibling `WebContentsView`s:
  - notes
  - splitter
  - browser chrome
  - browser content
- Main-process layout ownership lives in `src/features/workspace/main/WorkspaceController.ts`.
- Browser chrome actions in `src/features/browser/renderer/App.tsx` use arrow buttons with accessible back/forward names, plus a synchronized URL field.
- Browser URL and history availability stay synchronized from the remote browser `webContents`, so the local chrome reflects link clicks, redirects, and in-page navigation.
- Workspace startup registers the window bundle before renderer page loads finish, which prevents transient `workspace:get-state` errors during initialization.
- Notes, browser URL, and splitter width are persisted in `app.getPath('userData')/workspace-state.json`, while browser history availability remains live-only state.
- OpenCode uses its own `BaseWindow` with one local `WebContentsView` and a dedicated preload bridge on `window.opencode`.
- The main-process `OpenCodeService` starts a local `opencode serve` process rooted at the `tauri/` repo scope, creates a chat session, and publishes renderer-facing state.
- The OpenCode bridge is intentionally read-only for the repo scope:
  - reads, globbing, listing, and search are allowed
  - edits, write-style tools, and destructive shell or git actions are denied
- Terminal uses its own `BaseWindow` with one local `WebContentsView`, a dedicated preload bridge on `window.terminal`, and a main-process `TerminalPtyService` that owns the shell lifecycle.
- The Terminal app starts in the repo root by default, but it is a full local shell:
  - commands run with the current user account's filesystem permissions
  - shell selection follows `process.env.SHELL` on macOS/Linux and `COMSPEC` on Windows
  - host PATH tools such as `tmux` are available when installed
  - the renderer stays sandboxed and only talks through a narrow preload bridge
- Terminal appearance is Electron-owned and persisted under `app.getPath('userData')`:
  - the window is intentionally full-bleed and terminal-first, with no visible settings UI
  - the current implementation has a hard dependency on assuming Ghostty is the terminal config source
  - Electron reads Ghostty `font-family` for comparison/logging only
  - Electron renders the terminal with a bundled patched mono font for reliable prompt/file glyph coverage and cell metrics
  - if Ghostty's configured font differs from the bundled Electron render font, the main process logs a warning through `pino`

For the detailed file tree, diagrams, and responsibilities, see `docs/architecture.md`.
For terminal-specific wiring, Ghostty rationale, and PTY behavior, see `docs/ghostty.md`.

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
