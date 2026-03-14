## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Refactor the Electron app under `electron/` into a feature-first, process-aware structure.
- Keep Electron runtime boundaries explicit:
  - `main`
  - `preload`
  - `renderer`
- Organize business logic by feature/domain instead of flat technical folders.
- Move renderer HTML entry files out of the repo root into a renderer-owned location.
- Add `electron/docs/architecture.md` with:
  - helpful ASCII diagrams
  - a tree view of the file structure
  - responsibility notes for important files and folders
- Add `electron/README.md` and `electron/Justfile`.
- Keep the implementation isolated from the Tauri app in `hello-world/`.
- Preserve the implemented behavior:
  - launcher flow
  - workspace with sibling `WebContentsView`s
  - draggable splitter
  - notes persistence
  - browser navigation with the current security constraints

## Architecture Diagram

```text
Current shape
=============

electron/
|-- launcher.html
|-- notes.html
|-- splitter.html
`-- src/
    |-- main.ts
    |-- main/
    |-- preloads/
    |-- launcher/
    |-- notes/
    |-- splitter/
    `-- shared/

Problems:
- process boundaries exist, but feature ownership is flat and scattered
- renderer HTML entry files live at the project root
- business logic is not clearly grouped by domain
```

```text
Target shape
============

electron/
`-- src/
    |-- app/
    |   |-- main/          # bootstrap, window creation, IPC registration
    |   |-- preload/       # preload entry wiring only
    |   `-- renderer/
    |       `-- entries/   # launcher.html, notes.html, splitter.html
    |
    |-- features/
    |   |-- launcher/
    |   |   |-- renderer/
    |   |   `-- shared/
    |   |
    |   |-- workspace/
    |   |   |-- main/
    |   |   |-- preload/
    |   |   `-- shared/
    |   |
    |   |-- notes/
    |   |   |-- main/
    |   |   |-- renderer/
    |   |   `-- shared/
    |   |
    |   |-- browser/
    |   |   |-- main/
    |   |   `-- shared/
    |   |
    |   `-- splitter/
    |       |-- renderer/
    |       `-- shared/
    |
    `-- shared/
        |-- ipc/
        |-- types/
        `-- test/
```

```text
Runtime call flow
=================

app/main/index
   |
   +--> createLauncherWindow()
   |       |
   |       `--> loads launcher renderer entry
   |
   `--> registerIpc()
           |
           `--> open workspace request
                   |
                   v
             createWorkspaceWindow()
                   |
                   +--> notes WebContentsView
                   +--> splitter WebContentsView
                   +--> browser WebContentsView
                   |
                   `--> WorkspaceController.applyLayout()
```

```text
Feature ownership
=================

Launcher feature
  - renderer app card UI
  - open-workspace intent

Workspace feature
  - BaseWindow composition
  - layout controller
  - snapshot publication

Notes feature
  - notes UI
  - note persistence
  - browser URL form state

Browser feature
  - remote session policy
  - URL normalization
  - popup and permission restrictions

Splitter feature
  - pointer drag UI
  - drag delta emission
```

## Current Status

- The Electron app already exists under `electron/` and is separate from the Tauri app.
- The current source tree is partly process-aware but still mostly flat:
  - `src/main.ts`
  - `src/main/`
  - `src/preloads/`
  - `src/launcher/`
  - `src/notes/`
  - `src/splitter/`
  - `src/shared/`
- Renderer HTML entrypoints currently live at the project root:
  - `electron/launcher.html`
  - `electron/notes.html`
  - `electron/splitter.html`
- `electron/vite.renderer.config.ts` points directly at those root HTML files.
- `electron/forge.config.ts` still references the flat main and preload entrypoints.
- `electron/docs/reference.md` exists and already captures reusable Electron guidance from `electron/research.md`.
- `electron/docs/architecture.md` does not exist yet.
- `electron/README.md` does not exist yet.
- `electron/Justfile` does not exist yet.

## Short summary of changes

- Restructure the Electron app around business features instead of flat technical folders.
- Keep Electron process boundaries explicit, but nest them under feature ownership where appropriate.
- Move bootstrap and app wiring into `src/app/`.
- Move renderer HTML files into `src/app/renderer/entries/`.
- Extract browser-session and browser-security logic into a browser feature.
- Keep reusable cross-feature types, IPC constants, and test helpers in `src/shared/`.
- Add `electron/docs/architecture.md` with ASCII diagrams, a file tree, and responsibility notes.
- Add `electron/README.md` and `electron/Justfile`.

### Options considered

1. Keep the current mostly-flat structure and only improve docs
   - lowest migration cost
   - still weak on business ownership
   - rejected

2. Strict process-first structure only: `main/`, `preload/`, `renderer/`
   - aligns with Electron runtime boundaries
   - still scatters one feature across many folders
   - better than current, but not preferred

3. Feature-first, process-aware hybrid
   - preserves Electron runtime boundaries
   - groups business logic by domain
   - removes renderer HTML files from the project root
   - recommended

## Files to be changed

- `electron/package.json`
- `electron/tsconfig.json`
- `electron/forge.config.ts`
- `electron/vite.main.config.ts`
- `electron/vite.preload.config.ts`
- `electron/vite.renderer.config.ts`
- `electron/vitest.config.ts`
- `electron/playwright.config.js`
- `electron/src/main.ts`
- `electron/src/main/workspace-controller.ts`
- `electron/src/main/workspace-controller.test.ts`
- `electron/src/main/note-store.ts`
- `electron/src/main/note-store.test.ts`
- `electron/src/preloads/launcher.ts`
- `electron/src/preloads/workspace.ts`
- `electron/src/launcher/App.tsx`
- `electron/src/launcher/App.test.tsx`
- `electron/src/launcher/main.tsx`
- `electron/src/notes/App.tsx`
- `electron/src/notes/App.test.tsx`
- `electron/src/notes/main.tsx`
- `electron/src/splitter/SplitterHandle.tsx`
- `electron/src/splitter/SplitterHandle.test.tsx`
- `electron/src/splitter/main.tsx`
- `electron/src/shared/split-layout.ts`
- `electron/src/shared/split-layout.test.ts`
- `electron/src/types.d.ts`

## Files to be added

- `electron/docs/architecture.md`
- `electron/README.md`
- `electron/Justfile`
- `electron/src/app/main/index.ts`
- `electron/src/app/main/create-launcher-window.ts`
- `electron/src/app/main/create-workspace-window.ts`
- `electron/src/app/main/register-ipc.ts`
- `electron/src/app/preload/launcher.ts`
- `electron/src/app/preload/workspace.ts`
- `electron/src/app/renderer/entries/launcher.html`
- `electron/src/app/renderer/entries/notes.html`
- `electron/src/app/renderer/entries/splitter.html`
- `electron/src/features/launcher/renderer/App.tsx`
- `electron/src/features/launcher/renderer/App.test.tsx`
- `electron/src/features/launcher/renderer/main.tsx`
- `electron/src/features/workspace/main/WorkspaceController.ts`
- `electron/src/features/workspace/main/WorkspaceController.test.ts`
- `electron/src/features/workspace/shared/split-layout.ts`
- `electron/src/features/workspace/shared/split-layout.test.ts`
- `electron/src/features/notes/main/NoteStore.ts`
- `electron/src/features/notes/main/NoteStore.test.ts`
- `electron/src/features/notes/renderer/App.tsx`
- `electron/src/features/notes/renderer/App.test.tsx`
- `electron/src/features/notes/renderer/main.tsx`
- `electron/src/features/browser/main/browser-session.ts`
- `electron/src/features/browser/main/browser-session.test.ts`
- `electron/src/features/splitter/renderer/SplitterHandle.tsx`
- `electron/src/features/splitter/renderer/SplitterHandle.test.tsx`
- `electron/src/features/splitter/renderer/main.tsx`
- `electron/src/shared/ipc/channels.ts`
- `electron/src/shared/types/workspace.ts`
- `electron/src/shared/test/setup.ts`

## Verification Criteria

- The Electron source tree is reorganized around feature/domain ownership rather than flat technical folders.
- Electron runtime boundaries remain explicit:
  - main-only code is not imported into renderer code
  - preload code stays narrow and safe
  - renderer code stays UI-focused
- Root HTML entry files are removed from `electron/` and replaced with renderer-owned entry files.
- `electron/vite.renderer.config.ts` resolves the new renderer entrypoint locations correctly.
- `electron/forge.config.ts` points at the new main and preload entrypoints correctly.
- Existing behavior remains unchanged:
  - launcher opens workspace
  - workspace uses sibling `WebContentsView`s
  - splitter drag updates layout
  - notes persist
  - browser URL navigation still works
  - browser security restrictions still apply
- `electron/docs/architecture.md` exists and includes:
  - ASCII runtime diagrams
  - file tree view
  - file and folder responsibility descriptions
- `electron/README.md` exists and reflects the new structure.
- `electron/Justfile` exists and exposes the main development commands.
- Lint, unit tests, Playwright E2E tests, and production build all succeed.

## Acceptance Criteria

- The Electron app uses a feature-first, process-aware directory structure.
- Business logic is colocated with the feature it belongs to, not hidden in generic technical folders.
- App bootstrap and wiring live in `src/app/`.
- Feature code lives under `src/features/<feature-name>/`.
- Shared cross-feature code is limited to genuinely shared types, IPC constants, and test helpers.
- Renderer HTML entry files no longer live at the repo root.
- The workspace still uses sibling `WebContentsView`s rather than `BrowserView` or a primary `<webview>` architecture.
- The implementation remains covered by automated tests for:
  - layout math
  - splitter interaction
  - note persistence
  - launcher flow
  - workspace controller behavior
  - Electron end-to-end behavior with Playwright
- `electron/docs/architecture.md` documents the implemented structure with a tree view and responsibilities.
- `electron/README.md` and `electron/Justfile` are present and aligned with the new structure.
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Inspect the current Electron tree, build config, preload entrypoints, renderer entrypoints, and tests to lock down the exact flat structure being replaced.

2. Create the target folder map that separates:
   - app bootstrap
   - feature code
   - shared cross-feature code
   - renderer entry HTML files

3. Write failing tests for any new extracted seams before moving code, especially where refactoring introduces new modules such as:
   - browser security and session configuration
   - launcher and workspace entry resolution
   - IPC channel ownership
   - any new shared workspace snapshot types

4. Run those new tests immediately to confirm they actually fail; if they do not fail, tighten the assertions before moving code.

5. Refactor the pure logic first with minimal runtime risk:
   - move split-layout logic into the target feature/shared location
   - move note-store logic into the notes feature
   - move browser session and security logic into the browser feature

6. Rerun the affected unit tests after each logic extraction and keep fixing imports until they pass again.

7. Write failing tests for any new app-bootstrap modules such as:
   - launcher window creation
   - workspace window creation
   - IPC registration boundaries

8. Run those bootstrap/module tests to confirm failure first; if a test passes unexpectedly, correct the test before implementation.

9. Move app-wiring code out of `src/main.ts` into `src/app/main/` entry modules and rerun tests to passing.

10. Move preload bridges into `src/app/preload/` or feature-owned preload modules while keeping the exposed renderer API shape stable.

11. Write failing tests for the renderer entrypoint relocation if needed, such as path-resolution or config-level expectations around the moved HTML entry files.

12. Run the relevant tests or config validations immediately to verify the path assumptions fail before changing the Vite config; if they do not fail, strengthen the validation.

13. Move `launcher.html`, `notes.html`, and `splitter.html` into a renderer-owned entries folder and update Vite and Forge configuration to load them from the new location.

14. Rerun build-oriented checks after the HTML move to confirm entry resolution now works from the new structure.

15. Move flat renderer feature code into business-owned folders:
   - launcher
   - notes
   - splitter
   - workspace
   - browser

16. After each feature move, run the directly related unit and renderer tests so regressions are caught while the change set is still small.

17. Add `electron/docs/architecture.md` with:
   - runtime ASCII diagrams
   - file tree view
   - short responsibility notes for each important file and folder

18. Add `electron/README.md` documenting:
   - the new file structure
   - how to run the app
   - how to run tests
   - how to reason about process boundaries and feature ownership

19. Add `electron/Justfile` with the main commands used for development and verification.

20. Run `just --list` to confirm the Justfile parses; if it does not, fix the syntax before proceeding.

21. If a browser skill is available, use it for smoke checks of any PR-visible UI or doc changes. No browser skill is available in this environment, so use the Playwright Electron coverage as the browser-level smoke baseline.

22. Run lint after the structural refactor and doc additions to catch stale imports, dead files, config drift, and path mismatches.

23. Run the full unit and renderer test suite after the structural refactor to confirm behavior is unchanged.

24. Run Playwright Electron E2E tests after the refactor to verify:
   - launcher still renders
   - app card still opens workspace
   - splitter drag still works in both directions
   - resize stability still holds
   - notes still persist across relaunch
   - browser pane still loads the configured URL

25. Run the production build and package flow and fix any remaining config or entrypoint issues.

26. Stop only after the new structure is in place, docs are updated, all tests were proven to fail first where new seams were introduced, and every automated verification step passes.
