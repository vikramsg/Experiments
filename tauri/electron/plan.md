## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Keep the Electron app under `electron/` in its refactored feature-first, process-aware structure.
- Move the browser URL input and Go action out of the notes feature and into the right browser split.
- Preserve the implemented behavior:
  - launcher flow
  - workspace with sibling `WebContentsView`s
  - draggable splitter
  - notes persistence
  - browser navigation with the current security constraints
- Keep the implementation isolated from the Tauri app in `hello-world/`.
- Update docs and tests so they reflect the browser-owned renderer surface in the new structure.

## Architecture Diagram

```text
Current runtime
===============

+-------------------------------------------------------------------+
| Workspace BaseWindow                                              |
|                                                                   |
| +------------------+ +----------+ +----------------------------+  |
| | Notes view       | | Splitter | | Browser content view       |  |
| | - notes editor   | | handle   | | - remote site             |  |
| | - URL input      | |          | |                            |  |
| | - Go button      | |          | |                            |  |
| +------------------+ +----------+ +----------------------------+  |
|                                                                   |
+-------------------------------------------------------------------+
```

```text
Target runtime
==============

+-------------------------------------------------------------------+
| Workspace BaseWindow                                              |
|                                                                   |
| +------------------+ +----------+ +----------------------------+  |
| | Notes view       | | Splitter | | Browser chrome view        |  |
| | - notes editor   | | handle   | | - URL input               |  |
| |                  | |          | | - Go button               |  |
| |                  | |          | +----------------------------+  |
| |                  | |          | | Browser content view       |  |
| |                  | |          | | - remote site             |  |
| +------------------+ +----------+ +----------------------------+  |
|                                                                   |
| main process owns bounds for all four sibling views               |
+-------------------------------------------------------------------+
```

```text
Feature ownership
=================

src/features/notes/
  - note storage
  - notes renderer UI only

src/features/browser/
  - remote session policy
  - URL normalization
  - popup and permission restrictions
  - browser chrome renderer UI

src/features/workspace/
  - BaseWindow layout controller
  - split-layout math
  - workspace state publication
```

```text
Navigation flow
===============

browser renderer App
   |
   | window.workspace.setBrowserUrl(url)
   v
app/preload/workspace.ts
   |
   v
app/main/register-ipc.ts
   |
   +--> normalizeUrl(url)
   +--> browserView.webContents.loadURL(url)
   +--> persist snapshot
   `--> publish workspace state
```

## Current Status

- The `sm` branch refactor is already in place:
  - bootstrap is in `src/app/main/`
  - preloads are in `src/app/preload/`
  - renderer entries are in `src/app/renderer/entries/`
  - features are under `src/features/`
- The current implementation still renders the browser URL controls inside the notes renderer in `electron/src/features/notes/renderer/App.tsx`.
- The browser feature currently has only main-process logic in `electron/src/features/browser/main/` and no renderer surface yet.
- The workspace currently creates and lays out only three sibling views:
  - notes
  - splitter
  - browser content
- Existing renderer and E2E tests still assume the browser URL UI lives in the notes pane.
- `electron/docs/architecture.md`, `electron/README.md`, and `electron/Justfile` already exist, but they currently describe the old notes-owned URL UI and three-view workspace.

## Short summary of changes

- Add a browser renderer feature under `src/features/browser/renderer/` for the local URL bar UI.
- Add a new renderer entry HTML file under `src/app/renderer/entries/browser-chrome.html`.
- Remove browser URL controls from the notes renderer.
- Extend workspace composition and layout from three sibling views to four sibling views.
- Keep browser security and navigation logic in the browser feature main-process modules.
- Update docs so they reflect the browser-owned renderer surface and four-view workspace runtime.

### Options considered

1. Add `features/browser/renderer` and a fourth sibling `WebContentsView`
   - fits the refactored structure cleanly
   - keeps browser UI with browser ownership
   - preserves remote-content isolation
   - recommended

2. Keep URL UI in `features/notes/renderer`
   - lower code churn
   - incorrect feature ownership and wrong UX placement
   - rejected

3. Replace the right side with a larger local shell renderer
   - possible
   - larger architectural change than needed
   - rejected

## Files to be changed

- `electron/plan.md`
- `electron/vite.renderer.config.ts`
- `electron/src/app/main/create-workspace-window.ts`
- `electron/src/app/main/register-ipc.ts`
- `electron/src/app/preload/workspace.ts`
- `electron/src/features/workspace/main/WorkspaceController.ts`
- `electron/src/features/workspace/main/WorkspaceController.test.ts`
- `electron/src/features/notes/renderer/App.tsx`
- `electron/src/features/notes/renderer/App.test.tsx`
- `electron/e2e/launcher-workspace.spec.js`
- `electron/e2e/persistence.spec.js`
- `electron/e2e/splitter-drag.spec.js`
- `electron/e2e/workspace-resize.spec.js`
- `electron/docs/architecture.md`
- `electron/README.md`

## Files to be added

- `electron/src/app/renderer/entries/browser-chrome.html`
- `electron/src/features/browser/renderer/App.tsx`
- `electron/src/features/browser/renderer/App.test.tsx`
- `electron/src/features/browser/renderer/main.tsx`

## Verification Criteria

- The browser URL input and Go button are no longer rendered by `electron/src/features/notes/renderer/App.tsx`.
- A browser renderer feature exists under `electron/src/features/browser/renderer/`.
- A browser renderer entrypoint exists at `electron/src/app/renderer/entries/browser-chrome.html`.
- `electron/vite.renderer.config.ts` includes the browser-chrome entry.
- `electron/src/app/main/create-workspace-window.ts` creates and loads four sibling views:
  - notes
  - splitter
  - browser chrome
  - browser content
- `electron/src/features/workspace/main/WorkspaceController.ts` computes and applies four-view layout correctly.
- Browser URL changes still navigate the remote browser and persist across relaunch.
- Notes persistence still works unchanged.
- `electron/docs/architecture.md` reflects the browser renderer in the file tree and runtime diagrams.
- `electron/README.md` reflects the browser-owned renderer surface and current commands.
- Lint, unit tests, Playwright E2E tests, and production build all succeed.

## Acceptance Criteria

- The browser URL field lives in the browser split, not the notes split.
- The browser feature owns both:
  - browser main-process logic
  - browser renderer chrome UI
- The right pane is composed of:
  - a local browser chrome area
  - a remote browser content area
- The Electron workspace still uses sibling `WebContentsView`s rather than `BrowserView` or a primary `<webview>` architecture.
- The splitter remains draggable and persistent.
- Notes remain isolated to the left pane.
- Browser URL persistence still works across reopen and relaunch.
- The refactored file structure remains feature-first and process-aware.
- Docs and tests align with the implemented new structure.
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Inspect the refactored app wiring in:
   - `src/app/main/create-workspace-window.ts`
   - `src/app/main/register-ipc.ts`
   - `src/app/preload/workspace.ts`
   - `src/features/notes/renderer/App.tsx`
   - `src/features/workspace/main/WorkspaceController.ts`
   to identify every assumption that the browser URL UI lives in the notes feature.

2. Update `electron/plan.md` first so it reflects the current refactored structure and the new browser-renderer ownership before implementation begins.

3. Write failing renderer tests for `electron/src/features/notes/renderer/App.test.tsx` proving the notes surface no longer contains:
   - browser URL input
   - Go button

4. Run the notes renderer tests immediately to verify they fail for the expected reason; if they do not fail, tighten the assertions before implementing.

5. Write failing renderer tests for the new browser renderer in `electron/src/features/browser/renderer/App.test.tsx` covering:
   - saved browser URL rendering
   - URL editing
   - Go-triggered navigation
   - reaction to workspace state updates

6. Run the new browser renderer tests immediately to confirm they fail before implementation; if they do not fail, correct the tests first.

7. Write failing workspace controller tests in `electron/src/features/workspace/main/WorkspaceController.test.ts` for the new four-view layout:
   - initial bounds for notes, splitter, browser chrome, and browser content
   - updates on splitter drag
   - updates on window resize
   - cleanup of all child `webContents`

8. Run the workspace controller tests immediately to prove the new expectations fail against the current three-view implementation; if they pass unexpectedly, strengthen the assertions.

9. Add `electron/src/features/browser/renderer/` and `electron/src/app/renderer/entries/browser-chrome.html`.

10. Implement the browser renderer UI and entrypoint, keeping URL/chrome responsibility inside the browser feature.

11. Remove the URL controls from `electron/src/features/notes/renderer/App.tsx` while preserving notes load, autosave, and state-sync behavior.

12. Update `electron/src/app/main/create-workspace-window.ts` to create and load a fourth sibling `WebContentsView` for browser chrome and publish workspace state to it.

13. Update `electron/src/features/workspace/main/WorkspaceController.ts` to manage browser chrome and browser content bounds separately while preserving splitter behavior.

14. Re-run the directly related renderer and controller tests until they all pass.

15. Update Playwright Electron specs so browser URL interactions happen in the browser-chrome page instead of the notes page.

16. Run the Playwright tests to verify those new expectations fail first; if they do not fail, tighten the assertions and rerun.

17. Implement any remaining wiring needed for Playwright to discover and use the browser-chrome surface in:
   - launcher flow
   - persistence flow
   - resize flow
   - splitter flow

18. Re-run Playwright until launcher, persistence, resize, splitter, and browser navigation behavior all pass with the browser URL UI on the right.

19. Update `electron/docs/architecture.md` with:
   - the new file tree including `features/browser/renderer`
   - updated responsibility notes
   - updated four-view runtime diagrams

20. Update `electron/README.md` so it reflects the new browser-owned renderer surface and the current supported commands.

21. If a browser skill is available, use it for smoke checks of the visible UI change. No browser skill is available in this environment, so use the Playwright Electron coverage as the browser-level smoke baseline.

22. Run the full automated verification suite:
   - lint
   - unit and renderer tests
   - Playwright Electron E2E tests
   - build/package flow

23. Stop only after the browser URL UI is in the right split, the refactored structure remains coherent, all fail-first checks were exercised where new seams were introduced, and every automated verification step passes.
