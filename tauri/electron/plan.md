## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Fix the browser chrome layout so the URL bar is no longer vertically compressed or visually squashed.
- Fix the workspace startup race that logs `Workspace is not open` during renderer initialization.
- Fix the stuck `Loading workspace...` state in the notes pane so the UI transitions cleanly once workspace state becomes available.
- Keep the Electron app under `electron/` in its current feature-first, process-aware structure.
- Preserve the implemented behavior:
  - launcher flow
  - workspace with sibling `WebContentsView`s
  - draggable splitter
  - notes persistence
  - browser navigation with the current security constraints
- Keep the implementation isolated from the Tauri app in `hello-world/`.

## Architecture Diagram

```text
Current startup race
====================

openWorkspace()
   |
   v
createWorkspaceWindow() -------------------------------+
   |                                                   |
   +--> starts loading local renderer pages            |
   +--> browser/notes renderer mounts                  |
   +--> renderer calls workspace:get-state             |
   |                                                   |
   +--> createWorkspaceWindow() still resolving        |
   |                                                   |
   +--> workspaceBundle not assigned yet               |
   |                                                   |
   `--> requireWorkspace() throws ---------------------+
```

```text
Target startup flow
===================

openWorkspace()
   |
   v
createWorkspaceWindow()
   |
   +--> create window, views, controller, bundle
   +--> return bundle early
   |
   v
workspaceBundle assigned in main process
   |
   +--> renderer calls workspace:get-state
   |       |
   |       `--> snapshot resolves successfully
   |
   `--> local/remote page loads continue asynchronously
```

```text
Current right pane
==================

+----------------------------+
| Browser URL                |
| [input.................]Go |
+----------------------------+
| Remote browser content     |
|                            |
+----------------------------+

Problem:
- browser chrome height is too short
- content padding + caption + stretched button make the bar feel squashed
```

```text
Target right pane
=================

+----------------------------+
| Browser URL                |
| [input..................] Go |
+----------------------------+
| Remote browser content     |
|                            |
+----------------------------+

Fixes:
- taller browser chrome band
- compact single-row alignment
- non-stretched button sizing
```

## Current Status

- The browser chrome renderer currently lives in `electron/src/features/browser/renderer/App.tsx` and is visible, but its container height is too small for its content.
- `electron/src/features/workspace/main/WorkspaceController.ts` currently reserves only `64px` for the browser chrome band.
- The notes renderer in `electron/src/features/notes/renderer/App.tsx` still depends on `loadState()` resolving before it can clear the `Loading workspace...` text.
- The main process in `electron/src/app/main/index.ts` only assigns `workspaceBundle` after `createWorkspaceWindow(...)` resolves.
- `electron/src/app/main/create-workspace-window.ts` still waits on page-loading work before returning, which allows renderer IPC to arrive before the bundle is registered.
- Current logs show repeated `workspace:get-state` failures with `Workspace is not open`, matching the startup race.

## Short summary of changes

- Update the plan to cover the browser-chrome sizing fix, the startup race fix, and the loading-state fix.
- Write failing tests for:
  - taller browser chrome layout bounds
  - notes renderer recovery from initial `loadState()` failure via workspace state updates
  - Electron E2E confirmation that the loading text does not remain stuck
- Refactor workspace creation so the bundle becomes available to IPC before renderer page loads complete.
- Make the notes and browser chrome renderers robust against an initial `loadState()` miss.
- Adjust browser chrome styling and workspace layout height so the right-side header renders comfortably.
- Re-run full verification after the fixes.

### Options considered

1. Fix both the race and the layout in the current architecture
   - smallest targeted repair
   - preserves the current four-view workspace design
   - recommended

2. Only increase the browser chrome height
   - fixes the visual compression
   - leaves the startup errors and stuck loading state unresolved
   - rejected

3. Only suppress `workspace:get-state` errors in the renderer
   - hides the symptom
   - leaves the main-process lifecycle race intact
   - rejected

## Files to be changed

- `electron/plan.md`
- `electron/src/app/main/index.ts`
- `electron/src/app/main/create-workspace-window.ts`
- `electron/src/app/main/register-ipc.ts`
- `electron/src/features/workspace/main/WorkspaceController.ts`
- `electron/src/features/workspace/main/WorkspaceController.test.ts`
- `electron/src/features/notes/renderer/App.tsx`
- `electron/src/features/notes/renderer/App.test.tsx`
- `electron/src/features/browser/renderer/App.tsx`
- `electron/src/features/browser/renderer/App.test.tsx`
- `electron/e2e/launcher-workspace.spec.js`
- `electron/docs/architecture.md`
- `electron/README.md`

## Files to be added

- None required unless a dedicated startup-lifecycle test helper proves necessary during implementation.

## Verification Criteria

- The browser chrome band has enough height that the URL input and Go button render without obvious vertical compression.
- The notes pane no longer stays stuck on `Loading workspace...` after workspace startup succeeds.
- Normal startup no longer logs `Workspace is not open` for `workspace:get-state`.
- The browser chrome and notes renderers both recover cleanly if `loadState()` is temporarily unavailable during initialization.
- `WorkspaceController` applies the updated browser chrome height consistently on first layout, splitter drags, and window resize.
- The launcher/workspace E2E flow confirms the notes pane reaches its loaded state and the browser chrome controls are visible.
- Lint, unit tests, Playwright E2E tests, and production build all succeed.

## Acceptance Criteria

- The right-side browser chrome is visually stable and not squashed.
- The workspace opens without noisy `Workspace is not open` errors during normal startup.
- The notes pane transitions out of `Loading workspace...` once workspace state is available.
- The app still uses sibling `WebContentsView`s rather than `BrowserView` or a primary `<webview>` architecture.
- The splitter remains draggable and persistent.
- Notes persistence and browser URL persistence still work.
- Browser security restrictions still apply unchanged.
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Inspect the current startup path in:
   - `src/app/main/index.ts`
   - `src/app/main/create-workspace-window.ts`
   - `src/app/main/register-ipc.ts`
   - `src/features/notes/renderer/App.tsx`
   - `src/features/browser/renderer/App.tsx`
   - `src/features/workspace/main/WorkspaceController.ts`
   to identify the exact causes of the startup race, loading-state stall, and browser chrome compression.

2. Update `electron/plan.md` first so it reflects the startup-race fix, loading-state fix, and browser chrome layout fix before implementation begins.

3. Write failing workspace controller tests for the updated browser chrome height and bounds behavior.

4. Run the workspace controller tests immediately to verify the new height expectations fail against the current `64px` implementation; if they do not fail, tighten the assertions before implementing.

5. Write failing renderer tests for the notes app proving it can leave the loading state after an initial `loadState()` failure once workspace state arrives through the subscription path.

6. Run the notes renderer tests immediately to verify they fail for the expected reason; if they do not fail, strengthen the assertions before implementation.

7. Write failing renderer tests for the browser chrome app if needed to confirm it tolerates the same initialization path and still updates from workspace state changes.

8. Run those browser renderer tests immediately to confirm they fail before implementation; if they do not fail, correct the test first.

9. Update the launcher/workspace Playwright coverage to assert:
   - browser chrome controls are fully visible
   - the notes pane no longer remains on `Loading workspace...`

10. Run the relevant Playwright spec immediately to verify the new startup assertions fail first; if they do not fail, tighten the expectation before implementing.

11. Refactor workspace startup so the workspace bundle becomes available to IPC before page-load completion work finishes.

12. Update main-process workspace state handling so normal startup no longer throws `Workspace is not open` for `workspace:get-state`.

13. Update the notes renderer so an initial `loadState()` miss does not leave the status permanently stuck in the loading state.

14. Update the browser chrome renderer to tolerate the same startup timing and keep its URL state synchronized through workspace updates.

15. Increase the reserved browser chrome height in the workspace controller and update the browser chrome styling so the URL bar lays out comfortably in a single compact row.

16. Re-run the directly related unit and renderer tests until the controller, notes, and browser chrome suites all pass.

17. Re-run the updated Playwright startup spec and keep fixing behavior until the loading-state and visibility checks pass.

18. Update `electron/docs/architecture.md` and `electron/README.md` so they describe the improved startup behavior and current browser chrome layout accurately.

19. If a browser skill is available, use it for smoke checks of the visible browser chrome change. No browser skill is available in this environment, so use Playwright Electron coverage as the browser-level smoke baseline.

20. Run the full automated verification suite:
   - lint
   - unit and renderer tests
   - Playwright Electron E2E tests
   - build/package flow

21. Stop only after the browser chrome is visually stable, the startup race is resolved, the loading state clears correctly, all fail-first checks were exercised where new seams were introduced, and every automated verification step passes.
