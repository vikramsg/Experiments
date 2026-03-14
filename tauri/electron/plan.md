## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Build an Electron app under `electron/` that starts on a launcher window and opens a split workspace with:
  - a left notes editor
  - a center draggable splitter
  - a right browser pane for visiting sites
- Use `BaseWindow` with sibling `WebContentsView`s for the workspace, matching the research direction in `electron/research.md`.
- Keep the implementation isolated from the existing Tauri app in `hello-world/`.
- Persist both note content and splitter width under Electron `userData`.

## Architecture Diagram

```text
Runtime architecture
====================

+-------------------------------------------------------------+
| Launcher BrowserWindow                                      |
|  - local renderer                                           |
|  - one app card: "Browser + Notes"                         |
+----------------------------+--------------------------------+
                             |
                             | launch app
                             v
+-------------------------------------------------------------+
| Workspace BaseWindow                                         |
|                                                             |
|  +----------------+ +----------+ +------------------------+ |
|  | Notes          | | Splitter | | Browser                | |
|  | WebContentsView| | WebCV    | | WebContentsView        | |
|  | local UI       | | local UI | | remote site            | |
|  +----------------+ +----------+ +------------------------+ |
|                                                             |
|  main process owns bounds for all three sibling views       |
+-------------------------------------------------------------+
```

```text
Drag flow
=========

User drag on splitter
    |
    v
Splitter renderer pointer events
    |
    v
preload API -> IPC to main
    |
    v
workspace layout controller updates split position
    |
    v
main recomputes bounds for:
- notes view
- splitter view
- browser view
    |
    v
setBounds() on all sibling WebContentsViews
```

```text
Persistence flow
================

Notes renderer ----save/load----> preload ----IPC----> main store
                                                   |
                                                   v
                                          app.getPath('userData')

Splitter drag ----persist width----> preload ----IPC----> main store
```

## Current Status

- `electron/research.md` exists and already captures the recommended Electron architecture and citations.
- The existing app implementation in this repo is the Tauri app under `hello-world/`, not Electron.
- The Tauri app already shows patterns worth reusing conceptually:
  - launcher flow in `hello-world/src/features/app-shell/AppSelector.tsx`
  - editor app structure in `hello-world/src/features/editor/TextEditorApp.tsx`
  - Vitest + React Testing Library patterns in `hello-world/src/features/editor/TextEditorApp.test.tsx`
  - test configuration in `hello-world/vitest.config.ts`
- The Electron research currently treats draggable splitter support as a future enhancement; this plan promotes it into the MVP.

## Short summary of changes

- Scaffold a separate Electron app under `electron/`.
- Create `electron/docs/reference.md` by extracting stable Electron guidance and citations from `electron/research.md`.
- Keep `electron/plan.md` as the implementation-only plan for the current app.
- Create a launcher window with one app card, `Browser + Notes`.
- Create a workspace window using sibling `WebContentsView`s for:
  - notes pane
  - splitter handle
  - browser pane
- Add a layout controller that updates pane bounds on drag and on window resize.
- Add a persistent store for:
  - note content
  - splitter width
- Add preload and IPC boundaries for launcher actions, note persistence, and splitter dragging.
- Add tests first for layout math, controller behavior, splitter drag behavior, note persistence, launcher flow, and Playwright Electron end-to-end coverage.

### Options considered

1. `BaseWindow` + three sibling `WebContentsView`s for notes, splitter, and browser
   - aligns with `electron/research.md`
   - keeps remote browser content in a native Electron view
   - allows mouse dragging through a dedicated local splitter surface
   - recommended

2. DOM layout with `<webview>` inside a standard renderer
   - easier splitter dragging in pure HTML/CSS
   - conflicts with the research direction against using `<webview>` as the primary design
   - rejected

3. `BrowserView`
   - older examples exist
   - deprecated in current Electron guidance
   - rejected

## Files to be changed

- `electron/package.json`
- `electron/tsconfig.json`
- `electron/vitest.config.ts`
- `electron/src/main.ts`
- `electron/src/preloads/launcher.ts`
- `electron/src/preloads/notes.ts`
- `electron/src/preloads/splitter.ts`
- `electron/src/launcher/App.tsx`
- `electron/src/notes/App.tsx`

## Files to be added

- `electron/docs/reference.md`
- `electron/e2e/launcher-workspace.spec.ts`
- `electron/e2e/splitter-drag.spec.ts`
- `electron/e2e/persistence.spec.ts`
- `electron/playwright.config.ts`
- `electron/forge.config.ts`
- `electron/vite.main.config.ts`
- `electron/vite.preload.config.ts`
- `electron/vite.renderer.config.ts`
- `electron/src/shared/split-layout.ts`
- `electron/src/shared/split-layout.test.ts`
- `electron/src/main/workspace-controller.ts`
- `electron/src/main/workspace-controller.test.ts`
- `electron/src/main/note-store.ts`
- `electron/src/main/note-store.test.ts`
- `electron/src/launcher/main.tsx`
- `electron/src/launcher/App.test.tsx`
- `electron/src/notes/main.tsx`
- `electron/src/notes/App.test.tsx`
- `electron/src/splitter/main.tsx`
- `electron/src/splitter/SplitterHandle.tsx`
- `electron/src/splitter/SplitterHandle.test.tsx`
- `electron/launcher.html`
- `electron/notes.html`
- `electron/splitter.html`

## Verification Criteria

- `electron/docs/reference.md` exists and contains the reusable Electron guidance extracted from `electron/research.md`.
- The Electron app launches under Playwright Electron automation.
- The launcher window renders one app card.
- Clicking the app card opens a separate workspace window.
- The workspace contains:
  - left notes pane
  - center draggable splitter
  - right browser pane
- Playwright can drag the splitter with the mouse and verify pane widths update live in both directions.
- Splitter width is clamped to safe minimums and persisted across relaunch.
- Notes persist across workspace close/reopen and full app relaunch.
- Right-side browser content loads with the security constraints described in `electron/research.md`.
- All newly added tests fail first, then pass after implementation.
- Lint, unit tests, Playwright E2E tests, and production build all succeed.
- No manual verification step is required.

## Acceptance Criteria

- `electron/plan.md` exists and reflects the current implementation scope only.
- `electron/docs/reference.md` exists and holds the reusable Electron reference material extracted from `electron/research.md`.
- The Electron workspace uses sibling `WebContentsView`s rather than `BrowserView` or a primary `<webview>` architecture.
- The splitter is draggable by mouse left/right in the MVP.
- The layout remains stable on window resize and on repeated splitter drags.
- The notes pane saves and restores content from persistent storage.
- The browser pane can navigate to external sites.
- The app closes cleanly without leaving orphaned `webContents`.
- The implementation is covered by automated tests for:
  - layout math
  - splitter interaction
  - note persistence
  - launcher flow
  - workspace controller behavior
  - Electron end-to-end behavior with Playwright
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Create `electron/docs/reference.md` by extracting the stable Electron guidance and source citations from `electron/research.md`.

2. Keep `electron/plan.md` focused only on the present implementation, including draggable splitter support in MVP scope.

3. Scaffold a separate Electron app under `electron/` using the Forge/Vite direction already captured in `electron/research.md`.

4. Add test tooling before feature work:
   - wire Vitest and React Testing Library in the Electron app
   - add Playwright with Electron support
   - define scripts for unit, renderer, E2E, and full verification runs

5. Write failing tests for pure split-layout math first:
   - default split width
   - min/max clamping
   - window resize recomputation
   - bounds output for notes, splitter, and browser panes

6. Run the split-layout tests immediately to prove they fail; if they do not fail, correct the tests before writing logic.

7. Implement the shared split-layout module and rerun the tests until they pass.

8. Write failing tests for the splitter handle renderer:
   - pointer down starts drag
   - pointer move emits the correct delta or absolute x
   - pointer up stops drag

9. Run the splitter tests to verify they fail before implementation; if they pass unexpectedly, tighten the assertions.

10. Implement the splitter renderer and preload bridge, then rerun the tests to passing.

11. Write failing tests for the workspace controller in the main process:
   - creates three sibling views
   - applies initial bounds
   - updates bounds on drag
   - updates bounds on window resize
   - closes child `webContents` on teardown

12. Run the workspace controller tests to confirm failure, then implement the controller and rerun to passing.

13. Write failing tests for note persistence:
   - saves note content
   - reloads note content on reopen
   - saves splitter width
   - reloads splitter width on reopen

14. Run the persistence tests to verify they fail first, then implement the store against `userData` and rerun to passing.

15. Write failing launcher UI tests:
   - shows one app card
   - clicking launch triggers the workspace-open API

16. Run the launcher tests to prove they fail first, then implement the launcher renderer and rerun to passing.

17. Write failing notes UI tests:
   - renders saved content
   - updates content on edit
   - triggers save flow
   - surfaces saving state if included

18. Run the notes tests to verify they fail first, then implement the notes renderer and rerun to passing.

19. Wire all IPC and preload surfaces together:
   - launcher -> main open-workspace action
   - splitter -> main drag updates
   - notes -> main load/save actions

20. Implement remote browser security controls in the main process according to the constraints already captured in `electron/research.md`.

21. Write failing Playwright Electron E2E tests for:
   - launcher window renders
   - app card opens workspace
   - splitter drag changes pane widths
   - window resize preserves valid pane layout
   - note content persists across relaunch
   - browser pane loads a configured URL

22. Run the Playwright tests to verify they fail before implementation; if they do not fail, fix the assertions and rerun.

23. Implement any missing app wiring needed for the Playwright flows, then rerun E2E tests until they pass.

24. Run the full automated verification suite:
   - lint
   - unit tests
   - renderer tests
   - Playwright Electron E2E tests
   - build

25. Stop only after every automated verification step passes.
