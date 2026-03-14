## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria are met.
Start with a test first approach by writing failing tests, confirming they fail,
and only then proceeding with implementation.

- Add browser back and forward controls to the local browser chrome.
- Keep the browser chrome URL field synchronized with the actual remote page URL
  when navigation happens inside the embedded browser, including same-page and
  in-page navigation.
- Preserve the previously implemented fixes and behavior:
  - launcher flow
  - workspace startup race fix
  - notes loading-state recovery
  - workspace with sibling `WebContentsView`s
  - draggable splitter
  - notes persistence
  - browser URL persistence
  - current browser security constraints
- Keep the Electron app under `electron/` in its current feature-first,
  process-aware structure.
- Keep the implementation isolated from the Tauri app in `hello-world/`.

## Architecture Diagram

```text
Current browser navigation flow
===============================

Browser chrome renderer
   |
   +--> setBrowserUrl(url)
   |      |
   |      `--> main process loads the remote page
   |
   `--> remote page later changes itself ----------------------+
          |                                                    |
          +--> browser navigates via links/history/hash        |
          +--> no main-process listener publishes new URL      |
          +--> workspace snapshot stays stale                  |
          `--> browser chrome keeps showing the old URL <------+
```

```text
Target browser navigation flow
==============================

Browser chrome renderer
   |
   +--> setBrowserUrl(url) / goBack() / goForward()
   |      |
   |      `--> main process acts on browser webContents
   |
   `--> browser webContents emits navigation events
          |
          +--> main process reads getURL(), canGoBack(), canGoForward()
          +--> WorkspaceController updates browser navigation state
          +--> workspace state is published to browser chrome/notes renderers
          `--> browser chrome reflects the real page URL and history state
```

```text
Target right pane
=================

+---------------------------------------------------+
| [Back] [Forward] [input......................] Go |
+---------------------------------------------------+
| Remote browser content                           |
|                                                   |
+---------------------------------------------------+

Requirements:
- browser chrome remains visually stable and not squashed
- Back/Forward state reflects real history availability
- URL field tracks link clicks, redirects, and in-page navigation
```

## Current Status

- `electron/src/features/browser/renderer/App.tsx` currently renders only the
  URL field and Go button.
- `electron/src/types.d.ts`, `electron/src/app/preload/workspace.ts`,
  `electron/src/shared/ipc/channels.ts`, and
  `electron/src/app/main/register-ipc.ts` currently expose only
  `setBrowserUrl(...)` for browser navigation commands.
- `electron/src/app/main/create-workspace-window.ts` loads the remote browser
  page but does not publish live navigation updates from the browser
  `webContents` back into workspace state.
- `electron/src/shared/types/workspace.ts` currently has no browser history
  booleans, so the browser chrome cannot know whether back or forward is
  available.

## Short Summary Of Changes

- Update the plan to include browser back/forward controls, live URL sync, code
  comment updates, and documentation updates.
- Write failing tests for:
  - browser chrome back/forward controls and disabled state
  - browser chrome URL updates from workspace state changes
  - main-process browser navigation state synchronization
  - Playwright confirmation that the browser chrome tracks real navigation and
    exposes back/forward controls
- Extend the shared workspace contract to include live browser navigation state.
- Add preload and IPC support for back and forward actions.
- Publish browser navigation state changes from the remote browser
  `webContents` into the workspace controller.
- Update browser chrome UI and styling to support back/forward controls without
  regressing the existing layout fix.
- Update code comments and docs after implementation to match the final design.
- Re-run full verification after the fixes.

### Options Considered

1. Add browser navigation state synchronization in the current architecture
   - smallest targeted repair
   - preserves the current four-view workspace design
   - recommended

2. Only add visible back/forward buttons in the renderer
   - improves discoverability
   - leaves the buttons disconnected from real browser history
   - rejected

3. Poll the browser URL from the renderer
   - introduces unnecessary coupling and timing complexity
   - avoids the existing main-process state authority
   - rejected

## Files To Be Changed

- `electron/plan.md`
- `electron/src/shared/types/workspace.ts`
- `electron/src/types.d.ts`
- `electron/src/shared/ipc/channels.ts`
- `electron/src/app/preload/workspace.ts`
- `electron/src/app/main/create-workspace-window.ts`
- `electron/src/app/main/register-ipc.ts`
- `electron/src/features/workspace/main/WorkspaceController.ts`
- `electron/src/features/workspace/main/WorkspaceController.test.ts`
- `electron/src/features/browser/main/browser-session.ts`
- `electron/src/features/browser/main/browser-session.test.ts`
- `electron/src/features/browser/renderer/App.tsx`
- `electron/src/features/browser/renderer/App.test.tsx`
- `electron/e2e/launcher-workspace.spec.js`
- `electron/docs/architecture.md`
- `electron/README.md`

## Files To Be Added

- None required unless a dedicated browser navigation sync helper test proves
  necessary during implementation.

## Verification Criteria

- The browser chrome shows Back and Forward controls alongside the URL field and
  Go button without obvious compression.
- Back and Forward controls are enabled and disabled according to real browser
  history state.
- The browser chrome URL updates when the embedded browser navigates through
  links, redirects, back/forward actions, or in-page navigation.
- Browser URL persistence still works when reopening the workspace.
- Notes persistence, startup-race handling, loading-state recovery, splitter
  behavior, and current browser security restrictions still work unchanged.
- Lint, unit tests, Playwright E2E tests, and production build all succeed.

## Acceptance Criteria

- The right-side browser chrome visibly includes working Back and Forward
  controls.
- The URL field always reflects the actual current URL of the embedded browser.
- Back and Forward actions work from the browser chrome and keep the control
  state synchronized.
- The app still uses sibling `WebContentsView`s rather than `BrowserView` or a
  primary `<webview>` architecture.
- The splitter remains draggable and persistent.
- Notes persistence and browser URL persistence still work.
- Browser security restrictions still apply unchanged.
- Relevant code comments are updated where needed to explain the non-obvious
  browser navigation synchronization behavior.
- `electron/docs/architecture.md` and `electron/README.md` are updated to match
  the final browser navigation behavior and current UI.
- Work is not complete until all automated verification passes.

## Checklist Of Tasks To Be Done

1. Update `electron/plan.md` first so it reflects the back/forward control
   work, URL synchronization work, comment updates, docs updates, and complete
   verification scope before implementation begins.

2. Inspect the current browser navigation path in:
   - `electron/src/features/browser/renderer/App.tsx`
   - `electron/src/app/preload/workspace.ts`
   - `electron/src/shared/ipc/channels.ts`
   - `electron/src/app/main/register-ipc.ts`
   - `electron/src/app/main/create-workspace-window.ts`
   - `electron/src/features/workspace/main/WorkspaceController.ts`
   - `electron/src/shared/types/workspace.ts`
   to confirm the missing controls and missing navigation-state publication seam.

3. Write failing renderer tests in
   `electron/src/features/browser/renderer/App.test.tsx` for:
   - rendering Back and Forward buttons
   - calling new API methods for back/forward
   - disabling buttons when history is unavailable
   - reflecting URL and history updates from workspace state changes

4. Run the browser renderer test suite immediately to verify those new tests
   fail against the current implementation; if they do not fail, tighten the
   assertions before implementing.

5. Write failing main-process tests for the browser navigation synchronization
   seam in an existing or new focused test file so they prove navigation events
   update the workspace snapshot with current URL and history booleans.

6. Run the relevant main-process tests immediately to verify the navigation-sync
   assertions fail before implementation; if they do not fail, strengthen the
   test first.

7. Update the Playwright launcher/workspace coverage to assert:
   - browser chrome Back and Forward controls are visible
   - the URL field changes after navigating inside the remote page
   - Back returns to the prior page and Forward returns again

8. Run the updated Playwright spec immediately to verify the new assertions fail
   first; if they do not fail, tighten the expectation before implementing.

9. Extend the shared workspace state contract so browser navigation state can be
   published to the renderer safely.

10. Add preload and IPC support for browser back/forward actions.

11. Refactor main-process workspace/browser integration so the remote browser
    `webContents` publishes current URL plus history availability back into the
    workspace controller on initial load and on subsequent navigation changes.

12. Update `WorkspaceController` so browser navigation state changes publish a
    fresh workspace snapshot to subscribed renderers.

13. Update the browser chrome renderer to render Back and Forward controls,
    consume the new browser navigation state, and keep the input value in sync
    with workspace state updates.

14. Preserve existing startup resilience, notes recovery behavior, persistence,
    splitter layout behavior, and browser security rules while introducing the
    new navigation controls.

15. Add or update only the necessary code comments to explain the non-obvious
    browser navigation synchronization flow.

16. Re-run the directly related unit and renderer tests until all browser,
    workspace, and persistence suites pass.

17. Re-run the updated Playwright startup/navigation spec and keep fixing
    behavior until the visibility and URL synchronization checks pass.

18. Update `electron/docs/architecture.md` and `electron/README.md` so they
    describe the browser navigation synchronization behavior, available browser
    chrome controls, and current workspace architecture accurately.

19. Run the full automated verification suite:
    - lint
    - unit and renderer tests
    - Playwright Electron E2E tests
    - build/package flow

20. Stop only after the browser chrome includes working back/forward controls,
    the URL stays synchronized with real browser navigation, code comments and
    docs are updated, fail-first checks were exercised where new seams were
    introduced, and every automated verification step passes.
