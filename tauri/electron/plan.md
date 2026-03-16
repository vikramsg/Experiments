## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Change the existing `OpenCode` launcher so it opens a split `OpenCode + Browser`
  window instead of a standalone OpenCode-only window.
- Keep `Browser + Notes` as the separate launcher app and do not add a third
  launcher.
- Keep browser ownership and screenshot capture in Electron main.
- Keep OpenCode as the MCP client only; it must not own browser services or a
  second MCP client.
- Move browser-facing MCP ownership to the browser/app-main side and keep
  feature wiring in `app/*`, not through feature-to-feature imports.
- Add a real draggable splitter between OpenCode and the browser in the
  OpenCode launcher window.
- Store the last 10 visited browser URLs for shared browser-input autocomplete
  across both Browser + Notes and OpenCode + Browser. This history is browser UX
  only and must not be added to OpenCode MCP context.
- Simplify the OpenCode pane so the left side renders only the chat thread and
  composer, not dashboard-style headings or status cards.
- Update `electron/README.md` and `electron/docs/architecture.md` so they
  explicitly require clean domain-based separation:
  - browser-owned services live with browser or app composition
  - OpenCode-owned services live with OpenCode
  - `app/*` composes features
  - features must not import each other directly
- Add real integration coverage for the browser MCP path so we verify:
  - MCP server is connected inside OpenCode
  - `browser_browser_context_current` is registered with OpenCode
  - the model actually invokes the tool in a real session
  - current mock-mode E2E does not stand in for real MCP verification
- Preserve the existing browser security model, repo read-only OpenCode posture,
  and current Browser + Notes behavior.

## Architecture Diagram

```text
Launcher outcomes
=================

Launcher BrowserWindow
   |
   +--> Browser + Notes
   |      `--> existing split workspace
   |
   `--> OpenCode
          `--> split OpenCode + Browser window
```

```text
Target OpenCode window
======================

BaseWindow
+------------------------------------------------------------------+
| OpenCode chat                    | Browser chrome                |
| local renderer                   | local renderer                |
|                                  +-------------------------------+
| chat thread                      | Browser content               |
| composer                         | remote page                   |
|                                  |                               |
+------------------------------------------------------------------+

splitter lives between the left and right panes and adjusts their widths.

browser chrome input uses shared last-10 URL autocomplete.

All child views remain siblings under one BaseWindow.
```

```text
Ownership and composition
=========================

features/browser/main
   +--> browser-session.ts
   +--> browser-context.ts
   +--> BrowserHistoryStore.ts
   `--> BrowserMcpServer.ts

features/opencode/main
   `--> OpenCodeService.ts

app/main/create-opencode-window.ts
   +--> composes browser views
   +--> starts BrowserMcpServer
   +--> passes MCP connection info to OpenCodeService
   `--> owns split window wiring
```

```text
Tool call flow
==============

User asks: "what do you see in the browser?"
   |
   v
OpenCode model
   |
   `--> calls browser_context_current
         |
         v
OpenCode MCP client
   |
   v
BrowserMcpServer (Electron main)
   |
   +--> browserView.webContents.getURL()
   +--> browserView.webContents.capturePage()
   `--> returns text + screenshot attachment
```

```text
Verification flow for real MCP
==============================

OpenCodeService.initialize()
   |
   +--> GET /mcp
   |      `--> assert browser MCP status is connected
   |
   +--> GET /experimental/tool/ids
   |      `--> assert browser_browser_context_current exists
   |
   `--> only then allow browser-aware UX to be considered ready
```

```text
Shared URL history flow
=======================

browser navigation or direct URL submit
   |
   v
BrowserHistoryStore.remember(url)
   |
   +--> dedupe
   +--> keep last 10
   +--> persist to userData
   `--> publish updated suggestions to all browser chrome surfaces
```

## Current Status

- The current `OpenCode` launcher still opens a standalone full-window OpenCode
  view in `electron/src/app/main/create-opencode-window.ts`.
- The current browser MCP wiring points at `getWorkspace()?.browserView`, which
  means it inspects the browser in the separate Browser + Notes app instead of a
  browser that lives alongside OpenCode.
- Browser navigation IPC is still hard-wired to the Browser + Notes workspace in
  `electron/src/app/main/register-ipc.ts` through `requireWorkspace()`.
- The right-side browser composition already exists and works in
  `electron/src/app/main/create-workspace-window.ts`, including browser chrome,
  remote browser content, security policy, and navigation synchronization.
- Browser context capture already exists in
  `electron/src/features/browser/main/browser-context.ts`, but MCP ownership is
  currently misplaced because it was added under OpenCode instead of under the
  browser/app-main domain.
- The docs do not yet clearly state that services and adapters should always be
  owned by their domain and composed only from `app/*`.
- Current end-to-end coverage relies on `ELECTRON_OPENCODE_MOCK=1`, which proves
  window composition and mocked reply flow but does not prove real MCP
  registration, real tool visibility, or real tool invocation.

## Short summary of changes

- Replace the standalone OpenCode window with a split `OpenCode + Browser`
  `BaseWindow`.
- Reuse the browser host pattern from `create-workspace-window.ts` for the right
  side of the OpenCode launcher.
- Move `BrowserMcpServer` out of `features/opencode/*` into the browser/app-main
  ownership side.
- Keep `OpenCodeService` focused on OpenCode server lifecycle and MCP config
  only.
- Introduce browser-owned or root-boundary browser IPC/contracts as needed so
  browser chrome can control the OpenCode window's browser without relying on
  `workspace:*` semantics.
- Add explicit real-session MCP verification in app code and tests:
  - check MCP connection status
  - check tool registration
  - record actual tool invocation for browser-aware asks
- Add a shared browser history store for URL autocomplete only.
- Strip the OpenCode left pane down to chat messages and the prompt composer.
- Update docs so the architecture explicitly enforces domain ownership and
  composition rules.

### Options considered

1. Keep standalone OpenCode and separate Browser + Notes, connected through MCP
   - technically works
   - awkward user experience
   - wrong mental model for "OpenCode sees the browser next to it"
   - rejected

2. Add a third launcher app for `OpenCode + Browser`
   - isolates the new feature
   - user explicitly rejected this direction
   - rejected

3. Change the existing OpenCode launcher into `OpenCode + Browser`
   - matches requested product behavior
   - keeps Browser + Notes separate
   - reuses existing browser composition and MCP work
   - recommended

4. Let `features/opencode/*` import browser services directly
   - expedient
   - violates feature boundary rules
   - rejected

## Files to be changed

- `electron/plan.md`
- `electron/README.md`
- `electron/docs/architecture.md`
- `electron/src/app/main/index.ts`
- `electron/src/app/main/register-ipc.ts`
- `electron/src/app/main/create-opencode-window.ts`
- `electron/src/app/main/create-workspace-window.ts`
- `electron/src/features/browser/main/browser-session.ts`
- `electron/src/features/browser/main/browser-session.test.ts`
- `electron/src/features/browser/main/browser-context.ts`
- `electron/src/features/browser/main/browser-context.test.ts`
- `electron/src/features/browser/renderer/App.tsx`
- `electron/src/features/browser/renderer/App.test.tsx`
- `electron/src/features/opencode/main/OpenCodeService.ts`
- `electron/src/features/opencode/main/OpenCodeService.test.ts`
- `electron/src/features/opencode/renderer/App.tsx`
- `electron/src/features/opencode/renderer/App.test.tsx`
- `electron/src/features/splitter/renderer/main.tsx`
- `electron/src/ipc.ts`
- `electron/src/types.d.ts`
- `electron/src/workspace-contract.ts`
- `electron/src/workspace-model.ts`
- `electron/e2e/opencode.spec.js`
- `electron/e2e/opencode-real-mcp.spec.js`
- `electron/e2e/launcher-workspace.spec.js`

## Files to be added

- `electron/src/features/browser/main/BrowserMcpServer.ts`
- `electron/src/features/browser/main/BrowserMcpServer.test.ts`
- `electron/src/features/browser/main/BrowserHistoryStore.ts`
- `electron/src/features/browser/main/BrowserHistoryStore.test.ts`
- `electron/src/browser-contract.ts`
- `electron/src/browser-model.ts`
- `electron/src/app/preload/browser.ts`
- `electron/src/app/main/OpenCodeBrowserController.ts`
- `electron/src/app/main/opencode-splitter-preload.ts`
- `electron/src/app/renderer/entries/opencode-splitter.html`
- `electron/src/features/splitter/renderer/opencode-main.tsx`
- `electron/e2e/opencode-browser-layout.spec.js`
- `electron/e2e/opencode-mcp-registration.spec.js`

## Verification Criteria

- The `OpenCode` launcher opens a split window with OpenCode on the left and the
  browser on the right.
- The Browser + Notes launcher still opens the existing Browser + Notes app.
- The OpenCode window's browser chrome controls its own browser surface, not the
  Browser + Notes browser.
- The OpenCode window includes a draggable splitter between OpenCode and the
  browser.
- Browser URL inputs in both launchers offer shared autocomplete suggestions
  from the last 10 visited URLs.
- URL history stays in the browser UX domain only and is not added to MCP tool
  context.
- OpenCode can ask what it sees in the browser by calling the browser MCP tool.
- OpenCode startup verifies that browser MCP is connected through `/mcp`.
- OpenCode startup verifies that `browser_browser_context_current` is registered
  through `/experimental/tool/ids`.
- Real integration coverage proves the browser tool is actually invoked, not
  just that a mocked answer mentions browser context.
- The MCP server used for browser inspection is owned by the browser/app-main
  side, not by OpenCode feature ownership.
- No feature imports another feature directly.
- `app/*` remains the composition layer for wiring browser + OpenCode together.
- `electron/README.md` and `electron/docs/architecture.md` explicitly describe
  clean domain-based separation for features, services, adapters, and MCP
  ownership.
- The renderer still has no direct `webContents`, screenshot, shell, or raw
  filesystem access.
- Lint, unit tests, Playwright Electron E2E tests, and build all pass.

## Acceptance Criteria

- The `OpenCode` launcher no longer opens an OpenCode-only window.
- The `OpenCode` launcher opens a split `OpenCode + Browser` workspace.
- OpenCode explanations about the browser are based on the browser that is in
  the same window as OpenCode.
- Browser MCP ownership is moved out of OpenCode and into the correct domain.
- Real non-mock verification proves MCP is connected, the browser tool is
  registered, and browser-aware prompts invoke it.
- Browser + Notes still works as a separate launcher app.
- The OpenCode left pane shows only chat messages and the prompt composer.
- Docs clearly state the rule that services/adapters belong to their domain and
  must be composed from `app/*`.
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Update `electron/plan.md` first so it reflects:
   - `OpenCode` launcher becomes `OpenCode + Browser`
   - Browser + Notes remains separate
   - MCP server ownership moves to browser/app-main
   - browser/OpenCode composition stays in `app/*`
   - docs must explicitly require clean domain-based separation
   - real MCP verification must not rely on mock mode

2. Inspect the current seams in:
   - `electron/src/app/main/create-opencode-window.ts`
   - `electron/src/app/main/create-workspace-window.ts`
   - `electron/src/app/main/index.ts`
   - `electron/src/app/main/register-ipc.ts`
   - `electron/src/features/browser/main/browser-context.ts`
   - `electron/src/features/opencode/main/OpenCodeService.ts`
   so the exact ownership and composition changes are clear before writing
   tests.

3. Write failing unit tests for the browser MCP ownership move in
   `electron/src/features/browser/main/BrowserMcpServer.test.ts` that assert:
   - the MCP server exposes `browser_context_current`
   - it returns URL + screenshot
   - it does not depend on OpenCode feature code

4. Run those targeted browser MCP tests immediately to confirm they fail first;
   if they do not fail, tighten them before implementing.

5. Write failing tests for the new OpenCode window composition in the relevant
   window/controller test files so they assert:
   - OpenCode view is present on the left
   - browser chrome and browser content are present on the right
   - resize/layout still works
   - teardown closes all child views

6. Run those targeted composition tests immediately to confirm they fail first;
   if not, refine the assertions before implementing.

7. Write failing tests for browser IPC / contract decoupling so browser chrome
   is no longer tied only to `requireWorkspace()` and can control the OpenCode
   launcher's browser surface.

8. Run those targeted IPC/browser contract tests immediately to confirm the
   current ownership is wrong before implementation.

9. Write failing renderer tests for the OpenCode app that assert:
   - the browser-aware helper copy still appears
   - the user can ask what OpenCode sees in the browser
   - browser-aware answers continue to render correctly

10. Run the targeted OpenCode renderer tests immediately to confirm they fail
    for the expected reason if the window/browser ownership is not yet wired.

11. Write a failing Playwright Electron spec in
    `electron/e2e/opencode-browser-layout.spec.js` that asserts:
    - launcher opens OpenCode
    - the OpenCode app opens with browser visible on the right
    - a splitter is visible and draggable
    - browser chrome works in that same window
    - asking what OpenCode sees in the browser returns a browser-aware answer

12. Run the targeted Playwright spec immediately to confirm the assertions fail
     first; if they do not fail, tighten them before continuing.

13. Write failing integration tests for real MCP registration, separate from UI
    layout tests, that assert:
    - `GET /mcp` reports the browser MCP client as connected
    - `GET /experimental/tool/ids` contains `browser_browser_context_current`
    - these checks run without `ELECTRON_OPENCODE_MOCK=1`

14. Run those targeted MCP registration tests immediately to confirm they fail
    first; if they do not fail, tighten them before continuing.

15. Write failing integration tests for real tool invocation that assert:
    - a browser-aware prompt causes the browser MCP tool invocation counter or
      telemetry to increment
    - the test does not pass purely because of a mocked answer string

16. Run those targeted invocation tests immediately to confirm they fail first;
    if they do not fail, strengthen the telemetry/assertions before continuing.

17. Move `BrowserMcpServer` into browser/app-main ownership, keeping screenshot
    capture in browser-owned code and keeping OpenCode free of browser service
    ownership.

18. Add a browser-owned shared history store for the last 10 visited URLs and
    wire it into all browser hosts for autocomplete only.

19. Refactor `electron/src/app/main/create-opencode-window.ts` so it composes a
    split `BaseWindow` with:
    - OpenCode view
    - splitter view
    - browser chrome view
    - browser content view

20. Reuse or extract the browser host/composition seam from
    `electron/src/app/main/create-workspace-window.ts` so browser setup,
    security policy, and navigation sync do not get duplicated carelessly.

21. Introduce a browser-owned or root-boundary browser contract/preload/IPC
    path so the browser chrome renderer can control either launcher's browser
    surface without relying on notes-specific workspace semantics.

22. Add an OpenCode-specific splitter adjustment path and controller so the left
    and right panes can be resized without notes-specific workspace IPC.

23. Update `electron/src/app/main/index.ts` so the OpenCode launcher creates the
    split OpenCode window and wires the browser MCP server to that window's own
    browser view.

24. Extend `electron/src/features/opencode/main/OpenCodeService.ts` so it:
    - verifies MCP connection status through `/mcp`
    - verifies tool registration through `/experimental/tool/ids`
    - exposes a visible failure state if browser MCP is unavailable
    - records or surfaces enough telemetry to confirm real tool invocation in
      integration tests

25. Simplify `electron/src/features/opencode/renderer/App.tsx` so the left pane
    only renders the chat thread and prompt composer, with no dashboard headers
    or side cards.

26. Keep `electron/src/features/opencode/main/OpenCodeService.ts` limited to:
    - OpenCode process lifecycle
    - MCP client config injection
    - read-only repo permission config
    and do not let it import browser feature services directly.

27. Re-run the targeted browser MCP, composition, splitter, browser-history,
    IPC, renderer, registration,
    and real invocation tests
    immediately after implementation; if they do not pass, fix the
    implementation before moving on.

28. Re-run the targeted Playwright layout/browser-awareness spec immediately
    after implementation; if it does not pass, continue refining the split
    window wiring before moving on.

29. Update `electron/README.md` so it explicitly states:
    - services, adapters, and integrations belong to their domain
    - cross-domain wiring belongs in `app/*`
    - features must not import other features directly
    - shared browser URL history powers autocomplete only and is not MCP context

30. Update `electron/docs/architecture.md` so it explicitly documents:
    - domain-based ownership for browser, OpenCode, and app composition
    - where MCP server ownership belongs
    - shared browser history ownership and persistence
    - OpenCode splitter ownership
    - why feature-to-feature imports remain forbidden

31. Add only the minimal necessary inline comments where browser host reuse,
    browser history persistence, MCP ownership, or split-window composition
    would otherwise be non-obvious.

32. If a browser skill is available, use it for browser-level smoke checks of
    the final OpenCode + Browser layout. No browser skill is available in this
    environment, so use Playwright Electron coverage as the browser-level smoke
    baseline.

33. Run the full automated verification suite:
   - lint
   - unit and renderer tests
   - real MCP registration / invocation integration tests
   - Playwright Electron E2E tests
   - build/package flow

34. Stop only after:
   - OpenCode launcher opens OpenCode + Browser in one window
   - a splitter resizes OpenCode and browser panes
   - Browser + Notes remains separate
   - browser MCP ownership is in the correct domain
   - shared browser autocomplete shows the last 10 visited URLs across both browser surfaces
   - recent URL history is not added to MCP context
   - real MCP registration is verified through OpenCode APIs
   - real tool invocation is verified without mock mode
   - the OpenCode left pane shows only chat and composer
   - docs clearly state clean domain-based separation rules
   - all automated verification passes
