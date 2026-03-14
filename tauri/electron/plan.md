## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Implement the cleanest version of option 2: OpenCode gets browser access
  through a dedicated local MCP server hosted by Electron main, not through
  prompt-time context injection.
- Ensure OpenCode can explicitly call a tool that:
  - reads the current browser URL
  - takes a screenshot of the browser pane
  - returns both to the model so it can explain what it sees
- Keep all privileged browser access in Electron main.
- Update OpenCode config and permissions so the MCP browser tool is actually
  callable while preserving the app's existing read-only posture for repo tools.
- Avoid relying on MCP `resources/read` for this first version; use an MCP tool
  that returns the current browser context directly, including the screenshot.
- Preserve the existing browser security model and OpenCode read-only repo
  permissions.

## Architecture Diagram

```text
Target architecture
===================

User prompt in OpenCode pane
   |
   v
OpenCode session / model
   |
   | decides to call tool: browser_context_current
   v
OpenCode server
   |
   | configured remote MCP server
   v
Electron main hosted MCP endpoint
   |
   +--> browserView.webContents.getURL()
   +--> browserView.webContents.capturePage()
   |
   v
Tool result
   - text: current URL / metadata
   - attachment: screenshot image
   |
   v
OpenCode model
   |
   v
Assistant explains what it sees
```

```text
Privilege boundary
==================

OpenCode renderer
   |
   | no direct browser access
   v
preload / IPC
   |
   v
Electron main
   |
   +--> owns workspace browserView
   +--> owns local MCP server
   +--> owns screenshot capture
```

```text
Why MCP instead of prompt injection
===================================

LLM prompt
   |
   +--> normal question
   |      `--> no browser tool call
   |
   `--> "what do you see in the browser?"
          `--> model calls browser_context_current
                 `--> browser URL + screenshot returned on demand
```

## Current Status

- The current workspace already owns the real browser pane as a
  `WebContentsView` named `browserView`, returned on `WorkspaceBundle`, which is
  the clean seam for browser inspection.
- Browser-specific main-process logic already lives in
  `browser-session.ts`, so browser context capture belongs in the
  browser/main boundary instead of the OpenCode renderer.
- OpenCode is currently spawned as a separate local server process with config
  passed through `OPENCODE_CONFIG_CONTENT`, which makes it straightforward to
  register an MCP server in that config.
- OpenCode already uses a strict deny-by-default permission object, so the new
  MCP server tool will need an explicit allow rule.
- The current OpenCode app still answers ordinary repo questions, but it has no
  way to inspect the live browser pane or capture screenshots.

## Short summary of changes

- Add a small localhost MCP server hosted by Electron main.
- Register that MCP server in the OpenCode server config when spawning
  OpenCode.
- Implement a first MCP tool:
  - `browser_context_current`
- Make that tool:
  - read the current URL
  - capture a screenshot of the browser pane
  - return text metadata plus screenshot attachment
- Keep the OpenCode renderer unchanged from a privilege perspective; the model
  gets browser access by tool call, not by direct bridge access.
- Add targeted tests for:
  - screenshot capture
  - MCP server tool behavior
  - OpenCode config and permission wiring
  - end-to-end "what do you see?" flow

### Options considered

1. Prompt-time injection of URL + screenshot
   - simpler at first glance
   - not actually option 2
   - noisy, expensive, and not model-driven
   - rejected

2. OpenCode custom tool that calls back into Electron
   - possible
   - awkward screenshot/media fit
   - requires extra bridge plumbing anyway
   - rejected

3. Electron-main-hosted MCP server with browser tools
   - matches OpenCode's first-class extensibility model
   - keeps browser privilege in main
   - cleanest path for screenshot-capable tool results
   - recommended

4. Browser context exposed through MCP resources
   - attractive long-term
   - current OpenCode gap around `resources/read`
   - rejected for v1

## Files to be changed

- `electron/plan.md`
- `electron/README.md`
- `electron/docs/architecture.md`
- `electron/package.json`
- `electron/src/app/main/index.ts`
- `electron/src/app/main/register-ipc.ts`
- `electron/src/app/main/create-workspace-window.ts`
- `electron/src/features/browser/main/browser-session.ts`
- `electron/src/features/browser/main/browser-session.test.ts`
- `electron/src/features/opencode/main/OpenCodeService.ts`
- `electron/src/features/opencode/main/OpenCodeService.test.ts`
- `electron/src/opencode-model.ts`
- `electron/src/opencode-contract.ts`
- `electron/src/ipc.ts`
- `electron/src/app/preload/opencode.ts`
- `electron/src/features/opencode/renderer/App.tsx`
- `electron/src/features/opencode/renderer/App.test.tsx`
- `electron/e2e/opencode.spec.js`

## Files to be added

- `electron/src/features/browser/main/browser-context.ts`
- `electron/src/features/browser/main/browser-context.test.ts`
- `electron/src/features/opencode/main/BrowserMcpServer.ts`
- `electron/src/features/opencode/main/BrowserMcpServer.test.ts`
- `electron/e2e/opencode-browser-vision.spec.js`

## Verification Criteria

- OpenCode is configured with a local MCP server for browser tools.
- The MCP server exposes a `browser_context_current` tool.
- That tool returns:
  - current browser URL
  - screenshot of the active browser pane
- The screenshot is produced from Electron `webContents.capturePage()`.
- OpenCode permissions explicitly allow the new MCP browser tool while keeping
  unrelated write and destructive tools denied.
- OpenCode can answer a prompt like `what do you see in the browser?` by calling
  the tool rather than relying on injected prompt context.
- The renderer still does not get direct access to `browserView`, `webContents`,
  or screenshot APIs.
- If the browser is unavailable, loading, or not yet painted, the tool fails
  gracefully with a useful error or fallback message.
- Existing Browser + Notes behavior remains intact.
- Lint, unit tests, Playwright Electron E2E tests, and build all pass.

## Acceptance Criteria

- OpenCode can directly use a tool to inspect the browser.
- That inspection includes a real browser screenshot, not just the URL.
- Asking OpenCode what it sees in the browser results in a screenshot-backed
  explanation.
- The browser screenshot path is owned entirely by Electron main.
- No continuous screenshot or URL injection is required for normal prompts.
- The implementation uses the MCP tool path rather than a custom
  prompt-enrichment shortcut.
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Update `electron/plan.md` first so it reflects:
   - Electron-main-hosted MCP server
   - `browser_context_current` tool
   - URL + screenshot browser inspection
   - explicit OpenCode permission/config updates
   - no continuous context injection
   - fail-first verification flow

2. Inspect the current browser and OpenCode seams in:
   - `electron/src/app/main/create-workspace-window.ts`
   - `electron/src/features/browser/main/browser-session.ts`
   - `electron/src/features/opencode/main/OpenCodeService.ts`
   - `electron/src/app/main/index.ts`
   so the exact boundaries for browser ownership, OpenCode server spawn, and
   config wiring are confirmed before writing tests.

3. Write failing unit tests in
   `electron/src/features/browser/main/browser-context.test.ts` that assert:
   - current URL can be read from the browser pane
   - screenshot capture uses the browser `webContents`
   - screenshot output is serialized into a tool-safe attachment shape
   - loading or blank-browser cases fail predictably

4. Run the targeted browser-context unit tests immediately to confirm they fail
   for the correct reason; if they do not fail, tighten the assertions before
   implementation.

5. Write failing unit tests in
   `electron/src/features/opencode/main/BrowserMcpServer.test.ts` that assert:
   - the MCP server exposes `browser_context_current`
   - the tool returns text metadata plus screenshot attachment
   - the tool never exposes raw Electron internals
   - tool output degrades gracefully when browser context is unavailable

6. Run the targeted MCP server unit tests immediately to confirm they fail
   first; if not, refine the tests before proceeding.

7. Write failing tests in
   `electron/src/features/opencode/main/OpenCodeService.test.ts` that assert:
   - OpenCode server config includes the local MCP server entry
   - permission wiring explicitly allows the MCP browser tool
   - OpenCode startup still succeeds with the new MCP server configuration
   - mock-mode behavior can simulate browser tool answers

8. Run the targeted OpenCode service tests immediately to confirm they fail
   before implementation.

9. Write failing renderer tests in
   `electron/src/features/opencode/renderer/App.test.tsx` that assert:
   - the user can ask what OpenCode sees in the browser
   - the UI handles browser-tool-backed answers cleanly
   - tool-failure states are surfaced to the user if screenshot capture fails

10. Run the targeted OpenCode renderer tests immediately to confirm the UI
    expectations fail first; if they do not, strengthen the tests.

11. Write a failing Playwright Electron E2E spec in
    `electron/e2e/opencode-browser-vision.spec.js` that asserts:
    - launcher opens the relevant workspace/app
    - browser navigates to a known page
    - user asks `what do you see in the browser?`
    - OpenCode responds with an explanation that reflects tool-backed browser
      context
    - the flow remains stable across repeated asks

12. Run the targeted Playwright spec immediately to confirm the assertions
    actually fail; if they do not fail, tighten them before implementation.

13. Install any missing runtime dependencies needed to host an MCP server in the
    Electron main process.

14. Implement `electron/src/features/browser/main/browser-context.ts` so the
    browser feature owns:
    - reading the current URL
    - taking a screenshot with `capturePage()`
    - converting that screenshot into an MCP tool attachment payload
    - returning a serializable browser-context object

15. Extend `electron/src/features/browser/main/browser-session.ts` only where
    needed so browser context reading stays browser-owned and does not leak into
    renderer code.

16. Implement `electron/src/features/opencode/main/BrowserMcpServer.ts` to host
    a localhost MCP endpoint from Electron main that exposes
    `browser_context_current`.

17. Ensure `browser_context_current` returns:
    - a textual summary block with URL and lightweight metadata
    - the screenshot as an attachment/image payload
    so the model can reason over both forms of context.

18. Extend `electron/src/features/opencode/main/OpenCodeService.ts` so spawned
    OpenCode config includes:
    - the local MCP server registration
    - the explicit allow rule for the MCP browser tool
    - any required localhost auth headers

19. Keep permissions tight in OpenCode config so:
    - browser MCP tool access is explicitly enabled
    - unrelated write/destructive tools remain denied
    - repo read-only posture is preserved

20. Update `electron/src/app/main/index.ts` so the browser MCP server lifecycle
    is owned alongside workspace and OpenCode lifecycles:
    - start when needed
    - stop on teardown
    - avoid orphaned local ports

21. Re-run the targeted browser-context, MCP-server, and OpenCode service tests
    immediately after implementation; if they do not pass, fix the
    implementation before moving on.

22. Re-run the targeted Playwright browser-vision spec immediately after
    implementation; if it does not pass, continue refining screenshot transport
    and MCP tool behavior before moving on.

23. Update `electron/README.md` and `electron/docs/architecture.md` to describe:
    - why MCP is used instead of prompt injection
    - how browser screenshots are captured
    - why screenshot capture is main-process only
    - what follow-up work remains

24. Review the implementation for non-obvious screenshot timing,
    serialization, or local MCP auth behavior and add only the minimal
    necessary inline comments.

25. If a browser skill is available, use it for browser-level smoke checks of
    the screenshot-backed browser-inspection flow. No browser skill is available
    in this environment, so use Playwright Electron coverage as the
    browser-level smoke baseline.

26. Run the full automated verification suite:
   - lint
   - unit and renderer tests
   - Playwright Electron E2E tests
   - build/package flow

27. Stop only after:
   - OpenCode can call `browser_context_current`
   - the tool returns both URL and screenshot
   - OpenCode can explain what it sees in the browser
   - browser access remains main-process only
   - docs are updated
   - all automated verification passes
