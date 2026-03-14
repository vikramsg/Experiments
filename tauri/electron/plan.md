## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria are met.
Start with a test-first approach by writing failing tests, confirming they fail,
and then proceeding with implementation.

- Keep the root-boundary architecture in place:
  - `app/*` composes features
  - features must not import other features
  - cross-cutting runtime boundaries stay in shallow root `src/*.ts` files
- Preserve the existing browser navigation sync, startup-race fix, notes
  recovery, splitter behavior, persistence, and browser security.
- Add `OpenCode` as a second launcher app alongside `Browser + Notes`.
- Keep the launcher on `BrowserWindow`, but open the new `OpenCode` app in a
  `BaseWindow` using one local `WebContentsView`.
- Use OpenCode's client/server model for the new app instead of stretching the
  existing workspace contract to cover chat behavior.
- Restrict the OpenCode app to read-only repo access so users can chat with
  repository context without editing files or mutating git state.
- Update `electron/README.md` and `electron/docs/architecture.md` so they
  document the two-app launcher, the root-boundary rules, and the OpenCode
  runtime boundary clearly.

## Architecture Diagram

```text
Launcher BrowserWindow
   |
   +--> Browser + Notes
   |      `--> existing BaseWindow workspace
   |
   `--> OpenCode
          `--> new BaseWindow with one local WebContentsView
```

```text
OpenCode renderer
   |
   | uses OpenCodeApi from src/opencode-contract.ts
   v
window.opencode
   |
   v
src/app/preload/opencode.ts
   |
   v
src/app/main/register-ipc.ts
   |
   +--> load app state
   +--> send prompt
   +--> publish chat updates
   `--> talk to local OpenCode server service
```

```text
Electron main process
   |
   +--> starts local `opencode serve` for the tauri/ repo scope
   +--> applies a tightly scoped read-only config
   +--> allows read/search/list style repository access only
   +--> denies edit/write/destructive shell or git behavior
   `--> exposes only a narrow app IPC to the renderer
```

## Current Status

- The launcher now needs to support two app cards and two launch actions.
- The root boundary refactor is already in place on this branch, so OpenCode
  work must build on `src/ipc.ts`, `src/workspace-contract.ts`, and
  `src/workspace-model.ts` rather than reintroducing root `src/shared/*`
  dependencies.
- Main-process lifecycle currently owns the existing workspace bundle and must
  be extended to own an OpenCode bundle as well.
- The OpenCode CLI is available locally, so the app can start a local server
  process instead of vendoring the OpenCode source tree.

## Short Summary Of Changes

- Update the plan first to describe the new OpenCode app, its read-only repo
  boundary, and the full fail-first verification flow.
- Add a second launcher card for `OpenCode` while preserving the existing
  `Browser + Notes` card and behavior.
- Add shallow root runtime boundaries for OpenCode in:
  - `src/opencode-contract.ts`
  - `src/opencode-model.ts`
- Add a dedicated OpenCode preload bridge and renderer entry.
- Add a dedicated OpenCode `BaseWindow` with one local `WebContentsView`.
- Add a main-process `OpenCodeService` that manages a local OpenCode server,
  session lifecycle, prompt submission, and renderer-facing state publication.
- Configure the local OpenCode server for read-only use within the `tauri/`
  repo scope.
- Write failing tests first for launcher behavior, OpenCode renderer behavior,
  and OpenCode service behavior.
- Add an Electron Playwright smoke test for opening the OpenCode app and
  chatting successfully.

## Files To Be Changed

- `electron/plan.md`
- `electron/README.md`
- `electron/docs/architecture.md`
- `electron/forge.config.ts`
- `electron/vite.renderer.config.ts`
- `electron/src/ipc.ts`
- `electron/src/workspace-contract.ts`
- `electron/src/types.d.ts`
- `electron/src/features/launcher/renderer/App.tsx`
- `electron/src/features/launcher/renderer/App.test.tsx`
- `electron/src/features/launcher/renderer/main.tsx`
- `electron/src/app/preload/launcher.ts`
- `electron/src/app/main/index.ts`
- `electron/src/app/main/register-ipc.ts`

## Files To Be Added

- `electron/src/opencode-contract.ts`
- `electron/src/opencode-model.ts`
- `electron/src/app/main/create-opencode-window.ts`
- `electron/src/app/preload/opencode.ts`
- `electron/src/app/renderer/entries/opencode.html`
- `electron/src/features/opencode/renderer/App.tsx`
- `electron/src/features/opencode/renderer/App.test.tsx`
- `electron/src/features/opencode/renderer/main.tsx`
- `electron/src/features/opencode/main/OpenCodeService.ts`
- `electron/src/features/opencode/main/OpenCodeService.test.ts`
- `electron/e2e/opencode.spec.js`

## Verification Criteria

- The launcher shows two distinct app cards:
  - `Browser + Notes`
  - `OpenCode`
- Launching `Browser + Notes` still works as before.
- Launching `OpenCode` opens a dedicated `BaseWindow` with one local
  `WebContentsView`.
- The OpenCode renderer can load state and send prompts through `window.opencode`.
- The OpenCode app can start a session for the `tauri/` repo scope and display a
  response to a simple repository question.
- The OpenCode integration is read-only for the repo scope:
  - no file edits
  - no write tools
  - no destructive shell or git behavior
- The renderer has no direct Node access and no raw shell or filesystem bridge.
- New OpenCode runtime boundaries live in shallow root files under `src/`, not
  in an expanded root `src/shared/` bucket.
- Lint, unit tests, Playwright E2E tests, and production build all pass.

## Acceptance Criteria

- The launcher visibly offers `OpenCode` as a second app alongside
  `Browser + Notes`.
- The `OpenCode` app opens from the launcher without regressing the existing
  workspace launcher flow.
- The `OpenCode` app uses a dedicated preload/API boundary and does not reuse
  the Browser + Notes workspace contract.
- The `OpenCode` app can chat with repository context from the `tauri/` repo
  scope.
- The OpenCode integration is read-only for the repo:
  - no file edits
  - no write tools
  - no destructive shell or git actions
- The implementation respects the architecture direction that `app/*` composes
  features and features stay isolated.
- `electron/README.md` and `electron/docs/architecture.md` describe the two-app
  launcher and the OpenCode runtime boundary clearly.
- Work is not complete until all automated verification passes.

## Checklist Of Tasks To Be Done

1. Update `electron/plan.md` first so it reflects the new OpenCode app, the
   read-only repo scope, the dedicated preload/API boundary, and the full
   fail-first verification flow before implementation begins.
2. Inspect the launcher, main-process, and root-boundary seams before writing
   tests.
3. Write failing launcher renderer tests for two app cards and two launch
   actions, then confirm they fail.
4. Write failing OpenCode renderer tests and failing OpenCode service tests,
   then confirm they fail.
5. Implement the root OpenCode boundary files, preload bridge, renderer entry,
   window creation, server lifecycle, and IPC wiring.
6. Add Playwright Electron coverage for opening OpenCode and chatting
   successfully.
7. Update docs for the two-app launcher, root-boundary rules, and read-only
   OpenCode boundary.
8. Run the full automated verification suite and stop only when everything
   passes.
