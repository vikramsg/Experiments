## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Replace the literal `Back` / `Forward` labels with arrow buttons in the browser chrome while keeping accessible names.
- Enforce a simple code-organization rule: `app/*` may compose features, but features must not import other features.
- Avoid turning `src/shared/` into a dumping ground by moving cross-cutting runtime boundaries into a few shallow root files at `electron/src/`.
- Add substantial top-of-file docstrings to the root `src/*.ts` boundary files explaining what each file owns, what may import it, and why it exists.
- Update `electron/README.md` and `electron/docs/architecture.md` so they clearly describe the new code organization and import rules.
- Keep the existing browser navigation sync, startup-race fix, notes recovery, splitter behavior, persistence, and browser security intact.

## Architecture Diagram

```text
Target code organization
========================

electron/src/
|-- app/                      # composition only
|-- features/                 # isolated business features
|-- ipc.ts                    # IPC channel constants only
|-- workspace-contract.ts     # public workspace API contract
|-- workspace-model.ts        # workspace data model + helpers
|-- test-setup.ts             # shared Vitest setup only
|-- types.d.ts                # global window declarations only
`-- vite-env.d.ts

Rules:
- app/*        -> may import features + root boundary files
- feature A    -> may import itself + root boundary files
- feature A    -> may NOT import feature B
```

```text
Runtime boundary flow
=====================

Browser Chrome Renderer
   |
   | uses WorkspaceApi from src/workspace-contract.ts
   v
window.workspace (global bridge)
   |
   v
src/app/preload/workspace.ts
   |
   v
src/app/main/register-ipc.ts
   |
   +--> set URL
   +--> go back
   +--> go forward
   `--> publish workspace state
```

```text
Navigation synchronization
==========================

remote browser webContents
   |
   +--> did-navigate
   +--> did-navigate-in-page
   |
   +--> read getURL()
   +--> read canGoBack()
   +--> read canGoForward()
   |
   v
WorkspaceController.setBrowserNavigationState(...)
   |
   +--> publish snapshot to browser chrome / notes
   `--> persist durable workspace fields only
```

```text
Import boundary intent
======================

GOOD
----
src/features/browser/renderer/App.tsx
  -> ../../../workspace-contract
  -> ../../../workspace-model

src/app/main/create-workspace-window.ts
  -> ../../features/browser/...
  -> ../../features/notes/...
  -> ../../features/workspace/...

BAD
---
src/features/browser/renderer/App.tsx
  -> ../../../features/notes/...
  -> ../../../features/workspace/...
```

## Current Status

- `electron/src/features/browser/renderer/App.tsx` currently shows literal `Back` and `Forward` text buttons rather than arrow buttons.
- Cross-cutting runtime types and IPC names currently live under `electron/src/shared/`, and `README.md` / `docs/architecture.md` currently describe that structure.
- `WorkspaceApi` is currently sourced through `electron/src/types.d.ts`, which is not an ideal primary home for a runtime-facing contract.
- Root-level organization is not yet explicit enough about what is a model, what is a contract, and what is transport.
- TypeScript is already `strict`, but lint rules do not yet explicitly enforce the stronger constraints requested, such as no explicit `any`, no ts-comment bypasses, and no architecture-boundary enforcement.
- Current app-level imports from `src/app/*` into `src/features/*` are valid under the intended orchestration rule, but the rule is not yet documented or lint-enforced.
- The root `src/shared/` folder is at risk of becoming a catch-all, even though `src/features/workspace/shared/` is acceptable because it is still inside a single feature.

## Short summary of changes

- Replace the browser chrome word buttons with arrow buttons that keep accessible labels and preserve disabled-state behavior.
- Move cross-cutting runtime boundaries out of `src/shared/` into a minimal set of shallow root files:
  - `src/ipc.ts`
  - `src/workspace-contract.ts`
  - `src/workspace-model.ts`
  - `src/test-setup.ts`
- Keep `src/types.d.ts` as the ambient global bridge only, not the main home of the workspace API contract.
- Update imports across app, preload, main, and features so features depend only on themselves plus the shallow root boundary files.
- Add strong top-of-file docstrings to the root `src/*.ts` files explaining ownership, usage, and import-boundary intent.
- Tighten lint rules to enforce:
  - no feature-to-feature imports
  - no explicit `any`
  - no ts-comment bypasses
  - no unnecessary assertions or weak import style
- Update `electron/README.md` and `electron/docs/architecture.md` so code organization and boundary rules are clear and intentional.
- Re-run targeted fail-first checks, then full verification.

### Options considered

1. Move cross-cutting runtime boundaries into a few shallow root files at `src/`
   - minimal folder growth
   - avoids `shared` becoming a junk drawer
   - keeps imports short and obvious
   - recommended

2. Keep expanding `src/shared/`
   - easy in the short term
   - weak file ownership boundaries
   - trends toward an unstructured bucket
   - rejected

3. Add several new top-level folders like `contracts/`, `models/`, `protocol/`, `testing/`
   - clean on paper
   - too much structure for this repo size
   - rejected

## Files to be changed

- `electron/plan.md`
- `electron/eslint.config.mjs`
- `electron/tsconfig.json`
- `electron/vitest.config.ts`
- `electron/README.md`
- `electron/docs/architecture.md`
- `electron/src/types.d.ts`
- `electron/src/app/preload/launcher.ts`
- `electron/src/app/preload/workspace.ts`
- `electron/src/app/main/create-workspace-window.ts`
- `electron/src/app/main/register-ipc.ts`
- `electron/src/features/browser/main/browser-session.ts`
- `electron/src/features/browser/main/browser-session.test.ts`
- `electron/src/features/browser/renderer/App.tsx`
- `electron/src/features/browser/renderer/App.test.tsx`
- `electron/src/features/notes/main/NoteStore.ts`
- `electron/src/features/notes/main/NoteStore.test.ts`
- `electron/src/features/notes/renderer/App.tsx`
- `electron/src/features/notes/renderer/App.test.tsx`
- `electron/src/features/workspace/main/WorkspaceController.ts`
- `electron/src/features/workspace/main/WorkspaceController.test.ts`
- `electron/e2e/launcher-workspace.spec.js`
- `electron/src/shared/ipc/channels.ts`
- `electron/src/shared/types/workspace.ts`
- `electron/src/shared/test/setup.ts`

## Files to be added

- `electron/src/ipc.ts`
- `electron/src/workspace-contract.ts`
- `electron/src/workspace-model.ts`
- `electron/src/test-setup.ts`

## Verification Criteria

- Browser chrome renders arrow buttons for back and forward instead of literal `Back` / `Forward` text.
- Back/forward buttons remain accessible through semantic button roles and accessible labels.
- Browser URL stays synchronized with real browser navigation, including link clicks, back/forward, and in-page navigation.
- No feature imports another feature directly.
- `app/*` remains the only composition layer allowed to import feature entrypoints across feature boundaries.
- Cross-cutting runtime boundaries live in shallow root files under `electron/src/`, not in an expanding root `src/shared/` bucket.
- The new root `src/*.ts` files contain substantial top-of-file docstrings explaining purpose, ownership, and intended import usage.
- `electron/README.md` and `electron/docs/architecture.md` clearly explain code organization, import boundaries, and where cross-cutting types now live.
- Lint explicitly enforces no explicit `any`, no ts-comment bypasses, and the chosen import-boundary rules.
- No `as unknown` bypasses are introduced.
- Existing startup resilience, notes recovery, splitter behavior, persistence, and browser security remain intact.
- Lint, unit tests, Playwright E2E, and build all pass.

## Acceptance Criteria

- The browser chrome visibly uses arrow buttons rather than text labels for navigation controls.
- The arrow buttons have correct accessible labels and correct enabled/disabled behavior.
- The URL field always reflects the actual URL of the embedded browser.
- `src/shared/` is no longer the home of cross-cutting runtime contracts, models, or IPC definitions.
- The cross-cutting runtime boundary is represented by a minimal shallow set of root files:
  - `src/ipc.ts`
  - `src/workspace-contract.ts`
  - `src/workspace-model.ts`
  - `src/test-setup.ts`
- Root boundary files at `src/*.ts` have substantial top-of-file docstrings describing what they do and how they should be used.
- `src/types.d.ts` remains ambient-only and is not the primary home of the runtime workspace contract.
- Feature-to-feature imports are lint-forbidden.
- `app/*` composition imports remain allowed and documented.
- No explicit `any` is present in the implementation.
- No `eslint-disable`, `@ts-ignore`, or `@ts-expect-error` bypass comments are added.
- No `as unknown` bypassing is added.
- `electron/README.md` and `electron/docs/architecture.md` are updated to describe code organization and import rules clearly.
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Update `electron/plan.md` first so it reflects:
   - arrow-button navigation controls
   - shallow root boundary files at `src/`
   - no feature-to-feature imports
   - top-of-file docstrings for root `src/*.ts` files
   - README / architecture documentation updates
   - full fail-first and verification flow

2. Inspect all current imports that rely on `electron/src/shared/*`, `electron/src/types.d.ts`, and any cross-feature paths so the refactor scope is exact before writing tests.

3. Write failing browser renderer tests in `electron/src/features/browser/renderer/App.test.tsx` that assert:
   - arrow buttons are rendered instead of literal text labels
   - buttons still expose accessible names for back and forward
   - disabled/enabled state still follows workspace state
   - the URL input still updates from workspace snapshot changes

4. Run the targeted browser renderer tests immediately to confirm they fail for the right reason; if they do not fail, tighten the assertions before implementing.

5. Write or extend failing tests around the cross-cutting contract migration so browser, notes, preload, and controller consumers still compile against the new root files instead of `src/shared/*` or ambient-only contract definitions.

6. Run the targeted unit suites immediately after adding those migration tests so the failures prove the new boundary files are actually required; if the tests still pass unexpectedly, strengthen the migration seam.

7. Add failing tests for persistence and workspace state behavior where needed, especially around `WorkspaceSnapshot`, `PersistedWorkspaceSnapshot`, and the durable-vs-live browser state split.

8. Run the relevant persistence and controller tests immediately to confirm the contract/model migration causes the expected failures before implementation.

9. Add or update lint rules in plan form for:
   - no feature-to-feature imports
   - no explicit `any`
   - no ts-comment bypasses
   - no unnecessary type assertions
   - consistent type imports

10. Validate the lint-rule plan by running lint after the initial config changes; if lint does not fail where expected during the transition, refine the rule set before proceeding further.

11. Create the new shallow root files:
   - `electron/src/ipc.ts`
   - `electron/src/workspace-contract.ts`
   - `electron/src/workspace-model.ts`
   - `electron/src/test-setup.ts`

12. Add substantial top-of-file docstrings to each new root `src/*.ts` file explaining:
   - what the file owns
   - who is allowed to import it
   - why it exists at the root instead of inside a feature
   - how it helps avoid `src/shared/` becoming a dumping ground

13. Update `electron/src/types.d.ts` so it becomes an ambient global bridge only, importing the runtime API type from `src/workspace-contract.ts`.

14. Move imports across app, preload, and features from:
   - `src/shared/ipc/channels.ts`
   - `src/shared/types/workspace.ts`
   - ambient-only `src/types.d.ts`
   to the new root boundary files.

15. Replace the browser chrome literal text controls with arrow buttons in `electron/src/features/browser/renderer/App.tsx`, while preserving:
   - accessibility labels
   - keyboard behavior
   - disabled state
   - current URL synchronization

16. Run the targeted browser renderer tests again immediately after the UI change; if they do not pass, fix the implementation before moving on.

17. Refactor or retire the old root `src/shared/*` runtime files so cross-cutting IPC, contracts, models, and test setup no longer depend on that bucket.

18. Update `electron/vitest.config.ts` to use the new root test setup file, then run the full unit/renderer suite to confirm the new root organization works.

19. Update `electron/README.md` with a dedicated code-organization section that explains:
   - why the root boundary files exist
   - what belongs in `app/`
   - what belongs in `features/`
   - why features must not import each other
   - why root `src/shared/` is intentionally avoided for runtime boundaries

20. Update `electron/docs/architecture.md` with:
   - the new file tree
   - the import-boundary rule
   - runtime flow diagrams using the new root file names
   - explicit explanation that `app/*` composes features and features stay isolated

21. Add or update only the necessary inline code comments in the implementation files where the browser navigation sync or import-boundary intent would otherwise be non-obvious.

22. If a browser skill is available, use it for browser-level smoke checks of the updated arrow-button navigation UX. No browser skill is available in this environment, so use Playwright Electron coverage as the browser-level smoke baseline.

23. Run the updated Playwright spec immediately after the browser chrome work to verify:
   - arrow-button controls are visible
   - URL sync still works after remote navigation
   - back/forward still work end to end

24. If the Playwright assertions do not fail first where expected, tighten them before continuing; once they do fail, implement until they pass.

25. Run the full automated verification suite:
   - lint
   - unit and renderer tests
   - Playwright Electron E2E tests
   - build/package flow

26. Stop only after:
   - arrow buttons replace the text labels
   - root boundary files exist and have substantial docstrings
   - feature-to-feature imports are lint-forbidden
   - docs explain the code organization clearly
   - `src/shared/` is no longer the runtime boundary bucket
   - all automated verification passes
