## Goal

**Guidance**
`Do Not` stop until all verification and acceptance criteria is met.
Start with a test first approach with writing failing tests, making sure they fail
and then proceeding with implementation.

- Fix the first-pass OpenCode chat usability issues in the Electron app:
  - oversized send button
  - Enter does not submit
  - Shift+Enter should keep multiline input
  - overflowing chat pushes the composer down and makes the header disappear
  - message area should own scrolling instead of the whole page
- Keep the existing OpenCode app architecture intact:
  - launcher opens a separate OpenCode app
  - renderer talks through `window.opencode`
  - main process remains the only privileged boundary
- Update `electron/README.md` to include a small `TODO / Follow-up` section
  listing what should be improved after this current plan is finished.
- Preserve existing Browser + Notes behavior, OpenCode read-only behavior, and
  current app verification expectations.

## Architecture Diagram

```text
OpenCode layout target
======================

OpenCode window
+------------------------------------------------------+
| header / hero (always visible)                       |
| repo scope / status                                  |
+------------------------------------------------------+
| chat shell                                           |
| +--------------------------------------------------+ |
| | message list (only this area scrolls)            | |
| | user / assistant / system messages               | |
| +--------------------------------------------------+ |
| +--------------------------------------------------+ |
| | prompt textarea + compact send button            | |
| | Enter submits, Shift+Enter inserts newline       | |
| +--------------------------------------------------+ |
+------------------------------------------------------+
```

```text
Prompt submission flow
======================

textarea keydown / form submit
   |
   +--> Enter without Shift
   |      `--> preventDefault()
   |      `--> submit prompt
   |
   `--> Shift+Enter
          `--> allow newline in textarea
```

```text
Scroll ownership
================

page shell (fixed viewport height)
   |
   +--> header row (non-scrolling)
   `--> chat shell
          |
          +--> message list (overflow-y: auto)
          `--> composer row (always visible)
```

## Current Status

- The OpenCode chat renderer currently uses a page-level flex/grid layout that
  can grow taller than the viewport, so long conversations push the composer
  down and can hide the header.
- The send button is visually oversized because it stretches against the prompt
  row height.
- The prompt uses a `textarea`, but there is no keyboard submit behavior, so
  Enter inserts a newline instead of sending.
- Current tests cover basic prompt submission by click and state rendering, but
  they do not yet verify keyboard submit behavior, sticky visibility
  expectations, or overflow ownership.
- `electron/README.md` currently documents structure and verification, but it
  does not include a follow-up `TODO` section for post-plan improvements.

## Short summary of changes

- Tighten the OpenCode renderer layout so it behaves like a fixed-height app
  shell.
- Keep the header visible while making the message list the only scrolling
  region.
- Redesign the composer row so the send button is compact and aligned
  intentionally.
- Add Enter-to-submit and Shift+Enter-for-newline behavior.
- Add tests for keyboard behavior and overflow-safe chat layout.
- Add a small `TODO / Follow-up` section to `electron/README.md` listing
  deferred improvements after this plan is complete.

### Options considered

1. Keep the current structure and only tweak button sizing
   - easy
   - does not fix hidden header or scroll ownership
   - rejected

2. Make the whole page scroll while leaving the composer inline
   - simple
   - poor chat UX because the input can move off-screen
   - rejected

3. Convert OpenCode into a fixed-height shell with header + scrollable messages
   + pinned composer
   - best chat UX
   - fixes all reported usability issues together
   - recommended

4. Replace the textarea with a single-line input
   - simplifies Enter behavior
   - removes useful multiline prompt support
   - rejected

## Files to be changed

- `electron/plan.md`
- `electron/README.md`
- `electron/src/features/opencode/renderer/App.tsx`
- `electron/src/features/opencode/renderer/App.test.tsx`
- `electron/e2e/opencode.spec.js`

## Files to be added

- None expected unless a renderer helper becomes necessary for keyboard or
  auto-scroll behavior.

## Verification Criteria

- The OpenCode send button is visually compact and no longer stretches to the
  textarea height.
- Pressing Enter in the OpenCode prompt submits the prompt.
- Pressing Shift+Enter inserts a newline and does not submit.
- The OpenCode header remains visible while long chat histories accumulate.
- The message list owns vertical scrolling; the whole page does not expand
  downward with messages.
- The composer remains visible and usable at the bottom of the chat shell during
  long conversations.
- Existing OpenCode chat behavior still works by button click.
- Existing Browser + Notes flows remain unaffected.
- `electron/README.md` includes a `TODO / Follow-up` section describing post-plan
  improvements.
- Lint, unit tests, Playwright E2E, and build all pass.

## Acceptance Criteria

- The OpenCode app no longer has the oversized send button shown in the
  screenshot.
- Enter submits prompts in the OpenCode composer.
- Shift+Enter preserves multiline authoring.
- Overflowing conversations no longer hide the header or push the composer out
  of view.
- The OpenCode message list is the primary scrolling surface.
- `electron/README.md` clearly includes follow-up work that is intentionally
  deferred until after this plan is complete.
- Work is not complete until all automated verification passes.

## Checklist of tasks to be done

1. Update `electron/plan.md` first so it reflects:
   - OpenCode chat-shell usability fixes
   - Enter / Shift+Enter keyboard behavior
   - fixed-height shell with message-list scroll ownership
   - `README.md` follow-up `TODO` section
   - full fail-first verification flow

2. Inspect the current OpenCode renderer and test seams in:
   - `electron/src/features/opencode/renderer/App.tsx`
   - `electron/src/features/opencode/renderer/App.test.tsx`
   - `electron/e2e/opencode.spec.js`
   - `electron/README.md`
   so the exact change surface is confirmed before writing tests.

3. Write failing renderer tests in `electron/src/features/opencode/renderer/App.test.tsx` that assert:
   - Enter submits the prompt
   - Shift+Enter does not submit
   - submit is ignored when the trimmed prompt is empty
   - the send button keeps a compact non-stretched layout contract where
     testable
   - the prompt remains present after message updates

4. Run the targeted OpenCode renderer tests immediately to confirm they fail for
   the right reason; if they do not fail, tighten the assertions before
   implementing.

5. Extend the Playwright OpenCode smoke test in `electron/e2e/opencode.spec.js`
   so it asserts:
   - prompt submission works with Enter
   - repeated messages do not cause the header to disappear
   - the composer remains visible after chat growth

6. Run the targeted Playwright OpenCode spec immediately to confirm the new
   assertions fail first; if they do not fail, tighten them before implementing.

7. Refactor `electron/src/features/opencode/renderer/App.tsx` to use a
   fixed-height app shell that:
   - keeps the header in a stable top region
   - constrains the chat area beneath it
   - makes only the message list scroll vertically
   - keeps the composer visible

8. Refactor the OpenCode composer layout in
   `electron/src/features/opencode/renderer/App.tsx` so:
   - the send button has an intentional compact height
   - the textarea and button align cleanly
   - the layout works on both desktop and narrow widths

9. Add keyboard handling in `electron/src/features/opencode/renderer/App.tsx`
   so:
   - Enter without Shift submits
   - Shift+Enter inserts newline
   - empty trimmed prompts still do not submit
   - responding / connecting states still prevent invalid submission

10. If needed, add lightweight message-list scroll management in
    `electron/src/features/opencode/renderer/App.tsx` so new messages remain
    visible without making the entire page scroll.

11. Re-run the targeted OpenCode renderer tests immediately after the renderer
    changes; if they do not pass, fix the implementation before moving on.

12. Re-run the targeted OpenCode Playwright spec immediately after the renderer
    changes; if it does not pass, continue refining the layout and keyboard
    behavior before moving on.

13. Update `electron/README.md` to add a `TODO / Follow-up` section that lists
    intentionally deferred improvements after this plan completes, such as:
    - markdown / code-block rendering
    - improved auto-scroll behavior heuristics
    - dynamic textarea auto-grow
    - loading indicator polish
    - stronger message differentiation / avatars
    - quick-start prompts / better empty state

14. Review the implementation for any non-obvious keyboard or scroll-ownership
    logic and add only the minimal necessary inline comments.

15. If a browser skill is available, use it for browser-level smoke checks of
    the updated OpenCode UX. No browser skill is available in this environment,
    so use Playwright Electron coverage as the browser-level smoke baseline.

16. Run the full automated verification suite:
   - lint
   - unit and renderer tests
   - Playwright Electron E2E tests
   - build/package flow

17. Stop only after:
   - Enter submit works
   - Shift+Enter newline works
   - the send button is no longer oversized
   - the header stays visible during overflow
   - the composer stays visible during overflow
   - the message list owns scrolling
   - `electron/README.md` includes the follow-up `TODO` section
   - all automated verification passes
