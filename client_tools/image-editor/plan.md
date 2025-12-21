Plan for converting the React app to TypeScript, refactoring, adding tests, Makefile automation, and GitHub Pages hosting.

Current plan (updated):
1) Inventory current features and define the single-image crop flow; identify component boundaries and shared state types.
2) Refactor into modules/components (hooks, UI panels, canvas), preserving behavior and adding the new single-image tab.
3) Add Playwright config, fixtures, and E2E tests with screenshots; run tests in Chromium only and review outputs.
4) Document how to run dev/build/test/e2e and finalize hosting/deploy steps.
