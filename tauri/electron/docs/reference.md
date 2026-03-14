# Electron Reference

This document captures the stable Electron guidance for the launcher-plus-split-workspace app: notes on the left, browser pane on the right.

## Recommended Architecture

Build the app as a small local launcher window that opens a dedicated split-workspace window. For the split workspace, use `BaseWindow` with sibling `WebContentsView`s, not `BrowserView`, and not `<webview>` as the primary architecture. Electron's docs position `BaseWindow` and `WebContentsView` for composing multiple web surfaces, while `BrowserView` is deprecated and `<webview>` is discouraged for stability and security reasons. [W1][W2][W3][W7]

### Launcher window

Use a normal local Electron window for the launcher UI and show one app tile such as `Browser + Notes`. Electron's quick start pattern is to create a window in `createWindow()` and load a local HTML entrypoint with a preload script. [F1]

### Notes persistence

Persist notes under `app.getPath('userData')`, because Electron documents that as the conventional location for app configuration and user data. [W6]

### IPC boundary

Use a preload bridge plus `ipcMain.handle` and `ipcRenderer.invoke` for note load/save and any launcher-to-main commands. Electron's IPC tutorial and examples show this as the standard request/response pattern. [W5][F3]

### `WebContentsView` lifecycle constraints

Two implementation details are mandatory with `WebContentsView`:

- manually recompute bounds on window resize, because `WebContentsView` does not currently have `setAutoResize` parity with old `BrowserView`; an open Electron issue explicitly calls out the need to resize these views yourself today, typically from the parent window's resize event. [I1]
- manually close child `webContents` on window close, because Electron's `BaseWindow` docs say `WebContentsView` contents are not destroyed automatically and can leak if you do nothing. [W2]

Keep the views as siblings under the `BaseWindow` content view, not nested inside each other. An open issue reports rendering problems when a `WebContentsView` is added inside another `WebContentsView`, so sibling composition is the safer plan. [W2][I2]

## Security Guidance For Remote Content

Because the browser pane loads remote sites, apply Electron's remote-content rules:

- disable Node integration for any remote content, [W4]
- enable `contextIsolation`, [W4]
- enable sandboxing, [W4]
- use `session.setPermissionRequestHandler()` to explicitly approve or deny permissions like notifications, camera, or location, [W4]
- limit navigation and limit new-window creation with handlers, [W4]
- if you ever fall back to `<webview>`, validate options with `will-attach-webview` and deny unwanted popups or new windows. [W4][F4]

## Why Not Use `<webview>` As The Main Design

Electron's docs say they do not recommend WebViews, and the security guide treats them as something that must be locked down carefully. [W1][W4] There is also a confirmed open issue showing `<webview>` flicker in larger scrollable layouts, which is especially relevant for a split app UI. [I3] For a browser-on-the-right and notes-on-the-left app, `WebContentsView` is the cleaner long-term choice. [W1][W3][I3]

## Why Not Base New Work On `BrowserView`

Electron's own API docs mark `BrowserView` as deprecated and replaced by `WebContentsView`. [W7] The older navigation-history fiddle still shows how view-based browsing and IPC can work conceptually, but it uses `BrowserView`, so it should be treated as a historical reference only, not the architecture to start from now. [F2][W7]

## Citations

### Web Docs

- [W1] Electron Web Embeds docs - recommends against WebViews and compares `<iframe>`, `<webview>`, and `WebContentsView`: https://www.electronjs.org/docs/latest/tutorial/web-embeds
- [W2] Electron BaseWindow API - shows composing multiple `WebContentsView`s and warns that child `webContents` are not destroyed automatically: https://www.electronjs.org/docs/latest/api/base-window
- [W3] Electron WebContentsView API - current API for view-based embedded web content: https://www.electronjs.org/docs/latest/api/web-contents-view
- [W4] Electron Security tutorial - remote content rules, context isolation, sandboxing, permission handlers, webview verification, navigation limits, and window-creation limits: https://www.electronjs.org/docs/latest/tutorial/security
- [W5] Electron IPC tutorial - preload bridge, `ipcMain.handle`, `ipcRenderer.invoke`, and safe API exposure patterns: https://www.electronjs.org/docs/latest/tutorial/ipc
- [W6] Electron app API - `app.getPath('userData')` as conventional user/config data location: https://www.electronjs.org/docs/latest/api/app
- [W7] Electron BrowserView API - explicitly deprecated and replaced by `WebContentsView`: https://www.electronjs.org/docs/latest/api/browser-view
- [W8] Electron Forge CLI docs - init/start/make workflow and `create-electron-app` mention: https://www.electronforge.io/cli
- [W9] Electron Forge built-in templates - lists Vite and Vite + TypeScript templates: https://www.electronforge.io/templates

### Repository Files

- [F1] Canonical GitHub location for the cloned quick-start reference repo (`electron/electron-quick-start` now resolves to `electron/minimal-repro`), `main.js`, commit `e8b471eef693f0b5c721e12c2c36413599d28f12`, lines 5-16: https://github.com/electron/minimal-repro/blob/e8b471eef693f0b5c721e12c2c36413599d28f12/main.js#L5-L16
- [F2] `electron/electron` `docs/fiddles/features/navigation-history/main.js`, commit `26a3a8679a063623cf7e6bc1f5e07042fa953d7a`, lines 4-48: https://github.com/electron/electron/blob/26a3a8679a063623cf7e6bc1f5e07042fa953d7a/docs/fiddles/features/navigation-history/main.js#L4-L48
- [F3] `electron/electron` `docs/fiddles/ipc/pattern-2/main.js`, commit `26a3a8679a063623cf7e6bc1f5e07042fa953d7a`, lines 4-22: https://github.com/electron/electron/blob/26a3a8679a063623cf7e6bc1f5e07042fa953d7a/docs/fiddles/ipc/pattern-2/main.js#L4-L22
- [F4] `electron/electron` `docs/fiddles/ipc/webview-new-window/main.js`, commit `26a3a8679a063623cf7e6bc1f5e07042fa953d7a`, lines 5-24: https://github.com/electron/electron/blob/26a3a8679a063623cf7e6bc1f5e07042fa953d7a/docs/fiddles/ipc/webview-new-window/main.js#L5-L24

### GitHub Issues

- [I1] Electron issue `#43802` - `WebContentsView` lacks `setAutoResize`-style support; issue body shows manual resize-listener workaround and is still open: https://github.com/electron/electron/issues/43802
- [I2] Electron issue `#47990` - nested `WebContentsView` rendering problem (`addChildView` inside another `WebContentsView`), still open and recently updated: https://github.com/electron/electron/issues/47990
- [I3] Electron issue `#46519` - confirmed `<webview>` flickering-on-scroll bug in larger scrollable DOM layouts, still open: https://github.com/electron/electron/issues/46519
