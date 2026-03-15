# Ghostty Terminal In This App

```text
Launcher window
   |
   | click "Terminal"
   v
createTerminalWindow(repoRoot)
   |
   +--> BaseWindow
   +--> one local WebContentsView
   +--> preload bridge exposes window.terminal
   +--> TerminalPtyService stays in the main process
   |
   v
Terminal renderer (React)
   |
   +--> createGhosttyRuntime(...)
   +--> Ghostty.load(ghostty-vt.wasm)
   +--> new Terminal(...)
   +--> term.open(container)
   +--> FitAddon.fit()/observeResize()
   |
   +--> user input -> window.terminal.write(data)
   +--> resize -> window.terminal.resize(cols, rows)
   `--> terminal:data -> runtime.write(data)
```

## Summary

The Terminal app is a separate Electron window that provides a full local shell
experience through a narrow renderer bridge.

- `ghostty-web` renders the terminal surface in the renderer.
- A main-process `TerminalPtyService` owns the actual shell process in a PTY.
- A main-process `TerminalAppearanceStore` owns durable terminal appearance
  preferences such as font family, font size, and minimal chrome mode.
- The renderer never gets raw Node.js APIs, a shell handle, or unrestricted
  `ipcRenderer` access.
- The shell itself is a normal local shell session with the same effective user
  permissions as the Electron app.
- The visible window is intentionally terminal-first and full-bleed, with no
  visible settings UI.

## Why `ghostty-web` First

This app uses `ghostty-web` as the first Electron terminal implementation path
because it already matches the web-renderer model used by Electron.

- It provides a DOM-mounted terminal surface with a Ghostty-powered WASM parser.
- It is designed to open into an HTML element and stream terminal IO from a
  privileged backend.
- Its API is close to the xterm.js shape, which keeps the renderer integration
  compact.

That makes it a better fit for this codebase than trying to embed the standalone
Ghostty desktop app.

## Why Not `libghostty` For v1

`libghostty` is promising, but it is not the right first implementation here.

- Ghostty's own source still describes the current C API as not yet a general
  purpose embedding API.
- The public `libghostty` direction is still evolving, especially for broader
  non-macOS embedding scenarios.
- Using it from Electron would require a native addon or a separate host layer,
  which is much heavier than a renderer + PTY integration.

For this repo, `ghostty-web` gets us a useful terminal sooner while preserving a
future path to revisit native embedding if the upstream API matures.

## Window And Process Architecture

The launcher lives in `src/app/main/create-launcher-window.ts` and opens the
Terminal app through `openTerminal()` in `src/app/main/index.ts`.

`createTerminalWindow()` in `src/app/main/create-terminal-window.ts` builds the
Terminal app surface.

- It creates a `BaseWindow` titled `Terminal`.
- It creates one local `WebContentsView` with:
  - `sandbox: true`
  - `contextIsolation: true`
  - `nodeIntegration: false`
  - preload script `terminal.js`
- It creates one `TerminalPtyService` instance bound to the repo root.
- It subscribes to terminal state updates and raw output chunks, then forwards
  both back to the renderer over IPC.

This keeps the renderer focused on UI and terminal rendering while the main
process stays authoritative over shell creation, shell restart, and shell output
delivery.

## Renderer Boundary

The public renderer contract is defined in `src/terminal-contract.ts`.

The renderer gets these methods on `window.terminal`:

- `loadState()`
- `connect(cols, rows)`
- `write(data)`
- `resize(cols, rows)`
- `restart(cols, rows)`
- `onData(listener)`
- `onStateChange(listener)`

The preload implementation in `src/app/preload/terminal.ts` forwards those
calls through Electron IPC.

That boundary is intentionally narrow.

- The renderer cannot spawn processes.
- The renderer cannot access the local filesystem directly.
- The renderer cannot reach the PTY handle directly.
- The renderer cannot bypass the main process and issue raw shell commands on
  its own.

## Appearance Ownership And Precedence

Terminal appearance is owned by Electron, not inherited automatically from the
native Ghostty app window.

- Durable preferences are stored in `app.getPath('userData')/terminal-appearance.json`.
- On first load, when no Electron preference exists yet, the app tries to parse
  Ghostty config and import `font-family` as a seed value.
- After that first save, the Electron-owned preference wins over later Ghostty
  config changes unless the saved Electron preference is removed or replaced.

Precedence is:

1. saved Electron appearance preference
2. imported Ghostty `font-family` seed value
3. Electron terminal fallback defaults

This is why the Electron terminal can diverge from the native Ghostty app if it
uses a different configured font stack, even while the shell prompt itself is
identical.

## Ghostty Runtime Wiring

The renderer-side adapter lives in
`src/features/terminal/renderer/ghostty-runtime.ts`.

Current behavior:

- Load `ghostty-vt.wasm` through Vite-managed asset URLs.
- Wait for authoritative terminal state before first renderer paint so the
  initial prompt uses the persisted font instead of a temporary fallback.
- Create a shared `Ghostty` instance.
- Create a `ghostty-web` `Terminal` with initial columns and rows.
- Apply the persisted font family and font size from terminal appearance state.
- Expand the primary font into a browser-oriented font stack with symbol
  fallbacks so prompt glyphs have a better chance of rendering in Electron.
- Open the terminal inside a DOM container.
- Attach `FitAddon` so the terminal re-fits to the container.
- Forward user input and resize events back through the preload API.

The React app in `src/features/terminal/renderer/App.tsx` uses that adapter as a
thin runtime boundary so the UI stays testable without depending directly on the
WASM and canvas details.

## PTY Lifecycle

The main-process PTY owner lives in
`src/features/terminal/main/TerminalPtyService.ts`.

Current behavior:

1. `connect(cols, rows)` sets the state to `connecting`.
2. It resolves the shell path.
3. It spawns a PTY with the requested grid size.
4. It streams PTY output back to the renderer through `terminal:data`.
5. It updates renderer-facing state to `ready` once the session is active.
6. `resize(cols, rows)` resizes the PTY.
7. `restart(cols, rows)` disposes the current PTY and creates a fresh session.
8. `dispose()` tears the PTY down when the window closes.

## Appearance Persistence

The durable appearance owner lives in
`src/features/terminal/main/TerminalAppearanceStore.ts`.

Current behavior:

1. Try to read `terminal-appearance.json` from Electron `userData`.
2. If it exists, merge it with the terminal appearance defaults.
3. If it does not exist, try to parse Ghostty config for `font-family`.
4. Resolve the final Electron-owned appearance preference.
5. Persist that resolved preference so later launches use the same Electron
   terminal appearance even if Ghostty config changes later.

## Filesystem Access And Permissions

This is a **full local terminal**, not a read-only inspection surface like
OpenCode.

- Commands run with the current local user account.
- The shell has the same effective filesystem access as that user.
- If the user can read, write, rename, or delete something from a normal local
  terminal, this terminal can do the same.

The renderer is still sandboxed, but the shell process is not constrained to the
repo once it is running.

## Working Directory

The terminal starts in the repo root by default for convenience.

- Default repo root: `path.resolve(__dirname, '../../..')`
- Optional override: `ELECTRON_TERMINAL_REPO_ROOT`

This means a fresh terminal session opens in the local project tree, but it does
**not** mean the terminal is repo-sandboxed. Users can still `cd` elsewhere and
operate on the rest of the filesystem according to their local account
permissions.

## Shell Selection

Shell resolution happens in the main process.

- macOS/Linux: `process.env.SHELL || '/bin/bash'`
- Windows: `process.env.COMSPEC || 'cmd.exe'`
- Explicit constructor overrides win if a test or future feature provides one.

On this machine today, `$SHELL` is `/bin/zsh`, so the terminal would normally
start as `zsh` unless the environment is overridden.

## PATH And Host Tools

The PTY inherits the main process environment and then adds terminal-specific
values like:

- `TERM=xterm-256color`
- `COLORTERM=truecolor`

Because the environment is inherited, PATH-installed tools remain available.

- If `tmux` is installed and visible on PATH, the terminal can run `tmux`.
- The same is true for tools like `git`, `node`, `python`, `cargo`, and local
  editors launched from the shell.

On this machine today, `tmux 3.5a` is installed, so it is available to the full
terminal.

## Font Matching And Prompt Glyphs

If the Electron terminal prompt looks different from the native Ghostty prompt,
the usual cause is the renderer font configuration, not the shell itself.

- The shell prompt emits the same characters in both places.
- The native Ghostty app and the Electron renderer may still draw those
  characters with different primary fonts or fallback glyph resolution.
- The Electron terminal now treats font configuration as an explicit terminal
  preference and seeds the default from Ghostty config when possible.

In practical terms, if Ghostty uses `font-family = JetBrains Mono`, the Electron
terminal will seed `JetBrains Mono` on first run instead of keeping a separate
hardcoded demo font stack.

## Mock Mode

The terminal includes a deterministic mock path for packaged Electron E2E tests.

- Enable with `ELECTRON_TERMINAL_MOCK=1`
- `connect()` publishes a stable mock session state
- `write()` emits canned output for commands such as `pwd`, `echo $SHELL`, `ls`,
  and `tmux -V`

This keeps Playwright checks reliable without depending on an interactive shell
prompt timing model.

## Native Module Note

The PTY package is a native dependency.

- It must be installed in a way that is compatible with Electron's ABI.
- Development, build, and packaged verification all matter here.
- A dependency update that works in `vitest` but fails in a packaged app is not
  acceptable.

## Known Caveats

The terminal is intentionally useful in v1, but it is not claiming perfect TUI
parity with native Ghostty yet.

- `ghostty-web` still has upstream caveats around mouse tracking and some IME
  paths.
- Upstream reports also mention some Electron-specific rendering rough edges in
  some demo integrations.
- Basic shell IO, resize, restart, and common CLI tools are the priority for
  this repo's first terminal milestone.

## Verification Expectations

Terminal verification should include both automated and manual checks.

Automated:

- renderer tests for launcher and terminal UI behavior
- service tests for PTY spawn/write/resize/restart behavior
- packaged Playwright Electron smoke coverage with terminal mock mode

Manual:

- open Terminal from the launcher
- verify the prompt appears
- run `pwd`
- run `echo $SHELL`
- run `ls`
- run `tmux -V` when `tmux` is installed on the host
- resize the window and verify the terminal remains usable
- restart the session and verify a fresh prompt appears
