# Hello World Tauri App

This project is now a working desktop text editor built with React, TypeScript, Vite, Tauri, and CodeMirror.

## Setup

### Prerequisites

Install the machine-level dependencies with Homebrew:

```bash
xcode-select --install
brew install rustup
rustup default stable
```

If you installed `rustup` with Homebrew, make sure its bin directory is on your shell `PATH`:

```bash
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
```

Verify the toolchain:

```bash
node --version
npm --version
rustc --version
cargo --version
just --version
```

### Install project dependencies

From this directory, install the npm packages:

```bash
npm install
```

## Daily Commands

Use `just` as the main entrypoint:

```bash
just doctor
just dev
just check
just test
just build
just rebuild
just clean
```

Command meanings:

- `just dev` launches the Tauri desktop app in development mode
- `just check` runs lint only
- `just test` runs the Vitest suite
- `just build` creates the macOS app bundle and also runs the frontend production build through Tauri's `beforeBuildCommand`
- `just rebuild` removes build artifacts and rebuilds both outputs from scratch
- `just clean` removes `dist` and `src-tauri/target`

## Text Editor Features

The first app view is a text editor with:

- a desktop app shell
- a `Text Editor` app view
- CodeMirror editing
- native `Open`, `Save`, and `Save As` flows through Tauri
- dirty-state tracking
- support for `.txt`, `.md`, and `.markdown` files

## Run The App

Start the desktop app in development mode:

```bash
just dev
```

## Test The App

Run the frontend test suite:

```bash
just test
```

## Build The App

Create a production app bundle:

```bash
just build
```

The current bundle target is the macOS app bundle written under `src-tauri/target/`.

## Tooling Model

- `brew` installs machine prerequisites
- `npm` manages frontend dependencies and scripts
- `cargo` builds the Rust side under Tauri
- `just` is the preferred entrypoint for local development commands
- `npm run tauri ...` remains the underlying bridge into Tauri

## Recreate From Scratch

If you want to rebuild this setup manually from the parent directory:

```bash
npm create vite@latest hello-world -- --template react-ts --no-interactive
cd hello-world
npm install
npm install @uiw/react-codemirror @codemirror/lang-markdown
npm install @tauri-apps/plugin-dialog @tauri-apps/plugin-fs
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event
npm exec tauri -- init --ci \
  -A "Hello World" \
  -W "Hello World" \
  -D ../dist \
  -P http://localhost:5173 \
  --before-dev-command "npm run dev" \
  --before-build-command "npm run build"
PATH="/opt/homebrew/opt/rustup/bin:$PATH" npm exec tauri -- add dialog
PATH="/opt/homebrew/opt/rustup/bin:$PATH" npm exec tauri -- add fs
```

After that, add the `justfile`, the test setup, and the React editor files in this repo.

## Notes

- Apple signing and notarization are not required for local development
- Change `src-tauri/tauri.conf.json` identifier before distributing the app beyond this local setup
- Dialog and filesystem access are enabled through Tauri plugins and `src-tauri/capabilities/default.json`
