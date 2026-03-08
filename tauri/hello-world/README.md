# Hello World Tauri App

This project is now a working Tauri desktop app with a launcher screen and a separate `Text Editor` app built with React, TypeScript, Vite, and CodeMirror.

## Setup

### Prerequisites

Install the machine-level dependencies with Homebrew:

```bash
xcode-select --install
brew install rustup
rustup default stable
```

If you installed `rustup` with Homebrew, make sure your shell startup files expose `cargo` and `rustc` normally before using the project commands:

```bash
cargo --version
rustc --version
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

The app currently starts on a launcher screen and opens into a separate text editor experience with:

- an app selector / launcher screen
- a separate `Text Editor` app screen
- a `Back to Apps` return flow
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


## Notes

- Apple signing and notarization are not required for local development
- Change `src-tauri/tauri.conf.json` identifier before distributing the app beyond this local setup
- Dialog and filesystem access are enabled through Tauri plugins and `src-tauri/capabilities/default.json`
