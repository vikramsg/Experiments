# Hello World Tauri App

This project uses Homebrew for machine prerequisites, npm for JavaScript tooling, Vite for the frontend, React + TypeScript for the UI, and Cargo for the native Tauri layer.

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
```

### Install project dependencies

From this directory, install the npm packages:

```bash
npm install
```

### Run the app

Start the desktop app in development mode:

```bash
just dev
```

### Build the app

Create a production build:

```bash
just build
```

## Tooling Model

- `brew` installs machine prerequisites
- `npm` manages frontend dependencies and project scripts
- `cargo` builds the Rust side under Tauri
- `just` is the preferred entrypoint for day-to-day project commands

Common commands:

```bash
just doctor
just dev
just check
just build
just rebuild
just clean
```

Command meanings:

- `just check` runs lint only
- `just build` creates the macOS app bundle
- `just rebuild` removes build artifacts and builds again
- `just clean` removes `dist` and `src-tauri/target`

## Recreate From Scratch

If you want to rebuild this setup manually from the parent directory:

```bash
npm create vite@latest hello-world -- --template react-ts --no-interactive
cd hello-world
npm install
npm install @tauri-apps/api
npm install -D @tauri-apps/cli
npm exec tauri -- init --ci \
  -A "Hello World" \
  -W "Hello World" \
  -D ../dist \
  -P http://localhost:5173 \
  --before-dev-command "npm run dev" \
  --before-build-command "npm run build"
```

After that, add a `justfile` so the common commands are available through `just`.

## Notes

- Apple signing and notarization are not required for local development
- Change `src-tauri/tauri.conf.json` identifier before distributing the app
- The current bundle target is the macOS app bundle, which is written under `src-tauri/target/`
