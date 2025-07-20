# Development Container Setup

This devcontainer provides a complete development environment with your personalized Neovim and Tmux configurations.

## Features

- **Neovim**: Stable version with your mounted configuration
- **Tmux**: Terminal multiplexer with your custom config and plugins
- **Modern Shell Tools**: ripgrep, fd, bat, and other CLI utilities
- **Development Tools**: Git, GitHub CLI, Python 3.12, Node.js 20, Rust
- **Shell**: Zsh with Oh My Zsh and your custom `.zshrc`

## Prerequisites

- Docker installed and running
- Your dotfiles in place:
  - `~/.config/nvim/` (your Neovim configuration)
  - `~/.tmux.conf` (your Tmux configuration)
  - `~/.zshrc` (your Zsh configuration)

## Quick Start

```bash
# Install the Dev Container CLI (if not already installed)
npm install -g @devcontainers/cli

# Build and open the container
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . zsh
```

## First Time Setup

After the container starts:

1. **Install Tmux plugins**: 
   ```bash
   tmux
   # Press Ctrl+a + I to install plugins
   ```

2. **Setup Neovim**: Open nvim and run:
   ```bash
   nvim
   # Run :Mason to install LSP servers
   ```

## What Gets Mounted

- `~/.config/nvim` → `/home/vscode/.config/nvim`
- `~/.tmux.conf` → `/home/vscode/.tmux.conf`
- `~/.tmux/` → `/home/vscode/.tmux/`
- `~/.zshrc` → `/home/vscode/.zshrc`

## Installed Tools

### Via Features
- Git & GitHub CLI
- Python 3.12 with pip
- Node.js 20 with npm
- Rust toolchain
- Modern shell utilities (ripgrep, fd, bat, etc.)
- Tmux
- Neovim (stable)

### Via Setup Script
- Tmux Plugin Manager (TPM)
- Stylua (Lua formatter)

