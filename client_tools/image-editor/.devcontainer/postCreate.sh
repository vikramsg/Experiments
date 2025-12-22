#!/bin/bash
set -e

# Install OpenAI Codex CLI
npm install -g @openai/codex

# Install uv (Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install tmux
sudo apt-get update && sudo apt-get install -y tmux

# Install project dependencies
npm install

# Install Playwright browsers (Chromium only) and dependencies
npx playwright install chromium --with-deps
