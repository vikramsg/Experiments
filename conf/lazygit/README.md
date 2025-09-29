# Lazygit Configuration

This directory contains the lazygit configuration for this repository.

## Setup

The configuration is symlinked to the standard XDG config location:
However, note that `lazygit` does not directly use the config location so we launch `lazygit` using
`lazygit -ucf ~/.config/lazygit/config.yml`.

```bash
# Create the lazygit config directory if it doesn't exist
mkdir -p ~/.config/lazygit

# Create symlink from standard config location to this repo
ln -sf "/path/to/lazygit/config.yml" ~/.config/lazygit/config.yml
```

## Configuration Changes from Default

1. <c-f> custom command. Press `Ctrl+f` twice to see a diff in full screen mode. 

## Workflow Tips

Since lazygit doesn't support hiding staged files completely, here are the best practices:

1. **Navigation**: Use `Tab` to switch between "Unstaged Changes" and "Staged Changes" sections
2. **Staging**: Use `Space` to stage/unstage individual files
3. **View modes**: Use `+` to expand panels for better visibility
4. **File focus**: Arrow keys to navigate between files

## Why These Changes?

The `splitDiff: 'auto'` setting is the closest thing to your desired behavior:
- When you have only unstaged changes, you won't see the split view
- When you have only staged changes, you won't see the split view
- You only see the split when a file has both staged AND unstaged changes

This reduces the visual noise you were experiencing with unnecessary splits.

## Testing the Configuration

1. Open lazygit from within Neovim: `<leader>lg` or `:LazyGit`
2. Make some changes to files (unstaged only)
3. You should see a cleaner single-pane diff view
4. Stage some files and notice the interface adapts
