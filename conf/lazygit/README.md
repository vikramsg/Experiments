# Lazygit Configuration

This directory contains the lazygit configuration for this repository.

## Setup

The configuration is symlinked to the standard XDG config location:

```bash
# Create the lazygit config directory if it doesn't exist
mkdir -p ~/.config/lazygit

# Create symlink from standard config location to this repo
ln -sf "/Users/vikramsingh/Projects/Personal/Experiments/conf/lazygit/config.yml" ~/.config/lazygit/config.yml
```

## Configuration Changes from Default

The main customizations made to improve the diff view experience:

### 1. Simplified File View
- `showFileTree: false` - Disables the file tree view for a cleaner interface
- `expandFocusedSidePanel: false` - Keeps panels at consistent sizes
- `mainPanelSplitMode: 'flexible'` - Allows flexible panel sizing

### 2. Focus on Unstaged Changes
By default, lazygit shows both unstaged and staged changes simultaneously. This configuration:
- Emphasizes unstaged changes as the primary focus
- Provides easy ways to toggle between unstaged and staged views

### 3. Key Bindings
- `<space>` - Stage/unstage individual files
- `A` - Stage/unstage all files
- `<tab>` - Toggle between panels
- `t` - Toggle tree view
- `c` - Commit changes
- `C` - Commit without hooks

### 4. Improved Navigation
- `sidePanelWidth: 0.3333` - Sets optimal panel proportions
- `mouseEvents: true` - Enables mouse interaction
- Removed pager for faster navigation

## Why These Changes?

The default lazygit interface shows both unstaged and staged changes in separate blocks, which can be visually cluttered when you primarily want to focus on unstaged changes first. This configuration:

1. **Reduces visual noise** - Focuses on unstaged changes by default
2. **Maintains accessibility** - Staged changes are still easily accessible via keyboard shortcuts
3. **Improves workflow** - Matches the typical git workflow of reviewing unstaged changes first

## Testing the Configuration

1. Open lazygit from within Neovim: `<leader>lg` or `:LazyGit`
2. You should see a cleaner interface focused on unstaged changes
3. Use `<tab>` to navigate between different panels
4. Use `t` to toggle different view modes if needed