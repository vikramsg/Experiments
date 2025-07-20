# Usage

## Install

Use linking as in [Config init](## Config init) section.
After linking, setup is automatic. 
Start `nvim` for the first time, and `lazy` should automatically setup. 

## Keybindings

### Neovim

- `Ctrl + h` - To move between split 
- `Ctrl + l` - To move between split 

### Neotree 

- `Space + e`: Toggle File tree view 
- `a` in Neotree view to add a file.
- `d` in Neotree view to delete a file.
- `A` in Neotree view to create a dir.

#### Within Oil buffer:
- `Ctrl + s`: Open file in vertical split
- `Ctrl + h`: Open file in horizontal split  
- `Ctrl + t`: Open file in new tab

### Telescope 

- `Space + Space`: Open search to find files. 
- `Space + /`: Search for word in current buffer.
- `Space + sg`: Grep for word in project.
- `Space + sf`: Find files in project.

### Lazygit

- `Space + lg`: To open lazygit view
- `+` after `Enter` on a file to go to bigger diff view. 
- `Space`: To stage
- `c` to commit

### Mac issues

To bind keys with things like `Cmd`, we need to do a 2 step process. 
1. Go to `iterm2 -> Settings -> Keys -> Keybindings` and then attach each key combination you want with an escape sequence. 
2. Then go to Neovim, into insert mode, do `Ctrl + V, key combination` and note the output. 
3. Then go to `init.lua` and use that output for the shortcut you want.


## Python

To use with Python and `uv` just do 
`uv run nvim .` from your root workspace folder. It should automatically find the correct `venv`. 


## Config init 

We are trying to do a simple single file `init.lua` setup. 
The starting points was [this](https://github.com/khuedoan/nvim-minimal/tree/master).
We have linked it to our base system using 

```sh
ln -s <full path to nvim git dir> ~/.config/nvim
```

## References

Go look at some of the settings [here](https://github.com/nvim-lua/kickstart.nvim/blob/master/init.lua) for inspiration. 

## Debugging

- `:checkhealth` is useful
- `:checkhealth vim.lsp` shows what is happening with LSP 

## ToDo

1. Make `Cmd + .` work to suggest imports.
    - Did this break esc?
    - Always need to press esc twice now
2. Make `Cmd + p` work to show recent files. 
3. Make all files open show up as tabs. 

