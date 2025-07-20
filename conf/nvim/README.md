# Usage

## Install

Use linking as in [Config init](## Config init) section.
After linking, setup is automatic. 
Start `nvim` for the first time, and `lazy` should automatically setup. 
For `fzf`, if you try to execute a command, it will ask to install the binary and after you 
accept, it will just work!

## Keybindings

### Neovim

- `Ctrl + w + <-` - To move between split 
- `Ctrl + w + ->` - To move between split 

### Oil

- `-` :  Use `-` to show files as a vim buffer. Edit files exactly the way you would edit text
- `Space + e`: Open Oil in left split (VSCode-like file explorer)

#### Within Oil buffer:
- `Ctrl + s`: Open file in vertical split
- `Ctrl + h`: Open file in horizontal split  
- `Ctrl + t`: Open file in new tab

### Fzf

- `Space + Space`: Open search to find files. 

## Config init 

We are trying to do a simple single file `init.lua` setup. 
The starting points was [this](https://github.com/khuedoan/nvim-minimal/tree/master).
We have linked it to our base system using 

```sh
ln -s <full path to nvim git dir> ~/.config/nvim
```
