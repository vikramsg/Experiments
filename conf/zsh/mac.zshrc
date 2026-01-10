#########################################
# Install zinit
# Home is $HOME/.local/share/zinit/
# Directly copied from https://github.com/zdharma-continuum/zinit
ZINIT_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}/zinit/zinit.git"
[ ! -d $ZINIT_HOME ] && mkdir -p "$(dirname $ZINIT_HOME)"
[ ! -d $ZINIT_HOME/.git ] && git clone https://github.com/zdharma-continuum/zinit.git "$ZINIT_HOME"
source "${ZINIT_HOME}/zinit.zsh"

#########################################
# Plugins

## Prompt
### Starship

#### The following will install and then use starship. No need to install separately
#### To configure use `starship config`

#### Load starship theme
#### line 1: `starship` binary as command, from github release
#### line 2: starship setup at clone(create init.zsh, completion)
#### line 3: pull behavior same as clone, source init.zsh
zinit ice as"command" from"gh-r" \
          atclone"./starship init zsh > init.zsh; ./starship completions zsh > _starship" \
          atpull"%atclone" src"init.zsh"
zinit light starship/starship

### Syntax highlighting

#### Plugin history-search-multi-word loaded with investigating.
zinit ice wait lucid
zinit load zdharma-continuum/history-search-multi-word

#### More plugins
zinit ice wait lucid
zinit light zsh-users/zsh-completions

zinit ice wait lucid
zinit light zsh-users/zsh-autosuggestions

zinit ice wait lucid
zinit light zdharma-continuum/fast-syntax-highlighting

#### Dotenv plugin for automatic .env loading
# Disable confirmation prompt and auto-load .env files
ZSH_DOTENV_PROMPT=false
zinit snippet OMZP::dotenv

# Enable completions
autoload -Uz compinit && compinit

# -q is for quiet; actually run all the `compdef's saved before `compinit` call
# (`compinit' declares the `compdef' function, so it cannot be used until
# `compinit' is ran; Zinit solves this via intercepting the `compdef'-calls and
# storing them for later use with `zinit cdreplay')

zinit cdreplay -q

#########################################
# Tools

### Ripgrep
# Installs ripgrep binary from GitHub Releases
zinit ice from"gh-r" as"program" mv"ripgrep* -> ripgrep" pick"ripgrep/rg"
zinit light BurntSushi/ripgrep

### FZF
# Installs fzf binary from GitHub Releases
zinit ice from"gh-r" as"program"
zinit light junegunn/fzf
# Load FZF Keybindings (Ctrl+R, etc.)
zinit ice wait lucid
zinit snippet https://raw.githubusercontent.com/junegunn/fzf/master/shell/key-bindings.zsh

#########################################
# Aliases

[[ -f ~/.zshenv ]] && source ~/.zshenv

