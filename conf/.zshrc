# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded, in which case,
# to know which specific one was loaded, run: echo $RANDOM_THEME
# See https://github.com/ohmyzsh/ohmyzsh/wiki/Themes
ZSH_THEME="robbyrussell"

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
# HYPHEN_INSENSITIVE="true"

# Uncomment one of the following lines to change the auto-update behavior
# zstyle ':omz:update' mode disabled  # disable automatic updates
# zstyle ':omz:update' mode auto      # update automatically without asking
# zstyle ':omz:update' mode reminder  # just remind me to update when it's time

# Uncomment the following line to change how often to auto-update (in days).
# zstyle ':omz:update' frequency 13

# Would you like to use another custom folder than $ZSH/custom?
# ZSH_CUSTOM=/path/to/new-custom-folder

# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(
	git 
	python
	zsh-autosuggestions
	zsh-autocomplete
	kubectl
	per-directory-history
	zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# User configuration

# export MANPATH="/usr/local/man:$MANPATH"

# You may need to manually set your language environment
# export LANG=en_US.UTF-8

# Preferred editor for local and remote sessions
# if [[ -n $SSH_CONNECTION ]]; then
#   export EDITOR='vim'
# else
#   export EDITOR='mvim'
# fi

# Compilation flags
# export ARCHFLAGS="-arch x86_64"

# Set personal aliases, overriding those provided by oh-my-zsh libs,
# plugins, and themes. Aliases can be placed here, though oh-my-zsh
# users are encouraged to define aliases within the ZSH_CUSTOM folder.
# For a full list of active aliases, run `alias`.
#
# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"

# Aliases
alias vi="/usr/bin/vim"

alias python="python3"

alias lint="~/.local/bin/linter.sh"

alias aws="/home/vikramsg/.local/bin/aws"

alias awsdev="export AWS_DEFAULT_PROFILE=mocca-developer-dev"
alias awstest="export AWS_DEFAULT_PROFILE=mocca-developer-test"
alias awsprod="export AWS_DEFAULT_PROFILE=mocca-developer-prod"
alias awsmaster="export AWS_DEFAULT_PROFILE=mocca-developer-master"

alias omicsprod="export AWS_DEFAULT_PROFILE=omics-hub-developer-prod"
alias omicsdev="export AWS_DEFAULT_PROFILE=omics-hub-developer-dev"

alias k="kubectl"

# Use to get magic autocomplete
alias insh="inshellisense --shell bash"

# Poetry
export PATH="/home/vikramsg/.local/bin:$PATH"

# Usage: kubedel <word>
# Deletes all kubernets pods with "word" in their name
kubedel() {
  local search_term="$1"
  kubectl get pods | grep "$search_term" | awk '{print $1}' | xargs -I {pod} kubectl delete pod {pod}
}

## Limit autocompletion lines
# Autocompletion
zstyle -e ':autocomplete:list-choices:*' list-lines 'reply=( $(( LINES / 3 )) )'

# Override history search.
zstyle ':autocomplete:history-incremental-search-backward:*' list-lines 8

# History menu.
zstyle ':autocomplete:history-search-backward:*' list-lines 25
## Limit autocompletion lines

## Limit change menu select bindings 
zmodload zsh/complist
bindkey -M menuselect '^C' send-break 
bindkey -M menuselect '^[[C' send-break 
## Limit change menu select bindings 


