# Steps

1. Install Bitwardern - needed to find direct installer on website
2. Login 
    1. Email and calendar
    2. Gitlab
    3. Github
3. Install slack - website
4. Install tunnelblick - website
    1. Get ovpn file for using VPN
5. Terminal stuff
    1. VSCode - website
        1. Install extensions
    2. Install brew - website
    3. tmux - `brew install tmux`
    4. zsh config from GitHub
    5. pyenv - `curl [https://pyenv.run](https://pyenv.run/) | bash`
    6. Add ssh config - `ssh-keygen -t ed25519 -C "email"`
    7. tmux - from GitHub
    8. Install Alt - Tab from website
        1. Go to preferences - shortcut 1, change key from option key to command key to make it override system command + tab
6. Bookmarks
7. Keyboard setup
    1. Resolve pairing issue - [https://www.notion.so/Logitech-fc65efb895334e5da1bc61e9c4584eef](https://www.notion.so/Logitech-fc65efb895334e5da1bc61e9c4584eef?pvs=21) 
    2. Resolve option + right and option + left - [https://www.notion.so/Mac-1b5e41767540494e95f2dfcab6e8b0c2#44db6fca88a94dc5abab8d6c9989f519](https://www.notion.so/1b5e41767540494e95f2dfcab6e8b0c2?pvs=21)


## MX Keys pairing issue

### Not pairing with new machine

1. Is it not continuously blinking. Then its probably not going in pairing mode
    - Reset it by
        1. Switch off and on again
        2. Do esc + o, release, esc + o, release, esc + b
        3. Pair normally 

## Option keys navigation in terminal

The way to get the option keys to move a word left and right using arrow keys is slightly complicated. 

First, go to Settings → Profiles → Default → Keys → Key Mappings. Delete any existing key mapping for `Option + ->`as well as for left arrow. Then go back to General tab (which is right next  to Key Mappings)  make Option keys as `Esc+`

Again got to the Key Mappings. Add the Option + → as shortcut, and add Select Escape Sequence as input and in Esc+ add b for left arrow and f for right arrow. 
