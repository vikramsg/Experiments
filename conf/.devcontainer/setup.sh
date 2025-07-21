#!/bin/bash


mkdir /home/vscode/.ssh
sudo cp -r ~/.ssh-owner/* /home/vscode/.ssh/
sudo chown -R vscode:vscode /home/vscode/.ssh 2>/dev/null || true

# Install Tmux Plugin Manager (TPM) if tmux config exists
if [ -f "/home/vscode/.tmux.conf" ]; then
    git clone https://github.com/tmux-plugins/tpm /home/vscode/.tmux/plugins/tpm 2>/dev/null || true
    # Set correct ownership for mounted files
    sudo chown -R vscode:vscode /home/vscode/.tmux* 2>/dev/null || true
fi

# Set correct ownership for nvim config
sudo chown -R vscode:vscode /home/vscode/.config/nvim 2>/dev/null || true

# Install stylua for Lua formatting (used by your nvim config)
STYLUA_VERSION=$(curl -s "https://api.github.com/repos/JohnnyMorganz/StyLua/releases/latest" | grep -Po '"tag_name": "v\K[^"]*')
curl -L "https://github.com/JohnnyMorganz/StyLua/releases/latest/download/stylua-linux-x86_64.zip" -o stylua.zip
unzip stylua.zip
sudo mv stylua /usr/local/bin/
rm stylua.zip

# Install lazygit
LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | \grep -Po '"tag_name": *"v\K[^"]*')
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
tar xf lazygit.tar.gz lazygit
sudo install lazygit -D -t /usr/local/bin/
rm -rf lazygit lazygit.tar.gz

# Install gemini
npm install -g @google/gemini-cli


echo "Setup complete! You can now use Neovim with your mounted configuration."
echo "Run 'tmux' and then press 'Ctrl+a + I' to install tmux plugins."
