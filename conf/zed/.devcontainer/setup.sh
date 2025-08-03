set -e

# make the real SSH folder for sshd
SSH_DIR=/home/vscode/.ssh
mkdir -p "${SSH_DIR}"

cp /home/vscode/.ssh_host/* "${SSH_DIR}/"
# pull in your public key(s) from the host mount
# so that we can ssh using the host key
cp /home/vscode/.ssh_host/id_ed25519.pub "${SSH_DIR}/authorized_keys"

# lock down perms
sudo chown -R vscode:1000 "${SSH_DIR}"
sudo chmod 700         "${SSH_DIR}"
sudo chmod 600         "${SSH_DIR}/authorized_keys"

# restart sshd so it picks up the new authorized_keys
sudo systemctl restart sshd
