# Use a standard Ubuntu image as the base
FROM ubuntu:latest

# Avoid prompts from apt during image build
ENV DEBIAN_FRONTEND=noninteractive

# Install OpenSSH server, sudo, and common development tools
RUN apt-get update && apt-get install -y \
    openssh-server \
    sudo \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    --no-install-recommends

# Create a non-root user for SSH access
# The user will be 'zed' with password 'zed'
RUN useradd -m -s /bin/bash zed
RUN echo "zed:zed" | chpasswd
RUN adduser zed sudo

# Set up SSH for the zed user
# This will copy the authorized_keys file from the .devcontainer directory
RUN mkdir -p /home/zed/.ssh
COPY authorized_keys /home/zed/.ssh/authorized_keys
RUN chown -R zed:zed /home/zed/.ssh
RUN chmod 700 /home/zed/.ssh
RUN chmod 600 /home/zed/.ssh/authorized_keys

# Create sshd privilege separation directory
RUN mkdir -p /run/sshd

# The devcontainer will be kept alive by the devcontainer CLI.
# The SSH service will be started using a command in devcontainer.json,
# so no CMD or ENTRYPOINT is needed here.
