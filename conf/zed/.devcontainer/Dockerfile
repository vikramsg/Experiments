# Use a standard Ubuntu image as the base
FROM ubuntu:latest

# Avoid prompts from apt during image build
ENV DEBIAN_FRONTEND=noninteractive

# Install common development tools
# Git should also be installed by devcontainer
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    --no-install-recommends

# The devcontainer will be kept alive by the devcontainer CLI.
# The SSH service will be started by the 'sshd' feature in devcontainer.json,
# so no CMD or ENTRYPOINT is needed here.
