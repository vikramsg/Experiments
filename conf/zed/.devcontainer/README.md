# Zed Remote Development with Devcontainers

This guide provides detailed instructions on how to set up a development container for remote development with the Zed code editor using the Devcontainer CLI. By following these steps, you can create a containerized development environment with an SSH server, allowing you to edit files within the container directly from Zed.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Devcontainer CLI](https://code.visualstudio.com/docs/devcontainers/cli)
*   [Zed](https://zed.dev/)

## Setup Instructions

### Step 1: Prepare Your SSH Key

To securely connect to the Docker container, you need to provide it with your public SSH key.

1.  **Locate your public SSH key**: Your public SSH key is typically located at `~/.ssh/id_rsa.pub`. If you don't have one, you can generate a new key pair by running `ssh-keygen -t rsa`.

2.  **Copy your public key**: Open the `id_rsa.pub` file and copy its entire contents.

3.  **Update the `authorized_keys` file**: In this directory (`Experiments/conf/zed/.devcontainer/`), you will find a file named `authorized_keys`. Open it and paste your public SSH key into this file, replacing any placeholder text. The file should contain only your public key.

### Step 2: Build and Run the Devcontainer

The `devcontainer.json` and `Dockerfile` in this directory contain all the instructions needed to build and run the development container.

1.  **Open your terminal**.

2.  **Navigate to this directory**:
    ```bash
    cd /path/to/Experiments/conf/zed/.devcontainer
    ```

3.  **Build and run the devcontainer** by running the following command:
    ```bash
    devcontainer up --workspace-folder .
    ```
    This command reads the `devcontainer.json` file, builds the associated Docker image, and starts the container. The `-p 2222:22` argument in `devcontainer.json` maps port `2222` on your local machine to the container's SSH port `22`.

4.  **Verify the container is running**: You can check that the container is running with the `docker ps` command:
    ```bash
    docker ps
    ```
    You should see a container running that is based on the configuration in this directory.

## Connecting with Zed

With the container running, you can now connect to it from Zed.

1.  Open Zed.
2.  Open the command palette with `Cmd-Shift-P`.
3.  Type "Connect to SSH Host..." and select it from the list.
4.  Enter the following connection string:
    ```
    ssh://zed@localhost:2222
    ```
Zed will establish an SSH connection to the container, and you can now browse and edit files within its file system.

## Connecting with SSH from the Terminal

You can also SSH into the container from your terminal for command-line access.

1.  **Run the SSH command**:
    ```bash
    ssh zed@localhost -p 2222
    ```
    You will be logged in as the `zed` user inside the container's shell.

## Managing the Devcontainer

*   **To stop the container**, run the following command from this directory:
    ```bash
    devcontainer down --workspace-folder .
    ```

*   **To rebuild the container** after making changes to the `Dockerfile` or `devcontainer.json`:
    ```bash
    devcontainer rebuild --workspace-folder .
    ```

## Troubleshooting

*   **Permission denied (publickey)**: This error means that the SSH key in the `authorized_keys` file does not match the key you are using to connect. Ensure you have correctly copied your public key into `authorized_keys` and then rebuilt the devcontainer.

*   **Container is not running**: If `docker ps` does not show the container, it may have exited due to an error. Check the logs with:
    ```bash
    devcontainer logs --workspace-folder .
    ```
    This will help you diagnose any startup issues.

## File Descriptions

*   **`devcontainer.json`**: This is the primary configuration file for the devcontainer. It specifies how to build and run the container, including port mappings and commands to run after the container starts.

*   **`Dockerfile`**: This file contains the instructions for Docker to build the image. It sets up an Ubuntu environment, installs the SSH server and other development tools, creates a non-root user, and configures SSH.

*   **`authorized_keys`**: This file stores the public SSH keys that are authorized to connect to the container.