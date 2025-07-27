# Zed Remote Development with Devcontainers

This guide provides instructions on how to use this devcontainer for remote development with the Zed code editor.

This setup uses the [Devcontainer CLI](https://code.visualstudio.com/docs/devcontainers/cli) and the [devcontainer `sshd` feature](https://github.com/devcontainers/features/tree/main/src/sshd) to automatically create a containerized development environment with a secure SSH server. This allows you to edit files within the container directly from Zed.

The primary benefit of this new approach is simplicity and security. You no longer need to manually manage SSH keys in an `authorized_keys` file. The `sshd` feature automatically uses the keys available in your local SSH agent.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Devcontainer CLI](https://code.visualstudio.com/docs/devcontainers/cli)
*   [Zed](https://zed.dev/)
*   An SSH key added to your local SSH agent. You can add your default key by running `ssh-add`.

## Setup Instructions

### Step 1: Build and Run the Devcontainer

The `devcontainer.json` and `Dockerfile` in this directory contain all the instructions needed to build and run the development container.

1.  **Open your terminal**.

2.  **Navigate to the project's root directory**:
    ```bash
    cd /path/to/Experiments/conf/zed
    ```
    *Note: You run the command from the directory containing the `.devcontainer` folder, not from inside it.*

3.  **Build and run the devcontainer** by running the following command:
    ```bash
    devcontainer up --workspace-folder .
    ```
    This command reads the `devcontainer.json` file, builds the Docker image (if not already built), and starts the container. The configuration automatically forwards port `2222` on your local machine to the container's SSH port.

4.  **Verify the container is running**: You can check that the container is running with the `docker ps` command:
    ```bash
    docker ps
    ```
    You should see a container running that is based on this configuration.

## Connecting with Zed

With the container running, you can now connect to it from Zed.

1.  Open Zed.
2.  Open the command palette with `Cmd-Shift-P`.
3.  Type "Connect to SSH Host..." and select it from the list.
4.  Enter the following connection string:
    ```
    ssh://vscode@localhost:2222
    ```
Zed will establish an SSH connection to the container. You can now browse and edit files within the `Experiments/conf/zed` directory, which is mounted as the workspace.

## Connecting with SSH from the Terminal

You can also SSH into the container from your terminal for command-line access.

1.  **Run the SSH command**:
    ```bash
    ssh vscode@localhost -p 2222
    ```
    You will be logged in as the `vscode` user inside the container's shell.

## Managing the Devcontainer

*   **To stop the container**, run the following command from the `Experiments/conf/zed` directory:
    ```bash
    devcontainer down --workspace-folder .
    ```

*   **To rebuild the container** after making changes to the `Dockerfile` or `devcontainer.json`:
    ```bash
    devcontainer rebuild --workspace-folder .
    ```

## Troubleshooting

*   **Permission denied (publickey)**: This error likely means that your SSH agent isn't running or doesn't have your key loaded. Run `ssh-add` to add your default identity to the agent. The devcontainer feature automatically uses keys from the agent, so manual configuration is not needed.

*   **WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!**: This happens if you rebuild the container, as a new host key is generated. To fix this, run the following command to remove the old key from your `known_hosts` file:
    ```bash
    ssh-keygen -R "[localhost]:2222"
    ```

* **CHANGES NOT APPEARING**: `devcontainer up --workspace-folder . --remove-existing-container --build-no-cache`

## File Descriptions

*   **`devcontainer.json`**: This is the primary configuration file. It specifies the base image and uses a "feature" to automatically install and configure an SSH server, eliminating the need for manual setup. It also forwards the SSH port.

*   **`Dockerfile`**: This file contains the instructions to build the base Docker image. It now only installs essential development tools, as user and SSH setup are handled by the `sshd` feature.
