# Zed Remote Development with Docker

This guide provides detailed instructions on how to set up a Docker container for remote development with the Zed code editor. By following these steps, you can create a containerized development environment with an SSH server, allowing you to edit files within the container directly from Zed.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Zed](https://zed.dev/)

## Setup Instructions

### Step 1: Prepare Your SSH Key

To securely connect to the Docker container, you need to provide it with your public SSH key.

1.  **Locate your public SSH key**: Your public SSH key is typically located at `~/.ssh/id_rsa.pub`. If you don't have one, you can generate a new key pair by running `ssh-keygen -t rsa`.

2.  **Copy your public key**: Open the `id_rsa.pub` file and copy its entire contents.

3.  **Create the `authorized_keys` file**: In this directory (`conf/zed/Docker/`), you will find a file named `authorized_keys`. Open it and paste your public SSH key into this file, replacing the placeholder text. The file should contain only your public key.

### Step 2: Build the Docker Image

The `Dockerfile` in this directory contains all the instructions needed to build a Docker image with an OpenSSH server and a dedicated `zed` user.

1.  **Open your terminal**.

2.  **Navigate to this directory**:
    ```bash
    cd /path/to/Experiments/conf/zed/Docker
    ```

3.  **Build the Docker image** by running the following command:
    ```bash
    docker build -t zed-sshd .
    ```
    This command reads the `Dockerfile`, builds the image, and tags it with the name `zed-sshd`.

### Step 3: Run the Docker Container

Once the image is built, you can run it as a container.

1.  **Run the container** using the following command:
    ```bash
    docker run -d -p 2222:22 --name zed-container zed-sshd
    ```

    *   `-d`: Runs the container in detached mode (in the background).
    *   `-p 2222:22`: Maps port `2222` on your local machine to port `22` (the SSH port) inside the container.
    *   `--name zed-container`: Assigns a memorable name to the container.
    *   `zed-sshd`: The name of the image to use.

2.  **Verify the container is running**: You can check that the container is running with the `docker ps` command:
    ```bash
    docker ps
    ```
    You should see `zed-container` in the list of running containers.

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

## Troubleshooting

*   **Permission denied (publickey)**: This error means that the SSH key in the `authorized_keys` file does not match the key you are using to connect. Ensure you have correctly copied your public key into the `authorized_keys` file and then rebuilt the Docker image.

*   **Container is not running**: If `docker ps` does not show the `zed-container`, it may have exited due to an error. Check the logs of the stopped container with:
    ```bash
    docker logs zed-container
    ```
    This will help you diagnose any startup issues.

## File Descriptions

*   **`Dockerfile`**: This file contains the instructions for Docker to build the image. It sets up an Ubuntu environment, installs the SSH server, creates a non-root user, and configures SSH.

*   **`authorized_keys`**: This file stores the public SSH keys that are authorized to connect to the container.
