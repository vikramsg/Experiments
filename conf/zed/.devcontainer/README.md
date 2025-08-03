# Zed Remote Development with Devcontainers

This guide provides instructions on how to use this devcontainer for remote development with the Zed code editor.

This setup uses the [Devcontainer CLI](https://code.visualstudio.com/docs/devcontainers/cli) and the [devcontainer `sshd` feature](https://github.com/devcontainers/features/tree/main/src/sshd) to automatically create a containerized development environment with a secure SSH server. This allows you to edit files within the container directly from Zed.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Devcontainer CLI](https://code.visualstudio.com/docs/devcontainers/cli)
*   [Zed](https://zed.dev/)
*   `git` is already setup and uses `ssh` to connect to `origin`. `ssh` key is in `~/.ssh`.

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
    devcontainer down --workspace-folder ../../ --config .devcontainer/devcontainer.json
    ```

## Devcontainer gotchas

1. If we do not open the devcontainer from the root of the repo, the CLI will keep complaining about git.
  - So you have to set your workspace folder to a different location even though you want to use different devcontainer files.

  ```bash
  # To create the container
  devcontainer up --workspace-folder ../../ --config .devcontainer/devcontainer.json

  # To exec into the container
  devcontainer exec --workspace-folder ../../ --config .devcontainer/devcontainer.json /bin/bash
  ```

2. Rebuild

```bash
devcontainer up --workspace-folder ../../ --config .devcontainer/devcontainer.json --remove-existing-container --build-no-cache
```

3. Logging

```bash
devcontainer up --workspace-folder . --remove-existing-container --build-no-cache --log-level trace < /dev/null &> out.log &
```

4. `ssh` into devcontainer failed.

Sometimes we get errors like the following.

```bash
ssh vscode@localhost -p 2222
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
...
Host key verification failed.
```

Just pop the known hosts key from your local `~/.ssh/known_hosts`.

4. Terminal complains about `git` directory missing.
  - This seems to be an intermittent issue, and sometimes it just disappears after a while?
  - I should build from scratch to confirm.

## ssh

The biggest issues setting up was with `ssh`.
We want to be able to `ssh` into the container so the container must have an `ssh` server running.
So we used the `sshd` feature which does this for us.
However, there are some competing issues.

1. To enable us to `ssh` in, we need to add the client's public key as an authorized key on the server and change the permissions so that the server is the ownder of the key.
  - However this will change the permissions on the `ssh` folder on the host.
2. We also want to be able to do `git push` from inside the container, so ideally we are mounting the host `ssh` keys into the container.

### ssh prompts

Everytime we rebuild the container, we will get a prompt to allow `ssh`. So we have to automate that. One example is the following

```bash
alias zed-ssh="ssh-keygen -R '[localhost]:2222' && ssh-keyscan -p 2222 localhost >> ~/.ssh/known_hosts \
                    && zed ssh://vscode@localhost:2222/workspaces/Experiments"
```

### Remote

1. To enable using `devcontainer` on a remote machine, the remote machine devcontainer should have an ssh server running.
  - In addition it should have the client machine's key as an authorized key.
  - Also, the port of the ssh server on the devcontainer should be exposed on the VM to the internet.


## Issues

The devcontainer CLI does not support port forwarding.
So we will have to do custom port forwarding by parsing the devcontainer config as a post create command.
https://github.com/devcontainers/cli/issues/22

To figure out if ports have been forwarded use `docker inspect <container id>` and check the port mappings.
