# Claude code

Install using `npm install -g @anthropic-ai/claude-code`

### Usage

```sh
claude

# Let it create the API key and it will work.
/init # To create CLAUDE.md
```

## Initial Set Up

You will need to have at least `developer` role assigned to your user in our Claude console by an admin. Then, you may run `claude` in your CLI to initiate a session with Claude Code. This session will invite you to authenticate to the console, at which point you may follow the prompts to login. You are not expected to create an API key manually; claude code will generate and manage an API key on your behalf.

Sequential invocations of `claude` should allow you to use claude code without logging in.

## Config

Claude config is available at `~/.claude.json`.

### API key

If you use a new Devcontainer etc, then you may have to repeatedly authenticate. 
Instead just mount `~/.claude.json` which has the settings built in.
