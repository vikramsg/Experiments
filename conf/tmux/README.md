# Install

Install plugin manager from herei https://github.com/tmux-plugins/tpm

Put the following lines in ~/.tmux.conf

Then do

```
tmux conf ~/.tmux.conf
```

and then finally from inside a tmux session do

```
ctrl + b + I
```

This can also be done using 

```
tmux source-file ~/.tmux.conf
```

## Linking

We are going to link the `.tmux.conf` from here to our default config file to make it easy to manage config from git.

```sh
ln -s /Users/vikramsingh/Projects/Personal/Experiments/conf/tmux/.tmux.conf ~/.tmux.conf
```

## TODO

`tmux` is still not showing the git branch on the status line!
