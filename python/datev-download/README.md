# datev-download


## uv

Setup uv for the first time.

```bash
uv self update
uv init
```

Edit the `.python-version` file to the version you want to use.

Add a dependency by doing `uv add <package-name>`. The first time we do this it will download the package and create a virtual environment, as well as setup the lockfile `uv.lock`.


## playwright

```bash
uv add playwright
playwright install
```

## datev cannot be automated

Unfortunately, DATEV cannot be automated. It requires a 2FA login, which prevents us from automating it.
