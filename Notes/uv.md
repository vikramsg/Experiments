## uv

### Tools

We can install tools using `uv tool install ruff`. This will allow us to do stuff like `uvx ruff`.
We can then add aliases to run these tools via `uv` installed tools.

We can also add locally developed tools via `uv tool install .`.
`But`, this is very finicky and sometimes leads `uv` to search for the package on PyPI instead.
So, try to install the package from `$HOME` by giving the full path `uv tool install /path/to/pyproject.toml`.
