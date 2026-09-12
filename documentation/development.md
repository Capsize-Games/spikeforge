# Developer script

[`scripts/dev.sh`](../scripts/dev.sh) bundles the common tasks (setup, lint,
tests, dev servers, dataset cache, Docker),
and [`scripts/docker_server.sh`](../scripts/docker_server.sh) builds and runs the
server from the Docker container:

```bash
scripts/dev.sh help          # list every command
scripts/dev.sh check         # ruff + client type-check + client build
scripts/dev.sh bench         # run the benchmark suite (spikeforge-benchmark)
scripts/dev.sh dev           # run the API and Vite dev server together
scripts/dev.sh data          # show the dataset cache and sizes
scripts/dev.sh data-clear    # clear dataset caches (keeps models)
scripts/dev.sh docker-server # run the server from the Docker container
scripts/dev.sh docker-reset  # rebuild the Docker volume from scratch
```
