# docker/

| File | Purpose |
|---|---|
| `../Dockerfile` | Multi-stage build. Targets: `runtime` (production, lean), `dev` (runtime + pytest/ruff/mypy/uv). |
| `entrypoint.sh` | Runs under `tini`; makes `HOME`/cache dirs writable for any host UID, fails fast if live mode has no credentials. |
| `nginx/nginx.conf` | Unprivileged, read-only static server used by `make serve` / `make demo`; mirrors production headers. |
| `../docker-compose.yml` | Profiles `dev`, `ci`, `ops`, `serve`, `demo`. |

## Design decisions in one place

* **Ubuntu 24.04 base**, the same distribution as the `ubuntu-24.04` GitHub runner: same glibc, same Python 3.12, same GDAL 3.8.
* **CPU-only.** The `torch` wheel comes from the PyTorch CPU index (`pyproject.toml → [tool.uv.sources]`). CI asserts
  `torch.version.cuda is None` and a `+cpu` version string.
* **No libeccodes from apt.** The `eccodes` wheel depends on `eccodeslib`, which bundles ecCodes; a GRIB round-trip was verified.
* **Urdu without a browser.** Pillow's wheel bundles Raqm/HarfBuzz/FriBiDi; `fonts-noto-core` supplies Noto Nastaliq Urdu.
  Verified: Raqm layout joins Nastaliq correctly, the basic layout does not.
* **Non-root, read-only, capability-dropped** containers. The git token is never passed into a container.
* **Resource parity.** `ops` services default to 2 CPUs / 7 GB (smallest runner). Override with `SMOGSENSE_CPUS` / `SMOGSENSE_MEM`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build prints `WARNING: uv.lock not found` | Lock not generated yet | `make lock`, commit `uv.lock` |
| Files in `data/` owned by root | Docker created the bind-mount dir | `sudo chown -R "$USER" data models site .state`; always use `make dirs` first |
| `permission denied` writing `/app/...` | Container UID differs from file owner | `export HOST_UID=$(id -u) HOST_GID=$(id -g)` (done by `make`) |
| `docker pull ghcr.io/... denied` in CI | Package is private | GitHub → Packages → *Package settings* → set visibility **Public** (free for public packages) |
| `403 required licences not accepted` from ADS/CDS | Dataset licence not accepted | Open the dataset page while logged in and click *Accept* once |
| Out-of-memory kill locally | Limit is 7 GB by design | Reduce batch size; do not raise the limit just to pass locally — it must fit on the runner |
