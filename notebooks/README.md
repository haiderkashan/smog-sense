# notebooks/

Exploration only. Rules:

1. A notebook may **import** `smogsense` but must never contain logic that the pipeline depends on; promote code into `src/` with tests.
2. Strip outputs before committing (`jupyter nbconvert --clear-output`); notebooks with data or secrets in outputs are rejected by review.
3. Name as `NN_topic_owner.ipynb`, e.g. `01_station_audit_lahore.ipynb`.
4. Install with `uv sync --extra notebooks` (JupyterLab is not part of the production image).

Planned analyses: station and uptime audit of Lahore/Delhi on OpenAQ (Phase 1), AOD missingness during smog (research),
CAMS bias by regime (Phase 1), sensor-sparsity curve plots (Phase 3).
