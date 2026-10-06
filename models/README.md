# models/

Model weights are **never committed**. A trained model is a *bundle*: a tarball published as a GitHub Release asset
(`models-vMAJOR.MINOR.PATCH`) and downloaded by `models/registry.py`, which verifies every file's sha256 against the
manifest before use and refuses an incompatible bundle.

```
smogsense-model-<version>.tar.gz
  manifest.json            see below
  model_card.md            filled-in copy of model_card_template.md
  encoder.pt               torch state dict (meta-learned initialisation)
  head.pt                  torch state dict (non-crossing quantile head)
  lgbm/h{24,48,72}/q{05..95}.txt   LightGBM boosters, one per horizon and level (3 x 19 = 57 files)
  calibration.json         CQR/ACI state and parameters
  evaluation/              ladder + sparsity-curve outputs that justify promotion
```

`manifest.json` fields: `bundle_version`, `git_sha`, `config_hash`, `feature_set_version`, `cams_cycle_range_trained`
(forecast distributions shift at every CAMS model upgrade), `trained_domains`, `training_window_utc`, `torch_version`,
`lightgbm_version`, `files: [{path, sha256, bytes}]`, `metrics_summary`.

## Promotion

Promotion is a **reviewed pull request** that changes `promoted_bundle` in `configs/model.yaml` from `none` (baseline-only)
to a release tag. The PR must attach the ladder table and sparsity curve and tick the guardrail checklist. Rolling back is
reverting that one line.
