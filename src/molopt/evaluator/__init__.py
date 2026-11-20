"""Stage 3: the evaluator baseline — predict Δp from a molecule pair, protein and endpoint.

Modules:
    data      : read the dataset, filter by partition and by slice, pack batches
    features  : Morgan fingerprints and ESM2 embeddings, both cached on disk
    models    : shared ligand encoder, edit encoder, three output heads (direct,
                direct_no_edit, twin)
    metrics   : regression, per-query ranking, antisymmetry measurements
    train     : build caches, trivial baselines, the training loop, validation, checkpoints
    classical : classical ML baselines (Linear/Ridge/Lasso/XGBoost) on exactly the
                features, the split and the metrics of the training loop
    capacity  : the capacity check — can the model fit a very small sample at all?

`classical` likewise imports scikit-learn, scipy and xgboost inside function bodies only,
for the same reason.

Only `models` imports torch at module level. `metrics`, `data`, `features` and `train` do
not: torch is imported inside the functions that need it, so `--mode baseline` and
`--mode prepare --skip-protein` run on a machine without torch, and so do most of the
tests.
"""
