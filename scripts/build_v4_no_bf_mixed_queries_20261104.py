"""Copy of dataset v4 without every ranking query that mixes B and F measurements.

Query = (reference SMILES, protein_hash, endpoint), both directions, exactly as Stage 3
ranks (metrics.expand_directions + query_groups). A query is MIXED when its candidate rows
contain at least one row backed by a B assay and at least one row backed by an F assay
(a single B;F record counts as both). Every evaluator row that belongs to a mixed query,
on either side, is removed. The same pair_ids are removed from generator_training and
evidence_records so the three files stay consistent.

Assay types, documented links only:
  ChEMBL rows          ASSAYS.ASSAY_TYPE
  BindingDB rows       the ChEMBL assay id BindingDB writes in ASSAY_NAME
                       ('ChEMBL_<assay_id> (CHEMBL<id>)'), then that assay's ASSAY_TYPE
BindingDB labels known only by value match (audit route 2) are NOT used to delete; the
script reports how many more queries they would add.

Source files are only read. Output goes to a new directory with a manifest.

Options (before the positional arguments or after, any order):
  --rule=query        default: drop every row of a mixed query (5,732 queries, 41,163 rows on v4)
  --rule=row          drop only rows carrying both a B and an F assay (19,174 rows on v4); rows of
                      queries that stay mixed are KEPT and listed in under_review_pair_ids.parquet
  --salt-review=PATH  deprecated compatibility option; mass-based salt review is disabled.
                      ChEMBL parent selection determines the retained structure.
Salt stripping itself is not changed here.

Usage: python -I build_v4_no_bf_mixed_queries_20261104.py <dataset_dir> <audit_dir> <out_dir>
       python -I build_v4_no_bf_mixed_queries_20261104.py <dataset_dir> <audit_dir> --check <final_dir>
       (--check: count mixed queries in <final_dir>/evaluator_training.parquet, write nothing)
"""
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

RULE, SALT_REVIEW = "query", None
for _a in list(sys.argv[1:]):
    if _a.startswith("--rule="):
        RULE = _a.split("=", 1)[1]
        sys.argv.remove(_a)
    elif _a.startswith("--salt-review="):
        SALT_REVIEW = Path(_a.split("=", 1)[1])
        sys.argv.remove(_a)
if RULE not in ("query", "row"):
    raise SystemExit("--rule must be 'query' (drop whole mixed queries) or 'row' (drop only rows carrying both labels)")

DATA, AUDIT = Path(sys.argv[1]), Path(sys.argv[2])
CHECK = sys.argv[3] == "--check"
OUT = None if CHECK else Path(sys.argv[3])
FINAL = Path(sys.argv[4]) if CHECK else DATA / "final_dataset"
LINK = re.compile(r"^Ch(?:EMBL|EBML)_(\d+)")
PARTS = ("train", "validation", "test")
ENDPOINTS = ("Ki", "Kd", "IC50", "EC50")
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


def key(source, entry, assay):
    """Same construction as dataset.pipeline.assay_keys."""
    return f"{source}:{entry or ''}:{assay or ''}"


if not CHECK and (OUT / "final_dataset").exists():
    raise SystemExit(f"{OUT / 'final_dataset'} already exists; refusing to overwrite")

# ------------------------------------------------------------------ assay types
c = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_type", "assay_context"])
g = c.group_by(c.column_names).aggregate([])
TYPE, BY_NUMBER = {}, {}
for s, e, a, t, x in zip(*[g[n].to_pylist() for n in g.column_names]):
    if t:
        TYPE[key(s, e, a)] = t
        BY_NUMBER[json.loads(x).get("assay_id")] = t
del c, g
b = ds.dataset(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_name"])
g = b.group_by(b.column_names).aggregate([])
borrowed = 0
for s, e, a, n in zip(*[g[x].to_pylist() for x in g.column_names]):
    m = LINK.search(n) if n else None
    if m and int(m.group(1)) in BY_NUMBER:
        TYPE[key(s, e, a)] = BY_NUMBER[int(m.group(1))]
        borrowed += 1
del b, g
ROUTE2 = {}
labels_path = AUDIT / "bindingdb_assay_labels.parquet"
if labels_path.is_file():
    for row in pq.read_table(labels_path, columns=["assay_key", "route", "group"]).to_pylist():
        if row["route"].startswith("route 2") and row["group"][:2] in ("B ", "F "):
            ROUTE2[row["assay_key"]] = row["group"][0]
log("typed assays", len(TYPE), "BindingDB borrowed", borrowed, "route-2 labels", len(ROUTE2))

# ------------------------------------------------------------------ evaluator rows
ev = pq.read_table(FINAL / "evaluator_training.parquet",
                   columns=["pair_id", "partition", "endpoint", "protein_hash", "smiles_a", "smiles_b"])
n = ev.num_rows
row_of = {p: i for i, p in enumerate(ev["pair_id"].to_pylist())}
has_b = np.zeros(n, bool)
has_f = np.zeros(n, bool)
has_b2 = np.zeros(n, bool)
has_f2 = np.zeros(n, bool)
for path in sorted((DATA / "filter_merge_mmp" / "records").glob("*/*.parquet")):
    t = pq.read_table(path, columns=["pair_id", "assay_keys"])
    for p, keys in zip(t["pair_id"].to_pylist(), t["assay_keys"].to_pylist()):
        i = row_of.get(p)
        if i is None:
            continue
        kinds = {TYPE.get(k) for k in keys}
        has_b[i] = "B" in kinds
        has_f[i] = "F" in kinds
        extra = {ROUTE2.get(k) for k in keys}
        has_b2[i] = has_b[i] or "B" in extra
        has_f2[i] = has_f[i] or "F" in extra
del row_of
log("rows with B", int(has_b.sum()), "with F", int(has_f.sum()), "with both", int((has_b & has_f).sum()))

# query ids, both directions: (smiles code, protein code, endpoint code)
smiles = pa.chunked_array(list(ev["smiles_a"].chunks) + list(ev["smiles_b"].chunks)).combine_chunks()
smiles_code = pc.dictionary_encode(smiles).indices.to_numpy().astype(np.int64)
prot_code = pc.dictionary_encode(ev["protein_hash"].combine_chunks()).indices.to_numpy().astype(np.int64)
ep_code = pc.index_in(ev["endpoint"].combine_chunks(), value_set=pa.array(ENDPOINTS)).to_numpy().astype(np.int64)
P = int(prot_code.max()) + 1
side_keys = []
for side in (smiles_code[:n], smiles_code[n:]):
    side_keys.append((side * P + prot_code) * len(ENDPOINTS) + ep_code)
both = np.concatenate(side_keys)
uniq, inv = np.unique(both, return_inverse=True)
q_a, q_b = inv[:n], inv[n:]
size = np.bincount(inv, minlength=uniq.size)


def mixed_queries(row_b, row_f):
    qb = np.zeros(uniq.size, bool)
    qf = np.zeros(uniq.size, bool)
    for q in (q_a, q_b):
        np.logical_or.at(qb, q, row_b)
        np.logical_or.at(qf, q, row_f)
    return qb & qf


mixed = mixed_queries(has_b, has_f)
mixed2 = mixed_queries(has_b2, has_f2)
if RULE == "row":      # only rows whose own record carries both a B and an F assay
    drop, drop2 = has_b & has_f, has_b2 & has_f2
else:                   # every row of a mixed query
    drop, drop2 = mixed[q_a] | mixed[q_b], mixed2[q_a] | mixed2[q_b]
log("rule", RULE, "| mixed queries", int(mixed.sum()), "rows to drop", int(drop.sum()))

# Remaining mixed B/F queries are reported for review. Salt mass contributes no review flags.
# Zero-valued salt fields preserve the existing report schema for downstream readers.
mixed_left = mixed_queries(has_b & ~drop, has_f & ~drop)
review_bf = ~drop & (mixed_left[q_a] | mixed_left[q_b])
review_salt_total = np.zeros(n, bool)
review_salt_largest = np.zeros(n, bool)
if SALT_REVIEW is not None:
    log("--salt-review accepted for compatibility; ChEMBL parent selection is authoritative.")
log("kept under review:", {"mixed_B_F": int(review_bf.sum()), "salt_total_heavier": int(review_salt_total.sum()),
                           "salt_largest_heavier": int(review_salt_largest.sum())})
if CHECK:
    print(json.dumps({"file": str(FINAL / "evaluator_training.parquet"), "rule": RULE, "rows": n,
                      "rows_with_B_and_F_record": int((has_b & has_f).sum()),
                      "mixed_queries": int(mixed.sum()), "rows_dropped": int(drop.sum()),
                      "rows_after": int((~drop).sum()), "mixed_queries_remaining": int(mixed_left.sum()),
                      "under_review_rows": {"mixed_B_F": int(review_bf.sum()),
                                            "salt_total_heavier": int(review_salt_total.sum()),
                                            "salt_largest_heavier": int(review_salt_largest.sum())}}))
    raise SystemExit(0)

part = np.asarray(ev["partition"].to_pylist())
endpoint = np.asarray(ev["endpoint"].to_pylist())
prot = np.asarray(ev["protein_hash"].to_pylist())


def query_partition_counts(mask_q):
    """Mixed queries per partition (a query lives in one partition: one protein)."""
    first = np.full(uniq.size, -1)
    first[q_a] = np.arange(n)
    first[q_b] = np.arange(n)
    out = Counter(part[first[mask_q & (first >= 0)]].tolist())
    return {p: int(out.get(p, 0)) for p in PARTS}


def rankable(keep):
    cnt = np.bincount(q_a[keep], minlength=uniq.size) + np.bincount(q_b[keep], minlength=uniq.size)
    return cnt >= 2


rank_before, rank_after = rankable(np.ones(n, bool)), rankable(~drop)
first = np.full(uniq.size, -1)
first[q_a] = np.arange(n)
first[q_b] = np.arange(n)
q_part = part[first]

report = {
    "salt_policy": "ChEMBL Structure Pipeline parent selection; no mass-based exclusion or review",
    "rule_name": RULE,
    "rule": ("drop every evaluator row that belongs (either side) to a query whose candidates include "
             "both a B-assay row and an F-assay row; query = (reference SMILES, protein, endpoint), both directions"
             if RULE == "query" else
             "drop only the evaluator rows whose own record carries both a B and an F assay; rows that are B-only "
             "or F-only stay even when their query still mixes the two (those rows are listed in under_review_pair_ids)"),
    "mixed_queries_remaining": query_partition_counts(mixed_left),
    "under_review_rows_kept": {"mixed_B_F": int(review_bf.sum()), "salt_total_heavier": int(review_salt_total.sum()),
                               "salt_largest_heavier": int(review_salt_largest.sum())},
    "type_sources": "ChEMBL ASSAYS.ASSAY_TYPE; BindingDB via the ChEMBL assay id in ASSAY_NAME",
    "source_dataset": str(FINAL),
    "queries_total": {p: int((q_part == p).sum()) for p in PARTS},
    "mixed_queries": query_partition_counts(mixed),
    "mixed_queries_ranked_before": {p: int((mixed & rank_before & (q_part == p)).sum()) for p in PARTS},
    "rankable_queries_before": {p: int((rank_before & (q_part == p)).sum()) for p in PARTS},
    "rankable_queries_after": {p: int((rank_after & (q_part == p)).sum()) for p in PARTS},
    "rows": {}, "if_route2_labels_were_also_used": {
        "mixed_queries": query_partition_counts(mixed2),
        "rows_dropped": {p: int((drop2 & (part == p)).sum()) for p in PARTS}},
}
for p in PARTS:
    sel = part == p
    report["rows"][p] = {
        "before": int(sel.sum()), "dropped": int((sel & drop).sum()), "after": int((sel & ~drop).sum()),
        "dropped_B_and_F_record": int((sel & drop & has_b & has_f).sum()),
        "dropped_by_endpoint": {e: int((sel & drop & (endpoint == e)).sum()) for e in ENDPOINTS},
        "proteins_before": len(set(prot[sel])), "proteins_after": len(set(prot[sel & ~drop])),
        "after_rows_with_F_only_queries": int((sel & ~drop & has_f).sum()),
    }
log(json.dumps(report["rows"]))

# ------------------------------------------------------------------ write new files
dropped_ids = pa.array(sorted(pc.filter(ev["pair_id"], pa.array(drop)).to_pylist()), pa.string())
review_rows = []
for mask, reason in ((review_bf, "mixed_B_F_query_remaining"), (review_salt_total, "salt_total_removed_heavier"),
                     (review_salt_largest, "salt_largest_removed_heavier")):
    ids = pc.filter(ev["pair_id"], pa.array(mask))
    review_rows.append(pa.table({"pair_id": ids, "reason": pa.array([reason] * len(ids), pa.string())}))
review_table = pa.concat_tables(review_rows)
del ev, smiles, smiles_code
(OUT / "final_dataset").mkdir(parents=True)
written = {}
for name in ("evaluator_training", "generator_training", "evidence_records"):
    source = pq.ParquetFile(FINAL / f"{name}.parquet")
    writer = None
    kept = removed = 0
    for i in range(source.metadata.num_row_groups):
        t = source.read_row_group(i)
        mask = pc.invert(pc.is_in(t["pair_id"], value_set=dropped_ids))
        out = t.filter(mask)
        removed += t.num_rows - out.num_rows
        kept += out.num_rows
        if writer is None:
            writer = pq.ParquetWriter(OUT / "final_dataset" / f"{name}.parquet", t.schema,
                                      compression="zstd")
        writer.write_table(out)
    writer.close()
    written[name] = {"before": source.metadata.num_rows, "removed": removed, "after": kept}
    log(name, written[name])
for extra in ("protein_split.parquet", "protein_split_manifest.json", "protein_split_input.fasta",
              "export_manifest.json"):
    shutil.copy2(FINAL / extra, OUT / "final_dataset" / extra)
pq.write_table(pa.table({"pair_id": dropped_ids}), OUT / "final_dataset" / "dropped_pair_ids.parquet")
pq.write_table(review_table, OUT / "final_dataset" / "under_review_pair_ids.parquet")  # kept in the data, listed here
report["files"] = written
report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
report["script"] = "scripts/build_v4_no_bf_mixed_queries_20261104.py"
(OUT / "final_dataset" / "bf_mixed_query_manifest.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
log("done", OUT)
