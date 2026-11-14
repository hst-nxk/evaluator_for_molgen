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
drop = mixed[q_a] | mixed[q_b]
drop2 = mixed2[q_a] | mixed2[q_b]
log("mixed queries", int(mixed.sum()), "rows to drop", int(drop.sum()))

def cat(mask):
    b,f=has_b[mask],has_f[mask]
    return {"B_only":int((b&~f).sum()),"F_only":int((f&~b).sum()),"B_and_F":int((b&f).sum()),"no_B_or_F_label":int((~b&~f).sum()),"total":int(mask.sum())}
res={"before":cat(np.ones(n,bool)),"after":cat(~drop),"removed":cat(drop),"mixed_queries":int(mixed.sum())}
print("RESULT",json.dumps(res))
json.dump(res,open("audit_20261107/bf_traced_counts.json","w"),indent=1)
