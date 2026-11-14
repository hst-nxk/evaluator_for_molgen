"""How much of the evaluator file does dropping F touch once BindingDB rows that carry a
ChEMBL assay id (ASSAY_NAME 'ChEMBL_<assay_id> (CHEMBL<id>)') borrow ChEMBL's assay_type?

Two scenarios on the same v4 records, both removing F assays BEFORE aggregation and
recomputing delta_p_mean with the pipeline's own collapse_values:
  chembl_only     : F known only from ChEMBL rows (what PROPOSAL_DROP_F section 5 measured)
  with_bindingdb  : F also known from BindingDB rows linked to a ChEMBL assay
Also counts how many evaluator rows now labelled 'unknown' would get a type.
Test rows are counted only.

Usage: python -I audit_drop_f_with_bindingdb_20261104.py <dataset_dir> <out_dir>
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from molopt.dataset import aggregate  # noqa: E402

CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["stage2b"]
COLLAPSE = float(CONFIG["qc"]["collapse_tolerance"])
SLACK = float(CONFIG["replicate"]["float_slack"])
DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PARTS = ("train", "validation", "test")
ENDPOINTS = ("Ki", "Kd", "IC50", "EC50")
LINK = re.compile(r"^Ch(?:EMBL|EBML)_(\d+)")
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


def key(source, entry, assay):
    """Same construction as dataset.pipeline.assay_keys."""
    return f"{source}:{entry or ''}:{assay or ''}"


# ChEMBL: assay key -> type, and numeric assay_id -> type
chembl = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet")
c = chembl.to_table(columns=["source_database", "assay_entry_id", "assay_native_id",
                             "assay_type", "assay_context"])
g = c.group_by(["source_database", "assay_entry_id", "assay_native_id", "assay_type",
                "assay_context"]).aggregate([])
TYPE_CHEMBL, BY_NUMBER = {}, {}
for s, e, a, t, x in zip(*[g[n].to_pylist() for n in g.column_names]):
    if t:
        TYPE_CHEMBL[key(s, e, a)] = t
        BY_NUMBER[json.loads(x).get("assay_id")] = t
del c, g
log("chembl assays with type", len(TYPE_CHEMBL))

# BindingDB: assay key -> borrowed type through the ChEMBL assay id in assay_name
bdb = ds.dataset(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet", format="parquet")
b = bdb.to_table(columns=["source_database", "assay_entry_id", "assay_native_id", "assay_name"])
g = b.group_by(["source_database", "assay_entry_id", "assay_native_id", "assay_name"]).aggregate([])
TYPE_BDB = {}
link_status = Counter()
for s, e, a, n in zip(*[g[x].to_pylist() for x in g.column_names]):
    m = LINK.search(n) if n else None
    if not m:
        link_status["no ChEMBL link"] += 1
        continue
    t = BY_NUMBER.get(int(m.group(1)))
    link_status["linked, type found" if t else "linked, assay not in extract"] += 1
    if t:
        TYPE_BDB[key(s, e, a)] = t
del b, g
log("bindingdb assays", dict(link_status), "borrowed types", Counter(TYPE_BDB.values()))

SCENARIOS = {"chembl_only": TYPE_CHEMBL, "with_bindingdb": {**TYPE_CHEMBL, **TYPE_BDB}}

ev = ds.dataset(DATA / "final_dataset" / "evaluator_training.parquet", format="parquet").to_table(
    columns=["pair_id", "partition", "endpoint", "protein_hash", "smiles_a", "smiles_b",
             "delta_p_mean", "assay_types"])
pair_row = {p: i for i, p in enumerate(ev["pair_id"].to_pylist())}
log("evaluator rows", ev.num_rows)

# one pass over assay-level deltas: per evaluator pair, its (delta, type) list per scenario
f_keys = {name: {k for k, t in m.items() if t == "F"} for name, m in SCENARIOS.items()}
deltas = defaultdict(list)          # pair_id -> [(delta, key)] only for pairs touching F
types_seen = defaultdict(set)       # pair_id -> borrowed BindingDB types (unknown rows)
for path in sorted((DATA / "filter_merge_mmp" / "assay_deltas").glob("*/*.parquet")):
    t = pq.read_table(path, columns=["pair_id", "assay_key", "delta_p"])
    for p, k, d in zip(t["pair_id"].to_pylist(), t["assay_key"].to_pylist(),
                       t["delta_p"].to_numpy()):
        if p not in pair_row:
            continue
        if k in TYPE_BDB:
            types_seen[p].add(TYPE_BDB[k])
        deltas[p].append((d, k))
    # keep memory bounded: drop pairs without any F under the widest scenario
    for p in [p for p, v in deltas.items() if not any(k in f_keys["with_bindingdb"] for _, k in v)]:
        del deltas[p]
    log(path.parent.name, path.name, len(deltas))


def collapsed_mean(values):
    if not values:
        return None
    return float(aggregate.collapse_values(np.asarray(values, dtype=np.float64), COLLAPSE, SLACK).mean())


part = np.asarray(ev["partition"].to_pylist())
endpoint = np.asarray(ev["endpoint"].to_pylist())
prot = np.asarray(ev["protein_hash"].to_pylist())
old = ev["delta_p_mean"].to_numpy()
smiles_a = np.asarray(ev["smiles_a"].to_pylist(), dtype=object)
smiles_b = np.asarray(ev["smiles_b"].to_pylist(), dtype=object)


def rankable(mask):
    q = Counter(zip(np.concatenate([smiles_a[mask], smiles_b[mask]]),
                    np.concatenate([prot[mask], prot[mask]]),
                    np.concatenate([endpoint[mask], endpoint[mask]])))
    return sum(1 for v in q.values() if v >= 2)


result = {"bindingdb_assay_link_status": dict(link_status),
          "bindingdb_borrowed_types_assays": dict(Counter(TYPE_BDB.values()))}
for name, fset in f_keys.items():
    keep = np.ones(ev.num_rows, bool)
    changed = np.zeros(ev.num_rows, bool)
    new = old.copy()
    for p, v in deltas.items():
        if not any(k in fset for _, k in v):
            continue
        i = pair_row[p]
        m = collapsed_mean([d for d, k in v if k not in fset])
        if m is None:
            keep[i] = False
        else:
            changed[i], new[i] = True, m
    shift = np.abs(new[changed] - old[changed])
    block = {}
    for pt in PARTS:
        sel = part == pt
        block[pt] = {
            "rows_before": int(sel.sum()), "rows_removed": int((sel & ~keep).sum()),
            "rows_relabelled": int((sel & changed).sum()),
            "proteins_lost": int(len(set(prot[sel]) - set(prot[sel & keep]))),
            "removed_by_endpoint": {e: int((sel & ~keep & (endpoint == e)).sum()) for e in ENDPOINTS},
        }
    block["relabel_shift"] = {"n": int(changed.sum()),
                              "mean_abs": float(shift.mean()) if shift.size else None,
                              "frac_gt_0.3": float((shift > 0.3).mean()) if shift.size else None}
    vm = part == "validation"
    block["validation_rankable_queries_after"] = rankable(vm & keep)
    result[name] = block
    log(name, json.dumps(block))
result["validation_rankable_queries_before"] = rankable(part == "validation")

# evaluator rows labelled unknown that would get a borrowed type
unknown = pc.is_null(ev["assay_types"]).to_numpy(zero_copy_only=False)
pair_ids = ev["pair_id"].to_pylist()
gained = Counter()
for pt in PARTS:
    for i in np.flatnonzero(unknown & (part == pt)):
        kinds = types_seen.get(pair_ids[i])
        gained[(pt, ";".join(sorted(kinds)) if kinds else "still unknown")] += 1
result["unknown_rows_after_borrowing"] = {f"{a}|{b}": n for (a, b), n in sorted(gained.items())}
log("unknown rows", result["unknown_rows_after_borrowing"])

(OUT / "drop_f_with_bindingdb.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
log("written", OUT / "drop_f_with_bindingdb.json")
