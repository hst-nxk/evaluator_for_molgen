"""B / F (and other ChEMBL assay types) in dataset v4: where they come from, how far they
reach, and what dropping F would remove. Train / validation / test rows are COUNTED only
(no model, no metric on test).

  1. observations: ChEMBL assays.assay_type per endpoint (BindingDB has no such field)
  2. final files: assay_types per partition / endpoint (rows, proteins, validation queries)
  3. B vs F agreement inside the same record (same pair, same protein, same endpoint)
  4. simulated "drop F": per pair, recompute delta_p_mean from the non-F assays in
     filter_merge_mmp/assay_deltas; count rows removed / changed, proteins and queries lost

Usage: python -I audit_assay_type_bf_20261104.py <dataset_dir> <out_dir>
"""
import json
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
OUT.mkdir(parents=True, exist_ok=True)
ENDPOINTS = ("Ki", "Kd", "IC50", "EC50")
PARTS = ("train", "validation", "test")
TOL = 0.3
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


result = {"dataset": str(DATA)}

# ------------------------------------------------------------------ 1. observations
chembl = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet")
obs = chembl.to_table(columns=["source_database", "assay_entry_id", "assay_native_id",
                               "assay_type", "endpoint", "observation_id"])
log("chembl observations", obs.num_rows)
excluded = pq.read_table(DATA / "filter_merge_mmp" / "excluded_observations.parquet",
                         columns=["observation_id"])["observation_id"]
kept = pc.invert(pc.is_in(obs["observation_id"], value_set=excluded.combine_chunks()))
obs_types = pc.fill_null(obs["assay_type"], "null")


def crosstab(types, endpoints, mask=None):
    t = pa.table({"t": types, "e": endpoints})
    if mask is not None:
        t = t.filter(mask)
    g = t.group_by(["t", "e"]).aggregate([("t", "count")])
    out = defaultdict(dict)
    for a, b, n in zip(g["t"].to_pylist(), g["e"].to_pylist(), g["t_count"].to_pylist()):
        out[a][b] = n
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


result["observations_chembl_all"] = crosstab(obs_types, obs["endpoint"])
result["observations_chembl_after_filter"] = crosstab(obs_types, obs["endpoint"], kept)

# assay_key -> type, same construction as pipeline.assay_keys
keys = pc.binary_join_element_wise(obs["source_database"], pc.fill_null(obs["assay_entry_id"], ""),
                                   pc.fill_null(obs["assay_native_id"], ""), ":")
kt = pa.table({"k": keys, "t": obs["assay_type"]}).filter(pc.invert(pc.is_null(obs["assay_type"])))
kt = kt.group_by(["k", "t"]).aggregate([])
TYPE = dict(zip(kt["k"].to_pylist(), kt["t"].to_pylist()))
result["assays_by_type"] = dict(Counter(TYPE.values()))
log("assay types", result["assays_by_type"])
bdb = pq.ParquetFile(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet").metadata.num_rows
result["observations_bindingdb_rows_no_type_field"] = bdb
del obs, keys, kt, excluded, kept, obs_types

# ------------------------------------------------------------------ 2. final files
EVAL = DATA / "final_dataset" / "evaluator_training.parquet"
ev = ds.dataset(EVAL, format="parquet").to_table(
    columns=["pair_id", "partition", "endpoint", "protein_hash", "smiles_a", "smiles_b",
             "assay_types", "delta_p_mean", "n_assays"])
ev_types = pc.fill_null(ev["assay_types"], "unknown(BindingDB)").combine_chunks()
log("evaluator rows", ev.num_rows)


def group_label(t):
    if t == "unknown(BindingDB)":
        return t
    parts = set(t.split(";"))
    if parts == {"B"}:
        return "B only"
    if parts == {"F"}:
        return "F only"
    if "F" in parts and "B" in parts:
        return "B+F mixed"
    if "F" in parts:
        return "F + other (A/T/U)"
    return "B + other (A/T/U)" if "B" in parts else "other only (A/T/U/P)"


labels_vocab = pc.unique(ev_types).to_pylist()
label_of = {t: group_label(t) for t in labels_vocab}
result["evaluator_assay_types_raw"] = dict(Counter(ev_types.to_pylist()).most_common())
grp = pa.array([label_of[t] for t in ev_types.to_pylist()])
ev = ev.append_column("grp", grp)

tab = {}
for part in PARTS:
    sub = ev.filter(pc.equal(ev["partition"], part))
    g = sub.group_by(["grp", "endpoint"]).aggregate([("grp", "count")])
    d = defaultdict(dict)
    for a, b, n in zip(g["grp"].to_pylist(), g["endpoint"].to_pylist(), g["grp_count"].to_pylist()):
        d[a][b] = n
    tab[part] = {"rows": sub.num_rows, "by_group_endpoint": {k: dict(v) for k, v in d.items()}}
result["evaluator_by_partition"] = tab

for name in ("evidence_records", "generator_training"):
    t = pq.read_table(DATA / "final_dataset" / f"{name}.parquet", columns=["assay_types", "partition"])
    lab = [label_of.get(x, group_label(x)) if x else "unknown(BindingDB)"
           for x in t["assay_types"].to_pylist()]
    result[f"{name}_by_group"] = dict(Counter(lab).most_common())
    del t
log("final files counted")

# ------------------------------------------------------------------ 3 + 4. assay level
has_f = pc.match_substring(pc.fill_null(ev["assay_types"], ""), "F")
f_ids = set(pc.filter(ev["pair_id"], has_f).to_pylist())
ev_ids = set(ev["pair_id"].to_pylist())   # B-vs-B is compared on the same exported rows
log("evaluator pairs with any F assay", len(f_ids))
stat = defaultdict(list)   # pair_id -> [(delta_p of one assay, assay type or None)]
b_first = []   # for records with >=2 B assays: (first B delta, second B delta)
for path in sorted((DATA / "filter_merge_mmp" / "assay_deltas").glob("*/*.parquet")):
    t = pq.read_table(path, columns=["pair_id", "assay_key", "delta_p"])
    pid = t["pair_id"].to_pylist()
    ak = t["assay_key"].to_pylist()
    dp = t["delta_p"].to_numpy()
    per_pair_b = defaultdict(list)
    for p, k, d in zip(pid, ak, dp):
        kind = TYPE.get(k)
        if kind == "B":
            per_pair_b[p].append(d)
        if p in f_ids:
            stat[p].append((d, kind))
    for p, v in per_pair_b.items():
        if len(v) >= 2 and p not in f_ids and p in ev_ids:
            b_first.append((v[0], v[1]))
    log(path.parent.name, path.name, len(stat))


def collapsed_mean(values):
    """Mean exactly as Stage 2b exports it: near-identical copies merged first."""
    if not values:
        return None
    return float(aggregate.collapse_values(np.asarray(values, dtype=np.float64),
                                           COLLAPSE, SLACK).mean())


# sanity: recomputed all-assay mean must equal delta_p_mean of the export
ev_f = ev.filter(has_f)
dm = dict(zip(ev_f["pair_id"].to_pylist(), ev_f["delta_p_mean"].to_pylist()))
diff = [abs(collapsed_mean([d for d, _ in v]) - dm[p]) for p, v in stat.items() if p in dm and v]
result["sanity_recomputed_mean_max_abs_diff"] = float(max(diff)) if diff else None
result["sanity_pairs_checked"] = len(diff)

# 3. agreement B vs F inside one record, against B vs B inside one record
bf = []
for v in stat.values():
    b = collapsed_mean([d for d, k in v if k == "B"])
    f = collapsed_mean([d for d, k in v if k == "F"])
    if b is not None and f is not None:
        bf.append((b, f))
bf = np.array(bf) if bf else np.zeros((0, 2))
bb = np.array(b_first) if b_first else np.zeros((0, 2))


def agreement(x):
    if not len(x):
        return {"n": 0}
    a, b = x[:, 0], x[:, 1]
    decisive = (np.abs(a) > TOL) & (np.abs(b) > TOL)
    return {"n": int(len(x)), "mean_abs_diff": float(np.mean(np.abs(a - b))),
            "frac_diff_gt_0.3": float(np.mean(np.abs(a - b) > TOL)),
            "frac_diff_gt_1.0": float(np.mean(np.abs(a - b) > 1.0)),
            "sign_agree_when_both_|d|>0.3": float(np.mean(np.sign(a[decisive]) == np.sign(b[decisive])))
            if decisive.any() else None,
            "pearson": float(np.corrcoef(a, b)[0, 1])}


result["agreement_B_vs_F_same_record"] = agreement(bf)
result["agreement_B_vs_B_same_record"] = agreement(bb)
log("agreement", result["agreement_B_vs_F_same_record"], result["agreement_B_vs_B_same_record"])

# 4. drop-F simulation on the evaluator file
pair = ev["pair_id"].to_pylist()
new_mean = ev["delta_p_mean"].to_numpy().copy()
keep = np.ones(ev.num_rows, bool)
changed = np.zeros(ev.num_rows, bool)
for i in np.flatnonzero(has_f.to_numpy(zero_copy_only=False)):
    m = collapsed_mean([d for d, k in stat.get(pair[i], []) if k != "F"])
    if m is None:
        keep[i] = False
        continue
    changed[i] = True
    new_mean[i] = m
old = ev["delta_p_mean"].to_numpy()
shift = np.abs(new_mean[changed] - old[changed])
part = np.asarray(ev["partition"].to_pylist())
endpoint = np.asarray(ev["endpoint"].to_pylist())
prot = np.asarray(ev["protein_hash"].to_pylist())
sim = {}
for p in PARTS:
    m = part == p
    lost_prot = set(prot[m]) - set(prot[m & keep])
    sim[p] = {
        "rows_before": int(m.sum()), "rows_removed": int((m & ~keep).sum()),
        "rows_relabelled_B_part_only": int((m & changed).sum()),
        "proteins_before": int(len(set(prot[m]))), "proteins_lost": int(len(lost_prot)),
        "removed_by_endpoint": {e: int((m & ~keep & (endpoint == e)).sum()) for e in ENDPOINTS},
        "rows_by_endpoint_before": {e: int((m & (endpoint == e)).sum()) for e in ENDPOINTS},
    }
result["drop_F_simulation"] = sim
result["drop_F_relabel_shift"] = {
    "n": int(changed.sum()), "mean_abs": float(shift.mean()) if shift.size else None,
    "frac_gt_0.3": float((shift > TOL).mean()) if shift.size else None,
    "sign_flip_frac": float((np.sign(new_mean[changed]) != np.sign(old[changed])).mean()) if shift.size else None}

# validation queries before / after (Stage 3 protocol, both directions)
def queries(mask):
    a = np.asarray(ev["smiles_a"].to_pylist(), dtype=object)[mask]
    b = np.asarray(ev["smiles_b"].to_pylist(), dtype=object)[mask]
    pr, en = prot[mask], endpoint[mask]
    q = Counter(zip(np.concatenate([a, b]), np.concatenate([pr, pr]), np.concatenate([en, en])))
    return sum(1 for v in q.values() if v >= 2)


vmask = part == "validation"
result["validation_rankable_queries"] = {"before": queries(vmask), "after_drop_F": queries(vmask & keep)}
log("drop F", json.dumps(sim), result["validation_rankable_queries"])

(OUT / "assay_type_bf_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
log("written", OUT / "assay_type_bf_audit.json")
