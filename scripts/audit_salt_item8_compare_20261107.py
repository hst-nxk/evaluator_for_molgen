"""Item 8, v4 (salt stripped) versus v4 salt-retained: verify the desalting step.

Counts, per observation file and per final export:
  - rows whose parent has fewer fragments than the input (a fragment was dropped)
  - rows still multi-fragment after the step
  - rows containing a metal atom (input structure, parent structure)
  - for v4: rows where what was removed is heavier than what was kept (reuses the
    exception file of the 03/11 audit and re-derives its totals)
Read only; writes one JSON.

Usage: python audit_salt_item8_compare_20261107.py <v4_dir> <salt_retained_dir> <out_json>
  <v4_dir>             data/downloaded_20261103
  <salt_retained_dir>  data/v4_salt_retained_20261104/v4_salt_retained_20261104/data
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_requested_8_10 import METALS  # same metal definition as the 03/11 audit

V4, SR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
RDLogger.DisableLog("rdApp.*")
started = time.time()
PT = Chem.GetPeriodicTable()
SYMBOLS = sorted({PT.GetElementSymbol(n) for n in METALS}, key=lambda s: (-len(s), s))
BRACKET_METAL = re.compile(r"\[(?:\d+)?(?:" + "|".join(SYMBOLS) + r")(?![a-z])")


def log(*a):
    print(f"[{time.time() - started:6.0f}s]", *a, flush=True)


def has_metal(smiles):
    """Regex prefilter on bracket atoms, then RDKit decides."""
    if not smiles or not BRACKET_METAL.search(smiles):
        return False
    m = Chem.MolFromSmiles(smiles)
    return bool(m and any(a.GetAtomicNum() in METALS for a in m.GetAtoms()))


def metal_rows(col):
    """Rows (with multiplicity) whose SMILES carries a metal, via unique values."""
    vc = pc.value_counts(pc.drop_null(col))
    values, counts = vc.field("values").to_pylist(), vc.field("counts").to_pylist()
    rows = uniq = 0
    for v, n in zip(values, counts):
        if has_metal(v):
            rows += n
            uniq += 1
    return rows, uniq


def n_true(mask):
    return int(pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0)


def frags(col):
    return pc.add(pc.count_substring(pc.fill_null(col, ""), "."), 1)


def observation_stats(path):
    t = pq.read_table(path, columns=["molecule_input_canonical_smiles", "molecule_canonical_smiles"])
    a, p = t["molecule_input_canonical_smiles"].combine_chunks(), t["molecule_canonical_smiles"].combine_chunks()
    fa, fp = frags(a), frags(p)
    readable = pc.and_(pc.is_valid(a), pc.is_valid(p))
    r = {
        "rows": len(t),
        "readable_input_and_parent": n_true(readable),
        "input_multifragment": n_true(pc.and_(readable, pc.greater(fa, 1))),
        "parent_multifragment": n_true(pc.and_(readable, pc.greater(fp, 1))),
        "fragment_dropped": n_true(pc.and_(readable, pc.less(fp, fa))),
        "fragment_added": n_true(pc.and_(readable, pc.greater(fp, fa))),
    }
    r["input_metal_rows"], r["input_metal_unique"] = metal_rows(a)
    r["parent_metal_rows"], r["parent_metal_unique"] = metal_rows(p)
    return r


def export_stats(path):
    cols = [c for c in ("smiles_a", "smiles_b", "smiles_weak", "smiles_strong", "constant_smiles") if c in pq.ParquetFile(path).schema_arrow.names]
    t = pq.read_table(path, columns=cols)
    r = {"rows": len(t)}
    multi = metal = None
    for c in cols:
        col = t[c].combine_chunks()
        m = pc.fill_null(pc.match_substring(col, "."), False)
        multi = m if multi is None else pc.or_(multi, m)
    r["multifragment_pair_rows"] = n_true(multi)
    # metal rows: unique-value lookup, then mask by membership
    metal_values = set()
    for c in cols:
        for v in pc.unique(pc.drop_null(t[c].combine_chunks())).to_pylist():
            if has_metal(v):
                metal_values.add(v)
    vs = pa.array(sorted(metal_values), type=pa.string())
    for c in cols:
        m = pc.is_in(t[c].combine_chunks(), value_set=vs)
        metal = m if metal is None else pc.or_(metal, m)
    r["metal_pair_rows"] = n_true(metal)
    r["distinct_metal_smiles"] = len(metal_values)
    return r


report = {"observations": {}, "exports": {}}
for label, base in (("v4_stripped", V4 / "extract_observation_and_ion_split"), ("v4_salt_retained", SR / "observations_salt_retained")):
    for f in sorted(base.glob("*_observations.parquet")):
        report["observations"][f"{label}/{f.name}"] = observation_stats(f)
        log(label, f.name, report["observations"][f"{label}/{f.name}"])

for label, base in (("v4_stripped", V4 / "final_dataset"), ("v4_salt_retained", SR / "dataset_v4_salt_retained")):
    for name in ("evaluator_training", "generator_training", "evidence_records"):
        report["exports"][f"{label}/{name}"] = export_stats(base / f"{name}.parquet")
        log(label, name, report["exports"][f"{label}/{name}"])

# removed-heavier totals of the stripped version, from the 03/11 per-structure audit
exc = json.loads((ROOT / "audit_20261103/salt_mass_exceptions.json").read_text(encoding="utf-8"))
tot = Counter()
for r in exc:
    if r.get("removed_total_heavier"):
        tot["removed_total_heavier_rows"] += r["rows"]
        tot["removed_total_heavier_unique"] += 1
    if r.get("largest_removed_heavier"):
        tot["largest_removed_heavier_rows"] += r["rows"]
        tot["largest_removed_heavier_unique"] += 1
    if r.get("removal_match_unresolved"):
        tot["unresolved_rows"] += r["rows"]
report["v4_stripped_removed_heavier_from_0311_audit"] = dict(tot)

# do those heavier-removed cases reach the v4 evaluator export?
parents = {r["parent"] for r in exc if r.get("largest_removed_heavier") or r.get("removed_total_heavier")}
ev = pq.read_table(V4 / "final_dataset/evaluator_training.parquet", columns=["smiles_a", "smiles_b"])
vs = pa.array(sorted(parents), type=pa.string())
hit = pc.or_(pc.is_in(ev["smiles_a"].combine_chunks(), value_set=vs), pc.is_in(ev["smiles_b"].combine_chunks(), value_set=vs))
report["v4_stripped_heavier_parents_in_evaluator"] = {"flagged_parent_smiles": len(parents), "evaluator_pair_rows": n_true(hit)}

OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
log("done", OUT)
