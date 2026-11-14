"""After dropping every assay that ChEMBL types F (ChEMBL rows, and BindingDB rows that
carry a ChEMBL assay id), what in the kept evaluator rows could still be functional?

Only documented fields are used, nothing is read from free-text descriptions:
  ChEMBL assays.assay_type, bao_format (labels from ChEMBL's bioassay_ontology table),
  assay_test_type, assay_category; BindingDB Curation/DataSource.
Each kept row is tagged by the assays that remain in it after the F drop.

Usage: python -I audit_residual_functional_20261104.py <dataset_dir> <chembl_sqlite_db> <out_dir>
"""
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA = Path(sys.argv[1])
DB = Path(sys.argv[2])
OUT = Path(sys.argv[3])
PARTS = ("train", "validation", "test")
LINK = re.compile(r"^Ch(?:EMBL|EBML)_(\d+)")
# BAO formats that measure on an isolated molecular target (ChEMBL bioassay_ontology labels)
MOLECULAR_FORMATS = {"single protein format", "protein complex format", "protein format",
                     "cell-free format", "nucleic acid format", "cell membrane format",
                     "subcellular format", "microsome format"}
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


def key(source, entry, assay):
    return f"{source}:{entry or ''}:{assay or ''}"


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
BAO = dict(con.execute("select bao_id, label from bioassay_ontology"))
con.close()

# ChEMBL assay key -> documented attributes; numeric assay_id -> same
c = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_type", "assay_context"])
g = c.group_by(c.column_names).aggregate([])
ATTR, BY_NUMBER = {}, {}
for s, e, a, t, x in zip(*[g[n].to_pylist() for n in g.column_names]):
    ctx = json.loads(x)
    info = {"type": t, "format": BAO.get(ctx.get("bao_format"), ctx.get("bao_format") or "none"),
            "test_type": ctx.get("assay_test_type"), "category": ctx.get("assay_category"),
            "origin": "ChEMBL"}
    ATTR[key(s, e, a)] = info
    BY_NUMBER[ctx.get("assay_id")] = info
del c, g

# BindingDB assay key -> borrowed ChEMBL attributes, or curation source when no link
b = ds.dataset(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_name", "source_curation"])
g = b.group_by(b.column_names).aggregate([])
for s, e, a, n, cur in zip(*[g[x].to_pylist() for x in g.column_names]):
    m = LINK.search(n) if n else None
    linked = BY_NUMBER.get(int(m.group(1))) if m else None
    if linked:
        ATTR[key(s, e, a)] = {**linked, "origin": "BindingDB via ChEMBL id"}
    else:
        ATTR[key(s, e, a)] = {"type": None, "format": None, "test_type": None, "category": None,
                              "origin": f"BindingDB {cur}" + (" (ChEMBL id, assay not extracted)" if m else "")}
del b, g
log("assay attributes", len(ATTR))

ev = ds.dataset(DATA / "final_dataset" / "evaluator_training.parquet", format="parquet").to_table(
    columns=["pair_id", "partition", "endpoint"])
where = {p: (pt, e) for p, pt, e in zip(*[ev[x].to_pylist() for x in ev.column_names])}
log("evaluator rows", len(where))


def row_tag(assays):
    """Single tag per kept row, most worrying first (documented fields only)."""
    kinds = [ATTR.get(k, {"type": None, "origin": "missing"}) for k in assays]
    if any(i["type"] in ("A", "T", "P", "U") for i in kinds):
        return "2. ChEMBL type A/T/P/U (not binding, not functional)"
    if any(i["type"] == "B" and i.get("test_type") == "In vivo" for i in kinds):
        return "3. type B but assay_test_type In vivo"
    if any(i["type"] == "B" and i["format"] not in MOLECULAR_FORMATS for i in kinds):
        fmts = sorted({i["format"] for i in kinds if i["type"] == "B" and i["format"] not in MOLECULAR_FORMATS})
        return "4. type B but BAO format " + " / ".join(fmts)
    if any(i["type"] is None for i in kinds):
        unknown = sorted({i["origin"] for i in kinds if i["type"] is None})
        return "5. no documented type: " + " + ".join(unknown)
    return "1. type B, molecular-target BAO format"


tags = {pt: Counter() for pt in PARTS}
tags_ec50 = {pt: Counter() for pt in PARTS}
categories = Counter()
for path in sorted((DATA / "filter_merge_mmp" / "records").glob("*/*.parquet")):
    t = pq.read_table(path, columns=["pair_id", "assay_keys"])
    for p, assays in zip(t["pair_id"].to_pylist(), t["assay_keys"].to_pylist()):
        if p not in where:
            continue
        kept = [k for k in assays if ATTR.get(k, {}).get("type") != "F"]
        if not kept:
            continue            # dropped by the F filter
        pt, e = where[p]
        tag = row_tag(kept)
        tags[pt][tag] += 1
        if e == "EC50":
            tags_ec50[pt][tag] += 1
        for k in kept:
            cat = ATTR.get(k, {}).get("category")
            if cat and ATTR[k]["type"] == "B":
                categories[(pt, cat)] += 1
log("tagged")

result = {
    "molecular_target_formats": sorted(MOLECULAR_FORMATS),
    "kept_rows_by_tag": {pt: dict(sorted(v.items())) for pt, v in tags.items()},
    "kept_EC50_rows_by_tag": {pt: dict(sorted(v.items())) for pt, v in tags_ec50.items()},
    "type_B_assay_category_row_mentions": {f"{a}|{b}": n for (a, b), n in sorted(categories.items())},
}
(OUT / "residual_functional.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=1, ensure_ascii=False))
