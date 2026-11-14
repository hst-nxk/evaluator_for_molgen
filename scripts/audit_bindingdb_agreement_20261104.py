"""Does untyped BindingDB data behave like binding or like cell-based measurements?

Inside one evaluator record (same MMP pair, protein, endpoint) several assays may each give
a delta_p. Take one assay from group X and one from group Y, from DIFFERENT documents, and
compare their delta_p. "Same publication" = the two assays share a DOI, PMID or patent
number (document ids of the two databases are different namespaces, so they cannot be used
across sources); comparisons are reported separately for same and different publications.
Groups come from documented fields only:
  G1  ChEMBL type B, BAO protein-level format         (binding, both fields agree)
  G2  ChEMBL type B, BAO 'assay format' (unspecified)
  G3  ChEMBL type B, BAO cell / tissue / organism based
  F   ChEMBL type F (dropped from the dataset; kept here as the functional reference)
  BDB_patent / BDB_other  BindingDB with no ChEMBL assay link, by Curation/DataSource
BindingDB rows that carry a ChEMBL assay id take that assay's group.
Train and validation only; test is not read.

Usage: python -I audit_bindingdb_agreement_20261104.py <dataset_dir> <chembl_sqlite_db> <out_dir>
"""
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA, DB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LINK = re.compile(r"^Ch(?:EMBL|EBML)_(\d+)")
MOLECULAR = {"single protein format", "protein complex format", "protein format", "cell-free format",
             "nucleic acid format", "cell membrane format", "subcellular format", "microsome format"}
CELLULAR = {"cell-based format", "tissue-based format", "organism-based format"}
GROUPS = ("G1", "G2", "G3", "F", "BDB_patent", "BDB_other")
TOL = 0.3
rng = np.random.default_rng(20261104)
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


def key(source, entry, assay):
    return f"{source}:{entry or ''}:{assay or ''}"


def chembl_group(kind, fmt):
    if kind == "F":
        return "F"
    if kind != "B":
        return None                      # A/T/P/U: not compared
    if fmt in MOLECULAR:
        return "G1"
    if fmt in CELLULAR:
        return "G3"
    return "G2" if fmt == "assay format" else None


def publication(doi, pmid, patent):
    """Identifiers of the publication behind an assay, comparable across the two sources."""
    ids = set()
    if doi:
        ids.add("doi:" + doi.strip().lower())
    if pmid:
        ids.add("pmid:" + str(pmid).strip())
    if patent:
        # BindingDB 'US9447087' vs ChEMBL 'US-8889696-B2': keep country + number, drop kind code
        m = re.match(r"\s*([A-Z]{2})[\s-]*0*(\d+)", patent.upper())
        ids.add("patent:" + (m.group(1) + m.group(2) if m else patent.strip().upper()))
    return frozenset(ids)


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
BAO = dict(con.execute("select bao_id, label from bioassay_ontology"))
con.close()

GROUP, DOC, BY_NUMBER = {}, {}, {}
c = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_type", "assay_context",
             "document_doi", "document_pmid", "document_patent_number"])
g = c.group_by(c.column_names).aggregate([])
for s, e, a, t, x, doi, pmid, pat in zip(*[g[n].to_pylist() for n in g.column_names]):
    d = publication(doi, pmid, pat)
    ctx = json.loads(x)
    grp = chembl_group(t, BAO.get(ctx.get("bao_format")))
    k = key(s, e, a)
    GROUP[k] = grp
    DOC[k] = DOC.get(k, frozenset()) | d
    BY_NUMBER[ctx.get("assay_id")] = grp
del c, g
b = ds.dataset(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_name", "source_curation",
             "document_doi", "document_pmid", "document_patent_number"])
g = b.group_by(b.column_names).aggregate([])
for s, e, a, n, cur, doi, pmid, pat in zip(*[g[x].to_pylist() for x in g.column_names]):
    d = publication(doi, pmid, pat)
    k = key(s, e, a)
    m = LINK.search(n) if n else None
    if m and int(m.group(1)) in BY_NUMBER:
        GROUP[k] = BY_NUMBER[int(m.group(1))]
    elif m:
        GROUP[k] = None                  # ChEMBL id but assay not extracted: unknown
    else:
        GROUP[k] = "BDB_patent" if cur in ("US Patent", "WIPO") else "BDB_other"
    DOC[k] = DOC.get(k, frozenset()) | d
del b, g
log("assays grouped", len(GROUP))

ev = ds.dataset(DATA / "final_dataset" / "evaluator_training.parquet", format="parquet").to_table(
    columns=["pair_id", "partition", "endpoint"])
where = {p: e for p, pt, e in zip(*[ev[x].to_pylist() for x in ev.column_names]) if pt != "test"}
del ev
log("train+validation records", len(where))

PAIRS = list(combinations_with_replacement(GROUPS, 2))
COPY = 0.05                 # |delta_p difference| treated as the same measurement
copies = Counter()
SPLITS = ("same_pub", "diff_pub")
diffs = {(x, y, e, s): [] for x, y in PAIRS for e in ("all", "IC50", "EC50", "Ki") for s in SPLITS}
for path in sorted((DATA / "filter_merge_mmp" / "assay_deltas").glob("*/*.parquet")):
    t = pq.read_table(path, columns=["pair_id", "assay_key", "delta_p"])
    per = defaultdict(list)
    for p, k, d in zip(t["pair_id"].to_pylist(), t["assay_key"].to_pylist(), t["delta_p"].to_numpy()):
        if p in where:
            per[p].append((GROUP.get(k), DOC.get(k), float(d)))
    for p, items in per.items():
        if len(items) < 2:
            continue
        endpoint = where[p]
        by = defaultdict(list)
        for grp, doc, d in items:
            doc = doc or frozenset()
            if grp:
                by[grp].append((doc, d))
        # near-copy linkage: an untyped BindingDB delta that equals a ChEMBL delta of the
        # same record within COPY log is the same measurement reported twice; tally the
        # ChEMBL group it copies (closest match), or 'no near copy'
        chembl_items = [(grp, d) for grp, _, d in items if grp in ("G1", "G2", "G3", "F")]
        for grp, _, d in items:
            if grp not in ("BDB_patent", "BDB_other") or not chembl_items:
                continue
            best = min(chembl_items, key=lambda item: abs(item[1] - d))
            match = best[0] if abs(best[1] - d) <= COPY else "no near copy"
            copies[(grp, endpoint, match)] += 1
            copies[(grp, "all", match)] += 1
        for x, y in PAIRS:
            if x not in by or y not in by:
                continue
            # one comparison per record, group pair and split; random choice, not key order
            left = by[x][rng.integers(len(by[x]))]
            others = [r for r in by[y] if r is not left]
            for split in SPLITS:
                same = split == "same_pub"
                right = [r for r in others if bool(r[0] & left[0]) == same]
                if not right:
                    continue
                r = right[rng.integers(len(right))]
                for e in ("all", endpoint):
                    if (x, y, e, split) in diffs:
                        diffs[(x, y, e, split)].append((left[1], r[1]))
    log(path.parent.name, path.name)


def summary(rows):
    if len(rows) < 30:
        return {"n": len(rows)}
    a = np.array(rows)
    d = np.abs(a[:, 0] - a[:, 1])
    decisive = (np.abs(a[:, 0]) > TOL) & (np.abs(a[:, 1]) > TOL)
    boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(500)]
    return {"n": int(d.size), "mean_abs_diff": float(d.mean()),
            "frac_diff_le_0.05": float((d <= 0.05).mean()),
            "mean_abs_diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "frac_diff_gt_0.3": float((d > TOL).mean()),
            "sign_agree_both_gt_0.3": float((np.sign(a[decisive, 0]) == np.sign(a[decisive, 1])).mean())
            if decisive.sum() >= 30 else None,
            "pearson": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])}


result = {f"{x}~{y}|{e}|{s}": summary(v) for (x, y, e, s), v in diffs.items()}
result["near_copy_linkage"] = {f"{g}|{e}|{m}": n for (g, e, m), n in sorted(copies.items())}
for k, n in result["near_copy_linkage"].items():
    if "|all|" in k:
        print("copy", k, n)
(OUT / "bindingdb_agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
for k, v in result.items():
    if "|all|" in k and v.get("n", 0) >= 30:
        print(k, json.dumps(v))
