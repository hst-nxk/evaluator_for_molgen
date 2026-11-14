"""How many BindingDB rows are binding, functional, or still unclassified?

Each BindingDB assay gets at most one label, from the first route that applies:
  route 1  ChEMBL assay id written by BindingDB in ASSAY_NAME ('ChEMBL_<assay_id> ...'):
           take that ChEMBL assay's ASSAY_TYPE and BAO format            (source-documented)
  route 2  value match: inside the same record (pair, protein, endpoint) the BindingDB
           delta_p equals a ChEMBL assay's delta_p within COPY log, i.e. the same
           measurement published twice. An assay is labelled only if it has >= MIN_VOTES
           such matches and >= MIN_SHARE of them point to one ChEMBL group  (data linkage)
  else     unclassified
ChEMBL groups: B protein-level BAO format / B cell-tissue-organism format / B format not
specified / F / other (A,T,P,U). BAO labels come from ChEMBL 37 bioassay_ontology.

Counted at two levels: BindingDB observations in scope (dedup kept, not excluded by the
v4 cleaning rules), and evaluator rows that contain at least one BindingDB assay.
Linkage uses every partition (curation step, like dedup); test rows are only counted.

Usage: python -I audit_bindingdb_classified_20261104.py <dataset_dir> <chembl_sqlite_db> <out_dir>
"""
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA, DB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LINK = re.compile(r"^Ch(?:EMBL|EBML)_(\d+)")
MOLECULAR = {"single protein format", "protein complex format", "protein format", "cell-free format",
             "nucleic acid format", "cell membrane format", "subcellular format", "microsome format"}
CELLULAR = {"cell-based format", "tissue-based format", "organism-based format"}
COPY, MIN_VOTES, MIN_SHARE = 0.05, 3, 0.8
PARTS = ("train", "validation", "test")
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


def key(source, entry, assay):
    return f"{source}:{entry or ''}:{assay or ''}"


def chembl_group(kind, fmt):
    if kind == "F":
        return "F functional"
    if kind == "B":
        if fmt in MOLECULAR:
            return "B binding, protein-level format"
        if fmt in CELLULAR:
            return "B binding, cell/tissue/organism format"
        return "B binding, format not specified"
    return "other (A/T/P/U)"


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
BAO = dict(con.execute("select bao_id, label from bioassay_ontology"))
con.close()

CHEMBL, BY_NUMBER = {}, {}
c = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet").to_table(
    columns=["source_database", "assay_entry_id", "assay_native_id", "assay_type", "assay_context"])
g = c.group_by(c.column_names).aggregate([])
for s, e, a, t, x in zip(*[g[n].to_pylist() for n in g.column_names]):
    ctx = json.loads(x)
    grp = chembl_group(t, BAO.get(ctx.get("bao_format")))
    CHEMBL[key(s, e, a)] = grp
    BY_NUMBER[ctx.get("assay_id")] = grp
del c, g

# BindingDB observations in scope
b = ds.dataset(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet", format="parquet").to_table(
    columns=["observation_id", "source_database", "assay_entry_id", "assay_native_id", "assay_name",
             "source_curation", "endpoint"], filter=ds.field("dedup_keep") == True)  # noqa: E712
excluded = pq.read_table(DATA / "filter_merge_mmp" / "excluded_observations.parquet",
                         columns=["observation_id"])["observation_id"].combine_chunks()
b = b.filter(pc.invert(pc.is_in(b["observation_id"], value_set=excluded)))
log("BindingDB observations in scope", b.num_rows)
ROUTE1, CURATION = {}, {}
for s, e, a, n, cur in zip(*[b[x].to_pylist() for x in
                             ("source_database", "assay_entry_id", "assay_native_id", "assay_name",
                              "source_curation")]):
    k = key(s, e, a)
    CURATION[k] = cur
    m = LINK.search(n) if n else None
    if m and int(m.group(1)) in BY_NUMBER:
        ROUTE1[k] = BY_NUMBER[int(m.group(1))]

# route 2 votes from assay-level deltas
votes = defaultdict(Counter)
cooccur = Counter()                      # BindingDB assay -> records shared with a ChEMBL assay
record_assays = {}                       # evaluator pair_id -> BindingDB assay keys
in_scope = set()                         # BindingDB observation ids behind an evaluator row
ev = ds.dataset(DATA / "final_dataset" / "evaluator_training.parquet", format="parquet").to_table(
    columns=["pair_id", "partition"])
part_of = dict(zip(ev["pair_id"].to_pylist(), ev["partition"].to_pylist()))
del ev
for path in sorted((DATA / "filter_merge_mmp" / "assay_deltas").glob("*/*.parquet")):
    t = pq.read_table(path, columns=["pair_id", "assay_key", "delta_p", "observation_ids_a",
                                     "observation_ids_b"])
    per = defaultdict(list)
    for p, k, d, oa, ob in zip(t["pair_id"].to_pylist(), t["assay_key"].to_pylist(),
                               t["delta_p"].to_numpy(), t["observation_ids_a"].to_pylist(),
                               t["observation_ids_b"].to_pylist()):
        per[p].append((k, float(d)))
        if p in part_of and k in CURATION:
            in_scope.update(oa)
            in_scope.update(ob)
    for p, items in per.items():
        bdb = [(k, d) for k, d in items if k in CURATION]
        if p in part_of and bdb:
            record_assays[p] = {k for k, _ in bdb}
        chembl = [(CHEMBL[k], d) for k, d in items if k in CHEMBL]
        if not chembl:
            continue
        for k, d in bdb:
            if k in ROUTE1:
                continue
            cooccur[k] += 1
            grp, value = min(chembl, key=lambda item: abs(item[1] - d))
            if abs(value - d) <= COPY:
                votes[k][grp] += 1
    log(path.parent.name, path.name)

# scope = observations that actually enter an evaluator row (the stage2b.filter and the
# cleaning rules both already applied); the earlier 'not excluded' scope also counted rows
# that stage2b.filter drops, e.g. 57,008 rows without any BindingDB assay link
b = b.filter(pc.is_in(b["observation_id"], value_set=pa.array(sorted(in_scope), pa.string())))
log("BindingDB observations behind evaluator rows", b.num_rows)
CURATION = {key(s, e, a): cur for s, e, a, cur in zip(*[b[x].to_pylist() for x in
            ("source_database", "assay_entry_id", "assay_native_id", "source_curation")])}

LABEL, REASON = {}, {}
for k in CURATION:
    if k in ROUTE1:
        LABEL[k] = ("route 1: ChEMBL id in ASSAY_NAME", ROUTE1[k])
        continue
    v = votes.get(k)
    if v and sum(v.values()) >= MIN_VOTES:
        grp, n = v.most_common(1)[0]
        if n / sum(v.values()) >= MIN_SHARE:
            LABEL[k] = ("route 2: value match", grp)
            continue
        LABEL[k] = ("route 2: value match", "ambiguous (matches disagree)")
        continue
    if not cooccur[k]:
        why = "never shares a record with a ChEMBL assay"
    elif not v:
        why = "shares records with ChEMBL, but no value matches"
    else:
        why = f"fewer than {MIN_VOTES} value matches"
    LABEL[k] = ("unclassified", "unclassified")
    REASON[k] = why


def bucket(grp):
    if grp.startswith("B "):
        return "binding"
    if grp.startswith("F "):
        return "functional"
    if grp.startswith("other"):
        return "other (A/T/P/U)"
    return grp


obs = Counter()
for s, e, a, ep, cur in zip(*[b[x].to_pylist() for x in
                              ("source_database", "assay_entry_id", "assay_native_id", "endpoint",
                               "source_curation")]):
    route, grp = LABEL[key(s, e, a)]
    curation = "patent" if cur in ("US Patent", "WIPO") else "non-patent"
    for scope in ("all", ep):
        obs[(scope, bucket(grp), route, grp)] += 1
        obs[(scope + "|" + curation, bucket(grp), route, grp)] += 1

unclassified = Counter()
for s, e, a, ep, cur in zip(*[b[x].to_pylist() for x in
                              ("source_database", "assay_entry_id", "assay_native_id", "endpoint",
                               "source_curation")]):
    k = key(s, e, a)
    if k in REASON:
        unclassified[(cur, REASON[k])] += 1
        unclassified[(cur, "total")] += 1
        unclassified[("all", REASON[k])] += 1
assays_unclassified = Counter(CURATION[k] for k in REASON)

rows = Counter()
for p, keys in record_assays.items():
    kinds = {bucket(LABEL[k][1]) for k in keys if k in LABEL}
    if "functional" in kinds:
        tag = "has functional BindingDB assay"
    elif kinds == {"binding"}:
        tag = "all BindingDB assays binding"
    elif "unclassified" in kinds or "ambiguous (matches disagree)" in kinds:
        tag = "has unclassified BindingDB assay"
    else:
        tag = "other (A/T/P/U) / mixed"
    rows[(part_of[p], tag)] += 1

result = {
    "settings": {"copy_log": COPY, "min_votes": MIN_VOTES, "min_share": MIN_SHARE},
    "bindingdb_assays_by_route": dict(Counter(r for r, _ in LABEL.values())),
    "observations": {"|".join(k): n for k, n in sorted(obs.items())},
    "evaluator_rows_with_bindingdb": {f"{a}|{b}": n for (a, b), n in sorted(rows.items())},
    "unclassified_observations_by_source_and_reason": {f"{a}|{b}": n for (a, b), n in sorted(unclassified.items())},
    "unclassified_assays_by_source": dict(assays_unclassified.most_common()),
}
n_obs = Counter(key(s, e, a) for s, e, a in zip(*[b[x].to_pylist() for x in
                                                  ("source_database", "assay_entry_id", "assay_native_id")]))
assay_keys = sorted(LABEL)
pq.write_table(pa.table({
    "assay_key": assay_keys,
    "route": [LABEL[k][0] for k in assay_keys],
    "group": [LABEL[k][1] for k in assay_keys],
    "unclassified_reason": [REASON.get(k) for k in assay_keys],
    "source_curation": [CURATION[k] for k in assay_keys],
    "n_observations": [n_obs[k] for k in assay_keys],
}), OUT / "bindingdb_assay_labels.parquet")
(OUT / "bindingdb_classified.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
log("written")
