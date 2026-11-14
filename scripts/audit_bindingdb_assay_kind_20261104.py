"""BindingDB has no binding/functional field. What could stand in for one?

Counts, on BindingDB observations that survive dedup and the v4 cleaning rules:
  - source_curation x endpoint (PDSP Ki is radioligand binding by construction)
  - presence of assay_description text and of a PubChem AID
and how many survivors share a document (DOI or PMID) with ChEMBL, where a ChEMBL
assay_type could in principle be borrowed.

Usage: python -I audit_bindingdb_assay_kind_20261104.py <dataset_dir> <out_dir>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])

bdb = ds.dataset(DATA / "deduplicate" / "bindingdb202610_observations_dedup.parquet", format="parquet")
t = bdb.to_table(columns=["observation_id", "source_curation", "endpoint", "assay_name",
                          "assay_description", "assay_context", "document_doi", "document_pmid",
                          "dedup_keep"],
                 filter=ds.field("dedup_keep") == True)  # noqa: E712
excluded = pq.read_table(DATA / "filter_merge_mmp" / "excluded_observations.parquet",
                         columns=["observation_id"])["observation_id"].combine_chunks()
t = t.filter(pc.invert(pc.is_in(t["observation_id"], value_set=excluded)))
print("BindingDB rows kept by dedup and cleaning:", t.num_rows, flush=True)

src = t["source_curation"].to_pylist()
ep = t["endpoint"].to_pylist()
desc = t["assay_description"].to_pylist()
ctx = t["assay_context"].to_pylist()
doi = t["document_doi"].to_pylist()
pmid = t["document_pmid"].to_pylist()

by_src = defaultdict(Counter)
has_aid = Counter()
has_desc = Counter()
for s, e, d, c in zip(src, ep, desc, ctx):
    by_src[s][e] += 1
    has_aid[bool(c and "pubchem_aid" in c)] += 1
    has_desc[bool(d)] += 1

ch = ds.dataset(DATA / "deduplicate" / "chembl37_observations_dedup.parquet", format="parquet")
cdocs = ch.to_table(columns=["document_doi", "document_pmid"])
chembl_doi = set(x.lower() for x in pc.unique(cdocs["document_doi"].combine_chunks()).to_pylist() if x)
chembl_pmid = set(x for x in pc.unique(cdocs["document_pmid"].combine_chunks()).to_pylist() if x)
shared = Counter()
for d, p in zip(doi, pmid):
    shared[bool((d and d.lower() in chembl_doi) or (p and p in chembl_pmid))] += 1

result = {
    "rows": t.num_rows,
    "by_source_curation_endpoint": {k: dict(v) for k, v in sorted(by_src.items(), key=lambda kv: -sum(kv[1].values()))},
    "has_assay_description": {str(k): v for k, v in has_desc.items()},
    "has_pubchem_aid": {str(k): v for k, v in has_aid.items()},
    "shares_document_with_chembl": {str(k): v for k, v in shared.items()},
}
(OUT / "bindingdb_assay_kind.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
print(json.dumps(result, indent=1, ensure_ascii=False))
