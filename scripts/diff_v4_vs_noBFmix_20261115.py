"""Row-level diff between the v4 dataset of record (Khai's handoff) and the noBFmix variant.

Checks: is noBFmix a strict subset of v4, do the shared rows stay identical column by column,
and what exactly was removed (partition, endpoint, protein, assay-type labels).
"""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "data/molopt_v4_clean_20261104/final_dataset/evaluator_training.parquet"
NB = ROOT / "data/molopt_v4_noBFmix_20261104/final_dataset/evaluator_training.parquet"
OUT = ROOT / "audit_20261115"
OUT.mkdir(exist_ok=True)

names = pq.ParquetFile(V4).schema_arrow.names
v4_id = pq.read_table(V4, columns=["pair_id"])["pair_id"].combine_chunks()
nb_id = pq.read_table(NB, columns=["pair_id"])["pair_id"].combine_chunks()
res = {"v4_rows": len(v4_id), "nobf_rows": len(nb_id)}

in_nb = pc.is_in(v4_id, value_set=nb_id)
removed = pc.invert(in_nb)
res["removed_rows"] = int(pc.sum(pc.cast(removed, pa.int64())).as_py())
res["added_rows_not_in_v4"] = int(pc.sum(pc.cast(pc.invert(pc.is_in(nb_id, value_set=v4_id)), pa.int64())).as_py())
res["nobf_is_strict_subset"] = res["added_rows_not_in_v4"] == 0
res["same_order_after_filter"] = bool(pc.all(pc.equal(v4_id.filter(in_nb), nb_id)).as_py())
print(res, flush=True)

# shared rows: compare every column
diff_cols = {}
for c in names:
    a = pq.read_table(V4, columns=[c])[c].combine_chunks().filter(in_nb)
    b = pq.read_table(NB, columns=[c])[c].combine_chunks()
    eq = pc.fill_null(pc.equal(a, b), False)
    both_null = pc.and_(pc.is_null(a), pc.is_null(b))
    same = pc.or_(eq, both_null)
    diff_cols[c] = int(pc.sum(pc.cast(pc.invert(same), pa.int64())).as_py())
res["shared_rows_differing_by_column"] = {c: n for c, n in diff_cols.items() if n}
res["all_shared_rows_identical"] = not res["shared_rows_differing_by_column"]
print("shared-row diff:", res["shared_rows_differing_by_column"] or "none", flush=True)

# what was removed
rem = pq.read_table(V4, columns=["pair_id", "partition", "endpoint", "protein_hash", "assay_types", "mixed_assay_type"]).filter(removed)
res["removed_by_partition"] = {k: int(v) for k, v in zip(*[pc.value_counts(rem["partition"]).field(x).to_pylist() for x in ("values", "counts")])}
res["removed_by_endpoint"] = {k: int(v) for k, v in zip(*[pc.value_counts(rem["endpoint"]).field(x).to_pylist() for x in ("values", "counts")])}
vc = pc.value_counts(rem["protein_hash"])
cnt = sorted(vc.field("counts").to_pylist(), reverse=True)
res["removed_proteins"] = len(cnt)
res["removed_top10_protein_rows"] = cnt[:10]
res["removed_median_rows_per_protein"] = cnt[len(cnt) // 2]
res["removed_mixed_assay_type_flag_counts"] = {str(k): int(v) for k, v in zip(*[pc.value_counts(rem["mixed_assay_type"]).field(x).to_pylist() for x in ("values", "counts")])}
(OUT / "v4_vs_noBFmix_diff.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: res[k] for k in res if k.startswith("removed")}, indent=1, ensure_ascii=False))
