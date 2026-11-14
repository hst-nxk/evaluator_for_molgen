"""Append explicitly defined flags to copies of all three exports; originals read-only."""
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'data/downloaded_20261103'
OUT=BASE/'final_dataset_flagged';OUT.mkdir(exist_ok=True)
report={}
for name in ['evidence_records','evaluator_training','generator_training']:
    src=BASE/'final_dataset'/(name+'.parquet');dst=OUT/src.name;pf=pq.ParquetFile(src)
    schema=pf.schema_arrow.append(pa.field('mixed_binding_functional',pa.bool_())).append(pa.field('assay_types_missing',pa.bool_()))
    n=positive=missing=0
    with pq.ParquetWriter(dst,schema,compression='zstd') as writer:
        for batch in pf.iter_batches(batch_size=65536):
            table=pa.Table.from_batches([batch]);types=table['assay_types']
            bf=pc.and_(pc.fill_null(pc.match_substring_regex(types,'(^|;)B(;|$)'),False),pc.fill_null(pc.match_substring_regex(types,'(^|;)F(;|$)'),False))
            miss=pc.is_null(types);table=table.append_column('mixed_binding_functional',bf).append_column('assay_types_missing',miss)
            writer.write_table(table);n+=len(table);positive+=pc.sum(bf).as_py();missing+=pc.sum(miss).as_py()
    assert pq.ParquetFile(dst).metadata.num_rows==pf.metadata.num_rows==n
    report[name]={'file':str(dst),'rows':n,'mixed_binding_functional':positive,'assay_types_missing':missing}
    print(name,n,positive,flush=True)
(OUT/'flag_manifest.json').write_text(json.dumps({'originals_unchanged':True,'flag_definition':'observed assay_types contains both B and F; false does not prove absence when types unknown','files':report},indent=2),encoding='utf-8')
