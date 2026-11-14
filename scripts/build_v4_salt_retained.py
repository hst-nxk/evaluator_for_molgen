"""Explicit v4 ablation: retain every standardized component instead of salt stripping.
Recompute only structure-derived columns from saved ORIGINAL source SMILES, then rerun
author dedup/curation/MMP/aggregate/export/verify. No raw activity/protein fields change.
Intended Colab high-RAM CPU. Existing v4 protein partitions are locked for comparison.
"""
import os,sys,json,time,hashlib,shutil,subprocess,datetime
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(os.environ.get('MOLOPT_SALT_ROOT','/content/molopt_v4_salt_retained'))
PAYLOAD=Path(os.environ.get('MOLOPT_SALT_PAYLOAD',str(ROOT/'ablation_payload')))
BACKUP=Path(os.environ.get('MOLOPT_SALT_BACKUP','/content/drive/MyDrive/molopt_out/v4_salt_retained_20261104'))
STATE={};START=time.time()
FIELDS={'canonical_smiles':'molecule_canonical_smiles','inchikey':'molecule_inchikey','input_canonical_smiles':'molecule_input_canonical_smiles','input_inchikey':'molecule_input_inchikey','cx_extension':'molecule_cx_extension','stereo_groups':'molecule_stereo_groups','stereo_centers':'molecule_stereo_centers','unspecified_stereo_centers':'molecule_unspecified_stereo_centers','parent_fragments':'molecule_parent_fragments','parent_excluded':'molecule_parent_excluded','status':'structure_status','reason':'structure_reason'}

def retaining_parent(molecule):
    from rdkit import Chem
    from chembl_structure_pipeline import standardizer
    from chembl_structure_pipeline.exclude_flag import exclude_flag
    m=standardizer.standardize_mol(Chem.Mol(molecule))
    m=standardizer.get_isotope_parent_mol(m)
    excluded=exclude_flag(m,includeRDKitSanitization=False)
    before=len(Chem.GetMolFrags(m))
    result=standardizer.uncharge_mol(m)
    assert len(Chem.GetMolFrags(result))==before,'Unexpected component loss during uncharge'
    return result,excluded

def worker_init():
    sys.path.insert(0,str(ROOT/'src'))
    from molopt.observations import normalize
    normalize.chembl_parent=retaining_parent

def structure_chunk(sources):
    from molopt.observations import normalize
    rows=[]
    for source in sources:
        s=normalize.describe_structure(source)
        row={'source_smiles':source,**{new:getattr(s,old) for old,new in FIELDS.items()}}
        reason=row['structure_reason']
        row['structure_reason']=('ablation_keep_all_fragments'+(';' +reason if reason else ''))
        rows.append(row)
    return rows

def status(stage,**extra):
    STATE.update(stage=stage,updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=round(time.time()-START,1),**extra)
    (ROOT/'runs/salt_retained').mkdir(parents=True,exist_ok=True)
    text=json.dumps(STATE,indent=2,ensure_ascii=False)
    (ROOT/'runs/salt_retained/status.json').write_text(text,encoding='utf-8')
    if BACKUP.parent.exists():
        BACKUP.mkdir(parents=True,exist_ok=True);(BACKUP/'status.json').write_text(text,encoding='utf-8')
    print('STATUS',stage,json.dumps(extra),flush=True)

def run(stage,args,allowed=(0,)):
    status(stage)
    path=ROOT/'runs/salt_retained'/(stage+'.log')
    with path.open('a',encoding='utf-8') as log:
        log.write('COMMAND '+repr(args)+'\n');log.flush()
        p=subprocess.Popen([str(x) for x in args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        code=p.wait()
    if code not in allowed:raise RuntimeError(f'{stage} failed, exit={code}; {path}')

def backup(relative):
    if not BACKUP.parent.exists():raise RuntimeError('Drive not mounted')
    target=BACKUP/relative;source=ROOT/relative
    if source.is_dir():shutil.copytree(source,target,dirs_exist_ok=True)
    else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copytree(ROOT/'runs',BACKUP/'runs',dirs_exist_ok=True)

def smoke():
    worker_init()
    from molopt.observations import normalize
    from molopt.dataset import mmp
    from rdkit import Chem
    import dataclasses
    c=json.loads((ROOT/'config.json').read_text(encoding='utf-8'));settings=mmp.FragmentSettings(**{f.name:c['stage2b']['mmp'][f.name] for f in dataclasses.fields(mmp.FragmentSettings)})
    cases=['CN(C)CCc1ccc(Cl)cc1','C[NH+](C)CCc1ccc(Cl)cc1.[Cl-]','C[S+](C)C.[I-]']
    rows=[]
    for source in cases:
        result=normalize.describe_structure(source);assert result.status=='resolved'
        assert len(Chem.GetMolFrags(Chem.MolFromSmiles(result.canonical_smiles)))==len(Chem.GetMolFrags(Chem.MolFromSmiles(source)))
        rows.append({'input':source,'output':result.canonical_smiles,'key':result.inchikey,'cuts':len(mmp.fragment_molecule(result.canonical_smiles,settings).cuts)})
    assert rows[0]['key']!=rows[1]['key'];assert '[I-]' in rows[2]['output']
    (ROOT/'runs/salt_retained').mkdir(parents=True,exist_ok=True)
    (ROOT/'runs/salt_retained/smoke.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print('SMOKE PASS',json.dumps(rows),flush=True)

def prepare_config():
    c=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    f=json.loads((PAYLOAD/'baseline_full_manifest.json').read_text(encoding='utf-8'));e=json.loads((PAYLOAD/'baseline_export_manifest.json').read_text(encoding='utf-8'))
    for key,value in f['parameters'].items():c['stage2b'][key]=value
    c['stage2b']['export']=e['parameters']
    c['structure']['ablation']={'fragment_stripping':False,'other_standardization':'ChEMBL standardize + isotope parent + uncharge, retained','note':'custom ablation wrapper, not author-default v4 preprocessing'}
    c['paths']['observations_dir']='data/observations_salt_retained'
    c['stage2a']['paths']['dedup_dir']='data/dedup_salt_retained'
    c['stage2b']['inputs']={'chembl':'data/dedup_salt_retained/chembl37_observations_dedup.parquet','bindingdb':'data/dedup_salt_retained/bindingdb202610_observations_dedup.parquet'}
    c['stage2b']['paths'].update(dataset_dir='data/dataset_v4_salt_retained',full_dir='data/dataset_v4_salt_retained/full',version_note='Explicit ablation: no fragment stripping; all other v4 parameters restored from downloaded manifests; original v4 split locked')
    (ROOT/'config.json').write_text(json.dumps(c,indent=2,ensure_ascii=False),encoding='utf-8');backup('config.json')
    return c

def rebuild_structures(source_dir):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sys.path.insert(0,str(ROOT/'src'))
    from molopt.observations import normalize
    remote_files=sorted(source_dir.glob('*observations.parquet'));assert len(remote_files)==2
    expected=json.loads((PAYLOAD/'input_sizes.json').read_text())
    local=ROOT/'source_observations';local.mkdir(exist_ok=True);files=[]
    for source in remote_files:
        f=local/source.name
        if not f.exists():
            status('copy_source',source=source.name);partial=f.with_suffix('.partial');shutil.copy2(source,partial);partial.replace(f)
        assert f.stat().st_size==expected[f.name]['bytes'],(f,'size mismatch')
        with f.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==expected[f.name]['sha256'],(f,'SHA256 mismatch')
        assert pq.ParquetFile(f).metadata.num_rows==expected[f.name]['rows'],(f,'row mismatch');files.append(f)
    lookup_path=ROOT/'data/structure_cache_salt_retained.parquet'
    if not lookup_path.exists():
        sources=pc.unique(pa.chunked_array([p.chunk(0) if p.num_chunks==1 else p.combine_chunks() for p in [pq.read_table(f,columns=['molecule_source_smiles'])['molecule_source_smiles'] for f in files]])).to_pylist()
        status('structure_normalization',unique_sources=len(sources),processed=0)
        schema=pq.ParquetFile(files[0]).schema_arrow
        new_schema=pa.schema([pa.field('source_smiles',pa.string())]+[schema.field(x) for x in FIELDS.values()])
        lookup_path.parent.mkdir(parents=True,exist_ok=True);partial=lookup_path.with_suffix('.partial');sharddir=BACKUP/'structure_shards';sharddir.mkdir(parents=True,exist_ok=True)
        chunk_size=5000;pending=[i for i in range(0,len(sources),chunk_size) if not (sharddir/f'{i:09d}.parquet').exists()];processed=len(sources)-sum(min(chunk_size,len(sources)-i) for i in pending)
        with ProcessPoolExecutor(max_workers=min(8,os.cpu_count() or 1),initializer=worker_init) as pool:
            for i,rows in zip(pending,pool.map(structure_chunk,(sources[i:i+chunk_size] for i in pending),chunksize=1)):
                shard=sharddir/f'{i:09d}.parquet';part=shard.with_suffix('.partial');pq.write_table(pa.Table.from_pylist(rows,schema=new_schema),part,compression='zstd');part.replace(shard);processed+=len(rows)
                status('structure_normalization',unique_sources=len(sources),processed=processed)
        with pq.ParquetWriter(partial,new_schema,compression='zstd') as writer:
            for i in range(0,len(sources),chunk_size):writer.write_table(pq.read_table(sharddir/f'{i:09d}.parquet'))
        partial.replace(lookup_path)
    cache=pq.read_table(lookup_path);keys=cache['source_smiles'];dst=ROOT/'data/observations_salt_retained';dst.mkdir(exist_ok=True)
    for source in files:
        dest=dst/source.name
        if dest.exists():continue
        status('write_observations',source=source.name);pf=pq.ParquetFile(source);partial=dest.with_suffix('.partial');total=0
        with pq.ParquetWriter(partial,pf.schema_arrow,compression='zstd') as writer:
            for batch in pf.iter_batches(batch_size=50000):
                original=pa.Table.from_batches([batch]);indices=pc.index_in(original['molecule_source_smiles'],value_set=keys);assert indices.null_count==0
                table=original
                for column in FIELDS.values():table=table.set_column(table.schema.get_field_index(column),column,pc.take(cache[column],indices))
                checks=[normalize.compare_source_inchikey(a,b) for a,b in zip(original['molecule_source_inchikey'].to_pylist(),table['molecule_input_inchikey'].to_pylist())]
                table=table.set_column(table.schema.get_field_index('molecule_source_inchikey_check'),'molecule_source_inchikey_check',pa.array(checks,pa.string()))
                # Verify all nonstructure columns, including activity, protein and observation IDs.
                unchanged=[x for x in original.column_names if x not in FIELDS.values() and x!='molecule_source_inchikey_check']
                assert original.select(unchanged).equals(table.select(unchanged))
                writer.write_table(table);total+=len(table)
        assert total==pf.metadata.num_rows;partial.replace(dest)
    backup('data/observations_salt_retained')

def fixed_split(c):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from molopt.dataset import pipeline,split
    full= pipeline.Layout(ROOT/c['stage2b']['paths']['full_dir'],c['stage2b']['files'])
    counts=pipeline.dataset_pair_counts(full,c['endpoints'])
    baseline=pq.read_table(PAYLOAD/'baseline_protein_split.parquet').to_pylist();byid={r['protein_sequence_hash']:r for r in baseline}
    missing=set(counts)-set(byid)
    if missing:raise RuntimeError(f'{len(missing)} new proteins lack original v4 partition; stop rather than invent/reassign split')
    active=[dict(byid[k],pairs=int(counts[k])) for k in sorted(counts)]
    cp=Counter();cn=Counter()
    for row in active:cp[row['protein_cluster_id']]+=row['pairs'];cn[row['protein_cluster_id']]+=1
    for row in active:row['cluster_pairs']=cp[row['protein_cluster_id']];row['cluster_proteins']=cn[row['protein_cluster_id']]
    dest=ROOT/c['stage2b']['paths']['dataset_dir'];dest.mkdir(exist_ok=True);table=pa.Table.from_pylist(active,schema=split.SPLIT_SCHEMA);pq.write_table(table,dest/'protein_split.parquet')
    manifest={'stage':'stage2b_protein_split','protocol':'original v4 membership and partitions locked, active proteins/counts updated','source_manifest':json.loads((PAYLOAD/'baseline_split_manifest.json').read_text(encoding='utf-8')),'proteins':len(active),'pairs':sum(counts.values()),'new_unassigned_proteins':0}
    (dest/'protein_split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def main():
    ROOT.mkdir(parents=True,exist_ok=True);os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
    if len(sys.argv)>1 and sys.argv[1]=='smoke':smoke();return
    assert Path('/content/drive/MyDrive').is_dir(),'Mount Drive first'
    mem_kb=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))
    assert mem_kb>=24*1024*1024,'Use CPU + High RAM (at least 24 GB) before full rebuild'
    candidates=[Path('/content/drive/MyDrive/extract_observation_and_ion_split'),Path('/content/drive/MyDrive/data_check/extract_observation_and_ion_split')]
    source=next((x for x in candidates if (x/'chembl37_observations.parquet').is_file()),None)
    if source is None:raise FileNotFoundError('Add shortcut of data_check/extract_observation_and_ion_split to My Drive; no data copied or overwritten')
    STATE.update(status='running',protocol='v4_salt_retained',source=str(source),fragment_stripping=False,other_standardization=True,split='locked original v4',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    identity={'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_commit':STATE['base_commit'],'input_manifest_sha256':hashlib.sha256((PAYLOAD/'input_sizes.json').read_bytes()).hexdigest()}
    BACKUP.mkdir(parents=True,exist_ok=True);receipt=BACKUP/'resume_identity.json'
    if receipt.exists():assert json.loads(receipt.read_text())==identity,'Resume identity differs: do not mix artifacts from different protocols'
    else:receipt.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    c=prepare_config();smoke();rebuild_structures(source)
    dedup=ROOT/'data/dedup_salt_retained/cross_source_duplicates.parquet'
    if not dedup.exists():run('dedup',[sys.executable,'scripts/dedup_cross_source.py'])
    backup('data/dedup_salt_retained')
    run('dataset_smoke',[sys.executable,'run_dataset.py','--mode','smoke'])
    full=ROOT/'data/dataset_v4_salt_retained/full/full_manifest.json'
    if not full.exists():run('dataset_full',[sys.executable,'run_dataset.py','--mode','full'])
    backup('data/dataset_v4_salt_retained/full')
    run('split_prepare',[sys.executable,'run_dataset.py','--mode','split'],allowed=(0,2));fixed_split(c)
    run('export',[sys.executable,'run_dataset.py','--mode','export'])
    run('verify',[sys.executable,'run_dataset.py','--mode','verify'])
    backup('data/dataset_v4_salt_retained');status('complete',status='complete')

if __name__=='__main__':
    try:main()
    except Exception as e:
        if ROOT.exists():status('failed',status='failed',error=repr(e))
        raise
