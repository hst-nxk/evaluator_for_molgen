"""Read-only audits 8 and 10. Does not rewrite the training dataset."""
import json,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_20261103';OUT.mkdir(exist_ok=True)
# Explicit chemical definition: alkali, alkaline earth, transition, post-transition,
# lanthanide and actinide metals. B/Si/Ge/As/Sb/Te are not classified as metals here.
METALS=set([3,4,11,12,13,19,20,31,37,38,49,50,55,56,81,82,83,87,88]+list(range(21,31))+list(range(39,49))+list(range(57,81))+list(range(89,113)))

def inspect_structures(items):
    from rdkit import Chem,RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog('rdApp.*')
    result=[]
    for s,parent,n,example in items:
        rec={'input':s,'parent':parent,'rows':n,'observation_id':example}
        try:
            m=Chem.MolFromSmiles(s);p=Chem.MolFromSmiles(parent) if parent else None
            if m is None:rec['parse_error']=True;result.append(rec);continue
            rec['source_has_metal']=any(a.GetAtomicNum() in METALS for a in m.GetAtoms())
            rec['parent_has_metal']=bool(p and any(a.GetAtomicNum() in METALS for a in p.GetAtoms()))
            if p and len(Chem.GetMolFrags(m))>len(Chem.GetMolFrags(p)):
                # Only whole-component subtraction is considered verified removal.
                # Proton/isotope changes are ignored for matching, not for reported mass.
                from chembl_structure_pipeline import standardizer
                def key(x):
                    x=standardizer.standardize_mol(Chem.Mol(x))
                    x=standardizer.get_isotope_parent_mol(x)
                    x=standardizer.uncharge_mol(x);Chem.SanitizeMol(x)
                    return Chem.MolToInchiKey(x)
                keep=Counter(key(x) for x in Chem.GetMolFrags(p,asMols=True))
                removed=[];retained=[]
                for frag in Chem.GetMolFrags(m,asMols=True):
                    k=key(frag)
                    if keep[k]>0:keep[k]-=1;retained.append(Descriptors.MolWt(frag))
                    else:removed.append(Descriptors.MolWt(frag))
                if any(keep.values()) or not retained:rec['removal_match_unresolved']=True
                else:
                    rec.update(removal_verified=True,removed_total_mw=sum(removed),retained_total_mw=sum(retained),largest_removed_mw=max(removed,default=0),largest_retained_mw=max(retained))
                    rec['removed_total_heavier']=sum(removed)>sum(retained)
                    rec['largest_removed_heavier']=max(removed,default=0)>max(retained)
        except Exception as e:rec['error']=str(e)
        result.append(rec)
    return result

def main():
    start=time.time();report={'metal_atomic_numbers':sorted(METALS),'mass_definition':'RDKit MolWt, original input components; component matching after standardize/isotope removal/uncharge; unresolved matches not classified','item10':{}}
    data=ROOT/'data/downloaded_20261103'
    final=data/'final_dataset'
    for path in sorted(final.glob('*.parquet')):
        pf=pq.ParquetFile(path)
        if 'assay_types' not in pf.schema_arrow.names:continue
        counts=Counter();bfrows=[];flagrows=[]
        cols=[x for x in ['pair_id','protein_hash','endpoint','partition','assay_types','mixed_assay_type'] if x in pf.schema_arrow.names]
        for b in pf.iter_batches(columns=cols):
            for r in b.to_pylist():
                types=set((r.get('assay_types') or '').split(';'));bf={'B','F'}<=types
                counts['rows']+=1
                if r.get('mixed_assay_type'):counts['mixed_any_types']+=1
                if bf:
                    counts['binding_and_functional']+=1;counts['BF_'+str(r.get('partition'))]+=1;bfrows.append(r)
                if bf != bool(r.get('mixed_assay_type')):counts['BF_differs_from_mixed_any_types']+=1
                flagrows.append({'pair_id':r['pair_id'],'partition':r.get('partition'),'mixed_binding_functional':bf})
        report['item10'][path.name]=dict(counts)
        if flagrows:pq.write_table(pa.Table.from_pylist(flagrows),OUT/(path.stem+'_BF_flags.parquet'))
        if bfrows:pq.write_table(pa.Table.from_pylist(bfrows),OUT/(path.stem+'_BF_evidence.parquet'))
        print('item10',path.name,dict(counts),flush=True)
    (OUT/'audit_8_10.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    structs={}
    paths=list((data/'deduplicate').glob('*observations_dedup.parquet'))
    report['item8_scope']={'files':[str(x) for x in paths],'counting':'all rows in downloaded dedup observation files, including dedup_keep=False; row multiplicities retained'}
    for path in paths:
        pf=pq.ParquetFile(path);cols=['observation_id','molecule_input_canonical_smiles','molecule_canonical_smiles']
        for b in pf.iter_batches(columns=cols):
            for r in b.to_pylist():
                s=r['molecule_input_canonical_smiles'];p=r['molecule_canonical_smiles']
                if not s:continue
                k=(s,p)
                if k in structs:structs[k][0]+=1
                else:structs[k]=[1,r['observation_id']]
        print('loaded structures',path.name,len(structs),flush=True)
    items=[(s,p,n,q) for (s,p),(n,q) in structs.items()];counts=Counter();exceptions=[];examples=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(inspect_structures,items[i:i+10000]) for i in range(0,len(items),10000)]
        for j,f in enumerate(as_completed(futures),1):
            for r in f.result():
                counts['unique_input_parent_pairs']+=1;counts['observation_rows']+=r['rows']
                for k in ['source_has_metal','parent_has_metal','parse_error','removal_verified','removal_match_unresolved','removed_total_heavier','largest_removed_heavier','error']:
                    if r.get(k):counts[k+'_rows']+=r['rows'];counts[k+'_unique']+=1
                if r.get('removed_total_heavier') or r.get('largest_removed_heavier') or r.get('error') or r.get('removal_match_unresolved'):exceptions.append(r)
                elif r.get('source_has_metal') and len(examples)<20:examples.append(r)
            report['item8']=dict(counts);report['seconds']=round(time.time()-start,1);report['complete']=False
            (OUT/'audit_8_10.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            if j%5==0:print('item8',counts['unique_input_parent_pairs'],'/',len(items),flush=True)
    (OUT/'salt_mass_exceptions.json').write_text(json.dumps(exceptions,indent=2,ensure_ascii=False),encoding='utf-8')
    report['metal_examples']=examples;report['complete']=True
    (OUT/'audit_8_10.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print('COMPLETE',flush=True)
if __name__=='__main__':main()
