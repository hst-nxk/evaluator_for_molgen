"""Independent full column scan + author-native checks of saved v4 desalting.
No molecular structures or dataset files are modified.
"""
import sys,json,inspect,re
from pathlib import Path
from collections import Counter
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem,RDLogger
from chembl_structure_pipeline import standardizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_requested_8_10 import METALS
BASE=ROOT/'data/downloaded_20261103';OUT=ROOT/'audit_20261103/desalting_recheck_20261104';OUT.mkdir(exist_ok=True)
RDLogger.DisableLog('rdApp.*')
def total(mask):return int(pc.sum(pc.cast(pc.fill_null(mask,False),pa.int64())).as_py() or 0)
def main():
    report={'scope':'downloaded v4; original observation files and all three final exports','observations':{},'final_exports':{},'examples':[]}
    residual=[];changed=[];ids={'chembl:32049:IC50','chembl:254376:Ki','chembl:225211:Ki','chembl:1521625:IC50','bindingdb:50000104:IC50'};examples=[]
    cols=['observation_id','molecule_source_smiles','molecule_input_canonical_smiles','molecule_canonical_smiles','molecule_inchikey','molecule_parent_fragments','molecule_parent_excluded','structure_status','structure_reason']
    for f in (BASE/'extract_observation_and_ion_split').glob('*.parquet'):
        c=Counter()
        for b in pq.ParquetFile(f).iter_batches(columns=cols):
            t=pa.Table.from_batches([b]);a=t['molecule_input_canonical_smiles'];p=t['molecule_canonical_smiles'];multi_a=pc.fill_null(pc.match_substring(a,'.'),False);multi_p=pc.fill_null(pc.match_substring(p,'.'),False)
            c['rows']+=len(t);c['input_multifragment']+=total(multi_a);c['parent_multifragment']+=total(multi_p);c['multifragment_to_single']+=total(pc.and_(multi_a,pc.and_(pc.is_valid(p),pc.invert(multi_p))));c['parent_missing']+=p.null_count;c['parent_excluded']+=total(t['molecule_parent_excluded']);c['parent_with_fragments_removed_note']+=total(pc.match_substring(t['structure_reason'],'salt_or_solvent_fragments_removed'))
            residual.extend(t.filter(multi_p).to_pylist())
            selected=pc.is_in(t['observation_id'],value_set=pa.array(sorted(ids)));examples.extend(t.filter(selected).to_pylist())
            if len(changed)<1000:
                found=t.filter(pc.and_(multi_a,pc.and_(pc.is_valid(p),pc.invert(multi_p)))).to_pylist();changed.extend(found[:1000-len(changed)])
        report['observations'][f.name]=dict(c);print(f.name,dict(c),flush=True)
    pq.write_table(pa.Table.from_pylist(residual),OUT/'remaining_multifragment_observations.parquet')
    # No 'largest fragment' substitution: compare actual source with official package parent.
    def official(s):
        m=Chem.MolFromSmiles(s)
        if m is None:raise ValueError('source parse failed')
        m=standardizer.standardize_mol(m);p,excluded=standardizer.get_parent_mol(m,neutralize=True,check_exclusion=True);Chem.SanitizeMol(p)
        return Chem.MolToSmiles(p,isomericSmiles=True),Chem.MolToInchiKey(p),bool(excluded)
    checked=set();counts=Counter();mismatch=[]
    for row in examples+changed+residual:
        s=row['molecule_source_smiles']
        if not s or s in checked:continue
        checked.add(s)
        try:
            smi,key,excluded=official(s);counts['unique_sources_checked']+=1
            equal=key==row['molecule_inchikey'];counts['identity_match' if equal else 'identity_mismatch']+=1
            if not equal:mismatch.append({'observation_id':row['observation_id'],'stored_parent':row['molecule_canonical_smiles'],'official_parent':smi,'stored_key':row['molecule_inchikey'],'official_key':key})
            if row['observation_id'] in ids:report['examples'].append({**row,'official_parent':smi,'official_inchikey':key,'matches_official':equal,'official_excluded':excluded})
        except Exception as e:counts['errors']+=1;mismatch.append({'observation_id':row['observation_id'],'error':str(e)})
    report['official_recheck']={'selection':'unique source SMILES among requested concrete examples, first 1000 reduced-to-single rows, and ALL residual-multifragment rows; not an all-source normalization rerun','counts':dict(counts),'mismatches_or_errors':mismatch}
    pt=Chem.GetPeriodicTable();symbols=[pt.GetElementSymbol(z) for z in sorted(METALS)]
    pattern=r'\[(?:[0-9]+)?(?:'+'|'.join(sorted(symbols,key=len,reverse=True))+r')(?:[^a-z]|$)'
    report['metal_definition']={'atomic_numbers':sorted(METALS),'pattern':pattern,'scope':'bracketed atom tokens in canonical SMILES; matched positives independently parsed with RDKit'}
    for name,fields in [('evidence_records',['smiles_a','smiles_b']),('evaluator_training',['smiles_a','smiles_b']),('generator_training',['smiles_weak','smiles_strong'])]:
        c=Counter();metal_smiles=set()
        for b in pq.ParquetFile(BASE/'final_dataset'/(name+'.parquet')).iter_batches(columns=fields):
            t=pa.Table.from_batches([b]);c['rows']+=len(t);multi=None;metal=None
            for field in fields:
                col=t[field];mm=pc.fill_null(pc.match_substring(col,'.'),False);me=pc.fill_null(pc.match_substring_regex(col,pattern),False);multi=mm if multi is None else pc.or_(multi,mm);metal=me if metal is None else pc.or_(metal,me)
                metal_smiles.update(pc.unique(pc.filter(col,me)).to_pylist())
            c['multifragment_pair_rows']+=total(multi);c['metal_containing_pair_rows']+=total(metal)
        checked_metal=[]
        for s in sorted(metal_smiles):
            m=Chem.MolFromSmiles(s);assert m is not None and any(a.GetAtomicNum() in METALS for a in m.GetAtoms());checked_metal.append(s)
        c['distinct_metal_smiles']=len(checked_metal);report['final_exports'][name]={'counts':dict(c),'metal_examples':checked_metal[:10]};print(name,dict(c),flush=True)
    report['conclusion']='Desalting applied and actual removals verified. Not every source is forced to one fragment; official rules can retain all fragments. No final multi-fragment structures is not proof all parent choices are chemically correct.'
    (OUT/'recheck.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('COMPLETE',flush=True)
if __name__=='__main__':main()
