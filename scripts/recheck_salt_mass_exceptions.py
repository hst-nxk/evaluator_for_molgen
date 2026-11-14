"""Resolve mass accounting failures without standardizing discarded PF6 fragments.
First match unchanged components exactly; normalize only if needed to match retained
components. A discarded fragment need not be standardizable to have its mass measured.
"""
import json
from pathlib import Path
from collections import Counter
from rdkit import Chem,RDLogger
from rdkit.Chem import Descriptors
from chembl_structure_pipeline import standardizer
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_20261103'
path=OUT/'salt_mass_exceptions.json';rows=json.loads(path.read_text(encoding='utf-8'))
report=json.loads((OUT/'audit_8_10.json').read_text(encoding='utf-8'));c=report['item8'];rechecked=[]
for r in rows:
    if not (r.get('error') or r.get('removal_match_unresolved')):continue
    m=Chem.MolFromSmiles(r['input']);p=Chem.MolFromSmiles(r['parent']) if r['parent'] else None
    if not m or not p or len(Chem.GetMolFrags(m))<=len(Chem.GetMolFrags(p)):continue
    source=list(Chem.GetMolFrags(m,asMols=True));parents=list(Chem.GetMolFrags(p,asMols=True));available=list(range(len(source)));matched=[];unmatched=[]
    for parent in parents:
        k=Chem.MolToInchiKey(parent);found=next((i for i in available if Chem.MolToInchiKey(source[i])==k),None)
        if found is None:unmatched.append(parent)
        else:available.remove(found);matched.append(found)
    def key(x):
        x=standardizer.standardize_mol(Chem.Mol(x));x=standardizer.get_isotope_parent_mol(x);x=standardizer.uncharge_mol(x);Chem.SanitizeMol(x);return Chem.MolToInchiKey(x)
    for parent in unmatched:
        k=key(parent);found=None
        for i in available:
            try:
                if key(source[i])==k:found=i;break
            except (RuntimeError,ValueError):pass
        if found is None:break
        available.remove(found);matched.append(found)
    if len(matched)!=len(parents):continue
    n=r['rows'];old_error=r.pop('error',None);old_unresolved=r.pop('removal_match_unresolved',None)
    for old,k in [(old_error,'error'),(old_unresolved,'removal_match_unresolved')]:
        if old:c[k+'_rows']-=n;c[k+'_unique']-=1
    removed=[Descriptors.MolWt(source[i]) for i in available];kept=[Descriptors.MolWt(source[i]) for i in matched]
    r.update(removal_verified=True,removed_total_mw=sum(removed),retained_total_mw=sum(kept),largest_removed_mw=max(removed,default=0),largest_retained_mw=max(kept),removed_total_heavier=sum(removed)>sum(kept),largest_removed_heavier=max(removed,default=0)>max(kept),recheck_method='exact components first; normalized only unmatched retained components',initial_error=old_error)
    for k in ['removal_verified','removed_total_heavier','largest_removed_heavier']:
        if r[k]:c[k+'_rows']=c.get(k+'_rows',0)+n;c[k+'_unique']=c.get(k+'_unique',0)+1
    rechecked.append(r)
path.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
report['item8_recheck']={'recovered_unique':len(rechecked),'recovered_rows':sum(r['rows'] for r in rechecked),'method':'exact component matching first avoids requiring discarded fragments to standardize','note':'full scan initially recorded 101 unique PF6 valence exceptions (321 rows) plus two unmatched records; residuals remain explicit'}
(OUT/'audit_8_10.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'recheck':report['item8_recheck'],'final_counts':c},indent=2))
