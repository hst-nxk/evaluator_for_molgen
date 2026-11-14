"""Second look at salt stripping in v4: what exactly was removed, and does ChEMBL agree.
Read only. Usage: python audit_salt_second_look_20261107.py <obs_dir> <chembl_db> <out_json>"""
import json, sqlite3, sys, time
from collections import Counter
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
RDLogger.DisableLog('rdApp.*')
OBS, DB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
cols = ['molecule_chembl_id', 'molecule_input_canonical_smiles', 'molecule_canonical_smiles', 'molecule_inchikey']
df = pq.read_table(f'{OBS}/chembl37_observations.parquet', columns=cols).to_pandas().dropna(subset=cols[1:])
df['fi'] = df.molecule_input_canonical_smiles.str.count(r'\.') + 1
df['fp'] = df.molecule_canonical_smiles.str.count(r'\.') + 1
d = df[df.fp < df.fi]
g = d.groupby(cols).size().reset_index(name='rows')
print('rows with a dropped fragment', len(d), 'unique', len(g), flush=True)

def key(m):
    try: return inchi.MolToInchiKey(m)[:14]
    except Exception: return None

cat = Counter(); cat_u = Counter(); examples = []
for r in g.itertuples():
    a = Chem.MolFromSmiles(r.molecule_input_canonical_smiles); p = Chem.MolFromSmiles(r.molecule_canonical_smiles)
    if a is None or p is None: cat['unparsable'] += r.rows; cat_u['unparsable'] += 1; continue
    keep = Counter(key(x) for x in Chem.GetMolFrags(p, asMols=True))
    removed = []
    for fr in Chem.GetMolFrags(a, asMols=True):
        k = key(fr)
        if keep[k] > 0: keep[k] -= 1
        else: removed.append(fr)
    if not removed: c = 'only_duplicate_copies_removed'
    else:
        organic = [x for x in removed if any(at.GetAtomicNum() == 6 for at in x.GetAtoms())]
        big = [x for x in organic if x.GetNumHeavyAtoms() >= 10]
        if big: c = 'removed_organic_fragment_10plus_heavy_atoms'
        elif organic: c = 'removed_small_organic_fragment'
        else: c = 'removed_inorganic_only'
        if any(at.GetAtomicNum() > 20 and at.GetAtomicNum() not in (35, 53) for x in removed for at in x.GetAtoms()) and c == 'removed_inorganic_only': c = 'removed_inorganic_heavy_element'
    cat[c] += r.rows; cat_u[c] += 1
    if c == 'removed_organic_fragment_10plus_heavy_atoms' and len(examples) < 12:
        examples.append({'input': r.molecule_input_canonical_smiles, 'parent': r.molecule_canonical_smiles, 'rows': int(r.rows), 'chembl_id': r.molecule_chembl_id})
print('categories (rows):', dict(cat), flush=True)

# agreement with ChEMBL's own curated parent
con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
ids = sorted(set(g.molecule_chembl_id.dropna()))
parent_key = {}
for i in range(0, len(ids), 5000):
    chunk = ids[i:i + 5000]; q = ','.join('?' * len(chunk))
    for cid, own, par in con.execute(f"""select md.chembl_id, cs_own.standard_inchi_key, cs_par.standard_inchi_key
        from molecule_dictionary md join molecule_hierarchy mh on mh.molregno=md.molregno
        left join compound_structures cs_own on cs_own.molregno=md.molregno
        left join compound_structures cs_par on cs_par.molregno=mh.parent_molregno where md.chembl_id in ({q})""", chunk):
        parent_key[cid] = (own, par)
agree = Counter(); mism = []
for r in g.itertuples():
    own, par = parent_key.get(r.molecule_chembl_id, (None, None))
    if par is None: agree['no_chembl_parent_record'] += r.rows; continue
    ours = r.molecule_inchikey[:14]
    if par[:14] == ours: agree['ours_equals_chembl_parent'] += r.rows
    elif own and own[:14] == par[:14]: agree['chembl_kept_whole_(its_parent_is_itself)'] += r.rows
    else:
        agree['ours_differs_from_chembl_parent'] += r.rows
        if len(mism) < 12: mism.append({'input': r.molecule_input_canonical_smiles, 'ours': r.molecule_canonical_smiles, 'rows': int(r.rows), 'chembl_id': r.molecule_chembl_id})
print('agreement (rows):', dict(agree), flush=True)
json.dump({'dropped_rows': int(len(d)), 'unique_pairs': int(len(g)), 'what_was_removed_rows': dict(cat), 'what_was_removed_unique': dict(cat_u),
           'vs_chembl_parent_rows': dict(agree), 'examples_big_organic_removed': examples, 'examples_differs_from_chembl': mism, 'seconds': round(time.time() - t0)},
          open(OUT, 'w'), indent=1, ensure_ascii=False)
print('done', round(time.time() - t0), 's')
