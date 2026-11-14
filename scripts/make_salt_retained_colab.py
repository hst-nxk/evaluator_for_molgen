import json,zipfile,hashlib,base64,shutil
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'colab_salt_retained_20261104';OUT.mkdir(exist_ok=True)
BASE=ROOT/'data/downloaded_20261103';payload=OUT/'payload';payload.mkdir(exist_ok=True)
for source,name in [(ROOT/'scripts/build_v4_salt_retained.py','build_v4_salt_retained.py'),(BASE/'filter_merge_mmp/full_manifest.json','baseline_full_manifest.json'),(BASE/'final_dataset/export_manifest.json','baseline_export_manifest.json'),(BASE/'final_dataset/protein_split_manifest.json','baseline_split_manifest.json'),(BASE/'final_dataset/protein_split.parquet','baseline_protein_split.parquet')]:shutil.copy2(source,payload/name)
sizes={}
for f in (BASE/'extract_observation_and_ion_split').glob('*observations.parquet'):
    with f.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    sizes[f.name]={'bytes':f.stat().st_size,'rows':pq.ParquetFile(f).metadata.num_rows,'sha256':digest}
(payload/'input_sizes.json').write_text(json.dumps(sizes,indent=2),encoding='utf-8')
archive=OUT/'MOLOPT_V4_SALT_RETAINED_payload.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for f in payload.iterdir():z.write(f,f.name)
encoded=base64.b64encode(archive.read_bytes()).decode()
setup="""import os,sys,subprocess,json,pathlib,base64,io,zipfile
from google.colab import drive
drive.mount('/content/drive')
ROOT=pathlib.Path('/content/molopt_v4_salt_retained')
COMMIT='8e75726575e60c3031a644ae3f898c5c12399546'
if not ROOT.exists(): subprocess.run(['git','clone','https://github.com/hst-nxk/evaluator_for_molgen',str(ROOT)],check=True)
subprocess.run(['git','-C',str(ROOT),'checkout',COMMIT],check=True)
assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==COMMIT
subprocess.run([sys.executable,'-m','pip','install','-q','rdkit==2026.9.1','chembl_structure_pipeline==1.2.4','pyarrow==25.0.1','numpy==2.2.6','pytest==8.4.2'],check=True)
os.chdir(ROOT)
print('SETUP PASS',COMMIT,'CPU',os.cpu_count(),flush=True)
"""
unpack="""PAYLOAD=ROOT/'ablation_payload'
PAYLOAD.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode('"""+encoded+"""'))) as z:
    for name in z.namelist():
        assert (PAYLOAD/name).resolve().is_relative_to(PAYLOAD.resolve())
    z.extractall(PAYLOAD)
runner=PAYLOAD/'build_v4_salt_retained.py'
subprocess.run([sys.executable,str(runner),'smoke'],check=True)
print('PAYLOAD + ABLATION SMOKE PASS',flush=True)
"""
run="""# Full run: unique-source normalization, fresh dedup, v4 curation + MMP,
# locked original protein partitions, export, verify, Drive backup.
subprocess.run([sys.executable,str(runner)],check=True)
"""
def cell(kind,text):
    c={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':c.update(outputs=[],execution_count=None)
    return c
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'provenance':[]},'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},'cells':[cell('markdown','# MOLOPT v4 salt retained — controlled ablation\nCPU high-RAM. Source: MyDrive/extract_observation_and_ion_split shortcut (read-only). Keeps all standardized components; still standardizes representation/isotopes/charge. Custom ablation, not default author pipeline. Raw activity/protein columns preserved. Rebuild dedup/MMP; original v4 protein split locked. Output: MyDrive/molopt_out/v4_salt_retained_20261104. Source datasets not overwritten. Partial structure cache is resumable on Drive. Do not call complete until final verify PASS.\n'),cell('code',setup),cell('code',unpack),cell('code',run)]}
target=OUT/'MOLOPT_V4_SALT_RETAINED_20261104.ipynb';target.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8');print(target)
