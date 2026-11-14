"""Pack dataset v4 + the B/F-mixed-query fix into ZIP parts with the original Drive layout.

Layout inside every part (same top-level folders as the 03/11 Drive download):
  deduplicate/  extract_observation_and_ion_split/  filter_merge_mmp/   -> unchanged v4
  final_dataset/                                                        -> fixed files
  README_CHANGES.md, SHA256SUMS.txt, scripts/<build script>
Each part is a complete ZIP on its own; extracting all parts into one folder gives the
whole dataset. Files are spread over parts by size, never split across parts.

Usage: python -I package_dataset_noBFmix_20261104.py <v4_dir> <fixed_final_dir> <out_dir> <part_gb>
"""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

V4, FIXED, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PART_BYTES = int(float(sys.argv[4]) * 1024 ** 3)
ROOT = Path(__file__).resolve().parents[1]
NAME = "molopt_dataset_v4_noBFmix_20261104"
started = time.time()


def log(*parts):
    print(f"[{time.time() - started:7.1f}s]", *parts, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f"{OUT} is not empty; refusing to overwrite")
OUT.mkdir(parents=True, exist_ok=True)

# (path inside zip, source file)
entries = []
for top in ("deduplicate", "extract_observation_and_ion_split", "filter_merge_mmp"):
    for f in sorted((V4 / top).rglob("*")):
        if f.is_file():
            entries.append((f.relative_to(V4).as_posix(), f))
for f in sorted(FIXED.iterdir()):
    if f.is_file():
        entries.append((f"final_dataset/{f.name}", f))
for script in ("build_v4_no_bf_mixed_queries_20261104.py", "package_dataset_noBFmix_20261104.py"):
    entries.append((f"scripts/{script}", ROOT / "scripts" / script))

manifest = json.loads((FIXED / "bf_mixed_query_manifest.json").read_text(encoding="utf-8"))
rows = manifest["rows"]
files = manifest["files"]
readme = f"""# {NAME}

Dataset MOLOPT v4 (Stage 2b export 31/10/2026) with one curation fix applied on 04/11/2026.

## What changed
Only `final_dataset/` differs from the v4 download of 03/11/2026. Every other folder is
the v4 data byte for byte.

Rule: a ranking query = (reference SMILES, protein, endpoint), both directions, as Stage 3
ranks. A query is mixed when its candidates include a row backed by a B (binding) assay
and a row backed by an F (functional) assay. Every row of every mixed query is removed.
Assay types: ChEMBL `ASSAYS.ASSAY_TYPE`; BindingDB rows only through the ChEMBL assay id
BindingDB writes in `ASSAY_NAME` (`ChEMBL_<assay_id> (CHEMBL<id>)`). No text-based guessing.

| file | before | removed | after |
|---|---:|---:|---:|
""" + "\n".join(f"| {k} | {v['before']:,} | {v['removed']:,} | {v['after']:,} |" for k, v in files.items()) + f"""

Evaluator rows per partition: train {rows['train']['before']:,} -> {rows['train']['after']:,};
validation {rows['validation']['before']:,} -> {rows['validation']['after']:,};
test {rows['test']['before']:,} -> {rows['test']['after']:,}. No protein lost; split unchanged.
Mixed queries removed: {manifest['mixed_queries']}.

## What did NOT change
- Rows whose queries contain only F assays are still present (decision 04/11/2026).
- `filter_merge_mmp/` intermediates still contain the removed pairs; the removed
  pair_ids are listed in `final_dataset/dropped_pair_ids.parquet`.
- `protein_split*` and `export_manifest.json` are the v4 originals.

## Files
- `final_dataset/bf_mixed_query_manifest.json`: rule, counts, sources.
- `final_dataset/dropped_pair_ids.parquet`: every removed pair_id.
- `scripts/`: the scripts that built and packed this version.
- `SHA256SUMS.txt`: checksum of every file.

To use: extract every ZIP part into the same folder.
"""
# newline="\n": on Windows write_text would emit CRLF and `sha256sum -c` on Linux/Colab
# would then read every file name with a trailing \r
(OUT / "README_CHANGES.md").write_text(readme, encoding="utf-8", newline="\n")
entries.append(("README_CHANGES.md", OUT / "README_CHANGES.md"))

log("hashing", len(entries), "files")
sums = [(arc, sha256(src), src.stat().st_size) for arc, src in entries]
(OUT / "SHA256SUMS.txt").write_text("".join(f"{h}  {arc}\n" for arc, h, _ in sums), encoding="utf-8",
                                    newline="\n")
entries.append(("SHA256SUMS.txt", OUT / "SHA256SUMS.txt"))
size = {arc: s for arc, _, s in sums}
size["SHA256SUMS.txt"] = (OUT / "SHA256SUMS.txt").stat().st_size

# spread files over parts: largest first into the emptiest part
total = sum(size.values())
n_parts = max(1, -(-total // PART_BYTES))
parts = [[] for _ in range(n_parts)]
load = [0] * n_parts
small = {"README_CHANGES.md", "SHA256SUMS.txt"}
for arc, src in sorted(entries, key=lambda e: -size[e[0]]):
    if arc in small:
        continue
    i = load.index(min(load))
    parts[i].append((arc, src))
    load[i] += size[arc]
for part in parts:                         # README and checksums travel with every part
    part += [(a, s) for a, s in entries if a in small]

for i, part in enumerate(parts, 1):
    path = OUT / f"{NAME}-part{i:02d}of{n_parts:02d}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as z:
        for arc, src in sorted(part):
            z.write(src, arc)
    log(path.name, f"{len(part)} files", f"{sum(size[a] for a, _ in part) / 1024 ** 3:.2f} GiB raw",
        f"{path.stat().st_size / 1024 ** 3:.2f} GiB zipped")
log("done")
