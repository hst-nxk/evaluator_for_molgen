# MOLOPT — Raw to Observations Pipeline

Context: this is a reproducible data-engineering step for an academic benchmark paper, using public ChEMBL and BindingDB releases.

You are an experienced Python developer specializing in scientific data processing and cheminformatics.

## Objective

Build a clean, readable, reproducible Python pipeline that extracts molecular activity observations from ChEMBL37 and BindingDB202610.

**Only implement Raw → Observations.** Do not generate MMP pairs, aggregate measurements, calculate Δp, or train models.

## Paths

Read-only raw data (never modify, delete, move or overwrite anything here):
- `E:\eval_training\raw\chembl37`
- `E:\eval_training\raw\bindingdb202610`

All code, outputs, logs and temporary files stay inside `E:\eval_training`.

Inspect actual files and schemas first. Do not assume filenames or column names.

**Archive handling:**
- BindingDB TSVs: stream directly from the ZIPs with `zipfile`. Never extract to disk.
- ChEMBL is already extracted. Open it read-only at
  `E:\eval_training\work\chembl_37\chembl_37_sqlite\chembl_37.db`
  using URI mode (`file:...?mode=ro`). The code must never extract archives; if the `.db` is missing, fail with a clear message.
- Do not read either SDF file in this stage. Say so explicitly rather than parsing them.

## Observation requirements

Each observation is one measurement of one endpoint for one molecule and one protein. Support **Ki, Kd, IC50, EC50**.

Preserve, in separate columns (never overwrite source values with derived ones):

1. **Source:** database, release, native measurement ID, document/publication ID.
2. **Molecule:** source ligand ID, original SMILES/CXSMILES verbatim, RDKit canonical isomeric SMILES, full InChIKey, stereochemistry metadata.
3. **Protein:** the sequence attached to THAT measurement, sequence hash, accession, construct/variant metadata.
4. **Assay:** native assay identifier plus whatever experimental context exists.
5. **Activity:** endpoint, original value, original unit, qualifier, concentration in mol/L, pActivity.
6. **Status:** per-field resolution status with exact reasons (see below).

`pActivity = -log10(concentration_in_mol/L)` for valid exact concentrations.

Do not merge or aggregate anything at this stage.

### observation_id

Stable, derived from source database + native record ID + endpoint. **Never hash anything containing the protein sequence** — if correcting a protein changes the ID, every link back to evidence breaks.

### Status columns

Separate status per field: `protein_status`, `assay_status`, `structure_status`, `activity_status`. Each is `resolved` / `unresolved` / `invalid` plus a reason string. A row with a clear protein but an unknown assay must only be blocked at the step that needs the assay — do not collapse these into one global flag.

### Units and qualifiers

Convert to mol/L via an explicit unit table (nM → 1e-9, µM → 1e-6, etc.). Units outside the table → `unresolved`, never a guessed factor. Non-positive and non-finite values cannot be log-transformed → `invalid` with reason. Keep the qualifier (`=`, `<`, `>`) as its own column; censored rows stay in the table but are marked as censored. PMIDs must be stored as integer-like strings, never `"12345.0"`.

## Critical correctness rules

These are the exact defects found in the historical pipeline. Avoid every one:

- **Protein:** use the sequence tied to the individual measurement. Never look up one representative sequence by accession/target ID and apply it to many rows — one accession can span constructs of very different length.
- **ChEMBL variants:** resolve `assays.variant_id` → `variant_sequences` first; fall back to component sequence only when no variant is declared. If a variant is declared but its sequence is absent → `unresolved`. Never substitute wild type.
- **BindingDB assay mapping:** the per-measurement key (Reactant_set_id) is a DIFFERENT namespace from the assay table key (ENTRYID). Join `BindingDB_All` → `BindingDB_rsid_eaids` (Reactant_set_id → ENTRYID + ASSAYID), then → `BindingDB_Assays` on (ENTRYID, ASSAYID). Never join Reactant_set_id to ENTRYID directly. Assert the key domains actually intersect at load time and fail loudly if they don't.
- **Stereochemistry:** keep source CXSMILES verbatim in its own column. `MolToSmiles` drops enhanced-stereo markers such as `|r|`, and that information is unrecoverable once lost. Never infer absolute stereo from relative markers.
- Never invent missing protein, assay or stereochemical information.
- Do not drop rows merely because something is unresolved — keep the row, record the reason.
- Reject empty or invalid computed identifiers from the valid-identity category.
- Do not introduce any scientific filter not stated here.
- Keep the source-database column intact. BindingDB All contains records imported from ChEMBL; do NOT deduplicate across sources at this stage — a later stage needs to be able to detect it.

## Code quality

Scientific correctness first, then simplicity and readability. The code will be read and maintained by a human, so clarity beats cleverness.

- Clean Python, PEP 8, type hints on every function, meaningful names.
- Small functions with a single responsibility. No classes unless state genuinely requires one; no abstraction layers added "for later".
- Separate ChEMBL and BindingDB readers; share normalization utilities in `normalize.py`. No duplicated conversion logic between the two readers.
- No magic numbers or inline literals: unit factors, endpoint names and column names live in named constants or `config.json`.
- Pure functions for normalization (value in → value out), so they are testable without touching raw data.
- Bounded-memory streaming suitable for a 16GB RAM laptop: iterate rows, write Parquet in chunks, never build a full DataFrame of either source.
- Errors must name the row and the reason. No bare `except:`, no silent `pass`.
- Clear progress messages (rows processed, current stage) for runs that take a long time.
- Focused unit tests for protein mapping, assay mapping, stereochemistry preservation and activity conversion, including the edge cases: missing variant sequence, unknown unit, non-positive value, censored qualifier.

## Project structure

```text
E:\eval_training\
├── raw\                  (already exists; read-only)
├── work\                 (ChEMBL .db already extracted here; gitignored)
├── src\molopt\observations\
│   ├── bindingdb.py
│   ├── chembl.py
│   ├── normalize.py
│   └── pipeline.py
├── tests\
├── data\observations\
├── runs\observations\
├── run_observations.py
├── config.json
├── requirements.txt
├── .gitignore
└── README.md
```

Do not create future modules or placeholder files. Keep the root minimal. No separate projects, extra virtualenvs, redundant datasets or unnecessary intermediate files. `.gitignore` excludes `raw/`, `work/`, `data/`, `runs/`, caches and venvs.

## CLI

```powershell
python run_observations.py --mode inspect
python run_observations.py --mode smoke
python run_observations.py --mode full
python run_observations.py --mode verify
```

- `inspect`: discover raw files, list schemas/columns, report which columns supply per-measurement protein sequence, native assay ID, variant info and CXSMILES. If anything required is missing, say so explicitly instead of working around it.
- `smoke`: small real-data sample from both sources → `runs/observations/`.
- `full`: all relevant observations → `data/observations/` (Parquet only).
- `verify`: integrity checks (below).

### verify must include

1. **Conservation:** rows selected from raw == rows written.
2. **Random sample reconciliation:** draw ~5000 rows and compare every field back to raw. Internal consistency alone cannot catch a systematically wrong join.
3. **Fixed regression cases**, asserted on every run:
   - accession `P03367`: the stored sequence must match the measurement's own construct (the historical bug stored 1,447 aa where raw had 99 aa).
   - BindingDB monomer `13590`: the `|r|` marker must survive in the source-CXSMILES column.
   - no PMID stored in `.0` form.
4. **Counts by endpoint, by source, and by each unresolved/invalid reason.**

## Execution

**Write code only. Do not run anything except `--mode inspect`.** No smoke, no full, no verify, no tests. I will run those myself.

`inspect` is read-only: list the raw directory, read ZIP central directories and TSV header lines, and open the ChEMBL SQLite schema read-only. It must not extract archives or write anything under `data/` or `runs/`.

If a required column cannot be found, say so explicitly and leave a clearly marked TODO. Do not guess silently and do not invent workarounds.

After writing the code, explain what each module does and how data flows through it, then give me the exact PowerShell commands to run, in order: tests → smoke → full → verify.

No Git commits, no MMP construction, no model training. Do not touch anything outside `E:\eval_training`.

Report in Vietnamese: concise summary of what you wrote, the actual inspect output (real column names found), and any assumptions or TODOs needing my confirmation. Do not claim anything was tested or verified beyond inspect — you did not run it.
