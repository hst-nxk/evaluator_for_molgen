"""Stage 3: the input representations - Morgan fingerprints and ESM2 embeddings, cached.

Two groups of functions, deliberately kept apart:

Fingerprints (RDKit, no torch needed)
    The dataset holds 1.3 million distinct molecules across 10.5 million rows, so every
    molecule appears about 8 times. A fingerprint is therefore computed ONCE per distinct
    SMILES and cached on disk; the training loop only looks rows up by index. The bits are
    packed with np.packbits (2,048 bits -> 256 bytes) so the whole table fits in RAM.

Protein embeddings (ESM2, needs torch + transformers)
    Only 6,315 proteins take part in a pair, so the `frozen` configuration has to run ESM2
    6,315 times, once ever, and then reads the cache. The `finetune` configuration uses no
    cache at all: there ESM2 is part of the computation graph and runs again every batch.

THE EMBEDDING DIMENSION IS NEVER HARDCODED. It is read off the loaded model
(`config.hidden_size`), written into the cache's manifest, and checked again by train and
eval.

The caches are parquet, not pickle: every row is a key together with its vector, so the
vocabulary and the matrix always travel together and cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# The metadata key inside a cache file's parquet schema.
METADATA_KEY = b"molopt_stage3"


# -------------------------------------------------------------------- the caches themselves


def write_packed_cache(path: Path, keys: Sequence[str], packed: np.ndarray,
                       metadata: Mapping[str, Any]) -> None:
    """Write a packed-bit cache: every row is (key, the bit bytes).

    Takes: the path; the list of keys; a uint8 matrix (rows x bytes); the metadata.
    The column is fixed_size_binary, so every row has the same width and a read can take
    the whole matrix out of one contiguous buffer instead of stitching rows together.
    """
    if packed.dtype != np.uint8:
        raise ValueError(f"uint8 is required, got {packed.dtype}")
    if packed.shape[0] != len(keys):
        raise ValueError(f"row mismatch: {packed.shape[0]} vectors and {len(keys)} keys")
    width = int(packed.shape[1])
    table = pa.table(
        {"key": pa.array(list(keys), pa.string()),
         "packed": pa.array([row.tobytes() for row in packed], pa.binary(width))},
        metadata={METADATA_KEY: json.dumps(dict(metadata), ensure_ascii=False).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_packed_cache(path: Path) -> tuple[pa.Array, np.ndarray, dict[str, Any]]:
    """Read a packed-bit cache; returns (keys, uint8 matrix, metadata)."""
    if not path.is_file():
        raise FileNotFoundError(f"no cache at {path} yet; run --mode prepare")
    table = pq.read_table(path)
    values = table.column("packed").combine_chunks()
    width = values.type.byte_width
    # fixed_size_binary: buffers() = [validity, data]; data is the contiguous buffer.
    buffer = values.buffers()[1]
    matrix = np.frombuffer(buffer, dtype=np.uint8, count=len(values) * width)
    raw = table.schema.metadata.get(METADATA_KEY) if table.schema.metadata else None
    metadata = json.loads(raw.decode()) if raw else {}
    return table.column("key").combine_chunks(), matrix.reshape(len(values), width), metadata


def write_vector_cache(path: Path, keys: Sequence[str], vectors: np.ndarray,
                       metadata: Mapping[str, Any]) -> None:
    """Write a float32 vector cache: every row is (key, a fixed-length vector)."""
    if vectors.dtype != np.float32:
        raise ValueError(f"float32 is required, got {vectors.dtype}")
    if vectors.shape[0] != len(keys):
        raise ValueError(f"row mismatch: {vectors.shape[0]} vectors and {len(keys)} keys")
    dimension = int(vectors.shape[1])
    flat = pa.array(np.ascontiguousarray(vectors).reshape(-1), pa.float32())
    table = pa.table(
        {"key": pa.array(list(keys), pa.string()),
         "vector": pa.FixedSizeListArray.from_arrays(flat, dimension)},
        metadata={METADATA_KEY: json.dumps(dict(metadata), ensure_ascii=False).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_vector_cache(path: Path) -> tuple[pa.Array, np.ndarray, dict[str, Any]]:
    """Read a vector cache; returns (keys, float32 matrix, metadata)."""
    if not path.is_file():
        raise FileNotFoundError(f"no cache at {path} yet; run --mode prepare")
    table = pq.read_table(path)
    values = table.column("vector").combine_chunks()
    dimension = values.type.list_size
    matrix = np.asarray(values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    raw = table.schema.metadata.get(METADATA_KEY) if table.schema.metadata else None
    metadata = json.loads(raw.decode()) if raw else {}
    return (table.column("key").combine_chunks(),
            matrix.reshape(len(values), dimension), metadata)


# --------------------------------------------------------------------------- fingerprints


def morgan_generator(radius: int, bits: int, use_chirality: bool):
    """RDKit's Morgan fingerprint generator.

    Uses rdFingerprintGenerator (the current API) rather than AllChem.GetMorganFingerprint
    (which RDKit has deprecated), so the project does not depend on a function that may be
    removed.
    """
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits, includeChirality=use_chirality)


def pack_fingerprints(smiles: Sequence[str], radius: int, bits: int,
                      use_chirality: bool) -> tuple[np.ndarray, list[int]]:
    """The fingerprints of a list of SMILES, with the bits packed.

    Takes: the list of SMILES; the Morgan parameters.
    Returns: (a uint8 matrix of rows x bits/8, the indices of the SMILES RDKit could not
    parse).
    A SMILES that cannot be parsed is NOT dropped: its row stays all zeros and its index
    is returned, for the caller to count and write into the manifest. Dropping it quietly
    would put the cache out of step with the dataset. Stage 2b measured 0 unparseable
    SMILES, so this list has to come back empty; if it does not, something changed and that
    has to be known.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    generator = morgan_generator(radius, bits, use_chirality)
    matrix = np.zeros((len(smiles), bits // 8), dtype=np.uint8)
    failed: list[int] = []
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            failed.append(index)
            continue
        matrix[index] = np.packbits(generator.GetFingerprintAsNumPy(molecule))
    return matrix, failed


def unpack_rows(packed: np.ndarray, rows: np.ndarray, bits: int) -> np.ndarray:
    """Unpack the bits of a few rows into a 0/1 float32 matrix.

    Takes: the packed matrix; the row indices wanted; the fingerprint's bit count.
    Returns: a float32 matrix (rows x bits).
    Only the batch's own rows are unpacked, so the full table stays packed in RAM.
    """
    selected = packed[rows]
    return np.unpackbits(selected, axis=1, count=bits).astype(np.float32)


# --------------------------------------------------------------------------- ESM2


def load_protein_encoder(settings: Mapping[str, Any], device: str = "cpu"):
    """Load the ESM2 tokenizer and model; returns (tokenizer, model, the MEASURED
    embedding dimension).

    Takes: the stage3.protein_encoder config block; the device.
    Returns: (tokenizer, model, dimension).
    The dimension comes from `model.config.hidden_size` of the loaded model itself - the
    one place in the project that knows the real number - and it is written into the
    cache's manifest for every later step to check against. If the config holds an
    `embedding_dim` that differs from the measured one, this raises instead of quietly
    picking one of the two.
    """
    from transformers import AutoTokenizer, EsmModel

    name = settings["model"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name)
    model.to(device)
    dimension = int(model.config.hidden_size)
    declared = settings.get("embedding_dim")
    if declared is not None and int(declared) != dimension:
        raise ValueError(f"the config says embedding_dim = {declared} but {name} really "
                         f"has {dimension}; fix the config rather than ignore this")
    return tokenizer, model, dimension


def sequence_windows(sequence: str, max_residues: int, strategy: str) -> list[str]:
    """Cut a protein sequence into pieces that fit ESM2's window.

    Takes: the sequence; the maximum residues per piece; 'windows' or 'truncate'.
    Returns: the list of pieces.
    'truncate' keeps the first piece only (which finetune needs, because the memory for
    gradients grows with the length); 'windows' keeps the whole sequence as consecutive
    pieces, losing no tail.
    """
    if max_residues < 1:
        raise ValueError(f"max_residues has to be >= 1, got {max_residues}")
    if strategy == "truncate":
        return [sequence[:max_residues]]
    if strategy != "windows":
        raise ValueError(f"unknown long_sequence: {strategy!r}; only 'windows' or "
                         f"'truncate'")
    return [sequence[start:start + max_residues]
            for start in range(0, max(len(sequence), 1), max_residues)]


def embed_one_sequence(sequence: str, tokenizer, model, settings: Mapping[str, Any],
                       device: str) -> np.ndarray:
    """One protein's vector: mean-pooled over residues, special tokens left out.

    A sequence longer than the window is cut into pieces and averaged WEIGHTED by each
    piece's length, so the result equals a mean-pool over the whole sequence if the model
    has no edge effects - a more honest summary than the unweighted mean of pieces that
    differ in length.
    """
    import torch

    windows = sequence_windows(sequence, int(settings["max_residues"]),
                               str(settings["long_sequence"]))
    totals = None
    weight_sum = 0.0
    with torch.no_grad():
        for window in windows:
            encoded = tokenizer(window, return_tensors="pt", add_special_tokens=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[0]
            # Drop the first (CLS) and last (EOS) token: keep only the residues.
            residues = hidden[1:-1]
            if residues.shape[0] == 0:
                continue
            pooled = residues.mean(dim=0).to("cpu").numpy().astype(np.float64)
            weight = float(residues.shape[0])
            totals = pooled * weight if totals is None else totals + pooled * weight
            weight_sum += weight
    if totals is None or weight_sum == 0.0:
        raise ValueError(f"no embedding could be taken for a sequence of length "
                         f"{len(sequence)}")
    return (totals / weight_sum).astype(np.float32)


def embed_proteins(sequences: Mapping[str, str], settings: Mapping[str, Any], device: str,
                   progress_every: int = 500) -> tuple[list[str], np.ndarray, int, int]:
    """Embeddings for many proteins.

    Takes: {protein hash: sequence}; the protein_encoder config; the device; how often to
    log progress.
    Returns: (the hashes in order, a float32 matrix, the measured dimension, how many
    sequences had to be cut into pieces or truncated).
    The hashes are sorted so that building the cache twice gives the same file.
    """
    import logging

    tokenizer, model, dimension = load_protein_encoder(settings, device)
    model.eval()
    keys = sorted(sequences)
    matrix = np.zeros((len(keys), dimension), dtype=np.float32)
    long_sequences = 0
    limit = int(settings["max_residues"])
    for position, key in enumerate(keys):
        sequence = sequences[key]
        if len(sequence) > limit:
            long_sequences += 1
        matrix[position] = embed_one_sequence(sequence, tokenizer, model, settings, device)
        if progress_every and (position + 1) % progress_every == 0:
            logging.info("embedded %s/%s proteins", position + 1, len(keys))
    return keys, matrix, dimension, long_sequences


# --------------------------------------------------- molecule embeddings (MoLFormer)


def encoder_revision(settings: Mapping[str, Any], name: str) -> str | None:
    """The revision (HuggingFace commit or tag) to load for model `name`; None = latest.

    Takes: the stage3.molecule_encoder config block; the model name.
    Returns: the revision string, or None.
    Pinned PER MODEL, because every repository has its own history: one shared revision
    would be wrong the moment another model from allowed_models is used.
    """
    revisions = settings.get("revisions") or {}
    value = revisions.get(name)
    return None if value is None else str(value)


def load_molecule_encoder(settings: Mapping[str, Any], device: str = "cpu"):
    """Load the SMILES encoder's tokenizer and model; returns (tokenizer, model, the
    MEASURED dimension).

    Takes: the stage3.molecule_encoder config block; the device.
    Returns: (tokenizer, model, dimension).
    Same rule as load_protein_encoder: the dimension comes from the loaded model and not
    from the config, and a config that disagrees raises. `trust_remote_code` is required
    for MoLFormer because that model ships its own code on HuggingFace; that is why the
    model name has to appear in the config's allowed list instead of being taken from the
    command line. The downloaded code is PINNED by the config (see revisions_note): the
    repository's owner has changed that code, and the newer version needs transformers v5.
    """
    from transformers import AutoModel, AutoTokenizer

    name = str(settings["model"])
    allowed = tuple(settings.get("allowed_models", ()))
    if allowed and name not in allowed:
        raise ValueError(f"the molecule encoder {name!r} is not in allowed_models "
                         f"{allowed}; the model's own code is downloaded and executed, so "
                         f"it has to be declared in the config beforehand")
    trust = bool(settings.get("trust_remote_code", False))
    revision = encoder_revision(settings, name)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust,
                                              revision=revision)
    model = AutoModel.from_pretrained(name, trust_remote_code=trust,
                                      revision=revision, deterministic_eval=True)
    model.to(device)
    dimension = int(model.config.hidden_size)
    declared = settings.get("embedding_dim")
    if declared is not None and int(declared) != dimension:
        raise ValueError(f"the config says embedding_dim = {declared} but {name} really "
                         f"has {dimension}; fix the config rather than ignore this")
    return tokenizer, model, dimension


def pool_tokens(outputs: Any, mask: Any, pooling: str) -> Any:
    """Pool a batch's tokens into one vector per molecule.

    Takes: the model's output; the attention mask; 'pooler' or 'mean'.
    Returns: a tensor (batch, dimension).
    'pooler' uses pooler_output, as MoLFormer's model card instructs. 'mean' averages the
    REAL tokens itself (padding excluded), so a model without a pooler can be used too.
    """
    if pooling == "pooler":
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise ValueError("this model has no pooler_output; set pooling='mean'")
        return pooled
    if pooling != "mean":
        raise ValueError(f"unknown pooling: {pooling!r}; only 'pooler' or 'mean'")
    hidden = outputs.last_hidden_state
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def embed_molecules(smiles: Sequence[str], settings: Mapping[str, Any], device: str,
                    progress_every: int = 20_000
                    ) -> tuple[list[str], np.ndarray, int, int]:
    """Embeddings for many SMILES, computed in batches.

    Takes: the list of SMILES; the stage3.molecule_encoder config; the device; how often to
    log progress.
    Returns: (the sorted SMILES, a float32 matrix, the measured dimension, how many SMILES
    were truncated at the token ceiling).
    Unlike embed_proteins this runs in BATCHES: a molecule is hundreds of times shorter
    than a protein, so one at a time would waste the GPU. The keys are sorted so that
    building the cache twice gives the same file.
    A SMILES that is too long is truncated by the tokenizer at max_tokens; every truncation
    is counted and logged, because truncating a SMILES loses atoms rather than a less
    important tail as it does for a protein.
    """
    import logging

    import torch

    tokenizer, model, dimension = load_molecule_encoder(settings, device)
    model.eval()
    keys = sorted(set(smiles))
    matrix = np.zeros((len(keys), dimension), dtype=np.float32)
    batch_size = int(settings["batch_size"])
    max_tokens = int(settings["max_tokens"])
    pooling = str(settings.get("pooling", "pooler"))
    truncated = 0
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            chunk = keys[start:start + batch_size]
            encoded = tokenizer(chunk, padding=True, truncation=True,
                                max_length=max_tokens, return_tensors="pt")
            truncated += sum(1 for row in encoded["attention_mask"]
                             if int(row.sum()) >= max_tokens)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = pool_tokens(outputs, encoded["attention_mask"], pooling)
            matrix[start:start + len(chunk)] = pooled.to("cpu").numpy()
            done = start + len(chunk)
            if progress_every and done % progress_every < batch_size:
                logging.info("embedded %s/%s molecules", f"{done:,}", f"{len(keys):,}")
    if truncated:
        logging.warning("%s SMILES hit the ceiling of %s tokens and were truncated",
                        f"{truncated:,}", max_tokens)
    return keys, matrix, dimension, truncated
