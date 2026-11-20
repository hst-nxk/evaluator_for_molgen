"""Stage 3: read the evaluator dataset, filter by partition and by slice, build batches.

The input dataset (stage3.inputs.evaluator in the config) is READ ONLY, never written.

This module imports no torch: it returns numpy arrays of INDICES into the cache tables
(molecule fingerprints, substituent fingerprints, protein embeddings) rather than vectors.
A batch is therefore just an array of integers, and looking the vectors up is train.py's
job.

The `test` partition is blocked in code: every reading function goes through
`check_partition`, and the only way to read test is to pass allow_test=True - which nothing
in the project does.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

# The columns to read from the dataset. The Stage 2b names are kept exactly as they are,
# so a number can always be traced back.
PAIR_COLUMNS = ["pair_id", "endpoint", "protein_hash", "partition", "smiles_a", "smiles_b",
                "variable_a", "variable_b", "delta_p_mean", "n_assays", "n_documents",
                "n_distinct_values", "qc_status", "unseen_pair", "plausible_range",
                "source_quality_warning"]

# The slice keys the config may use; an unknown key is a configuration error and is never
# ignored in silence.
SLICE_KEYS = ("unseen_pair", "plausible_range", "no_source_warning", "min_n_assays",
              "min_n_documents", "min_distinct_values", "abs_delta_p_above",
              "abs_delta_p_at_most")
# The two slice keys on the size of |Δp|, which separate the activity-cliff cases. The
# definition of a cliff comes from MoleculeACE (van Tilborg, Alenicheva, Grisoni 2022,
# J. Chem. Inf. Model. 62(23): 5938-5951, DOI 10.1021/acs.jcim.2c01073; source file
# benchmark/cliffs.py): a pair is a cliff when the two molecules are similar enough AND
# their activity differs by more than tenfold. Every pair in this dataset is an MMP pair
# (one scaffold, exactly one substituent apart), so the similarity condition holds by
# construction - the mmp=True branch of find_cliffs. What is left is the activity
# condition: more than tenfold on a concentration scale = |Δp| > 1. That paper's
# `find_cliffs` uses a STRICT >, so abs_delta_p_above is strict too, and
# abs_delta_p_at_most uses <= so that the two slices cover everything without overlapping.
ABSOLUTE_DELTA_KEYS = ("abs_delta_p_above", "abs_delta_p_at_most")
# The "at least N" slice keys -> the integer dataset column each one compares against.
MINIMUM_KEYS = {"min_n_assays": "n_assays", "min_n_documents": "n_documents",
                "min_distinct_values": "n_distinct_values"}

# The key and value columns of a partition override table (see load_partition_override).
OVERRIDE_COLUMNS = ("protein_hash", "partition")
# The columns of a PAIR-keyed partition override table (see
# load_pair_partition_override). Two things differ from the protein-keyed table: the key is
# pair_id, and it overwrites the unseen_pair column as well.
PAIR_OVERRIDE_COLUMNS = ("pair_id", "partition", "unseen_pair")
# Pair-keyed override tables already read, by absolute path. Cached for the same reason.
_PAIR_OVERRIDE_CACHE: dict[Path, tuple[pa.Array, pa.Array, pa.Array]] = {}
# The key column of a train-subset table (see load_train_subset). Just one: pair_id.
SUBSET_COLUMN = "pair_id"
# Subset tables already read, by absolute path; cached for the same reason as
# _OVERRIDE_CACHE.
_SUBSET_CACHE: dict[Path, pa.Array] = {}
# Override tables already read, by absolute path. Cached because read_columns is called
# again and again within one run (train, validation, every slice) and this table is only a
# few thousand rows but lives on disk; reading it every time is waste.
_OVERRIDE_CACHE: dict[Path, tuple[pa.Array, pa.Array]] = {}


def load_partition_override(config: Mapping[str, Any], root: Path
                            ) -> tuple[pa.Array, pa.Array] | None:
    """Read the partition override table if the config declares one; None if it does not.

    Takes: the config, the project root. Returns: (protein_hash array, partition array) or
    None.

    WHY THIS PATH EXISTS: the experiment that changes the protein-similarity threshold used
    to split the data only changes WHICH PARTITION each protein belongs to. The dataset, the
    molecule pairs, the labels, the fingerprint cache and the embedding cache all stay the
    same (the caches are keyed by molecule and by protein, not by partition), so rebuilding
    the dataset just to change one column would be both expensive and one more version of
    the data to keep track of.
    """
    setting = str(config["stage3"]["inputs"].get("partition_override") or "").strip()
    if not setting:
        return None
    path = (root / setting).resolve()
    cached = _OVERRIDE_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"no partition override table at {path} yet; build one "
                                f"with scripts/build_identity_split.py build")
    table = pq.read_table(path, columns=list(OVERRIDE_COLUMNS))
    keys = table.column("protein_hash").combine_chunks()
    values = table.column("partition").combine_chunks()
    if isinstance(keys, pa.ChunkedArray):
        keys = keys.chunk(0) if keys.num_chunks == 1 else pa.concat_arrays(keys.chunks)
    if isinstance(values, pa.ChunkedArray):
        values = values.chunk(0) if values.num_chunks == 1 else pa.concat_arrays(values.chunks)
    if len(set(keys.to_pylist())) != len(keys):
        raise ValueError(f"{path} holds a repeated protein_hash; every protein gets "
                         f"exactly one partition")
    _OVERRIDE_CACHE[path] = (keys, values)
    return keys, values


def _flat(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """One parquet column as ONE contiguous array, which pc.take and pc.index_in need."""
    column = column.combine_chunks()
    if isinstance(column, pa.ChunkedArray):
        return column.chunk(0) if column.num_chunks == 1 else pa.concat_arrays(column.chunks)
    return column


def load_pair_partition_override(config: Mapping[str, Any], root: Path
                                 ) -> tuple[pa.Array, pa.Array, pa.Array] | None:
    """Read the PAIR-keyed partition override table if the config declares one, else None.

    Takes: the config, the project root. Returns: (pair_id array, partition array,
    unseen_pair array).

    WHY THIS PATH EXISTS (task 13 of the 2026-11-09 meeting): the minutes asked for the
    OPPOSITE split - let protein P appear on both sides, but forbid a validation pair
    (A, B) from also appearing in train. The existing `partition_override` table is keyed
    by protein_hash and cannot express that: all it can say is which partition a protein
    belongs to. This table is keyed by pair_id, so the split happens at pair level and a
    protein may appear on both sides.

    It may not be combined with the other two paths: stacking two override tables makes the
    result depend on the order they are applied in, and the train-subset table was built
    under the OLD split, so under the new one it no longer matches. Better an error than a
    split nobody can describe.
    """
    setting = str(config["stage3"]["inputs"].get("partition_override_pairs") or "").strip()
    if not setting:
        return None
    for other, name in ((config["stage3"]["inputs"].get("partition_override"),
                         "partition_override"),
                        (config["stage3"]["inputs"].get("train_subset"), "train_subset")):
        if str(other or "").strip():
            raise ValueError(f"partition_override_pairs (pair-keyed) and {name} cannot "
                             f"both be on: the two tables overwrite each other, so the "
                             f"split would depend on the order they are applied in")
    path = (root / setting).resolve()
    cached = _PAIR_OVERRIDE_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"no pair-keyed partition override table at {path} yet; "
                                f"build one with scripts/build_pair_split.py")
    table = pq.read_table(path, columns=list(PAIR_OVERRIDE_COLUMNS))
    keys = _flat(table.column("pair_id"))
    values = _flat(table.column("partition"))
    unseen = _flat(table.column("unseen_pair"))
    if len(set(keys.to_pylist())) != len(keys):
        raise ValueError(f"{path} holds a repeated pair_id; every pair gets exactly one "
                         f"partition")
    _PAIR_OVERRIDE_CACHE[path] = (keys, values, unseen)
    return keys, values, unseen


def apply_pair_partition_override(table: pa.Table,
                                  override: tuple[pa.Array, pa.Array, pa.Array]
                                  ) -> pa.Table:
    """Replace 'partition' AND 'unseen_pair' by pair_id; a pair not in the table -> 'test'.

    Two things differ sharply from the protein-keyed table, both on purpose:

    1. A MISSING KEY IS NOT AN ERROR. The protein-keyed table has to cover the whole
       dataset, and a missing protein means it was built from a different export. The
       pair-keyed table lists only the pairs that belong to the experiment, and every pair
       it does not list is pushed to 'test' - that is, out of both train and validation,
       rather than LEFT under the old split. That way half the data cannot end up split one
       way and half the other without anyone noticing.
    2. IT ALSO OVERWRITES unseen_pair. That dataset column was computed under the OLD split
       ("did this pair appear in the OLD train?"), so under the new split it is wrong. Stage
       3's `strict` slice reads exactly that column, so not overwriting it would mean every
       strict number was measured over the wrong set of rows.
    """
    keys, values, unseen = override
    position = pc.index_in(table.column("pair_id"), value_set=keys)
    found = pc.is_valid(position)
    partition = pc.if_else(found, pc.take(values, position), pa.scalar(TEST))
    table = table.set_column(table.schema.get_field_index("partition"),
                             "partition", partition)
    if "unseen_pair" in table.column_names:
        flag = pc.if_else(found, pc.take(unseen, position), pa.scalar(False))
        table = table.set_column(table.schema.get_field_index("unseen_pair"),
                                 "unseen_pair", flag)
    return table


def apply_partition_override(table: pa.Table, override: tuple[pa.Array, pa.Array]
                             ) -> pa.Table:
    """Replace the table's 'partition' column with the partition looked up by
    'protein_hash'.

    A protein missing from the override table IS an error and is not skipped: a missing
    protein means the table was built from a different export, and then part of the data
    would follow the old split and part the new one - a wrong result with no warning.
    """
    keys, values = override
    position = pc.index_in(table.column("protein_hash"), value_set=keys)
    missing = pc.sum(pc.is_null(position)).as_py() or 0
    if missing:
        unknown = pc.filter(table.column("protein_hash"), pc.is_null(position))
        raise ValueError(f"{missing} rows hold a protein that is not in the partition "
                         f"override table, for example {unknown.to_pylist()[:3]}")
    replaced = pc.take(values, position)
    return table.set_column(table.schema.get_field_index("partition"), "partition", replaced)


def load_train_subset(config: Mapping[str, Any], root: Path) -> pa.Array | None:
    """Read the train subset's pair_id table if the config declares one, else None.

    Takes: the config, the project root. Returns: the array of pair_ids to keep, or None.

    WHY THIS PATH EXISTS (2026-11-09 meeting, task 2): the request was to train on a small
    subset (~100,000 rows drawn from the 200 Context P groups with the most data) instead of
    6.6 million rows, so that one experiment takes 30 minutes rather than 2-3 hours. Like
    the override table, a subset does NOT rebuild the dataset: it is only a list of rows to
    keep, and both caches (fingerprints, embeddings) are shared as they are, because they
    are keyed by molecule and by protein and not by row.

    IT APPLIES TO THE TRAIN PARTITION ONLY. Validation has to stay as it is or no earlier
    number can be compared against - exactly as settled on line 99 of the minutes ("leave
    the current validation set as it is").
    """
    setting = str(config["stage3"]["inputs"].get("train_subset") or "").strip()
    if not setting:
        return None
    path = (root / setting).resolve()
    cached = _SUBSET_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"no train subset table at {path} yet; build one with "
                                f"scripts/build_train_subset.py")
    table = pq.read_table(path, columns=[SUBSET_COLUMN])
    keys = table.column(SUBSET_COLUMN).combine_chunks()
    if isinstance(keys, pa.ChunkedArray):
        keys = keys.chunk(0) if keys.num_chunks == 1 else pa.concat_arrays(keys.chunks)
    if not len(keys):
        raise ValueError(f"{path} is empty; a subset has to hold at least one row")
    if len(set(keys.to_pylist())) != len(keys):
        raise ValueError(f"{path} holds a repeated pair_id; every row appears once")
    _SUBSET_CACHE[path] = keys
    return keys


@dataclass(frozen=True)
class PairArrays:
    """One partition encoded as indices, ready for the training loop.

    molecule_a / molecule_b : row index into the molecule fingerprint table
    fragment_a / fragment_b : row index into the substituent fingerprint table
    protein                 : row index into the protein embedding table
    endpoint                : endpoint code, in the order of config["endpoints"]
    target                  : the true Δp (the label)
    slices                  : {slice name: boolean mask}, for reporting metrics per slice
    """

    molecule_a: np.ndarray
    molecule_b: np.ndarray
    fragment_a: np.ndarray
    fragment_b: np.ndarray
    protein: np.ndarray
    endpoint: np.ndarray
    target: np.ndarray
    slices: Mapping[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.target.size)


def check_partition(partition: str, allow_test: bool = False) -> str:
    """Block any reading of the test partition.

    WHY IT IS BLOCKED IN CODE: test is used once, at the end, for the final report. Every
    choice of model, of hyper-parameter and every early stop uses validation only. A
    reminder in the documentation is easier to overlook than a ValueError.
    """
    if partition == TEST and not allow_test:
        raise ValueError("the 'test' partition may not be used for training, for choosing "
                         "hyper-parameters or for early stopping; only 'train' and "
                         "'validation'")
    if partition not in (TRAIN, VALIDATION, TEST):
        raise ValueError(f"unknown partition: {partition!r}")
    return partition


def slice_mask(table: pa.Table, filters: Mapping[str, Any]) -> np.ndarray:
    """The boolean mask of one slice.

    Takes: the pair table; the slice's conditions (see SLICE_KEYS).
    Returns: a boolean array with one entry per table row.
    A slice with no conditions gives an all-True mask, meaning "the whole partition".
    """
    mask = np.ones(table.num_rows, dtype=bool)
    for key, wanted in filters.items():
        if key not in SLICE_KEYS:
            raise ValueError(f"unknown slice key: {key!r}; allowed: {SLICE_KEYS}")
        if key in ABSOLUTE_DELTA_KEYS:
            values = np.abs(np.asarray(
                table.column("delta_p_mean").to_numpy(zero_copy_only=False),
                dtype=np.float64))
            mask &= (values > float(wanted) if key == "abs_delta_p_above"
                     else values <= float(wanted))
        elif key == "no_source_warning":
            column = table.column("source_quality_warning")
            found = np.asarray(pc.is_null(column).to_numpy(zero_copy_only=False), dtype=bool)
            mask &= found if wanted else ~found
        elif key in MINIMUM_KEYS:
            values = np.asarray(table.column(MINIMUM_KEYS[key]).to_numpy(zero_copy_only=False))
            mask &= values >= int(wanted)
        else:
            values = np.asarray(
                pc.fill_null(table.column(key), False).to_numpy(zero_copy_only=False),
                dtype=bool)
            mask &= values if wanted else ~values
    return mask


def read_columns(config: Mapping[str, Any], root: Path, partition: str,
                 columns: Sequence[str], limit: int | None = None,
                 allow_test: bool = False) -> pa.Table:
    """Read one partition, but only the columns asked for.

    Takes: the config, the project root, the partition name, the column list, a row cap,
    allow_test.
    Returns: an Arrow table of exactly those columns (always with 'partition', so it can be
    filtered).
    This exists as its own function because the trivial baseline needs only two columns
    (endpoint and label) over 8.4 million train rows; reading every column for that is
    waste.
    `limit` takes the FIRST rows after the partition filter, not a random sample: the
    dataset is written in (endpoint, protein bucket) order, so the opening rows still hold
    enough proteins and enough pairs to exercise the pipeline, and the result is fixed from
    one smoke run to the next.

    This is the ONLY place in Stage 3 that filters by partition, which is why applying the
    config's override table and train subset here is enough for every reading path (train,
    eval, errors, features) to follow the same split and the same subset.
    """
    check_partition(partition, allow_test)
    path = root / config["stage3"]["inputs"]["evaluator"]
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist yet; run Stage 2b --mode export "
                                f"first")
    # The PAIR-keyed table is read first: it carries the "these cannot both be on" check,
    # so reading it second would mean a protein table with a mistyped path raises a
    # missing-file error before anyone learns that two splits were switched on at once.
    pair_override = load_pair_partition_override(config, root)
    override = load_partition_override(config, root)
    # The subset applies to train ONLY (see load_train_subset): skipped for validation and
    # test.
    subset = load_train_subset(config, root) if partition == TRAIN else None
    wanted = list(columns)
    if "partition" not in wanted:
        wanted.append("partition")
    extra: list[str] = []
    if override is not None and "protein_hash" not in wanted:
        extra.append("protein_hash")
    if pair_override is not None and "pair_id" not in wanted:
        extra.append("pair_id")
    if subset is not None and SUBSET_COLUMN not in wanted:
        extra.append(SUBSET_COLUMN)
    wanted.extend(extra)
    handle = pq.ParquetFile(path)
    chunks: list[pa.Table] = []
    rows = 0
    for batch in handle.iter_batches(batch_size=200_000, columns=wanted):
        table = pa.Table.from_batches([batch])
        if override is not None:
            table = apply_partition_override(table, override)
        if pair_override is not None:
            table = apply_pair_partition_override(table, pair_override)
        table = table.filter(pc.equal(table.column("partition"), partition))
        if subset is not None and table.num_rows:
            table = table.filter(pc.is_in(table.column(SUBSET_COLUMN), value_set=subset))
        if not table.num_rows:
            continue
        if limit is not None and rows + table.num_rows > limit:
            table = table.slice(0, limit - rows)
        chunks.append(table)
        rows += table.num_rows
        if limit is not None and rows >= limit:
            break
    if not chunks:
        raise ValueError(f"partition {partition!r} has no rows at all in {path}"
                         + ("; the train subset table may have been built from a different "
                            "export" if subset is not None else ""))
    result = pa.concat_tables(chunks)
    if extra:
        result = result.drop_columns(extra)
    if subset is not None and rows != len(subset) and limit is None:
        raise ValueError(f"the train subset table holds {len(subset):,} pair_ids but only "
                         f"{rows:,} rows of the train partition matched; either the subset "
                         f"was built from a different export, or it is being used together "
                         f"with a partition_override other than the one it was built under")
    return result


def read_partition(config: Mapping[str, Any], root: Path, partition: str,
                   limit: int | None = None, allow_test: bool = False) -> pa.Table:
    """Read one partition of the evaluator dataset, with all of PAIR_COLUMNS."""
    return read_columns(config, root, partition, PAIR_COLUMNS, limit, allow_test)


def code_column(values: pa.ChunkedArray | pa.Array, vocabulary: pa.Array,
                label: str) -> np.ndarray:
    """The index of every value in `vocabulary`; a missing value is an error.

    WHY A MISSING VALUE IS NOT SKIPPED: missing means the cache does not match the dataset
    (built from an earlier export, for example), and training on data that matches halfway
    gives a wrong result with no warning.
    """
    position = pc.index_in(values, value_set=vocabulary)
    missing = pc.sum(pc.is_null(position)).as_py() or 0
    if missing:
        examples = pc.filter(values, pc.is_null(position)).to_pylist()[:3]
        raise ValueError(f"{missing} {label} values are not in the cache, for example "
                         f"{examples}; rebuild the cache with --mode prepare")
    return np.asarray(position.to_numpy(zero_copy_only=False), dtype=np.int64)


def encode_pairs(table: pa.Table, molecules: pa.Array, fragments: pa.Array,
                 proteins: pa.Array, endpoints: Sequence[str],
                 slices: Sequence[Mapping[str, Any]]) -> PairArrays:
    """Encode a pair table into indices and labels.

    Takes: the pair table; the cache's three vocabularies (molecule SMILES, substituent
    SMILES, protein hashes); the endpoints in order; the slice definitions.
    Returns: a PairArrays.
    """
    endpoint_vocabulary = pa.array(list(endpoints), pa.string())
    masks = {}
    for definition in slices:
        name = definition["name"]
        masks[name] = slice_mask(table, definition.get("filters", {}))
    return PairArrays(
        molecule_a=code_column(table.column("smiles_a"), molecules, "smiles_a"),
        molecule_b=code_column(table.column("smiles_b"), molecules, "smiles_b"),
        fragment_a=code_column(table.column("variable_a"), fragments, "variable_a"),
        fragment_b=code_column(table.column("variable_b"), fragments, "variable_b"),
        protein=code_column(table.column("protein_hash"), proteins, "protein_hash"),
        endpoint=code_column(table.column("endpoint"), endpoint_vocabulary, "endpoint"),
        target=np.asarray(table.column("delta_p_mean").to_numpy(zero_copy_only=False),
                          dtype=np.float32),
        slices=masks,
    )


def query_key_columns(table: pa.Table, molecules: pa.Array, proteins: pa.Array,
                      endpoints: Sequence[str]) -> dict[str, np.ndarray]:
    """The key columns the ranking metrics build their queries from.

    Returns: {'molecule_a', 'molecule_b', 'protein', 'endpoint'}, encoded as integers.
    Doubling the rows into both directions is metrics.expand_directions' job, not this
    one's.
    """
    endpoint_vocabulary = pa.array(list(endpoints), pa.string())
    return {
        "molecule_a": code_column(table.column("smiles_a"), molecules, "smiles_a"),
        "molecule_b": code_column(table.column("smiles_b"), molecules, "smiles_b"),
        "protein": code_column(table.column("protein_hash"), proteins, "protein_hash"),
        "endpoint": code_column(table.column("endpoint"), endpoint_vocabulary, "endpoint"),
    }


def take_rows(arrays: PairArrays, rows: np.ndarray) -> PairArrays:
    """Take a subset of a PairArrays' rows, carrying the matching slice masks along."""
    return PairArrays(
        molecule_a=arrays.molecule_a[rows], molecule_b=arrays.molecule_b[rows],
        fragment_a=arrays.fragment_a[rows], fragment_b=arrays.fragment_b[rows],
        protein=arrays.protein[rows], endpoint=arrays.endpoint[rows],
        target=arrays.target[rows],
        slices={name: mask[rows] for name, mask in arrays.slices.items()},
    )


def permute_protein_rows(arrays: PairArrays, seed: int) -> PairArrays:
    """Shuffle the protein column ACROSS ROWS; every other column stays as it is.

    Takes: the arrays of ONE partition; the shuffle seed.
    Returns: a new PairArrays whose ``protein`` column is a permutation of the old one.

    WHY THIS EXISTS — assigned at the 2026-11-18 meeting. The question raised there was how
    the project can claim the protein branch is unused when the only evidence is "removing
    the branch barely changes the result". Removing the branch CHANGES the model (fewer
    parameters, fewer inputs), so that test cannot separate "the model does not use the
    protein" from "the model compensates for what is missing". The right control keeps the
    model, the parameter count and the distribution of protein embeddings all identical and
    only DESTROYS the link between a row and its own protein. If the result still does not
    move, the protein branch really carries no information into the prediction; if it drops,
    it does.

    Because this is a permutation of the very same column, the multiset of proteins is
    UNCHANGED: every protein still appears in exactly as many rows as before, so the result
    cannot move merely because the protein distribution moved. Only the row -> protein
    mapping is broken.

    TWO THINGS TO KNOW WHEN READING THE RESULTS:

    1. This function does NOT touch the query key columns (`query_key_columns` reads them
       straight from the Arrow table). Ranking queries are still grouped by the REAL
       protein, so the denominator and the grouping match the control run exactly and the
       two runs are comparable.
    2. The shuffle has to be applied to BOTH train and validation, with its own permutation
       for each. Shuffling train alone measures "noisy train, clean test" — a completely
       different experiment.
    """
    rows = len(arrays)
    if rows == 0:
        return arrays
    order = np.random.default_rng(seed).permutation(rows)
    return PairArrays(
        molecule_a=arrays.molecule_a, molecule_b=arrays.molecule_b,
        fragment_a=arrays.fragment_a, fragment_b=arrays.fragment_b,
        protein=arrays.protein[order], endpoint=arrays.endpoint,
        target=arrays.target, slices=arrays.slices,
    )


def swap_sides(arrays: PairArrays) -> PairArrays:
    """The same data with the A and B sides swapped; the label changes sign.

    TWO uses, both legitimate:

    1. MEASURING how asymmetric the `direct` head is: a perfectly antisymmetric model has
       f(A, B) = -f(B, A). `twin` and `antisymmetric` are antisymmetric by construction,
       `direct` is not (see docs/STAGE3_EVALUATOR.md).
    2. AUGMENTING the train data (2026-11-09 meeting, task 2), switched on with
       stage3.train.augment_reverse. Before 2026-11-09 this docstring said "never used for
       training", and that was wrong: the pair (B, A) with the label -Δp is not invented
       data, it is the same measurement read in the other direction. The dataset stores one
       direction only because Stage 2b fixes the direction by row index (see
       molopt.dataset.mmp), which means the stored direction is ARBITRARY with respect to
       the sign of the label - so the reverse direction is information the project already
       has and simply had not written down.

    TRAIN ONLY. Augmenting validation would double its denominator and make every earlier
    number incomparable.
    """
    return PairArrays(
        molecule_a=arrays.molecule_b, molecule_b=arrays.molecule_a,
        fragment_a=arrays.fragment_b, fragment_b=arrays.fragment_a,
        protein=arrays.protein, endpoint=arrays.endpoint,
        target=-arrays.target, slices=arrays.slices,
    )


def concat_arrays(first: PairArrays, second: PairArrays) -> PairArrays:
    """Join two PairArrays into one; they must carry the same slice names."""
    if set(first.slices) != set(second.slices):
        raise ValueError(f"the two arrays carry different slices: {sorted(first.slices)} "
                         f"and {sorted(second.slices)}")
    return PairArrays(
        molecule_a=np.concatenate([first.molecule_a, second.molecule_a]),
        molecule_b=np.concatenate([first.molecule_b, second.molecule_b]),
        fragment_a=np.concatenate([first.fragment_a, second.fragment_a]),
        fragment_b=np.concatenate([first.fragment_b, second.fragment_b]),
        protein=np.concatenate([first.protein, second.protein]),
        endpoint=np.concatenate([first.endpoint, second.endpoint]),
        target=np.concatenate([first.target, second.target]),
        slices={name: np.concatenate([mask, second.slices[name]])
                for name, mask in first.slices.items()},
    )


def augment_reverse(arrays: PairArrays, keys: Mapping[str, np.ndarray]
                    ) -> tuple[PairArrays, dict[str, np.ndarray]]:
    """Append the reverse direction (B, A, F_B, F_A, P) -> -Δp after the original rows.

    Takes: the arrays of the TRAIN partition; the query key columns of that same partition.
    Returns: (the doubled arrays, the doubled keys) - exactly twice as many rows.

    The query keys have to be swapped the same way, or the ranking metrics would gather the
    augmented rows into the forward direction's queries and every per-query number would be
    meaningless.
    """
    doubled = concat_arrays(arrays, swap_sides(arrays))
    swapped = {
        "molecule_a": np.concatenate([keys["molecule_a"], keys["molecule_b"]]),
        "molecule_b": np.concatenate([keys["molecule_b"], keys["molecule_a"]]),
        "protein": np.concatenate([keys["protein"]] * 2),
        "endpoint": np.concatenate([keys["endpoint"]] * 2),
    }
    return doubled, swapped


def identity_sides(arrays: PairArrays) -> PairArrays:
    """The pair (X, X): the B side is replaced by the A side; the label is exactly 0.

    Used to measure whether the model gives Δp = 0 when the "edit" edits nothing. Such a
    pair does NOT exist in the dataset (an MMP pair always holds two different molecules),
    so this measures extrapolation outside the training distribution and says nothing about
    the quality of the fit. It is an artificial input and is never used for training.
    """
    return PairArrays(
        molecule_a=arrays.molecule_a, molecule_b=arrays.molecule_a,
        fragment_a=arrays.fragment_a, fragment_b=arrays.fragment_a,
        protein=arrays.protein, endpoint=arrays.endpoint,
        target=np.zeros_like(arrays.target), slices=arrays.slices,
    )


def iter_batches(count: int, batch_size: int, rng: np.random.Generator | None = None
                 ) -> Iterator[np.ndarray]:
    """Cut 0..count-1 into batches of indices; shuffled first when an rng is given.

    The last, shorter batch is used and not dropped, so no row is lost in an epoch.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size has to be >= 1, got {batch_size}")
    indices = np.arange(count, dtype=np.int64)
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start:start + batch_size]


def iter_protein_batches(proteins: np.ndarray, batch_size: int,
                         rng: np.random.Generator | None = None,
                         max_proteins: int = 0) -> Iterator[np.ndarray]:
    """Batch so that EVERY BATCH TOUCHES FEW PROTEINS, while still using every row once.

    WHY IT IS NEEDED: in the finetune configuration ESM2 is part of the computation graph
    and runs once per DISTINCT protein in the batch, keeping its activations for the
    backward pass. With random batches, 64 rows reach about 54 different proteins (measured
    on train v4), which is 54 ESM2 passes - an estimated ~68 GB, beyond every card here.
    Grouping by protein leaves most batches with one or two proteins, so the memory drops by
    roughly an order of magnitude while THE NUMBER OF STEPS PER EPOCH DOES NOT CHANGE.

    How: shuffle the rows within each protein, shuffle the order of the proteins, join them
    and cut the result into consecutive batches. Every row appears exactly once per epoch,
    as in iter_batches.

    THE TRADE-OFF, which belongs in the paper: the rows of one batch are no longer
    independent (mostly one protein), so each step's gradient is noisier than it would be
    with random sampling. That is why this function is used for finetune ONLY, while frozen
    keeps random sampling: with frozen the protein vectors are precomputed, so there is
    nothing to save.

    With `max_proteins` > 0 a batch NEVER holds more than that many distinct proteins: on
    reaching the limit the batch is cut short, even with fewer than `batch_size` rows. Most
    batches hold a single protein so the cut is rare, but it turns the memory ceiling from a
    STATISTICAL bound into a CERTAIN one - which matters when the GPU is shared with someone
    else, because one rare batch touching many proteins is enough to exhaust the memory of
    both.

    Takes: `proteins`, the protein index of every row (PairArrays.protein); the batch size;
    an rng (None = no shuffling, so a test can repeat itself); the ceiling on proteins per
    batch (0 = no ceiling).
    Returns: the row-index arrays, one at a time.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size has to be >= 1, got {batch_size}")
    order = np.argsort(proteins, kind="stable")
    sorted_proteins = proteins[order]
    # Where one protein ends and the next begins in the sorted array.
    cuts = np.flatnonzero(np.diff(sorted_proteins)) + 1
    groups = np.split(order, cuts)
    if rng is not None:
        for group in groups:
            rng.shuffle(group)
        groups = [groups[i] for i in rng.permutation(len(groups))]
    if max_proteins < 1:
        flat = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
        for start in range(0, flat.size, batch_size):
            yield flat[start:start + batch_size]
        return
    held: list[np.ndarray] = []
    rows = 0
    seen = 0
    for group in groups:
        offset = 0
        # Has this protein been counted into the batch being built? A large protein can
        # span several batches, and it has to be counted once in EVERY one of them.
        counted = False
        while offset < group.size:
            if not counted and seen >= max_proteins:
                yield np.concatenate(held)
                held, rows, seen = [], 0, 0
            if not counted:
                seen += 1
                counted = True
            take = min(batch_size - rows, group.size - offset)
            held.append(group[offset:offset + take])
            rows += take
            offset += take
            if rows >= batch_size:
                yield np.concatenate(held)
                held, rows, seen = [], 0, 0
                counted = False
    if held:
        yield np.concatenate(held)


def distinct_strings(table: pa.Table, columns: Sequence[str]) -> pa.Array:
    """The distinct string values across several columns, sorted.

    Sorted so that the row order of a cache does not depend on the order the data was read
    in, which is what makes building the cache twice give the same file.
    """
    pieces = [table.column(name).combine_chunks() for name in columns]
    unique = pc.unique(pa.chunked_array(pieces).combine_chunks())
    return pc.take(unique, pc.array_sort_indices(unique))
