"""Logic thuần của scripts/audit_noise.py, trên dữ liệu tự tạo (không đọc dữ liệu thật)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_noise as audit  # noqa: E402  (scripts/ không phải package)

DDOF = 1
TOLERANCE = 1e-9


def measurements(rows: list[tuple], flags: list[bool] | None = None,
                 publications: list[int] | None = None) -> pa.Table:
    """Bảng tập audit nhỏ: (nguồn, ENTRYID, assay, protein, phân tử, pActivity) -> có assay_key.

    flags: cờ potential_duplicate từng dòng (mặc định False);
    publications: số hiệu publication từng dòng (mặc định PUBLICATION_MISSING).
    """
    names = ["source", "assay_entry_id", "assay_id", "protein_hash", "inchikey", "p_activity"]
    columns = list(zip(*rows)) if rows else [[] for _ in names]
    table = pa.table({name: pa.array(list(values),
                                     type=pa.float64() if name == "p_activity" else pa.string())
                      for name, values in zip(names, columns)})
    table = table.append_column("potential_duplicate", pa.array(
        flags if flags is not None else [False] * len(rows), type=pa.bool_()))
    table = table.append_column("publication", pa.array(
        publications if publications is not None
        else [audit.PUBLICATION_MISSING] * len(rows), type=pa.int64()))
    return audit.add_assay_key(table)


@pytest.fixture()
def table() -> pa.Table:
    return measurements([
        ("chembl", None, "CHEMBL1", "h1", "M", 6.0),
        ("chembl", None, "CHEMBL1", "h1", "M", 7.0),   # replicate của dòng trên
        ("chembl", None, "CHEMBL1", "h1", "N", 5.0),   # nhóm một phần tử
        ("chembl", None, "CHEMBL2", "h1", "M", 8.0),   # cùng phân tử, assay khác
        ("chembl", None, "CHEMBL2", "h1", None, 9.0),  # không có InChIKey -> bỏ khỏi nhóm
    ])


# --------------------------------------------------------------------------- khóa assay và nhóm


def test_bindingdb_assay_key_includes_entry() -> None:
    keyed = measurements([("bindingdb", "1025", "1", "h", "M", 6.0),
                          ("bindingdb", "2000", "1", "h", "M", 6.0)])
    assert keyed["assay_key"].to_pylist() == ["bindingdb:1025:1", "bindingdb:2000:1"]


def test_replicate_std_and_single_member_group(table: pa.Table) -> None:
    groups = {(row["assay_key"], row["inchikey"]): row
              for row in audit.replicate_groups(table, DDOF).to_pylist()}
    assert len(groups) == 3
    replicate = groups[("chembl::CHEMBL1", "M")]
    assert replicate["n"] == 2
    assert replicate["mean_p"] == pytest.approx(6.5)
    assert replicate["std_p"] == pytest.approx(math.sqrt(0.5))
    single = groups[("chembl::CHEMBL1", "N")]
    assert single["n"] == 1 and single["std_p"] is None


def test_replicate_summary_counts_only_groups_with_two_or_more(table: pa.Table) -> None:
    groups = audit.replicate_groups(table, DDOF)
    summary = audit.replicate_summary(groups, [0, 50, 100], [0.3, 0.8])
    assert summary["groups"] == 1 and summary["measurements"] == 2
    assert summary["std"]["median"] == pytest.approx(math.sqrt(0.5))
    assert summary["at_most"] == {"0.3": 0, "0.8": 1}


def test_replicate_summary_of_singletons_is_empty() -> None:
    groups = audit.replicate_groups(measurements([("chembl", None, "A", "h", "M", 6.0)]), DDOF)
    summary = audit.replicate_summary(groups, [50], [0.3])
    assert summary["groups"] == 0 and summary["std"] == {"median": None}


def test_assay_count_distribution(table: pa.Table) -> None:
    groups = audit.replicate_groups(table, DDOF)
    result = audit.assay_count_distribution(groups, [1, 2, 3])
    assert result == {"pairs": 2, "buckets": {"1": 1, "2": 1, "≥3": 0}}


def test_assay_offsets_use_assay_means(table: pa.Table, config: dict[str, Any]) -> None:
    groups = audit.replicate_groups(table, DDOF)
    result = audit.assay_offsets(groups, config["stage2a"]["audit"])
    # Phân tử M: assay 1 trung bình 6,5; assay 2 là 8,0 -> |Δ| = 1,5.
    assert result["pairs_with_multiple_assays"] == 1 and result["assay_pairs"] == 1
    assert result["abs_difference"]["min"] == pytest.approx(1.5)
    assert result["abs_difference"]["max"] == pytest.approx(1.5)
    assert result["above"] == {"1.0": 1, "2.0": 0, "3.0": 0}


def test_assay_offsets_without_multi_assay_pairs(config: dict[str, Any]) -> None:
    groups = audit.replicate_groups(measurements([("chembl", None, "A", "h", "M", 6.0)]), DDOF)
    result = audit.assay_offsets(groups, config["stage2a"]["audit"])
    assert result["assay_pairs"] == 0 and result["abs_difference"]["median"] is None


# --------------------------------------------------------------------------- hàm số học


def test_pair_indices() -> None:
    first, second = audit.pair_indices(np.array([4, 7, 9]))
    assert list(zip(first.tolist(), second.tolist())) == [(4, 7), (4, 9), (7, 9)]
    assert audit.pair_indices(np.array([5]))[0].size == 0


def test_bin_labels_and_counts() -> None:
    edges = [0.01, 0.1, 0.3, 0.5]
    assert audit.bin_labels(edges) == ["= 0", "(0; 0,01)", "[0,01; 0,1)", "[0,1; 0,3)",
                                       "[0,3; 0,5)", "≥ 0,5"]
    values = np.array([0.0, 0.005, 0.01, 0.2, 0.5, 0.7])
    counts = audit.bin_counts(values, values <= 1e-9, edges)
    assert counts.tolist() == [1, 1, 1, 1, 0, 2]


# --------------------------------------------------------------------------- mục 7


def test_publication_ids_bridge_shared_keys() -> None:
    pairs = [("A", "pmid:1"), ("B", "doi:x"), ("A", "doi:x"), ("C", "pmid:2")]
    ids = audit.publication_ids(pairs)
    assert ids["A"] == ids["B"] != ids["C"]


def test_add_publication_matches_normalized_doi(config: dict[str, Any]) -> None:
    table = audit.add_assay_key(pa.table({
        "source": ["chembl", "bindingdb", "chembl"],
        "assay_entry_id": [None, "77", None],
        "assay_id": ["CHEMBL1", "1", "CHEMBL2"],
        "pmid": ["100", None, None],
        "doi": ["10.1/X", "https://doi.org/10.1/x", None],
        "patent": pa.array([None, None, None], type=pa.string()),
        "document_id": ["CHEMBL9", "77", "CHEMBL8"],
    }))
    result, stats = audit.add_publication(table, config["stage2a"]["dedup"])
    publication = result["publication"].to_pylist()
    # Hai assay đầu chung DOI (sau chuẩn hóa) -> cùng publication; assay thứ ba riêng.
    assert publication[0] == publication[1] != publication[2]
    assert stats == {"assays": 3, "publications": 2, "largest_publication_assays": 2}


def test_replicate_diagnostics_zero_bin_and_flags(config: dict[str, Any]) -> None:
    table = measurements([
        ("chembl", None, "A", "h", "M", 6.0),
        ("chembl", None, "A", "h", "M", 6.0),       # nhóm '= 0', có cờ ở một dòng
        ("chembl", None, "A", "h", "N", 5.0),
        ("chembl", None, "A", "h", "N", 6.0),       # nhóm STD lớn, không cờ
        ("bindingdb", "1", "1", "h", "M", 7.0),
        ("bindingdb", "1", "1", "h", "M", 7.0),     # nhóm '= 0' của BindingDB
    ], flags=[True, False, False, False, False, False])
    groups = audit.replicate_groups(table, DDOF)
    result = audit.replicate_diagnostics(groups, config["stage2a"]["audit"])
    assert result["groups"] == 3 and result["bins"]["= 0"] == 2
    assert (result["chembl_groups"], result["chembl_near_zero"],
            result["chembl_near_zero_flagged"]) == (2, 1, 1)
    assert (result["chembl_other"], result["chembl_other_flagged"]) == (1, 0)


def test_assay_offsets_split_by_publication_and_flag(config: dict[str, Any]) -> None:
    table = measurements([
        ("chembl", None, "X", "h", "M", 6.0),
        ("chembl", None, "Y", "h", "M", 6.0),
        ("chembl", None, "Z", "h", "M", 7.5),
    ], flags=[False, False, True], publications=[1, 1, 2])
    result = audit.assay_offsets(audit.replicate_groups(table, DDOF),
                                 config["stage2a"]["audit"])
    views = result["views"]
    # X–Y: cùng publication, |Δ| = 0; X–Z và Y–Z: khác publication, có cờ, |Δ| = 1,5.
    assert result["assay_pairs"] == 3 and result["bins"]["= 0"] == 1
    assert views[audit.VIEW_SAME]["pairs"] == 1 and views[audit.VIEW_SAME]["bins"]["= 0"] == 1
    assert views[audit.VIEW_DIFFERENT]["pairs"] == 2
    assert views[audit.VIEW_DIFFERENT]["above"]["1.0"] == 2
    assert views[audit.VIEW_INDEPENDENT]["pairs"] == 0
    assert views[audit.VIEW_FLAGGED]["pairs"] == 2 and views[audit.VIEW_UNFLAGGED]["pairs"] == 1


def test_publication_section_prints_unflagged_view_only_for_chembl(
        config: dict[str, Any]) -> None:
    settings = config["stage2a"]["audit"]
    table = measurements([("chembl", None, "X", "h", "M", 6.0),
                          ("chembl", None, "Y", "h", "M", 6.5)], publications=[1, 2])
    offsets = audit.assay_offsets(audit.replicate_groups(table, DDOF), settings)
    results = {"Ki": {"publications": {"assays": 2, "publications": 2,
                                       "largest_publication_assays": 1},
                      "scopes": {"chembl": {"offsets": offsets},
                                 "bindingdb": {"offsets": offsets}}}}
    lines = audit.publication_section(results, settings)
    unflagged = [line for line in lines if line.startswith("Ki / ")
                 and audit.VIEW_INDEPENDENT in line]
    assert len(unflagged) == 1 and unflagged[0].startswith("Ki / chembl / ")


def test_potential_duplicate_flag_reading(config: dict[str, Any]) -> None:
    settings = config["stage2a"]["audit"]
    assert audit.is_potential_duplicate({"potential_duplicate": 1}, settings)
    assert not audit.is_potential_duplicate({"potential_duplicate": 0}, settings)
    assert not audit.is_potential_duplicate({}, settings)


def test_histogram_percentiles_within_half_a_bin() -> None:
    counts = audit.new_histogram(0.1, 1.0)
    counts = audit.add_to_histogram(counts, np.array([0.15, 0.25, 0.35]), 0.1)
    assert int(counts.sum()) == 3
    result = audit.histogram_percentiles(counts, 0.1, [50])
    assert abs(result["median"] - 0.25) <= 0.05


def test_histogram_overflow_bin_and_empty_histogram() -> None:
    counts = audit.add_to_histogram(audit.new_histogram(0.1, 1.0), np.array([50.0]), 0.1)
    assert counts[-1] == 1
    assert audit.histogram_percentiles(audit.new_histogram(0.1, 1.0), 0.1, [50]) == {
        "median": None}


def test_bucket_counts_and_empty_summary() -> None:
    assert audit.bucket_counts(np.array([1, 1, 2, 5, 7]), [1, 2, 3]) == {
        "1": 2, "2": 1, "≥3": 2}
    assert audit.summarize_values(np.empty(0), [0, 100]) == {"min": None, "max": None}


def test_p_activity_summary_counts_out_of_range(config: dict[str, Any]) -> None:
    table = pa.table({"p_activity": pa.array([-1.0, 6.0, 16.0, None], type=pa.float64())})
    result = audit.p_activity_summary(table, config["stage2a"]["audit"])
    assert result["rows"] == 3 and result["below_low"] == 1 and result["above_high"] == 1


# --------------------------------------------------------------------------- lọc


def test_filter_funnel_counts_each_step() -> None:
    table = pa.table({
        "keep": [True, True, True, True, False],
        "qualifier": ["=", "=", ">", "=", "="],
        "protein_status": ["resolved", "unresolved", "resolved", "resolved", "resolved"],
        "assay_status": ["resolved"] * 5,
        "p_activity": pa.array([6.0, 6.0, None, None, 6.0], type=pa.float64()),
    })
    kept, funnel = audit.apply_filters(table, audit.filter_steps(table, ["="], "resolved"))
    assert kept.num_rows == 1
    assert [count for _, count in funnel] == [5, 4, 3, 2, 2, 1]


# --------------------------------------------------------------------------- protein


def test_top_proteins_rank_and_median_std(table: pa.Table) -> None:
    extra = measurements([("chembl", None, "CHEMBL9", "h2", "Z", 7.0)])
    data = pa.concat_tables([table, extra])
    labelled = data.append_column("protein_accession", pa.array(["P1"] * 5 + ["P2"]))
    labelled = labelled.append_column("protein_name", pa.array(["one"] * 5 + ["two"]))
    groups = audit.replicate_groups(labelled, DDOF)
    top = audit.top_proteins(labelled, groups, 1)
    assert len(top) == 1
    assert (top[0]["accession"], top[0]["rows"], top[0]["replicate_groups"]) == ("P1", 5, 1)
    assert top[0]["median_replicate_std"] == pytest.approx(math.sqrt(0.5))


def test_most_common_labels_breaks_ties_alphabetically() -> None:
    rows = [{"protein_hash": "h", "protein_accession": "P2", "protein_name": "b",
             "p_activity_count": 3},
            {"protein_hash": "h", "protein_accession": "P1", "protein_name": "a",
             "p_activity_count": 3},
            {"protein_hash": "h", "protein_accession": "P9", "protein_name": "z",
             "p_activity_count": 1}]
    assert audit.most_common_labels(rows) == {"h": ("P1", "a")}


# -------------------------------------------------------------------------- ChEMBL gốc vs standard


@pytest.mark.parametrize(("context", "standard", "expected"), [
    ({"value": 1, "units": "uM", "relation": "="}, ("1000", "nM", "="),
     (True, True, False, False)),
    ({"value": 10.0, "units": "nM", "relation": "="}, ("10", "nM", "="),
     (False, False, False, False)),
    ({"value": 10.0, "units": "nM", "relation": ">"}, ("10", "nM", "="),
     (False, False, True, False)),
    ({"value": None, "units": None, "relation": None}, ("10", "nM", "="),
     (False, False, False, True)),
])
def test_compare_original_standard(context: dict, standard: tuple, expected: tuple) -> None:
    assert tuple(audit.compare_original_standard(context, *standard, TOLERANCE)) == expected


def test_invalid_activity_context_names_the_observation() -> None:
    with pytest.raises(ValueError, match="chembl:5:Ki"):
        audit.parse_context({"observation_id": "chembl:5:Ki", "activity_context": "{bad"})
