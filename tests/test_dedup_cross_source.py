"""Logic thuần của scripts/dedup_cross_source.py, trên dữ liệu tự tạo (không đọc dữ liệu thật)."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dedup_cross_source as dedup  # noqa: E402  (scripts/ không phải package)

EXACT_TOLERANCE = 1e-9
EQUAL_TOLERANCE = 0.01


@pytest.fixture()
def settings(config: dict[str, Any]) -> dict[str, Any]:
    return config["stage2a"]["dedup"]


def side_table(source: str, rows: list[dict[str, Any]],
               settings: dict[str, Any]) -> pa.Table:
    """Bảng một nguồn với đủ cột DEDUP_ROLES (cột không khai báo -> null), đã qua prepare_side."""
    data = {}
    for role in dedup.DEDUP_ROLES:
        dtype = pa.float64() if role == "molar" else pa.string()
        data[role] = pa.array([row.get(role) for row in rows], type=dtype)
    data["source"] = pa.array([source] * len(rows), type=pa.string())
    return dedup.prepare_side(pa.table(data), source, settings)


# --------------------------------------------------------------------------- chuẩn hóa


def test_doi_normalization_removes_prefix_and_case(settings: dict[str, Any]) -> None:
    values = pa.array(["https://doi.org/10.1021/JM1", " DOI:10.1/A ", None])
    result = dedup.normalize_doi(values, settings["doi_prefix_pattern"]).to_pylist()
    assert result == ["10.1021/jm1", "10.1/a", None]


def test_patent_normalization_drops_separators_and_kind_code(settings: dict[str, Any]) -> None:
    values = pa.array(["US-9718838-B2", "US9718838", "us 2014/0213628 a1"])
    result = dedup.normalize_patent(values, settings["patent_strip_pattern"],
                                    settings["patent_kind_code_pattern"],
                                    settings["patent_kind_code_replacement"]).to_pylist()
    assert result == ["US9718838", "US9718838", "US20140213628"]


def test_provenance_mask_is_case_insensitive_and_null_safe() -> None:
    curation = pa.array(["ChEMBL", "Curated from the literature by BindingDB", None])
    assert dedup.provenance_mask(curation, ["chembl"]).to_pylist() == [True, False, False]
    assert dedup.provenance_mask(curation, []).to_pylist() == [False, False, False]


def test_value_key_is_rounded_p_activity() -> None:
    keys = dedup.value_keys(pa.array([1e-9, None], type=pa.float64()), 6).to_pylist()
    assert keys == [9_000_000, None]


# --------------------------------------------------------------------------- so sánh một cặp


@pytest.mark.parametrize(("chembl", "bindingdb", "expected"), [
    (("=", 1e-8), ("=", 1.0e-8), dedup.EXACT),
    # 10 nM và 10,2 nM: |Δp| = 0,0086 ≤ 0,01 -> cùng giá trị ở độ phân giải pChEMBL.
    (("=", 1e-8), ("=", 1.02e-8), dedup.EXACT),
    # 10 nM và 11 nM: |Δp| = 0,041 > 0,01 -> khác giá trị.
    (("=", 1e-8), ("=", 1.1e-8), dedup.DIFFERS),
    (("=", 1e-8), ("=", 1e-6), dedup.DIFFERS),
    (("=", 1e-8), (">", 1e-8), dedup.QUALIFIER_DIFFERS),
    (("=", None), ("=", 1e-8), dedup.MISSING),
])
def test_compare_values(chembl: tuple, bindingdb: tuple, expected: str) -> None:
    label, _ = dedup.compare_values(*chembl, *bindingdb, EXACT_TOLERANCE, EQUAL_TOLERANCE)
    assert label == expected


def test_compare_values_reports_p_difference() -> None:
    _, difference = dedup.compare_values("=", 1e-8, "=", 1e-6, EXACT_TOLERANCE,
                                         EQUAL_TOLERANCE)
    assert difference == pytest.approx(2.0)


def test_compare_values_rejects_non_positive_concentration() -> None:
    with pytest.raises(ValueError, match="không dương"):
        dedup.compare_values("=", 0.0, "=", 1e-8, EXACT_TOLERANCE, EQUAL_TOLERANCE)


@pytest.mark.parametrize(("arguments", "expected"), [
    (("h1", "P1", "h1", None), dedup.SEQUENCE),
    (("h1", "P1", "h2", "P1"), dedup.ACCESSION),
    ((None, None, "h2", "P2"), dedup.MISSING),
    (("h1", "P1", "h2", "P2"), dedup.DIFFERS),
])
def test_compare_proteins(arguments: tuple, expected: str) -> None:
    assert dedup.compare_proteins(*arguments) == expected


@pytest.mark.parametrize(("arguments", "expected"), [
    ((True, dedup.SEQUENCE, dedup.EXACT),
     (dedup.LEVEL_CURATION, dedup.DUPLICATE, dedup.REASON_CURATION)),
    ((False, dedup.ACCESSION, dedup.EXACT),
     (dedup.LEVEL_PUBLICATION, dedup.DUPLICATE, dedup.REASON_PUBLICATION)),
    ((True, dedup.SEQUENCE, dedup.DIFFERS),
     (dedup.LEVEL_PARTIAL, dedup.UNCERTAIN, "value_differs")),
    ((True, dedup.MISSING, dedup.EXACT),
     (dedup.LEVEL_PARTIAL, dedup.UNCERTAIN, "protein_missing")),
])
def test_classify_pair(arguments: tuple, expected: tuple) -> None:
    assert dedup.classify_pair(*arguments) == expected


def test_compare_documents_lists_every_shared_kind() -> None:
    chembl = {"pmid": "1", "doi_norm": "10.1/a", "patent_norm": None}
    bindingdb = {"pmid": "1", "doi_norm": "10.1/a", "patent_norm": None}
    assert dedup.compare_documents(chembl, bindingdb, ["pmid", "doi", "patent"]) == "pmid+doi"
    assert dedup.compare_documents(chembl, {**bindingdb, "pmid": "2", "doi_norm": None},
                                   ["pmid", "doi"]) == dedup.NO_DOCUMENT


# --------------------------------------------------------------------------- nhóm và cờ


def test_duplicate_groups_merge_connected_pairs() -> None:
    groups = dedup.duplicate_groups([("c1", "b1"), ("c2", "b1"), ("c3", "b3")])
    assert groups["c1"] == groups["c2"] == groups["b1"] == "dup:b1"
    assert groups["c3"] == groups["b3"] == "dup:b3"


def pair(chembl: str | None, bindingdb: str | None, status: str, level: int,
         reason: str, group: str | None = None) -> dict[str, Any]:
    return {"chembl_observation_id": chembl, "bindingdb_observation_id": bindingdb,
            "dedup_status": status, "evidence_level": level, "reason": reason,
            "group_id": group, "endpoint": "Ki"}


def test_resolved_uncertain_pairs_are_dropped_and_counted() -> None:
    records = [pair("c1", "b1", dedup.DUPLICATE, 2, "publication_match"),
               pair("c2", "b2", dedup.DUPLICATE, 2, "publication_match"),
               pair("c1", "b2", dedup.UNCERTAIN, 3, "value_differs"),
               pair("c1", "b3", dedup.UNCERTAIN, 3, "value_differs")]
    kept, dropped = dedup.drop_resolved_uncertain(records)
    assert dropped == 1
    assert [(record["chembl_observation_id"], record["bindingdb_observation_id"])
            for record in kept] == [("c1", "b1"), ("c2", "b2"), ("c1", "b3")]


def test_observation_flags_keep_only_the_chosen_source() -> None:
    records = [pair("c1", "b1", dedup.DUPLICATE, 1, "x", "dup:b1"),
               pair("c2", "b1", dedup.UNCERTAIN, 3, "value_differs"),
               pair(None, "b9", dedup.UNCERTAIN, 3, dedup.REASON_NO_COUNTERPART)]
    flags = {flag["observation_id"]: flag
             for flag in dedup.observation_flags(records, keep_source=dedup.CHEMBL)}
    assert flags["b1"]["status"] == dedup.DUPLICATE and flags["b1"]["keep"] is False
    assert flags["b1"]["level"] == 1 and flags["b1"]["reason"] == "counterparts:1"
    assert flags["c1"]["status"] == dedup.DUPLICATE and flags["c1"]["keep"] is True
    assert flags["c2"]["status"] == dedup.UNCERTAIN and flags["c2"]["keep"] is True
    assert flags["b9"]["reason"] == dedup.REASON_NO_COUNTERPART
    assert flags["b9"]["keep"] is True


def test_group_summary_counts_level_and_multi_chembl_groups() -> None:
    records = [pair("c1", "b1", dedup.DUPLICATE, 1, "x", "dup:b1"),
               pair("c2", "b1", dedup.DUPLICATE, 2, "x", "dup:b1"),
               pair("c3", "b3", dedup.DUPLICATE, 2, "x", "dup:b3")]
    by_level, multi = dedup.summarize_groups(records)
    assert by_level == {(1, "Ki"): 1, (2, "Ki"): 1}
    assert multi == {1: 1}


# --------------------------------------------------------------------------- phân vùng


def test_partitions_cover_letters_other_and_null() -> None:
    partitions = dedup.build_partitions("ABCD", 2)
    assert [partition.label for partition in partitions] == ["AB", "CD", "other", "null"]
    keys = pa.array(["ABC", "DZZ", None, "1XY"])
    masks = {partition.label: dedup.partition_mask(keys, partition).to_pylist()
             for partition in partitions}
    assert masks == {"AB": [True, False, False, False], "CD": [False, True, False, False],
                     "other": [False, False, False, True], "null": [False, False, True, False]}


# --------------------------------------------------------------------------- ghép một phân vùng


def test_match_partition_end_to_end(settings: dict[str, Any]) -> None:
    common = {"endpoint": "Ki", "qualifier": "="}
    chembl = side_table(dedup.CHEMBL, [
        # Cùng bài (qua DOI), cùng protein, cùng giá trị với b10, và b10 nhập từ ChEMBL -> mức 1.
        {**common, "observation_id": "chembl:1:Ki", "native_id": "1", "inchikey": "AAA",
         "pmid": "100", "doi": "10.1/X", "protein_hash": "h1", "protein_accession": "P1",
         "molar": 1e-8},
        # Cùng bài, cùng protein với b10 nhưng khác giá trị -> uncertain value_differs.
        {**common, "observation_id": "chembl:2:Ki", "native_id": "2", "inchikey": "AAA",
         "pmid": "100", "doi": "10.1/X", "protein_hash": "h1", "protein_accession": "P1",
         "molar": 1e-6},
        # Thiếu protein, cùng bài và cùng giá trị với b11 -> uncertain protein_missing.
        {**common, "observation_id": "chembl:3:Ki", "native_id": "3", "inchikey": "BBB",
         "pmid": "200", "molar": 1e-7},
        # Cùng bài, protein, giá trị với b13, không có dấu hiệu nguồn nhập liệu -> mức 2.
        {**common, "observation_id": "chembl:4:Ki", "native_id": "4", "inchikey": "DDD",
         "pmid": "300", "protein_hash": "h2", "molar": 5e-9, "curation": "LITERATURE"},
    ], settings)
    bindingdb = side_table(dedup.BINDINGDB, [
        {**common, "observation_id": "bindingdb:10:Ki", "native_id": "10", "inchikey": "AAA",
         "doi": "https://doi.org/10.1/x", "protein_hash": "h1", "protein_accession": "P1",
         "molar": 1e-8, "curation": "ChEMBL"},
        {**common, "observation_id": "bindingdb:11:Ki", "native_id": "11", "inchikey": "BBB",
         "pmid": "200", "protein_hash": "h9", "protein_accession": "P9", "molar": 1e-7},
        # Nhập từ ChEMBL nhưng không có bản tương ứng -> uncertain, không đoán.
        {**common, "observation_id": "bindingdb:12:Ki", "native_id": "12", "inchikey": "CCC",
         "pmid": "999", "molar": 1e-7, "curation": "ChEMBL"},
        {**common, "observation_id": "bindingdb:13:Ki", "native_id": "13", "inchikey": "DDD",
         "pmid": "300", "protein_hash": "h2", "molar": 5e-9,
         "curation": "Curated from the literature by BindingDB"},
    ], settings)

    records, suppressed = dedup.match_partition(chembl, bindingdb, True, settings)
    by_pair = {(record["chembl_observation_id"], record["bindingdb_observation_id"]): record
               for record in records}

    level_1 = by_pair[("chembl:1:Ki", "bindingdb:10:Ki")]
    assert (level_1["evidence_level"], level_1["dedup_status"]) == (1, dedup.DUPLICATE)
    assert level_1["document_match"] == "doi"
    assert level_1["group_id"] == "dup:bindingdb:10:Ki"
    assert by_pair[("chembl:2:Ki", "bindingdb:10:Ki")]["reason"] == "value_differs"
    assert by_pair[("chembl:3:Ki", "bindingdb:11:Ki")]["reason"] == "protein_missing"
    assert by_pair[(None, "bindingdb:12:Ki")]["reason"] == dedup.REASON_NO_COUNTERPART
    level_2 = by_pair[("chembl:4:Ki", "bindingdb:13:Ki")]
    assert (level_2["evidence_level"], level_2["dedup_status"]) == (2, dedup.DUPLICATE)
    assert len(records) == 5 and suppressed == 0


def test_rows_without_inchikey_only_produce_unpaired_records(settings: dict[str, Any]) -> None:
    chembl = side_table(dedup.CHEMBL, [], settings)
    bindingdb = side_table(dedup.BINDINGDB, [
        {"observation_id": "bindingdb:20:Ki", "native_id": "20", "endpoint": "Ki",
         "curation": "ChEMBL"},
        {"observation_id": "bindingdb:21:Ki", "native_id": "21", "endpoint": "Ki"},
    ], settings)
    records, _ = dedup.match_partition(chembl, bindingdb, False, settings)
    assert [(record["bindingdb_observation_id"], record["reason"]) for record in records] == [
        ("bindingdb:20:Ki", dedup.REASON_NO_INCHIKEY)]


# --------------------------------------------------------------------------- báo cáo


def test_overlap_diagnosis_counts_shared_criteria(settings: dict[str, Any]) -> None:
    chembl = side_table(dedup.CHEMBL, [
        {"observation_id": "chembl:1:Ki", "endpoint": "Ki", "inchikey": "AAA", "pmid": "1",
         "qualifier": "=", "molar": 1e-8, "protein_accession": "P1"}], settings).to_pylist()
    bindingdb = side_table(dedup.BINDINGDB, [
        {"observation_id": "bindingdb:1:Ki", "endpoint": "Ki", "inchikey": "AAA", "pmid": "2",
         "qualifier": "=", "molar": 1e-8, "protein_accession": "P1",
         "curation": "ChEMBL"}], settings).to_pylist()
    diagnosis = dict(dedup.overlap_diagnosis(bindingdb, chembl, ["pmid", "doi"]))
    assert diagnosis["pmid chung"] == 0
    assert diagnosis["InChIKey chung"] == 1
    assert diagnosis["khóa protein chung (chuỗi hoặc accession)"] == 1
    assert diagnosis["(InChIKey, endpoint, qualifier, giá trị) chung"] == 1
    assert diagnosis["dòng BindingDB có dấu hiệu nguồn nhập liệu"] == 1


@pytest.mark.parametrize(("document_match", "expected"), [
    ("pmid", dedup.BRANCH_PUBLICATION),
    ("pmid+doi", dedup.BRANCH_PUBLICATION),
    ("patent", dedup.BRANCH_PATENT),
    ("doi+patent", dedup.BRANCH_BOTH),
    ("none", dedup.BRANCH_NONE),
    (None, dedup.BRANCH_NONE),
])
def test_document_branch(document_match: str | None, expected: str) -> None:
    assert dedup.document_branch(document_match) == expected


def test_branch_totals_split_levels_and_uncertain_reasons() -> None:
    def record(match: str | None, level: int, status: str, reason: str,
               record_type: str = dedup.RECORD_PAIR) -> dict[str, Any]:
        return {"record_type": record_type, "document_match": match, "evidence_level": level,
                "dedup_status": status, "reason": reason}

    records = [record("pmid", 1, dedup.DUPLICATE, dedup.REASON_CURATION),
               record("doi", 2, dedup.DUPLICATE, dedup.REASON_PUBLICATION),
               record("patent", 2, dedup.DUPLICATE, dedup.REASON_PUBLICATION),
               record("patent", 3, dedup.UNCERTAIN, "value_differs"),
               record("patent", 3, dedup.UNCERTAIN, "value_differs"),
               record("pmid+patent", 3, dedup.UNCERTAIN, "protein_missing"),
               # Dòng lẻ không thuộc nhánh nào, không được đếm.
               record(None, 3, dedup.UNCERTAIN, dedup.REASON_NO_COUNTERPART,
                      dedup.RECORD_UNPAIRED)]
    totals = dedup.branch_totals(dedup.branch_counts(records))
    publication = totals[dedup.BRANCH_PUBLICATION]
    assert (publication["pairs"], publication["level_1"], publication["level_2"],
            publication["uncertain"]) == (2, 1, 1, 0)
    patent = totals[dedup.BRANCH_PATENT]
    assert (patent["pairs"], patent["level_2"], patent["uncertain"]) == (3, 1, 2)
    assert patent["uncertain_reasons"] == {"value_differs": 2}
    assert totals[dedup.BRANCH_BOTH]["uncertain_reasons"] == {"protein_missing": 1}
    assert totals[dedup.BRANCH_NONE]["pairs"] == 0


def test_relative_stereo_mask_finds_r_field_only(settings: dict[str, Any]) -> None:
    extensions = pa.array(["|r|", "|r,&1:3|", "|&1:1|", "|c:1,2|", None])
    result = dedup.relative_stereo_mask(extensions, settings["relative_stereo_pattern"])
    assert result.to_pylist() == [True, True, False, False, False]


def test_provenance_stereo_counts_by_pairing_state(settings: dict[str, Any]) -> None:
    pattern = settings["relative_stereo_pattern"]
    bindingdb = side_table(dedup.BINDINGDB, [
        {"observation_id": "b1", "curation": "ChEMBL", "cx_extension": "|r|"},
        {"observation_id": "b2", "curation": "ChEMBL", "cx_extension": "|r|"},
        {"observation_id": "b3", "curation": "ChEMBL"},
        # Không nhập từ ChEMBL -> không được đếm dù có |r|.
        {"observation_id": "b4", "curation": "BindingDB", "cx_extension": "|r|"},
    ], settings)
    records = [{"record_type": dedup.RECORD_PAIR, "bindingdb_observation_id": "b1"}]
    counts = dedup.provenance_stereo_counts(bindingdb, records, True, pattern)
    assert counts == {(dedup.PAIRED, True): 1, (dedup.REASON_NO_COUNTERPART, True): 1,
                      (dedup.REASON_NO_COUNTERPART, False): 1}
    totals = dedup.stereo_totals(counts)
    assert totals[dedup.REASON_NO_COUNTERPART] == {"rows": 2, "with_r": 1}
    assert totals[dedup.REASON_NO_INCHIKEY] == {"rows": 0, "with_r": 0}
    unkeyed = dedup.provenance_stereo_counts(bindingdb, [], False, pattern)
    assert unkeyed == {(dedup.REASON_NO_INCHIKEY, True): 2, (dedup.REASON_NO_INCHIKEY, False): 1}


def test_reservoir_sample_is_reproducible() -> None:
    def draw() -> list[int]:
        rng, sample = random.Random(7), []
        for seen, item in enumerate(range(100)):
            dedup.reservoir_add(sample, seen, item, 3, rng)
        return sample

    first = draw()
    assert len(first) == 3 and first == draw()


def test_number_formatting_uses_vietnamese_separators() -> None:
    assert dedup.fmt_int(5338213) == "5.338.213"
    assert dedup.fmt_float(1234.5, 2) == "1.234,50"
    assert dedup.fmt_pct(1, 8) == "12,5 %"
    assert dedup.fmt_pct(1, 0) == "–"
