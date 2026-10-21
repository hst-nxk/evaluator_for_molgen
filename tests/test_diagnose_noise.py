"""Logic thuần của scripts/diagnose_noise.py, trên dữ liệu tự tạo (không đọc dữ liệu thật)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_noise as audit  # noqa: E402  (scripts/ không phải package)
import diagnose_noise as diagnose  # noqa: E402

COLUMNS = ("source", "assay_entry_id", "assay_id", "protein_hash", "inchikey", "p_activity",
           "pmid", "doi", "patent", "document_id", "cx_extension", "molecule_source_id",
           "potential_duplicate", "unspecified_stereo_centers")
TYPES = {"p_activity": pa.float64(), "potential_duplicate": pa.bool_(),
         "unspecified_stereo_centers": pa.int32()}


def dataset(rows: list[tuple], config: dict[str, Any]) -> pa.Table:
    """Bảng giống đầu ra của load_endpoint (đã qua add_assay_key và prepare)."""
    table = pa.table({name: pa.array(list(values), type=TYPES.get(name, pa.string()))
                      for name, values in zip(COLUMNS, zip(*rows))})
    return diagnose.prepare(audit.add_assay_key(table), config["stage2a"])


def bindingdb(entry: str, molecule: str, p: float, *, patent: str | None = None,
              pmid: str | None = None, document: bool = True,
              molecule_id: str | None = None, unspecified: int | None = 0) -> tuple:
    """Một dòng BindingDB: assay 1 của entry `entry`; document_id là ENTRYID."""
    return ("bindingdb", entry, "1", "h", molecule, p, pmid, None, patent,
            entry if document else None, None, molecule_id or molecule, False, unspecified)


def chembl(assay: str, molecule: str, p: float, pmid: str, *, flagged: bool = False,
           cx: str | None = None, molecule_id: str | None = None) -> tuple:
    """Một dòng ChEMBL; mỗi assay một tài liệu riêng."""
    return ("chembl", None, assay, "h", molecule, p, pmid, None, None, "D" + assay, cx,
            molecule_id or molecule, flagged, 0)


@pytest.fixture()
def data(config: dict[str, Any]) -> pa.Table:
    return dataset([
        # Hai patent chép cùng một bảng: 3 phân tử chung, cả 3 |Δ| = 0.
        bindingdb("E1", "M1", 6.0, patent="US-1-B2"),
        bindingdb("E1", "M2", 7.3, patent="US-1-B2"),
        bindingdb("E1", "M3", 8.0, patent="US-1-B2"),
        bindingdb("E2", "M1", 6.0, patent="US2"),
        bindingdb("E2", "M2", 7.3, patent="US2"),
        bindingdb("E2", "M3", 8.0, patent="US2"),
        # Hai bài báo đo lại độc lập: 3 phân tử chung, không cặp nào bằng nhau.
        bindingdb("E3", "M4", 6.0, pmid="10"),
        bindingdb("E3", "M5", 6.2, pmid="10"),
        bindingdb("E3", "M6", 6.4, pmid="10"),
        bindingdb("E4", "M4", 7.0, pmid="20"),
        bindingdb("E4", "M5", 7.2, pmid="20"),
        bindingdb("E4", "M6", 7.4, pmid="20"),
        # Entry không có khóa tài liệu nào, trùng giá trị M1 với E1 và E2.
        bindingdb("E6", "M1", 6.0, document=False),
        # Hai bài báo chỉ chung một phân tử: không đủ để phân biệt.
        bindingdb("E7", "M7", 5.5, pmid="30"),
        bindingdb("E8", "M7", 5.5, pmid="40"),
        # Nhóm replicate BindingDB '= 0', một mã phân tử.
        bindingdb("E9", "M9", 5.0, pmid="90"),
        bindingdb("E9", "M9", 5.0, pmid="90"),
        # ChEMBL: C2 có cờ nên mọi cặp với C2 bị bỏ; còn lại C1–C3, |Δ| = 1.
        chembl("C1", "M1", 6.0, "50"),
        chembl("C2", "M1", 6.0, "60", flagged=True),
        chembl("C3", "M1", 7.0, "70"),
        # ChEMBL: dạng |r| và dạng tuyệt đối chung InChIKey M8 trong cùng assay.
        chembl("C4", "M8", 6.0, "80", cx="|r|", molecule_id="CA"),
        chembl("C4", "M8", 7.0, "80", molecule_id="CB"),
    ], config)


@pytest.fixture()
def result(data: pa.Table, config: dict[str, Any]) -> dict[str, Any]:
    return diagnose.diagnose_data(data, config)


# --------------------------------------------------------------------------- hàm nhỏ


def test_bin_index_matches_audit_bin_counts() -> None:
    edges = [0.01, 0.1, 0.3, 0.5]
    values = np.array([0.0, 0.005, 0.01, 0.2, 0.5, 0.7])
    zero = values <= 1e-9
    index = diagnose.bin_index(values, zero, edges)
    counts = np.bincount(index, minlength=len(edges) + 2)
    assert counts.tolist() == audit.bin_counts(values, zero, edges).tolist()


def test_fraction_labels() -> None:
    assert diagnose.fraction_labels([0.1, 0.5, 0.9]) == [
        "[0; 0,1)", "[0,1; 0,5)", "[0,5; 0,9)", "≥ 0,9"]


def test_assay_doc_kinds() -> None:
    keys = pa.table({"assay_key": ["a", "a", "b", "c", "d"],
                     "key": ["patent:US1", "pmid:1", "doi:x", "patent:US2", "doc:chembl:D"]})
    kinds = dict(zip(*diagnose.assay_doc_kinds(keys).to_pydict().values()))
    assert kinds == {"a": 3, "b": 2, "c": 1, "d": 0}


# --------------------------------------------------------------------------- mục A


def test_copy_categories_separate_copied_tables_from_independent_pairs(
        result: dict[str, Any]) -> None:
    copies = result["copies"]["bindingdb"]
    assert (copies["pairs"], copies["zero_pairs"]) == (9, 6)
    by_code = {entry["category"]: entry for entry in copies["categories"]}
    top = by_code[3]                                          # tỷ lệ ≥ 0,9
    assert (top["publication_pairs"], top["pairs"], top["zero_pairs"]) == (1, 3, 3)
    assert top["zero_round_value"] == 2                       # 6,0 và 8,0; 7,3 không tròn
    low = by_code[0]                                          # tỷ lệ [0; 0,1)
    assert (low["publication_pairs"], low["pairs"], low["zero_pairs"]) == (1, 3, 0)
    missing = by_code[diagnose.CATEGORY_MISSING]
    assert (missing["publication_pairs"], missing["pairs"], missing["zero_pairs"]) == (None, 2, 2)
    small = by_code[diagnose.CATEGORY_SMALL]
    assert (small["publication_pairs"], small["pairs"], small["zero_pairs"]) == (1, 1, 1)


def test_copy_kinds_and_volume(result: dict[str, Any]) -> None:
    copies = result["copies"]["bindingdb"]
    kinds = {tuple(entry["kinds"]): entry for entry in copies["kinds"]}
    assert (kinds[(1, 1)]["pairs"], kinds[(1, 1)]["copied_zero_pairs"]) == (3, 3)
    assert (kinds[(2, 2)]["pairs"], kinds[(2, 2)]["zero_pairs"]) == (4, 1)
    assert (kinds[(0, 1)]["pairs"], kinds[(0, 1)]["zero_pairs"]) == (2, 2)
    assert copies["copied_volume"] == {"groups": 6, "measurements": 6,
                                       "scope_groups": 16, "scope_measurements": 17}


def test_copy_top_publication_pairs_have_labels(result: dict[str, Any]) -> None:
    top = result["copies"]["bindingdb"]["top_publication_pairs"]
    assert [entry["zero_pairs"] for entry in top] == [3, 1]
    assert {"patent:US1", "patent:US2"} == {label.split(",")[0] for label in top[0]["labels"]}


def test_chembl_control_drops_flagged_pairs(result: dict[str, Any]) -> None:
    copies = result["copies"]["chembl"]
    assert (copies["pairs"], copies["zero_pairs"]) == (1, 0)


# --------------------------------------------------------------------------- mục B


def test_largest_publication_reports_bridging_keys(config: dict[str, Any]) -> None:
    data = dataset([
        chembl("A", "M1", 6.0, "1"),
        chembl("B", "M2", 6.0, "1"),
        chembl("D", "M3", 6.0, "2"),
    ], config)
    data = data.set_column(data.schema.get_field_index("doi"), "doi",
                           pa.array([None, "10.1/X", None], type=pa.string()))
    _, assay_publication, key_counts = diagnose.publication_key_counts(
        data, config["stage2a"]["dedup"])
    largest = diagnose.largest_publications(data, assay_publication, key_counts, 1, 5)
    assert len(largest) == 1
    entry = largest[0]
    assert entry["assays"] == 2 and entry["assays_by_source"] == {"chembl": 2}
    assert entry["keys_by_tag"] == {"pmid": 1, "doi": 1, "doc": 2}
    assert entry["top_keys"][0] == ("pmid:1", 2)


# --------------------------------------------------------------------------- mục C


def test_stereo_breakdown_flags_mixed_groups(result: dict[str, Any]) -> None:
    chembl_bins = result["stereo"]["chembl"]
    assert chembl_bins["groups"] == 1 and chembl_bins["mixed"] == 1
    assert chembl_bins["bins"]["≥ 0,5"] == {"groups": 1, "any_relative": 1, "mixed": 1,
                                             "multiple_ids": 1,
                                             "multiple_ids_without_relative": 0}
    bindingdb_bins = result["stereo"]["bindingdb"]
    assert bindingdb_bins["groups"] == 1
    assert bindingdb_bins["bins"]["= 0"]["groups"] == 1
    assert bindingdb_bins["bins"]["= 0"]["multiple_ids"] == 0


def test_isomer_check_splits_by_ids_and_unspecified_stereo(config: dict[str, Any]) -> None:
    data = dataset([
        # Hai MonomerID, có tâm chưa xác định, giá trị khác nhau: kiểu "isomer 1/2".
        bindingdb("E10", "M10", 5.0, pmid="1", molecule_id="A", unspecified=1),
        bindingdb("E10", "M10", 6.0, pmid="1", molecule_id="B", unspecified=1),
        # Hai MonomerID, lập thể đủ, giá trị giống hệt: hai bản ghi của một chất.
        bindingdb("E11", "M11", 5.0, pmid="2", molecule_id="C"),
        bindingdb("E11", "M11", 5.0, pmid="2", molecule_id="D"),
        # Một mã, không đếm được tâm lập thể.
        bindingdb("E12", "M12", 5.0, pmid="3", unspecified=None),
        bindingdb("E12", "M12", 5.5, pmid="3", unspecified=None),
    ], config)
    result = diagnose.isomer_check(data, 1, config["stage2a"]["audit"])
    multi, single = result["> 1 mã phân tử"], result["1 mã phân tử"]
    assert multi["có tâm chưa xác định"] == {"groups": 1, "zero": 0, "high": 1}
    assert multi["không có tâm chưa xác định"] == {"groups": 1, "zero": 1, "high": 0}
    assert single["không rõ"] == {"groups": 1, "zero": 0, "high": 0}
    assert single["có tâm chưa xác định"]["groups"] == 0


# --------------------------------------------------------------------------- báo cáo


def test_report_renders(result: dict[str, Any], config: dict[str, Any]) -> None:
    lines = diagnose.build_report({"Ki": result}, config, 1.0)
    text = "\n".join(lines)
    for heading in ("## A.", "## B.", "## C.", "## D."):
        assert heading in text
    assert "patent – patent" in text and "patent:US1" in text
