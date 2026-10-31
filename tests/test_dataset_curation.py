"""Logic thuần của src/molopt/dataset/curation.py (không đọc dữ liệu thật)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from molopt.dataset import curation


@pytest.fixture()
def settings(config: dict[str, Any]) -> dict[str, Any]:
    return config["stage2b"]["curation"]


def chembl_context(**overrides: Any) -> tuple[str, str]:
    """(activity_context, assay_context) của một dòng ChEMBL sạch, định dạng như Stage 1."""
    activity = {"record_id": 1, "relation": "=", "value": 10, "units": "nM",
                "activity_comment": None, "data_validity_comment": None,
                "potential_duplicate": 0, "pchembl_value": 8.0}
    assay = {"assay_id": 1, "relationship_type": "D", "confidence_score": 9}
    for key, value in overrides.items():
        (assay if key in assay else activity)[key] = value
    return json.dumps(activity), json.dumps(assay)


def batch(rows: list[dict[str, Any]]) -> pa.Table:
    """Bảng có đủ cột mà row_reasons đọc."""
    defaults = {"activity_context": None, "assay_context": None, "assay_description": "x",
                "protein_variant_id": None, "dedup_reason": None, "protein_hash": "h1",
                "protein_sequence": "MKT", "p_activity": 7.0}
    filled = [{**defaults, **row} for row in rows]
    columns = {key: pa.array([row[key] for row in filled],
                             pa.float64() if key == "p_activity" else pa.string())
               for key in defaults}
    return pa.table(columns)


def chembl_row(description: str = "x", variant: str | None = None,
               **fields: Any) -> dict[str, Any]:
    activity, assay = chembl_context(**fields)
    return {"activity_context": activity, "assay_context": assay,
            "assay_description": description, "protein_variant_id": variant}


def test_rule_names_match_the_config(settings: dict[str, Any]) -> None:
    curation.check_rule_names(settings["rules"])


def test_rule_names_report_missing_rules(settings: dict[str, Any]) -> None:
    rules = dict(settings["rules"])
    rules.pop(curation.BINNED_ASSAY)
    with pytest.raises(KeyError, match=curation.BINNED_ASSAY):
        curation.check_rule_names(rules)


def test_json_field_reads_null_numbers_strings_and_escapes() -> None:
    texts = pa.array([json.dumps({"a": None}), json.dumps({"a": 9}),
                      json.dumps({"a": 'say "hi"'}), None, json.dumps({"b": 1})])
    assert curation.json_field(texts, "a") == [None, 9, 'say "hi"', None, None]


def test_check_json_fields_agrees_with_json_loads() -> None:
    table = batch([chembl_row(data_validity_comment="Outside typical range"),
                   chembl_row(confidence_score=8)])
    fields = curation.chembl_fields(table)
    assert curation.check_json_fields(table, fields, 10) == 2


def test_check_json_fields_stops_on_a_mismatch() -> None:
    table = batch([chembl_row()])
    fields = curation.chembl_fields(table)
    fields["confidence_score"] = [8]
    with pytest.raises(ValueError, match="confidence_score"):
        curation.check_json_fields(table, fields, 10)


def test_a_clean_chembl_row_has_no_reason(settings: dict[str, Any]) -> None:
    flags = curation.row_reasons(batch([chembl_row()]), True, settings, {}, 1)
    assert not any(values[0] for values in flags.values())


@pytest.mark.parametrize(("fields", "reason"), [
    ({"data_validity_comment": "Manually validated"}, curation.CHEMBL_DATA_VALIDITY),
    ({"pchembl_value": None}, curation.CHEMBL_NO_PCHEMBL),
    ({"activity_comment": " Not Active "}, curation.CHEMBL_INACTIVE_COMMENT),
    ({"potential_duplicate": 1}, curation.CHEMBL_POTENTIAL_DUPLICATE),
    ({"confidence_score": 8}, curation.CHEMBL_CONFIDENCE),
    ({"confidence_score": None}, curation.CHEMBL_CONFIDENCE),
    ({"relationship_type": "H"}, curation.CHEMBL_RELATIONSHIP),
])
def test_each_chembl_field_rule(fields: dict[str, Any], reason: str,
                                settings: dict[str, Any]) -> None:
    flags = curation.row_reasons(batch([chembl_row(**fields)]), True, settings, {}, 1)
    assert [name for name, values in flags.items() if values[0]] == [reason]


def test_uncertain_comment_is_not_an_inactive_comment(settings: dict[str, Any]) -> None:
    flags = curation.row_reasons(batch([chembl_row(activity_comment="uncertain")]), True,
                                 settings, {}, 1)
    assert not flags[curation.CHEMBL_INACTIVE_COMMENT][0]


def test_mutant_rule_needs_a_missing_variant(settings: dict[str, Any]) -> None:
    rows = [chembl_row(description="Inhibition of EGFR T790M MUTANT"),
            chembl_row(description="Inhibition of EGFR T790M mutant", variant="12"),
            chembl_row(description="Inhibition of wild type EGFR")]
    flags = curation.row_reasons(batch(rows), True, settings, {}, 3)
    assert flags[curation.CHEMBL_MUTANT].tolist() == [True, False, False]


def test_bindingdb_rows_skip_chembl_rules_and_get_the_typical_range(
        settings: dict[str, Any]) -> None:
    rows = [{"p_activity": 3.9}, {"p_activity": 4.0}, {"p_activity": 11.0},
            {"p_activity": 11.2}, {"p_activity": 1.5}]
    flags = curation.row_reasons(batch(rows), False, settings, {}, 0)
    assert flags[curation.BINDINGDB_TYPICAL_RANGE].tolist() == [True, False, False, True, True]
    assert flags[curation.P_OUTSIDE].tolist() == [False, False, False, False, True]
    assert not flags[curation.CHEMBL_CONFIDENCE].any()


def test_typical_range_does_not_apply_to_chembl(settings: dict[str, Any]) -> None:
    flags = curation.row_reasons(batch([chembl_row() | {"p_activity": 3.5}]), True, settings,
                                 {}, 1)
    assert not flags[curation.BINDINGDB_TYPICAL_RANGE][0]


def test_bindingdb_conflict_matches_whole_tokens(settings: dict[str, Any]) -> None:
    rows = [{"dedup_reason": "protein_missing;value_differs"},
            {"dedup_reason": "value_qualifier_differs"}, {"dedup_reason": "counterparts:1"},
            {"dedup_reason": None}, {"dedup_reason": "value_differs_not"}]
    flags = curation.row_reasons(batch(rows), False, settings, {}, 0)
    assert flags[curation.BINDINGDB_CONFLICT].tolist() == [True, True, False, False, False]


def test_nonstandard_sequence_uses_the_hash_normalisation(settings: dict[str, Any]) -> None:
    rows = [{"protein_hash": "h1", "protein_sequence": "mk t"},
            {"protein_hash": "h2", "protein_sequence": "MK1T"}]
    cache: dict[str, bool] = {}
    flags = curation.row_reasons(batch(rows), False, settings, cache, 0)
    assert flags[curation.NONSTANDARD_SEQUENCE].tolist() == [False, True]
    assert cache == {"h1": False, "h2": True}


# --------------------------------------------------------------------------- theo nhóm


def group_frame(rows: list[tuple[str, str, float]], protein: str = "P",
                endpoint: str = "Ki") -> pd.DataFrame:
    """(assay, phân tử, p) -> bảng nhóm như pipeline dựng."""
    return pd.DataFrame({"observation_id": [f"o{index}" for index in range(len(rows))],
                         "assay_key": [row[0] for row in rows],
                         "protein_hash": protein, "endpoint": endpoint,
                         "molecule_key": [row[1] for row in rows],
                         "p_activity": [row[2] for row in rows]})


def group_flags(frame: pd.DataFrame, settings: dict[str, Any],
                excluded: list[bool] | None = None) -> dict[str, np.ndarray]:
    """group_reasons với mảng 'đã bị quy tắc theo dòng bắt' (mặc định: không dòng nào)."""
    mask = np.zeros(len(frame), dtype=bool) if excluded is None else np.array(excluded)
    return curation.group_reasons(frame, settings, mask)


def test_unit_error_marks_both_assays_of_an_exact_shift(settings: dict[str, Any]) -> None:
    frame = group_frame([("a1", "M", 9.52), ("a2", "M", 3.52), ("a1", "N", 7.0),
                         ("a2", "N", 7.4), ("a3", "M", 9.0)])
    flags = group_flags(frame, settings)
    assert flags[curation.UNIT_ERROR].tolist() == [True, True, False, False, False]


def test_unit_error_allows_the_rounding_tolerance(settings: dict[str, Any]) -> None:
    frame = group_frame([("a1", "M", 8.00), ("a2", "M", 5.01), ("b1", "N", 8.00),
                         ("b2", "N", 5.03)])
    assert group_flags(frame, settings)[curation.UNIT_ERROR].tolist() == [
        True, True, False, False]


def test_unit_error_needs_the_same_protein() -> None:
    first = group_frame([("a1", "M", 9.0)], protein="P")
    second = group_frame([("a2", "M", 6.0)], protein="Q")
    means = curation.molecule_means(pd.concat([first, second], ignore_index=True))
    assert curation.unit_error_keys(means, [3.0], 0.01, 2).empty


def test_top_concentration_ties_marks_only_the_molecules_at_the_minimum(
        settings: dict[str, Any]) -> None:
    rows = [("a", f"M{index}", 5.0) for index in range(3)] + [("a", "X", 7.2),
                                                              ("a", "Y", 6.1)]
    flags = group_flags(group_frame(rows), settings)
    assert flags[curation.TOP_CONCENTRATION].tolist() == [True, True, True, False, False]


def test_two_molecules_at_the_minimum_are_not_a_ceiling(settings: dict[str, Any]) -> None:
    rows = [("a", "M1", 5.0), ("a", "M2", 5.0), ("a", "X", 7.2)]
    flags = group_flags(group_frame(rows), settings)
    assert not flags[curation.TOP_CONCENTRATION].any()


def test_binned_assay_marks_the_whole_assay(settings: dict[str, Any]) -> None:
    levels = [8.26, 7.26, 6.26, 5.26]
    binned = [("bin", f"M{index}", levels[index % 4]) for index in range(10)]
    measured = [("real", f"M{index}", 5.0 + 0.1 * index) for index in range(10)]
    flags = group_flags(group_frame(binned + measured), settings)
    assert flags[curation.BINNED_ASSAY].tolist() == [True] * 10 + [False] * 10


def binned_with_strays(strays: int, assay: str = "a") -> list[tuple[str, str, float]]:
    """(20 − strays) phân tử ở 4 mức chia khoảng, `strays` phân tử giá trị lẻ.

    Mức thấp nhất chỉ có 2 phân tử, để quy tắc nồng độ trần (>= 3) không đụng tới.
    """
    levels = [5.26] * 2 + [6.26] * 4 + [7.26] * 5 + [8.26] * (20 - strays - 11)
    rows = [(assay, f"{assay}{index}", level) for index, level in enumerate(levels)]
    return rows + [(assay, f"{assay}s{index}", 6.5 + 0.1 * index) for index in range(strays)]


def test_binned_assay_tolerates_a_few_stray_values(settings: dict[str, Any]) -> None:
    rows = binned_with_strays(1, "one") + binned_with_strays(2, "two")
    flags = group_flags(group_frame(rows), settings)
    assert flags[curation.BINNED_ASSAY].tolist() == [True] * 20 + [False] * 20


def test_binned_assay_works_on_categorical_columns(settings: dict[str, Any]) -> None:
    frame = group_frame(binned_with_strays(1) + [("m", f"M{index}", 5.0 + 0.1 * index)
                                                 for index in range(10)])
    for column in ["assay_key", "protein_hash", "endpoint", "molecule_key"]:
        frame[column] = frame[column].astype("category")
    flags = group_flags(frame, settings)
    assert flags[curation.BINNED_ASSAY].tolist() == [True] * 20 + [False] * 10


def test_ties_are_found_below_a_row_that_another_rule_removed(
        settings: dict[str, Any]) -> None:
    # Một dòng p 3,2 (đã bị quy tắc theo dòng bắt) không được che ba phân tử ghi đúng trần.
    rows = [("a", "W", 3.2)] + [("a", f"M{index}", 4.0) for index in range(3)] + [
        ("a", "X", 6.1), ("a", "Y", 7.3)]
    flags = group_flags(group_frame(rows), settings, [True] + [False] * 5)
    assert flags[curation.TOP_CONCENTRATION].tolist() == [False, True, True, True, False,
                                                          False]


def test_binned_assay_ignores_the_ceiling_values(settings: dict[str, Any]) -> None:
    # 12 phân tử ở trần 4,0 rồi 20 phân tử chia khoảng có 1 giá trị lẻ: tính cả trần thì
    # 4 giá trị phổ biến nhất chỉ chiếm 30/32 < 0,95; bỏ trần thì 19/20 = 0,95.
    ceiling = [("a", f"C{index}", 4.0) for index in range(12)]
    flags = group_flags(group_frame(ceiling + binned_with_strays(1)), settings)
    assert flags[curation.TOP_CONCENTRATION].tolist() == [True] * 12 + [False] * 20
    assert flags[curation.BINNED_ASSAY].all()


def test_unit_error_still_sees_rows_that_another_rule_removed(
        settings: dict[str, Any]) -> None:
    frame = group_frame([("a1", "M", 9.0), ("a2", "M", 6.0)])
    flags = group_flags(frame, settings, [True, False])
    assert flags[curation.UNIT_ERROR].tolist() == [True, True]


def test_binned_assay_needs_enough_molecules(settings: dict[str, Any]) -> None:
    rows = [("small", f"M{index}", 6.26) for index in range(9)]
    assert not group_flags(group_frame(rows), settings)[
        curation.BINNED_ASSAY].any()


def test_replicates_count_as_one_molecule(settings: dict[str, Any]) -> None:
    rows = [("a", "M1", 5.0), ("a", "M1", 5.0), ("a", "M1", 5.0), ("a", "X", 6.0)]
    flags = group_flags(group_frame(rows), settings)
    assert not flags[curation.TOP_CONCENTRATION].any()


# --------------------------------------------------------------------------- tổng hợp


def test_reason_lists_keep_the_declared_order() -> None:
    flags = {curation.BINNED_ASSAY: np.array([True, False]),
             curation.CHEMBL_DATA_VALIDITY: np.array([True, False])}
    assert curation.reason_lists(flags) == [
        [curation.CHEMBL_DATA_VALIDITY, curation.BINNED_ASSAY], []]


def test_reason_counts_separate_rows_caught_by_one_rule_only() -> None:
    flags = {curation.CHEMBL_DATA_VALIDITY: np.array([True, True, False]),
             curation.P_OUTSIDE: np.array([True, False, False])}
    counts = curation.reason_counts(flags)
    assert counts[curation.CHEMBL_DATA_VALIDITY] == {"rows": 2, "only_this_rule": 1}
    assert counts[curation.P_OUTSIDE] == {"rows": 1, "only_this_rule": 0}
