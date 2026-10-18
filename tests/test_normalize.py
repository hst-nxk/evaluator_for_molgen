"""Activity conversion, identifiers and stereochemistry handling in normalize.py."""

from __future__ import annotations

import math
from typing import Any

import pytest

from molopt.observations import normalize


def activity(config: dict[str, Any], value: str | None, unit: str | None,
             qualifier: str | None) -> normalize.ActivityResult:
    return normalize.parse_activity(value, unit, qualifier, config["unit_to_molar"],
                                    config["qualifiers"]["exact"],
                                    config["qualifiers"]["censored"])


def test_exact_nanomolar_converts_to_molar_and_pactivity(config: dict[str, Any]) -> None:
    result = activity(config, "1", "nM", "=")
    assert result.status == normalize.RESOLVED
    assert math.isclose(result.concentration_molar, 1e-9)
    assert math.isclose(result.p_activity, 9.0)
    assert result.censored is False


def test_micromolar_symbol_variants_use_table(config: dict[str, Any]) -> None:
    for unit in ("uM", "µM", "μM"):
        assert math.isclose(activity(config, "2", unit, "=").concentration_molar, 2e-6)


def test_unknown_unit_is_unresolved_not_guessed(config: dict[str, Any]) -> None:
    result = activity(config, "5", "ug.mL-1", "=")
    assert result.status == normalize.UNRESOLVED
    assert result.reason == "unit_not_in_table:ug.mL-1"
    assert result.concentration_molar is None and result.p_activity is None


def test_missing_unit_is_unresolved(config: dict[str, Any]) -> None:
    assert activity(config, "5", None, "=").reason == "missing_unit"


@pytest.mark.parametrize("value", ["0", "-3", "0.0"])
def test_non_positive_value_is_invalid(config: dict[str, Any], value: str) -> None:
    result = activity(config, value, "nM", "=")
    assert result.status == normalize.INVALID
    assert result.reason == f"non_positive_value:{value}"
    assert result.p_activity is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_value_is_invalid(config: dict[str, Any], value: str) -> None:
    result = activity(config, value, "nM", "=")
    assert result.status == normalize.INVALID
    assert result.reason == f"non_finite_value:{value}"


def test_non_numeric_value_is_invalid(config: dict[str, Any]) -> None:
    assert activity(config, "abc", "nM", "=").reason == "non_numeric_value:abc"


@pytest.mark.parametrize("qualifier", [">", "<", ">=", "<="])
def test_censored_qualifier_kept_and_marked(config: dict[str, Any], qualifier: str) -> None:
    result = activity(config, "10000", "nM", qualifier)
    assert result.status == normalize.RESOLVED
    assert result.qualifier == qualifier
    assert result.censored is True
    assert math.isclose(result.concentration_molar, 1e-5)
    assert result.p_activity is None
    assert result.reason == f"censored_qualifier:{qualifier}"


def test_unknown_qualifier_is_unresolved(config: dict[str, Any]) -> None:
    result = activity(config, "10", "nM", "~")
    assert result.status == normalize.UNRESOLVED
    assert result.reason == "qualifier_not_in_table:~"
    assert result.censored is None


def test_missing_qualifier_is_unresolved(config: dict[str, Any]) -> None:
    assert activity(config, "10", "nM", None).reason == "missing_qualifier"


@pytest.mark.parametrize(("text", "expected"), [
    (">10000", (">", "10000")),
    (">=5", (">=", "5")),
    ("< 0.1", ("<", "0.1")),
    ("12.5", ("=", "12.5")),
])
def test_split_embedded_qualifier(text: str, expected: tuple[str, str]) -> None:
    assert normalize.split_embedded_qualifier(text, ["=", "<", ">", "<=", ">="], "=") == expected


@pytest.mark.parametrize(("value", "expected"), [
    ("12345", "12345"), ("12345.0", "12345"), (12345, "12345"), (12345.0, "12345"),
    ("12345.5", None), ("", None), (None, None), ("abc", None), (float("nan"), None),
])
def test_pmid_is_integer_like_string(value: object, expected: str | None) -> None:
    assert normalize.normalize_integer_id(value) == expected


def test_observation_id_uses_source_native_id_and_endpoint_only() -> None:
    assert normalize.make_observation_id("bindingdb", "50012345", "Ki") == "bindingdb:50012345:Ki"


def test_sequence_hash_ignores_whitespace_and_case() -> None:
    assert normalize.sequence_hash("pqit lwqr\n") == normalize.sequence_hash("PQITLWQR")
    assert normalize.sequence_hash(None) is None


def test_cx_extension_kept_verbatim() -> None:
    assert normalize.cx_extension("C[C@H](N)C(=O)O |r|") == "|r|"
    assert normalize.cx_extension("CCO") is None


def test_relative_stereo_group_is_recorded_and_flagged() -> None:
    result = normalize.describe_structure("C[C@H](N)C(=O)O |&1:1|")
    assert result.status == normalize.RESOLVED
    assert result.cx_extension == "|&1:1|"
    assert result.stereo_groups is not None and result.stereo_groups.startswith("STEREO_AND")
    assert result.reason == "relative_or_racemic_stereo_groups_not_encoded_in_canonical_ids"
    assert "|" not in result.canonical_smiles


def test_plain_structure_gets_canonical_smiles_and_inchikey() -> None:
    result = normalize.describe_structure("OCC")
    assert result.status == normalize.RESOLVED
    assert result.canonical_smiles == "CCO"
    assert result.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert result.reason is None


def test_stereo_annotation_failure_keeps_identifiers_and_records_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cip(molecule: object) -> tuple[int, int]:
        raise RuntimeError("Digraph generation failed: more than 100000 nodes found.")

    monkeypatch.setattr(normalize, "count_stereo_centers", fail_cip)
    result = normalize.describe_structure("OCC")
    assert result.status == normalize.RESOLVED
    assert result.canonical_smiles == "CCO"
    assert result.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert result.stereo_centers is None and result.unspecified_stereo_centers is None
    assert result.reason == ("stereo_annotation_failed:"
                             "Digraph generation failed: more than 100000 nodes found.")


def test_unparseable_structure_is_invalid_without_identifiers() -> None:
    result = normalize.describe_structure("this is not smiles")
    assert result.status == normalize.INVALID
    assert result.canonical_smiles is None and result.inchikey is None


def test_missing_structure_is_unresolved() -> None:
    assert normalize.describe_structure("").reason == "missing_source_smiles"
