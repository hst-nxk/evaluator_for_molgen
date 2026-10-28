"""Activity conversion, identifiers and stereochemistry handling in normalize.py."""

from __future__ import annotations

import math
from typing import Any

import pytest
from chembl_structure_pipeline import standardizer
from rdkit import Chem

from molopt.observations import normalize


def canonical(smiles: str) -> str:
    """Canonical SMILES RDKit của một SMILES viết tay (để test không phụ thuộc cách viết)."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


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
    assert result.input_canonical_smiles == "CCO"
    assert result.input_inchikey == result.inchikey
    assert result.parent_fragments == 1 and result.parent_excluded is False
    assert result.reason is None


# --------------------------------------------------------------------------- parent (ChEMBL)

DRUG = "CN(C)CCOC(=O)c1ccccc1"


def test_counterion_is_removed_and_parent_is_neutral() -> None:
    result = normalize.describe_structure("C[NH+](C)CCOC(=O)c1ccccc1.[Cl-]")
    assert result.status == normalize.RESOLVED
    assert result.canonical_smiles == canonical(DRUG)
    assert result.inchikey == normalize.describe_structure(DRUG).inchikey
    assert "Cl" in result.input_canonical_smiles
    assert result.parent_fragments == 1
    assert normalize.REASON_FRAGMENTS_REMOVED in result.reason


def test_protonated_and_neutral_forms_share_parent_identifier() -> None:
    protonated = normalize.describe_structure("C[NH+](C)CCOC(=O)c1ccccc1")
    neutral = normalize.describe_structure(DRUG)
    assert protonated.inchikey == neutral.inchikey
    assert protonated.canonical_smiles == neutral.canonical_smiles
    assert protonated.input_inchikey != neutral.input_inchikey


def test_quaternary_ammonium_keeps_its_charge_after_iodide_removed() -> None:
    result = normalize.describe_structure("C[N+](C)(C)CCOC(=O)C.[I-]")
    assert result.canonical_smiles == canonical("C[N+](C)(C)CCOC(=O)C")
    assert result.parent_fragments == 1
    assert normalize.REASON_FRAGMENTS_REMOVED in result.reason


def test_isotopes_are_removed_and_recorded() -> None:
    labelled = normalize.describe_structure("[2H]c1ccc(CN2CCCC2)cc1")
    plain = normalize.describe_structure("c1ccc(CN2CCCC2)cc1")
    assert labelled.inchikey == plain.inchikey
    assert labelled.input_inchikey != plain.input_inchikey
    assert labelled.reason == normalize.REASON_ISOTOPES_REMOVED


def test_mixture_of_two_non_salt_components_is_kept_and_flagged() -> None:
    result = normalize.describe_structure("c1ccc(CN2CCCC2)cc1.Cc1ccc(N)cc1")
    assert result.status == normalize.RESOLVED
    assert result.parent_fragments == 2
    assert f"{normalize.REASON_MULTIPLE_FRAGMENTS}:2" in result.reason


def test_metal_compound_is_flagged_by_chembl_exclude_rule() -> None:
    result = normalize.describe_structure("N.N.Cl[Pt]Cl")
    assert result.status == normalize.RESOLVED
    assert result.parent_excluded is True
    assert normalize.REASON_EXCLUDED in result.reason


def test_parent_failure_is_unresolved_and_keeps_input_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(molecule: object) -> tuple[object, bool]:
        raise ValueError("boom")

    monkeypatch.setattr(normalize, "chembl_parent", fail)
    result = normalize.describe_structure("OCC")
    assert result.status == normalize.UNRESOLVED
    assert result.reason == "chembl_parent_failed:boom"
    assert result.canonical_smiles is None and result.inchikey is None
    assert result.input_canonical_smiles == "CCO"


@pytest.mark.parametrize("smiles", [
    "CCO", "NCCO", "CC[NH3+]", "CS(C)=O", "[O-][n+]1ccccc1", "C[N+](C)(C)CCOC(=O)C.[I-]",
    "CC(=O)[O-].[Na+]", "[2H]c1ccc(CN2CCCC2)cc1", "c1ccc(CN2CCCC2)cc1.Cl",
    "OC(=O)C(O)C(O)C(=O)O.c1ccc(CN2CCCC2)cc1", "N.N.Cl[Pt]Cl", "OC(=O)c1ccccc1",
    "C[C@H](N)C(=O)O", "[NH3+][C@@H](C)C(=O)[O-]", "O=[N+]([O-])c1ccccc1",
])
def test_single_fragment_parent_equals_package(smiles: str) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    reference, reference_excluded = standardizer.get_parent_mol(
        standardizer.standardize_mol(Chem.Mol(molecule)), neutralize=True,
        check_exclusion=True)
    parent, excluded = normalize.chembl_parent(molecule)
    assert excluded == reference_excluded
    assert (Chem.MolToSmiles(normalize.sanitized_copy(parent))
            == Chem.MolToSmiles(normalize.sanitized_copy(reference)))
    # Phân tử đầu vào không bị sửa.
    assert Chem.MolToSmiles(molecule) == canonical(smiles)


def test_structure_pipeline_version_mismatch_stops() -> None:
    normalize.check_structure_pipeline_version(normalize.chembl_structure_pipeline.__version__)
    with pytest.raises(RuntimeError, match="0.0.0"):
        normalize.check_structure_pipeline_version("0.0.0")


def test_config_pins_installed_structure_pipeline(config: dict[str, Any]) -> None:
    normalize.check_structure_pipeline_version(
        config["structure"]["chembl_structure_pipeline_version"])


@pytest.mark.parametrize(("source", "computed", "expected"), [
    ("LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", normalize.INCHIKEY_MATCH),
    ("XGEGDSLAQZJGCW-UHFFFAOYSA-N", "XGEGDSLAQZJGCW-HHGOQMMWSA-N",
     normalize.INCHIKEY_OTHER_LAYERS_DIFFER),
    ("POKZVJVCUGVHME-UHFFFAOYSA-N", "WWONKAWUQZQATD-UHFFFAOYSA-N",
     normalize.INCHIKEY_SKELETON_DIFFERS),
    (None, "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", None),
    ("LFQSCWFLJHTTHZ-UHFFFAOYSA-N", None, None),
])
def test_compare_source_inchikey(source: str | None, computed: str | None,
                                 expected: str | None) -> None:
    assert normalize.compare_source_inchikey(source, computed) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("CHEMBL25", "CHEMBL25"), (" CHEMBL25 ", "CHEMBL25"), ("", None), (None, None),
    ("CHEMBL25,CHEMBL26", None), ("chembl25", None), ("25", None),
])
def test_chembl_molecule_id_is_validated(text: str | None, expected: str | None) -> None:
    assert normalize.chembl_molecule_id(text) == expected


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
