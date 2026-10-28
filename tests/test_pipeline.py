"""Observation rows: source columns untouched, derived columns added alongside."""

from __future__ import annotations

import math
from typing import Any

import pytest

from molopt.observations import bindingdb, pipeline
from test_bindingdb import RESOLVED_ASSAY, records_for


def observations(config: dict[str, Any], values: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {record["endpoint"]: pipeline.build_observation(record, config)
            for record in records_for(values, config, RESOLVED_ASSAY)}


def test_enhanced_stereo_marker_survives_in_source_column(config: dict[str, Any]) -> None:
    rows = observations(config, {"reactant_set_id": "9", "monomer_id": "13590",
                                 "ligand_smiles": "C[C@H](N)C(=O)O |r|", "Ki": "5"})
    row = rows["Ki"]
    assert row["molecule_source_smiles"] == "C[C@H](N)C(=O)O |r|"
    assert row["molecule_cx_extension"] == "|r|"


def test_bindingdb_censored_value_and_pmid_form(config: dict[str, Any]) -> None:
    rows = observations(config, {"reactant_set_id": "9", "ligand_smiles": "CCO",
                                 "Ki": ">10000", "IC50": "1", "pmid": "12345.0"})
    censored, exact = rows["Ki"], rows["IC50"]
    assert censored["activity_value_source"] == ">10000"
    assert censored["activity_qualifier"] == ">"
    assert censored["activity_censored"] is True
    assert censored["activity_p_activity"] is None
    assert math.isclose(exact["activity_p_activity"], 9.0)
    assert exact["document_pmid"] == "12345"
    assert exact["document_pmid_source"] == "12345.0"


def test_observation_id_does_not_change_with_protein(config: dict[str, Any]) -> None:
    first = observations(config, {"reactant_set_id": "9", "chain_count": "1",
                                  "chain_1_sequence": "AAAA", "Ki": "5"})["Ki"]
    second = observations(config, {"reactant_set_id": "9", "chain_count": "1",
                                   "chain_1_sequence": "CCCCCC", "Ki": "5"})["Ki"]
    assert first["observation_id"] == second["observation_id"] == "bindingdb:9:Ki"
    assert first["protein_sequence_hash"] != second["protein_sequence_hash"]


def test_unknown_assay_does_not_block_protein(config: dict[str, Any]) -> None:
    unknown = bindingdb.AssayResolution(None, None, None, None, bindingdb.UNRESOLVED,
                                        "no_rsid_assay_link")
    values = {"reactant_set_id": "9", "chain_count": "1", "chain_1_sequence": "AAAA", "Ki": "5"}
    record = records_for(values, config, unknown)[0]
    row = pipeline.build_observation(record, config)
    assert row["assay_status"] == "unresolved"
    assert row["protein_status"] == "resolved"
    assert row["activity_status"] == "resolved"


def test_salt_row_gets_parent_identifiers_and_source_kept(config: dict[str, Any]) -> None:
    salt = "C[NH+](C)CCOC(=O)c1ccccc1.[Cl-]"
    row = observations(config, {"reactant_set_id": "9", "ligand_smiles": salt, "Ki": "5"})["Ki"]
    free_base = observations(config, {"reactant_set_id": "10", "ligand_smiles":
                                      "CN(C)CCOC(=O)c1ccccc1", "Ki": "5"})["Ki"]
    assert row["molecule_source_smiles"] == salt
    assert row["molecule_inchikey"] == free_base["molecule_inchikey"]
    assert row["molecule_input_inchikey"] != free_base["molecule_input_inchikey"]
    assert row["molecule_parent_fragments"] == 1


def test_bindingdb_molecule_type_comes_from_chembl_lookup(config: dict[str, Any]) -> None:
    asked: list[str] = []

    def lookup(chembl_id: str) -> str | None:
        asked.append(chembl_id)
        return "Protein"

    good, bad = (records_for({"reactant_set_id": rsid, "chembl_id": chembl_id, "Ki": "5"},
                             config, RESOLVED_ASSAY)[0]
                 for rsid, chembl_id in (("9", " CHEMBL2 "), ("10", "not-an-id")))
    row = pipeline.build_observation(good, config, lookup)
    assert (row["molecule_chembl_id"], row["molecule_chembl_type"]) == ("CHEMBL2", "Protein")
    row = pipeline.build_observation(bad, config, lookup)
    assert (row["molecule_chembl_id"], row["molecule_chembl_type"]) == (None, None)
    assert asked == ["CHEMBL2"]


def test_chembl_molecule_type_is_taken_from_its_own_row(config: dict[str, Any]) -> None:
    record = {name: None for name in pipeline.SOURCE_COLUMN_NAMES}
    record.update({"source_database": "chembl", "native_measurement_id": "1", "endpoint": "Ki",
                   "molecule_source_smiles": "CCO", "molecule_source_chembl_id": "CHEMBL545",
                   "molecule_source_type": "Small molecule", "activity_value_source": "5",
                   "activity_unit_source": "nM", "activity_qualifier_source": "="})

    def never(chembl_id: str) -> str | None:
        raise AssertionError("ChEMBL rows must not be looked up")

    row = pipeline.build_observation(record, config, never)
    assert (row["molecule_chembl_id"], row["molecule_chembl_type"]) == (
        "CHEMBL545", "Small molecule")
    assert set(row) == set(pipeline.OBSERVATION_SCHEMA.names)


def test_record_with_unexpected_keys_is_rejected(config: dict[str, Any]) -> None:
    record = records_for({"reactant_set_id": "9", "Ki": "5"}, config, RESOLVED_ASSAY)[0]
    with pytest.raises(ValueError, match="bindingdb:9"):
        pipeline.build_observation({**record, "extra": 1}, config)


def test_full_skips_finished_source_unless_forced(
    config: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_records(label: str) -> Any:
        def iter_records(config: Any, root: Any, limit: Any) -> Any:
            calls.append(label)
            return iter(records_for({"reactant_set_id": "9", "Ki": "5"}, config, RESOLVED_ASSAY))
        return iter_records

    monkeypatch.setattr(pipeline, "source_outputs", lambda config: [
        ("first", fake_records("first"), "first.parquet"),
        ("second", fake_records("second"), "second.parquet"),
    ])
    pipeline.write_observations(fake_records("setup")(config, None, None),
                                tmp_path / "first.parquet", config, "setup")
    calls.clear()

    counts = pipeline.run_extract(config, tmp_path, tmp_path, None, force=False)
    assert calls == ["second"]
    assert counts == {"first": 1, "second": 1}

    calls.clear()
    pipeline.run_extract(config, tmp_path, tmp_path, None, force=True)
    assert calls == ["first", "second"]
