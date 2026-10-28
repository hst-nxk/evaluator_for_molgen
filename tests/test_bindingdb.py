"""BindingDB protein mapping, assay mapping (rsid -> ENTRYID/ASSAYID) and record building."""

from __future__ import annotations

from typing import Any

import pytest

from molopt.observations import bindingdb

CONSTRUCT_99_AA = "P" * 99
POLYPROTEIN_1447_AA = "M" * 1447


def make_row(values: dict[str, str], roles: list[str]) -> tuple[list[str], dict[str, int]]:
    """A synthetic BindingDB_All row with one column per role."""
    columns = {role: index for index, role in enumerate(roles)}
    return [values.get(role, "") for role in roles], columns


def records_for(values: dict[str, str], config: dict[str, Any],
                assay: bindingdb.AssayResolution) -> list[dict[str, Any]]:
    endpoints = config["endpoints"]
    roles = list(config["bindingdb"]["all_columns"]) + endpoints
    fields, columns = make_row(values, roles)
    endpoint_columns = {endpoint: columns[endpoint] for endpoint in endpoints}
    units = {endpoint: spec["unit"]
             for endpoint, spec in config["bindingdb"]["endpoint_columns"].items()}
    return bindingdb.build_records(fields, columns, endpoint_columns, units, assay, "202610")


RESOLVED_ASSAY = bindingdb.AssayResolution("7", "1", "Name", "Desc", bindingdb.RESOLVED, None)


# --------------------------------------------------------------------------- protein


def test_single_chain_uses_its_own_sequence_and_swissprot() -> None:
    result = bindingdb.resolve_protein("1", CONSTRUCT_99_AA, "P03367", "Q00000")
    assert result.status == bindingdb.RESOLVED
    assert (result.sequence, result.accession, result.accession_db) == (
        CONSTRUCT_99_AA, "P03367", bindingdb.SWISSPROT)


def test_trembl_used_only_when_swissprot_absent() -> None:
    result = bindingdb.resolve_protein("1", CONSTRUCT_99_AA, None, "Q00000")
    assert (result.accession, result.accession_db) == ("Q00000", bindingdb.TREMBL)


def test_multichain_target_is_unresolved() -> None:
    result = bindingdb.resolve_protein("2", CONSTRUCT_99_AA, "P03367", None)
    assert result.status == bindingdb.UNRESOLVED
    assert result.reason == "target_chain_count_not_1:2"
    assert result.sequence is None


def test_missing_chain_sequence_is_unresolved() -> None:
    assert bindingdb.resolve_protein("1", None, "P03367", None).reason == "chain_sequence_missing"


def test_same_accession_keeps_each_measurements_own_construct(config: dict[str, Any]) -> None:
    short = records_for({"reactant_set_id": "1", "chain_count": "1",
                         "chain_1_sequence": CONSTRUCT_99_AA,
                         "chain_1_swissprot_id": "P03367", "Ki": "5"}, config, RESOLVED_ASSAY)
    long = records_for({"reactant_set_id": "2", "chain_count": "1",
                        "chain_1_sequence": POLYPROTEIN_1447_AA,
                        "chain_1_swissprot_id": "P03367", "Ki": "5"}, config, RESOLVED_ASSAY)
    assert len(short[0]["protein_sequence"]) == 99
    assert len(long[0]["protein_sequence"]) == 1447


# --------------------------------------------------------------------------- assay


def test_assay_resolved_through_rsid_link_not_entryid() -> None:
    links = {"100": [("7", "1")]}
    assays = {("100", "1"): ("WRONG: rsid used as ENTRYID", None), ("7", "1"): ("Right", "D")}
    result = bindingdb.resolve_assay("100", links, assays)
    assert result.status == bindingdb.RESOLVED
    assert (result.entry_id, result.assay_id, result.name) == ("7", "1", "Right")


def test_assay_without_link_is_unresolved() -> None:
    result = bindingdb.resolve_assay("100", {}, {("100", "1"): ("x", None)})
    assert result.status == bindingdb.UNRESOLVED
    assert result.reason == "no_rsid_assay_link"


def test_multiple_assay_links_are_unresolved() -> None:
    result = bindingdb.resolve_assay("100", {"100": [("7", "1"), ("7", "2")]}, {})
    assert result.reason == "multiple_assay_links:2"
    assert result.assay_id is None


def test_linked_pair_missing_from_assay_table_is_unresolved() -> None:
    result = bindingdb.resolve_assay("100", {"100": [("7", "1")]}, {})
    assert result.reason == "assay_pair_not_in_assay_table"
    assert (result.entry_id, result.assay_id) == ("7", "1")


def test_entry_assay_id_parsing_fails_loudly_on_unexpected_shape() -> None:
    assert bindingdb.parse_entry_assay_id("123_4", "_", 2) == ("123", "4")
    with pytest.raises(ValueError, match="line 9"):
        bindingdb.parse_entry_assay_id("1234", "_", 9)


def test_disjoint_key_domains_fail_loudly() -> None:
    with pytest.raises(ValueError, match="do not intersect"):
        bindingdb.assert_domains_intersect("test", ["1", "2"], {"3"})
    assert bindingdb.assert_domains_intersect("test", ["1", "2"], {"2"}) == 1


# --------------------------------------------------------------------------- records


def test_one_record_per_non_empty_endpoint_with_verbatim_values(config: dict[str, Any]) -> None:
    records = records_for({"reactant_set_id": "55", "ligand_smiles": "C[C@H](N)C(=O)O |r|",
                           "chain_count": "1", "chain_1_sequence": CONSTRUCT_99_AA,
                           "Ki": ">10000", "IC50": "3.2", "pmid": "12345.0"},
                          config, RESOLVED_ASSAY)
    assert sorted(record["endpoint"] for record in records) == ["IC50", "Ki"]
    for record in records:
        assert record["molecule_source_smiles"] == "C[C@H](N)C(=O)O |r|"
        assert record["document_pmid_source"] == "12345.0"
        assert record["activity_unit_source"] == "nM"
    assert {record["activity_value_source"] for record in records} == {">10000", "3.2"}


def test_ligand_chembl_id_kept_verbatim_and_type_left_empty(config: dict[str, Any]) -> None:
    record = records_for({"reactant_set_id": "55", "chembl_id": " CHEMBL25", "Ki": "5"},
                         config, RESOLVED_ASSAY)[0]
    assert record["molecule_source_chembl_id"] == " CHEMBL25"
    assert record["molecule_source_type"] is None


def test_row_wider_than_header_is_rejected() -> None:
    with pytest.raises(ValueError, match="line 4"):
        bindingdb.check_row_width(["a", "b", "c"], 2, 4, "BindingDB_All.tsv")
