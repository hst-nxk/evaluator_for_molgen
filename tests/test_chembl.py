"""ChEMBL protein mapping: declared variant first, never wild type as a substitute."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from molopt.observations import chembl

WILD_TYPE = "MWILDTYPESEQ"
VARIANT = "MVARIANTSEQ"


def test_declared_variant_sequence_wins_over_component() -> None:
    result = chembl.resolve_protein(7, VARIANT, "P00001", 1, WILD_TYPE, "P00001")
    assert result.status == chembl.RESOLVED
    assert result.sequence == VARIANT
    assert result.origin == chembl.SEQUENCE_ORIGIN_VARIANT


def test_declared_variant_without_sequence_is_unresolved_not_wild_type() -> None:
    result = chembl.resolve_protein(7, None, "P00001", 1, WILD_TYPE, "P00001")
    assert result.status == chembl.UNRESOLVED
    assert result.reason == "variant_declared_but_sequence_missing"
    assert result.sequence is None


def test_no_variant_uses_single_component_sequence() -> None:
    result = chembl.resolve_protein(None, None, None, 1, WILD_TYPE, "P00001")
    assert result.status == chembl.RESOLVED
    assert (result.sequence, result.accession) == (WILD_TYPE, "P00001")
    assert result.origin == chembl.SEQUENCE_ORIGIN_COMPONENT


def test_multi_component_target_is_unresolved() -> None:
    result = chembl.resolve_protein(None, None, None, 3, WILD_TYPE, "P00001")
    assert result.status == chembl.UNRESOLVED
    assert result.reason == "target_component_count_not_1:3"
    assert result.sequence is None


def test_target_without_protein_component_is_unresolved() -> None:
    result = chembl.resolve_protein(None, None, None, None, None, None)
    assert result.reason == "target_has_no_protein_component"


def test_text_or_none_keeps_values_verbatim() -> None:
    assert chembl.text_or_none(12345) == "12345"
    assert chembl.text_or_none(" ") is None
    assert chembl.text_or_none(None) is None


def test_molecule_type_lookup_by_chembl_id(tmp_path: Path) -> None:
    db_path = tmp_path / "mini.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE molecule_dictionary (chembl_id TEXT, molecule_type TEXT)")
    connection.executemany("INSERT INTO molecule_dictionary VALUES (?, ?)",
                           [("CHEMBL1", "Small molecule"), ("CHEMBL2", "Protein"),
                            ("CHEMBL3", None)])
    connection.commit()
    connection.close()
    lookup = chembl.MoleculeTypeLookup(db_path, cache_size=10)
    try:
        assert lookup("CHEMBL1") == "Small molecule"
        assert lookup("CHEMBL2") == "Protein"
        assert lookup("CHEMBL3") is None
        assert lookup("CHEMBL404") is None
    finally:
        lookup.close()


def test_molecule_type_lookup_does_not_open_database_until_needed(tmp_path: Path) -> None:
    lookup = chembl.MoleculeTypeLookup(tmp_path / "absent.db", cache_size=10)
    lookup.close()
    assert lookup.connection is None


def test_selected_activity_ids_are_ascending_and_limited() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE activities (activity_id INTEGER, standard_type TEXT)")
    connection.executemany("INSERT INTO activities VALUES (?, ?)",
                           [(30, "Ki"), (10, "IC50"), (20, "Potency"), (5, "Kd"), (40, "EC50")])
    endpoints = ["Ki", "Kd", "IC50", "EC50"]
    assert list(chembl.selected_activity_ids(connection, endpoints, None, 2)) == [5, 10, 30, 40]
    assert list(chembl.selected_activity_ids(connection, endpoints, 2, 2)) == [5, 10]
