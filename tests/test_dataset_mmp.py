"""Logic thuần của src/molopt/dataset/mmp.py (khóa định danh, cắt phân tử, tạo cặp)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pyarrow as pa
import pytest

from molopt.dataset import mmp

TOLUENE = "Cc1ccccc1"
ETHYLBENZENE = "CCc1ccccc1"
BENZENE = "c1ccccc1"
SUFFIX = "-REL"


@pytest.fixture()
def settings(config: dict[str, Any]) -> mmp.FragmentSettings:
    """FragmentSettings đúng tham số trong config.json."""
    mmp_settings = config["stage2b"]["mmp"]
    return mmp.FragmentSettings(
        min_cuts=mmp_settings["min_cuts"], max_cuts=mmp_settings["max_cuts"],
        max_cut_bonds=mmp_settings["max_cut_bonds"],
        cut_smarts=mmp_settings["cut_smarts"],
        min_core_fraction=mmp_settings["min_core_fraction"],
        max_variable_heavy_atoms=mmp_settings["max_variable_heavy_atoms"],
        attachment_points=mmp_settings["attachment_points"],
        attachment_degree=mmp_settings["attachment_degree"])


@pytest.fixture()
def cutter(settings: mmp.FragmentSettings):
    """Hàm cắt dùng cho find_pairs trong các test (không cache)."""
    return lambda smiles: mmp.fragment_molecule(smiles, settings)


def molecules(*items: tuple[int, str, bool]) -> list[mmp.GroupMolecule]:
    return [mmp.GroupMolecule(row, smiles, relative) for row, smiles, relative in items]


# --------------------------------------------------------------------------- khóa định danh


def test_relative_stereo_mask_uses_config_pattern(config: dict[str, Any]) -> None:
    pattern = config["stage2a"]["dedup"]["relative_stereo_pattern"]
    values = pa.array(["|r|", "|r,c:4|", "|&1:3|", None])
    assert mmp.relative_stereo_mask(values, pattern).to_pylist() == [True, True, False, False]


def test_identity_keys_adds_suffix_only_to_relative_rows() -> None:
    keys = mmp.identity_keys(pa.array(["AAA-BBB-N", "CCC-DDD-N", None]),
                             pa.array([True, False, True]), SUFFIX)
    assert keys.to_pylist() == ["AAA-BBB-N-REL", "CCC-DDD-N", None]


def test_is_relative_key_reads_the_suffix_back() -> None:
    assert mmp.is_relative_key("AAA-BBB-N-REL", SUFFIX)
    assert not mmp.is_relative_key("AAA-BBB-N", SUFFIX)


# --------------------------------------------------------------------------- cắt phân tử


def test_fragment_molecule_keeps_only_the_large_core(settings: mmp.FragmentSettings) -> None:
    # Toluene có 7 nguyên tử nặng; lõi phải ≥ 60 % = 4,2 nên chỉ vòng benzen làm lõi được,
    # còn nhóm metyl (1 nguyên tử) chỉ làm phần biến đổi.
    result = mmp.fragment_molecule(TOLUENE, settings)
    assert result.failed is False
    assert len(result.cuts) == 1
    (cut,) = result.cuts
    assert cut.variable_heavy == 1
    assert "*" in cut.constant and "*" in cut.variable


def test_fragment_molecule_shares_the_core_between_analogs(
        settings: mmp.FragmentSettings) -> None:
    first = mmp.fragment_molecule(TOLUENE, settings).cuts
    second = mmp.fragment_molecule(ETHYLBENZENE, settings).cuts
    assert {cut.constant for cut in first} & {cut.constant for cut in second}


def test_fragment_molecule_of_molecule_without_cut_bonds(
        settings: mmp.FragmentSettings) -> None:
    # Benzen không có liên kết đơn ngoài vòng nào để cắt.
    assert mmp.fragment_molecule(BENZENE, settings).cuts == ()


def test_fragment_molecule_reports_unreadable_smiles(settings: mmp.FragmentSettings) -> None:
    result = mmp.fragment_molecule("không phải SMILES", settings)
    assert result.failed is True and result.cuts == ()


# Mạch thẳng 22 carbon: 21 liên kết C-C đơn ngoài vòng, đều khớp mẫu cắt.
LONG_CHAIN = "C" * 22
LONG_CHAIN_CUT_BONDS = 21


def test_long_chain_has_the_expected_number_of_cut_bonds(
        settings: mmp.FragmentSettings) -> None:
    molecule = mmp.Chem.MolFromSmiles(LONG_CHAIN)
    pattern = mmp.cut_pattern(settings.cut_smarts)
    assert len(molecule.GetSubstructMatches(pattern)) == LONG_CHAIN_CUT_BONDS


def test_fragment_molecule_raises_instead_of_dropping_a_large_molecule(
        settings: mmp.FragmentSettings) -> None:
    # Với max_cut_bonds < số liên kết cắt được, RDKit trả rỗng cho CẢ phân tử; v1 mất
    # cặp thầm lặng vì vậy. v2 phải báo lỗi.
    small = dataclasses.replace(settings, max_cut_bonds=LONG_CHAIN_CUT_BONDS - 1)
    with pytest.raises(ValueError, match="max_cut_bonds"):
        mmp.fragment_molecule(LONG_CHAIN, small)


def test_fragment_molecule_cuts_a_large_molecule_under_config_limit(
        settings: mmp.FragmentSettings) -> None:
    assert settings.max_cut_bonds >= LONG_CHAIN_CUT_BONDS
    cuts = mmp.fragment_molecule(LONG_CHAIN, settings).cuts
    # Phép sửa vẫn nhỏ: giới hạn phần biến đổi không phụ thuộc max_cut_bonds.
    assert cuts
    assert max(cut.variable_heavy for cut in cuts) <= settings.max_variable_heavy_atoms


def test_cut_pattern_rejects_bad_smarts() -> None:
    with pytest.raises(ValueError, match="cut_smarts"):
        mmp.cut_pattern("[[[")


def test_fragment_molecule_respects_variable_size_limit(
        settings: mmp.FragmentSettings) -> None:
    small = dataclasses.replace(settings, max_variable_heavy_atoms=0)
    assert mmp.fragment_molecule(TOLUENE, small).cuts == ()


def test_cut_score_prefers_the_smaller_variable_part() -> None:
    small = mmp.Cut("core", "[*:1]C", 1)
    large = mmp.Cut("core", "[*:1]CCCC", 4)
    assert mmp.cut_score(small, small) < mmp.cut_score(small, large)


# --------------------------------------------------------------------------- tạo cặp


def test_find_pairs_matches_analogs_on_the_same_core(cutter) -> None:
    pairs, lost = mmp.find_pairs(molecules((0, TOLUENE, False), (1, ETHYLBENZENE, False)),
                                 cutter)
    assert lost == set()
    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair.row_x, pair.row_y) == (0, 1)
    assert pair.variable_x != pair.variable_y


def test_find_pairs_drops_pairs_mixing_relative_stereo(cutter) -> None:
    pairs, lost = mmp.find_pairs(molecules((0, TOLUENE, False), (1, ETHYLBENZENE, True)),
                                 cutter)
    assert pairs == []
    assert lost == {(0, 1)}


def test_find_pairs_keeps_pairs_when_both_sides_are_relative(cutter) -> None:
    pairs, lost = mmp.find_pairs(molecules((0, TOLUENE, True), (1, ETHYLBENZENE, True)),
                                 cutter)
    assert len(pairs) == 1 and lost == set()


def test_find_pairs_needs_two_molecules(cutter) -> None:
    assert mmp.find_pairs(molecules((0, TOLUENE, False)), cutter) == ([], set())


def test_find_pairs_ignores_molecules_without_a_common_core(cutter) -> None:
    pairs, lost = mmp.find_pairs(molecules((0, TOLUENE, False), (1, "CCCCO", False)), cutter)
    assert pairs == [] and lost == set()


def test_find_pairs_returns_each_unordered_pair_once(cutter) -> None:
    pairs, _ = mmp.find_pairs(molecules((0, TOLUENE, False), (1, ETHYLBENZENE, False),
                                        (2, "CCCc1ccccc1", False)), cutter)
    assert sorted((pair.row_x, pair.row_y) for pair in pairs) == [(0, 1), (0, 2), (1, 2)]


def test_process_task_requires_init_worker(settings: mmp.FragmentSettings) -> None:
    mmp._FRAGMENTS = None
    with pytest.raises(RuntimeError):
        mmp.process_task([(0, molecules((0, TOLUENE, False)))])


def test_process_task_counts_molecules_and_pairs(settings: mmp.FragmentSettings) -> None:
    mmp.init_worker(settings, 16)
    result = mmp.process_task([(7, molecules((0, TOLUENE, False), (1, ETHYLBENZENE, False))),
                               (8, molecules((2, TOLUENE, False), (3, "bad smiles", False)))])
    assert result.molecules == 4
    assert result.failed_smiles == 1
    assert [group for group, _ in result.pairs] == [7]
    assert result.lost == []
