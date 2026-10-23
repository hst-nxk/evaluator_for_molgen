"""Fingerprint, cache và cách chia chuỗi protein của Stage 3 (không cần torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from molopt.evaluator import features

BITS = 64


def test_pack_fingerprints_packs_eight_bits_into_one_byte() -> None:
    packed, failed = features.pack_fingerprints(["CCO", "c1ccccc1"], radius=2, bits=BITS,
                                                use_chirality=True)
    assert packed.shape == (2, BITS // 8)
    assert packed.dtype == np.uint8
    assert failed == []
    # Fingerprint của hai phân tử khác nhau không được trùng khít nhau.
    assert not np.array_equal(packed[0], packed[1])


def test_pack_fingerprints_reports_an_unreadable_smiles_instead_of_dropping_it() -> None:
    packed, failed = features.pack_fingerprints(["CCO", "khong-phai-smiles"], radius=2,
                                                bits=BITS, use_chirality=True)
    # Dòng vẫn tồn tại (toàn 0) để cache không lệch so với dataset.
    assert packed.shape == (2, BITS // 8)
    assert failed == [1]
    assert packed[1].tolist() == [0] * (BITS // 8)


def test_pack_fingerprints_is_deterministic() -> None:
    first, _ = features.pack_fingerprints(["CCO"], 2, BITS, True)
    second, _ = features.pack_fingerprints(["CCO"], 2, BITS, True)
    assert np.array_equal(first, second)


def test_use_chirality_changes_the_fingerprint_of_a_stereo_molecule() -> None:
    smiles = ["C[C@H](N)C(=O)O"]
    with_stereo, _ = features.pack_fingerprints(smiles, 2, 2048, True)
    without_stereo, _ = features.pack_fingerprints(smiles, 2, 2048, False)
    assert not np.array_equal(with_stereo, without_stereo)


def test_unpack_rows_returns_zero_one_floats_for_the_requested_rows() -> None:
    packed, _ = features.pack_fingerprints(["CCO", "c1ccccc1"], 2, BITS, True)
    unpacked = features.unpack_rows(packed, np.array([1]), BITS)
    assert unpacked.shape == (1, BITS)
    assert unpacked.dtype == np.float32
    assert set(np.unique(unpacked).tolist()) <= {0.0, 1.0}


def test_packed_cache_round_trip_keeps_keys_matrix_and_metadata(tmp_path: Path) -> None:
    packed, _ = features.pack_fingerprints(["CCO", "c1ccccc1"], 2, BITS, True)
    path = tmp_path / "fp.parquet"
    features.write_packed_cache(path, ["CCO", "c1ccccc1"], packed, {"bits": BITS})
    keys, matrix, metadata = features.read_packed_cache(path)
    assert keys.to_pylist() == ["CCO", "c1ccccc1"]
    assert np.array_equal(matrix, packed)
    assert metadata["bits"] == BITS


def test_packed_cache_refuses_a_key_count_that_does_not_match(tmp_path: Path) -> None:
    packed, _ = features.pack_fingerprints(["CCO"], 2, BITS, True)
    with pytest.raises(ValueError, match="lệch số dòng"):
        features.write_packed_cache(tmp_path / "fp.parquet", ["CCO", "CC"], packed, {})


def test_vector_cache_round_trip(tmp_path: Path) -> None:
    vectors = np.array([[0.5, -1.0, 2.0], [0.0, 1.0, 0.25]], dtype=np.float32)
    path = tmp_path / "prot.parquet"
    features.write_vector_cache(path, ["h1", "h2"], vectors, {"embedding_dim": 3})
    keys, matrix, metadata = features.read_vector_cache(path)
    assert keys.to_pylist() == ["h1", "h2"]
    assert np.allclose(matrix, vectors)
    assert matrix.shape == (2, 3)
    assert metadata["embedding_dim"] == 3


def test_vector_cache_refuses_a_non_float32_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="float32"):
        features.write_vector_cache(tmp_path / "prot.parquet", ["h1"],
                                     np.zeros((1, 3), dtype=np.float64), {})


def test_read_cache_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare"):
        features.read_packed_cache(tmp_path / "khong-co.parquet")


def test_sequence_windows_truncate_keeps_only_the_first_window() -> None:
    assert features.sequence_windows("ABCDEFG", 3, "truncate") == ["ABC"]


def test_sequence_windows_windows_keeps_the_whole_sequence() -> None:
    assert features.sequence_windows("ABCDEFG", 3, "windows") == ["ABC", "DEF", "G"]


def test_sequence_windows_leaves_a_short_sequence_untouched() -> None:
    assert features.sequence_windows("AB", 3, "windows") == ["AB"]


def test_sequence_windows_refuses_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="long_sequence"):
        features.sequence_windows("AB", 3, "trung-binh")
