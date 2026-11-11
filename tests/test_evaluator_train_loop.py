"""Vòng train thật của Stage 3, chạy trên dữ liệu giả nhỏ. Cần torch.

Vì sao có file này: trước ngày 10/11/2026 KHÔNG test nào gọi `train.run_train`, nên mọi
thay đổi trong hàm đó chỉ bị phát hiện khi chạy thật trên server — tức mất vài chục phút
mỗi lần sai một dòng. Các test ở đây thay phần đọc dữ liệu và phần cache bằng dữ liệu giả
(60 dòng, 2 epoch, CPU) rồi gọi đúng hàm thật, nên chúng bắt được lỗi nối dây mà không
cần GPU và không chạm dataset.

Chúng KHÔNG kiểm chất lượng mô hình: 60 dòng ngẫu nhiên thì không có gì để học. Chúng kiểm
đúng ba điều: hàm chạy hết, manifest ghi đúng những gì đã bật, và các tính chất toán học
(phản đối xứng, thang đo nhãn) còn đúng sau khi đi qua cả vòng train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch", reason="vòng train cần torch")

from molopt.evaluator import data, models, train  # noqa: E402

ROWS = 60
LIGAND_BITS = 32
FRAGMENT_BITS = 16
PROTEIN_DIM = 8
PROTEINS = 3
MOLECULES = 12
FRAGMENTS = 8


def fake_arrays(rng: np.random.Generator, slices: list[dict[str, Any]]
                ) -> tuple[data.PairArrays, dict[str, np.ndarray]]:
    """PairArrays nhỏ cùng các cột khóa query khớp với nó."""
    molecule_a = rng.integers(0, MOLECULES // 2, ROWS)
    molecule_b = rng.integers(MOLECULES // 2, MOLECULES, ROWS)
    protein = rng.integers(0, PROTEINS, ROWS)
    endpoint = rng.integers(0, 4, ROWS)
    arrays = data.PairArrays(
        molecule_a=molecule_a, molecule_b=molecule_b,
        fragment_a=rng.integers(0, FRAGMENTS // 2, ROWS),
        fragment_b=rng.integers(FRAGMENTS // 2, FRAGMENTS, ROWS),
        protein=protein, endpoint=endpoint,
        # Nhãn lệch tâm có chủ ý: minmax và zscore mới có phép dịch khác 0 để test bắt được.
        target=(rng.normal(size=ROWS) + 0.7).astype(np.float32),
        slices={definition["name"]: np.ones(ROWS, dtype=bool) for definition in slices})
    keys = {"molecule_a": molecule_a, "molecule_b": molecule_b, "protein": protein,
            "endpoint": endpoint}
    return arrays, keys


class FakeCaches:
    """Cache giả: cùng giao diện với train.Caches nhưng toàn bộ nằm trong RAM."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.molecule_packed = np.packbits(
            rng.integers(0, 2, (MOLECULES, LIGAND_BITS)).astype(np.uint8), axis=1)
        self.fragment_packed = np.packbits(
            rng.integers(0, 2, (FRAGMENTS, FRAGMENT_BITS)).astype(np.uint8), axis=1)
        self.ligand_bits = LIGAND_BITS
        self.fragment_bits = FRAGMENT_BITS
        self.molecule_keys = pa.array([f"m{index}" for index in range(MOLECULES)])
        self.fragment_keys = pa.array([f"f{index}" for index in range(FRAGMENTS)])
        self.molecule_vectors = None
        self.protein_keys = pa.array([f"h{index}" for index in range(PROTEINS)])
        self.protein_vectors = rng.normal(size=(PROTEINS, PROTEIN_DIM)).astype(np.float32)
        self.protein_dim = PROTEIN_DIM


@pytest.fixture()
def loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
         config: dict[str, Any]) -> dict[str, Any]:
    """Config nhỏ + mọi monkeypatch cần thiết để run_train chạy trên CPU."""
    rng = np.random.default_rng(0)
    run_config = json.loads(json.dumps(config))
    settings = run_config["stage3"]["train"]
    settings.update({"device": "cpu", "epochs": 2, "batch_size": 16,
                     "eval_batch_size": 32, "eval_every_steps": 0,
                     "early_stopping_patience": 5, "gpu_memory_limit_mib": None})
    run_config["stage3"]["metrics"]["asymmetry_rows"] = ROWS
    model_settings = run_config["stage3"]["model"]
    model_settings.update({"ligand_hidden": [16], "ligand_out": 8, "edit_hidden": [8],
                           "edit_out": 4, "protein_out": 4, "endpoint_dim": 3,
                           "head_hidden": [8], "dropout": 0.0,
                           "antisymmetric_hidden": [8], "antisymmetric_gate_hidden": [8],
                           "antisymmetric_gate_dim": 5})
    arrays, keys = fake_arrays(rng, run_config["stage3"]["metrics"]["slices"])
    caches = FakeCaches(rng)

    monkeypatch.setattr(train, "Caches", lambda *args, **kwargs: caches)
    monkeypatch.setattr(train, "prepare_arrays",
                        lambda *args, **kwargs: (arrays, dict(keys), None))
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    return {"config": run_config, "root": tmp_path, "arrays": arrays, "caches": caches}


def run(loop: dict[str, Any], head: str, **overrides: Any) -> dict[str, Any]:
    """Gọi run_train với một vài tham số ghi đè; trả về manifest."""
    loop["config"]["stage3"]["train"].update(overrides)
    return train.run_train(loop["config"], loop["root"], "frozen", head, smoke=False,
                           stamp="t", overrides=overrides)


@pytest.mark.parametrize("head", models.HEADS)
def test_every_head_completes_a_training_run(loop: dict[str, Any], head: str) -> None:
    manifest = run(loop, head)
    assert manifest["rows"]["train"] == ROWS
    assert manifest["checkpoint"] is not None
    assert Path(manifest["checkpoint"]).is_file()
    assert manifest["validation_metrics"]["all"]["regression"]["mae"] is not None
    assert manifest["augment_reverse"] is False
    assert manifest["label_scaler"]["name"] == train.LABEL_NONE


@pytest.mark.parametrize("head,expected", (
    (models.HEAD_DIRECT, False),
    (models.HEAD_DIRECT_NO_EDIT, False),
    (models.HEAD_TWIN, True),
    (models.HEAD_ANTISYMMETRIC, True),
))
def test_the_manifest_records_whether_the_head_is_antisymmetric(
        loop: dict[str, Any], head: str, expected: bool) -> None:
    manifest = run(loop, head)
    assert manifest["antisymmetric_by_construction"] is expected
    assert manifest["asymmetry"]["antisymmetric_by_construction"] is expected


@pytest.mark.parametrize("head", (models.HEAD_TWIN, models.HEAD_ANTISYMMETRIC))
def test_an_antisymmetric_head_meets_the_meeting_target_after_training(
        loop: dict[str, Any], head: str) -> None:
    """Mốc 1e-8 của anh Tiến (dòng 141 biên bản) phải ĐẠT ngay, do cấu tạo chứ không do train."""
    manifest = run(loop, head)
    asymmetry = manifest["asymmetry"]
    assert asymmetry["antisymmetry_max"] <= train.ASYMMETRY_TARGET
    assert asymmetry["meets_target"] is True
    # Và không được "đạt" bằng cách đoán hằng số 0.
    assert asymmetry["prediction_mean_abs"] > 0.0


@pytest.mark.parametrize("head", (models.HEAD_DIRECT, models.HEAD_DIRECT_NO_EDIT))
def test_a_joint_head_does_not_meet_the_target(loop: dict[str, Any], head: str) -> None:
    manifest = run(loop, head)
    assert manifest["asymmetry"]["meets_target"] is False


def test_reverse_augmentation_doubles_the_training_rows(loop: dict[str, Any]) -> None:
    manifest = run(loop, models.HEAD_DIRECT, augment_reverse=True)
    assert manifest["augment_reverse"] is True
    assert manifest["rows"]["train"] == 2 * ROWS
    # Validation KHÔNG được nhân đôi: nếu nhân thì mọi con số cũ không so được nữa.
    assert manifest["rows"]["validation"] == ROWS


def test_scale_normalization_keeps_the_reported_metrics_in_original_units(
        loop: dict[str, Any]) -> None:
    """Chuẩn hóa nhãn không được làm đổi THANG của chỉ số báo cáo.

    Cách kiểm: độ lệch chuẩn của nhãn mà báo cáo bất đối xứng ghi lại (label_spread) phải
    bằng độ lệch chuẩn của nhãn GỐC, không phải của nhãn đã chia. Nếu nó bằng 1,0 thì
    nghĩa là chỉ số đang được chấm trong không gian đã chuẩn hóa — và khi đó MAE 0,5879 của
    baseline "luôn đoán 0" không còn là cùng một mốc nữa.
    """
    original = float(loop["arrays"].target.astype(np.float64).std())
    manifest = run(loop, models.HEAD_DIRECT, label_normalization=train.LABEL_SCALE)
    assert manifest["label_scaler"]["name"] == train.LABEL_SCALE
    assert manifest["label_scaler"]["shift"] == 0.0
    assert manifest["label_scaler"]["scale"] == pytest.approx(original)
    assert manifest["label_scaler"]["preserves_antisymmetry"] is True
    assert manifest["asymmetry"]["label_spread"] == pytest.approx(original)


def test_minmax_normalization_is_refused_with_reverse_augmentation(
        loop: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="augment_reverse"):
        run(loop, models.HEAD_DIRECT, label_normalization=train.LABEL_MINMAX,
            augment_reverse=True)


@pytest.mark.parametrize("head", (models.HEAD_TWIN, models.HEAD_ANTISYMMETRIC))
def test_minmax_normalization_is_refused_with_an_antisymmetric_head(
        loop: dict[str, Any], head: str) -> None:
    with pytest.raises(ValueError, match="phản đối xứng"):
        run(loop, head, label_normalization=train.LABEL_MINMAX)


def test_the_linear_head_block_trains_and_stays_a_single_layer(
        loop: dict[str, Any]) -> None:
    loop["config"]["stage3"]["model"]["head_block"] = models.HEAD_BLOCK_LINEAR
    manifest = run(loop, models.HEAD_DIRECT)
    assert manifest["model"]["head_block"] == models.HEAD_BLOCK_LINEAR
    # Một lớp Linear không bias trên đặc trưng THÔ: toàn bộ mô hình chỉ còn hai thứ, lớp
    # Linear đó và bảng embedding endpoint. Không encoder, không MLP ghép.
    endpoint_dim = loop["config"]["stage3"]["model"]["endpoint_dim"]
    linear = 2 * LIGAND_BITS + 2 * FRAGMENT_BITS + PROTEIN_DIM + endpoint_dim
    embedding = 4 * endpoint_dim
    assert manifest["parameters"]["trainable"] == linear + embedding


def test_the_checkpoint_carries_its_own_label_scale(loop: dict[str, Any]) -> None:
    """eval phải lấy thang đo từ checkpoint, không từ config hiện tại."""
    manifest = run(loop, models.HEAD_DIRECT, label_normalization=train.LABEL_SCALE)
    state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    assert state["label_scaler"]["name"] == train.LABEL_SCALE
    restored = train.LabelScaler.from_state(state["label_scaler"])
    assert restored.scale == pytest.approx(manifest["label_scaler"]["scale"])


def test_the_manifest_records_the_subset_and_the_split_in_use(loop: dict[str, Any]) -> None:
    """Không ghi hai khóa này thì không truy lại được một lần chạy đã dùng subset nào."""
    loop["config"]["stage3"]["inputs"]["train_subset"] = "work/splits/x.parquet"
    manifest = run(loop, models.HEAD_DIRECT)
    assert manifest["train_subset"] == "work/splits/x.parquet"
    assert manifest["partition_override"] == ""
