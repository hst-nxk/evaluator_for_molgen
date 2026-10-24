"""Stage 3: dựng cache, vòng train, đánh giá trên validation, checkpoint và manifest.

Năm chế độ, gọi từ run_evaluator.py:
    prepare  : dựng cache fingerprint và cache embedding protein (chạy một lần)
    baseline : chỉ số của hai baseline tầm thường, làm mốc dưới cho mọi mô hình
    smoke    : train trên một mẫu nhỏ, chạy được trên CPU, chỉ để xác nhận pipeline thông
    train    : train thật trên partition train, chọn checkpoint theo validation
    eval     : nạp checkpoint rồi báo cáo chỉ số trên validation theo từng lát cắt

Module này KHÔNG import torch lúc nạp: torch chỉ được import trong thân những hàm thật
sự cần nó, và `models` (có import torch ở đầu file) cũng chỉ được import trong thân
run_train/run_eval. Nhờ vậy `--mode prepare --skip-protein` và `--mode baseline` chạy
được trên máy chưa cài torch.

Partition `test` không được dùng ở bất kỳ chế độ nào: mọi lần đọc dữ liệu đi qua
`data.read_partition`, hàm đó chặn test (xem data.check_partition).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import data, features, metrics

LOGGER = logging.getLogger(__name__)
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# --------------------------------------------------------------------------- tiện ích


def resolve_device(setting: str) -> str:
    """Thiết bị thực dùng: 'auto' chọn cuda nếu có, ngược lại cpu."""
    import torch

    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config yêu cầu cuda nhưng torch không thấy GPU")
    return setting


def set_seeds(seed: int) -> None:
    """Cố định seed của numpy và torch để hai lần chạy cùng cấu hình cho cùng kết quả.

    Không bật cudnn.deterministic: nó làm chậm đáng kể và v1 chỉ cần tái lập được ở mức
    seed. Chỗ này cần sửa nếu sau này phải tái lập từng bit trên GPU.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layout(config: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    """(thư mục cache, thư mục runs) của Stage 3."""
    paths = config["stage3"]["paths"]
    return root / paths["cache_dir"], root / paths["runs_dir"]


def cache_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    """Đường dẫn của từng file cache."""
    cache_dir, _ = layout(config, root)
    return {key: cache_dir / name for key, name in config["stage3"]["files"].items()}


# --------------------------------------------------------------------------- prepare


def distinct_values(path: Path, columns: Sequence[str], batch_rows: int = 200_000) -> list[str]:
    """Tập giá trị chuỗi phân biệt của một số cột, đọc theo lô, đã sắp xếp.

    Giữ một set Python nhưng chỉ thêm các giá trị ĐÃ phân biệt trong từng lô (pc.unique),
    nên số phép thêm nhỏ hơn số dòng nhiều lần. Sắp xếp ở cuối để cache dựng lại hai lần
    cho cùng thứ tự dòng.
    """
    handle = pq.ParquetFile(path)
    found: set[str] = set()
    for batch in handle.iter_batches(batch_size=batch_rows, columns=list(columns)):
        for name in columns:
            found.update(pc.unique(batch.column(name)).to_pylist())
    found.discard(None)
    return sorted(found)


def build_fingerprint_cache(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Dựng hai cache fingerprint: phân tử đầy đủ và nhóm thế.

    Cache được dựng trên TOÀN BỘ dataset, kể cả partition test. Đây KHÔNG phải rò rỉ:
    fingerprint là hàm thuần của cấu trúc phân tử, không có tham số nào được ước lượng từ
    dữ liệu, nên biết trước fingerprint của một phân tử test không cho mô hình thêm thông
    tin nào về nhãn. Nếu muốn chặt hơn thì dựng cache riêng cho train+validation, nhưng
    khi đó phải dựng lại cache trước khi đánh giá trên test.
    """
    settings = config["stage3"]["fingerprint"]
    source = root / config["stage3"]["inputs"]["evaluator"]
    paths = cache_paths(config, root)
    report: dict[str, Any] = {}
    for key, columns, bits in (("molecule_fingerprints", ("smiles_a", "smiles_b"),
                                int(settings["bits"])),
                               ("fragment_fingerprints", ("variable_a", "variable_b"),
                                int(settings["fragment_bits"]))):
        started = time.perf_counter()
        keys = distinct_values(source, columns)
        LOGGER.info("%s: %s SMILES phân biệt", key, f"{len(keys):,}")
        packed, failed = features.pack_fingerprints(keys, int(settings["radius"]), bits,
                                                    bool(settings["use_chirality"]))
        features.write_packed_cache(paths[key], keys, packed, {
            "radius": int(settings["radius"]), "bits": bits,
            "use_chirality": bool(settings["use_chirality"]),
            "source": str(source), "rows": len(keys), "unreadable_smiles": len(failed),
        })
        report[key] = {"rows": len(keys), "bits": bits, "unreadable_smiles": len(failed),
                       "seconds": round(time.perf_counter() - started, 1),
                       "path": str(paths[key])}
        if failed:
            LOGGER.warning("%s: %s SMILES RDKit không đọc được (dòng toàn 0)", key,
                           len(failed))
    return report


def canonical_sequence(sequence: str) -> str:
    """Chuỗi protein bỏ khoảng trắng và viết hoa — đúng chuẩn hóa của hash ở Stage 1.

    Vì sao: proteins.parquet giữ chuỗi nguyên văn của nguồn, và BindingDB có chuỗi viết
    thường (đo ngày 24/10/2026: 34 hash, 11 protein có cặp). Tokenizer ESM2 không nhận chữ
    thường: mỗi ký tự thường thành token <unk>, nên embedding của protein đó vô nghĩa.
    Hash của Stage 1 đã coi chuỗi thường và hoa là một (normalize.sequence_hash), nên
    chuẩn hóa ở đây không đổi danh tính protein nào.
    """
    return "".join(sequence.split()).upper()


def protein_sequences(config: Mapping[str, Any], root: Path) -> dict[str, str]:
    """{hash protein: chuỗi} cho các protein thực sự có cặp trong dataset.

    Chỉ lấy protein xuất hiện trong dataset (đo ngày 24/10/2026: 6.313 trên 11.835 chuỗi
    của Stage 2b), vì chạy ESM2 cho chuỗi không dùng đến là tốn vô ích. Hàm báo lỗi nếu có
    protein của dataset không có chuỗi, nên từ vựng cache luôn phủ hết dataset.
    Chuỗi trả về đã qua canonical_sequence: mọi đường dùng ESM2, frozen lẫn finetune,
    đều lấy chuỗi từ đây.
    """
    source = root / config["stage3"]["inputs"]["evaluator"]
    used = set(distinct_values(source, ("protein_hash",)))
    table = pq.read_table(root / config["stage3"]["inputs"]["proteins"],
                          columns=["protein_hash", "protein_sequence"])
    mapping = {}
    for hash_value, sequence in zip(table.column("protein_hash").to_pylist(),
                                    table.column("protein_sequence").to_pylist()):
        if hash_value in used:
            mapping[hash_value] = canonical_sequence(sequence)
    missing = used - set(mapping)
    if missing:
        raise ValueError(f"{len(missing)} protein của dataset không có chuỗi trong "
                         f"proteins.parquet, ví dụ {sorted(missing)[:3]}")
    return mapping


def build_protein_cache(config: Mapping[str, Any], root: Path, device: str) -> dict[str, Any]:
    """Dựng cache embedding protein bằng ESM2 (dùng cho cấu hình frozen).

    Chiều embedding được ĐO từ model đã nạp và ghi vào metadata của cache; không có số
    nào được giả định ở đây.
    """
    settings = config["stage3"]["protein_encoder"]
    paths = cache_paths(config, root)
    sequences = protein_sequences(config, root)
    LOGGER.info("sẽ embed %s protein bằng %s trên %s", f"{len(sequences):,}",
                settings["model"], device)
    started = time.perf_counter()
    keys, matrix, dimension, long_sequences = features.embed_proteins(sequences, settings,
                                                                      device)
    features.write_vector_cache(paths["protein_embeddings"], keys, matrix, {
        "model": settings["model"], "embedding_dim": dimension,
        "max_residues": int(settings["max_residues"]),
        "long_sequence": str(settings["long_sequence"]),
        "rows": len(keys), "sequences_longer_than_window": long_sequences,
    })
    LOGGER.info("CHIỀU EMBEDDING ĐO ĐƯỢC: %s (model %s)", dimension, settings["model"])
    return {"rows": len(keys), "embedding_dim": dimension,
            "sequences_longer_than_window": long_sequences,
            "model": settings["model"], "device": device,
            "seconds": round(time.perf_counter() - started, 1),
            "path": str(paths["protein_embeddings"])}


def run_prepare(config: Mapping[str, Any], root: Path, skip_protein: bool) -> dict[str, Any]:
    """Chế độ prepare: dựng cache fingerprint, và cache embedding protein nếu có thể.

    skip_protein = True thì chỉ dựng fingerprint (chạy được khi chưa cài torch và
    transformers), và ghi rõ vào manifest rằng chiều embedding vẫn CHƯA ĐO.
    """
    manifest: dict[str, Any] = {"stage": "stage3_prepare",
                                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "fingerprint": build_fingerprint_cache(config, root)}
    if skip_protein:
        manifest["protein_embeddings"] = {
            "built": False,
            "note": "bỏ qua theo --skip-protein; chiều embedding ESM2 CHƯA ĐO trên máy này",
        }
    else:
        device = resolve_device(config["stage3"]["train"]["device"])
        manifest["protein_embeddings"] = {"built": True,
                                          **build_protein_cache(config, root, device)}
    paths = cache_paths(config, root)
    paths["cache_manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["cache_manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    LOGGER.info("manifest cache: %s", paths["cache_manifest"])
    return manifest


# --------------------------------------------------------------------------- nạp cache


class Caches:
    """Ba bảng cache đã nạp, cùng từ vựng của chúng."""

    def __init__(self, config: Mapping[str, Any], root: Path, need_protein: bool) -> None:
        paths = cache_paths(config, root)
        self.molecule_keys, self.molecule_packed, molecule_meta = features.read_packed_cache(
            paths["molecule_fingerprints"])
        self.fragment_keys, self.fragment_packed, fragment_meta = features.read_packed_cache(
            paths["fragment_fingerprints"])
        self.ligand_bits = int(molecule_meta["bits"])
        self.fragment_bits = int(fragment_meta["bits"])
        settings = config["stage3"]["fingerprint"]
        if self.ligand_bits != int(settings["bits"]):
            raise ValueError(f"cache fingerprint có {self.ligand_bits} bit nhưng config ghi "
                             f"{settings['bits']}; dựng lại cache")
        self.protein_keys: pa.Array | None = None
        self.protein_vectors: np.ndarray | None = None
        self.protein_dim: int | None = None
        if need_protein:
            self.protein_keys, self.protein_vectors, protein_meta = (
                features.read_vector_cache(paths["protein_embeddings"]))
            self.protein_dim = int(protein_meta["embedding_dim"])
            if self.protein_dim != self.protein_vectors.shape[1]:
                raise ValueError("metadata cache ghi chiều khác với ma trận đã lưu")
            expected = config["stage3"]["protein_encoder"]["model"]
            if protein_meta.get("model") != expected:
                raise ValueError(f"cache embedding dựng bằng {protein_meta.get('model')!r} "
                                 f"nhưng config ghi {expected!r}; dựng lại cache")


def empty_protein_vocabulary(mapping: Mapping[str, str]) -> pa.Array:
    """Từ vựng hash protein khi không dùng cache embedding (cấu hình none/finetune)."""
    return pa.array(sorted(mapping), pa.string())


# --------------------------------------------------------------------------- batch


def batch_tensors(arrays: data.PairArrays, rows: np.ndarray, caches: Caches,
                  protein_source: Any, device: str, uses_edit: bool) -> dict[str, Any]:
    """Dựng tensor cho một batch.

    protein_source:
        None                       -> không có nhánh protein
        np.ndarray                 -> ma trận embedding đã cache (frozen)
        ProteinEncoderRuntime      -> ESM2 đang được train cùng (finetune)
    """
    import torch

    ligand_a = features.unpack_rows(caches.molecule_packed, arrays.molecule_a[rows],
                                    caches.ligand_bits)
    ligand_b = features.unpack_rows(caches.molecule_packed, arrays.molecule_b[rows],
                                    caches.ligand_bits)
    batch = {
        "ligand_a": torch.from_numpy(ligand_a).to(device),
        "ligand_b": torch.from_numpy(ligand_b).to(device),
        "endpoint": torch.from_numpy(arrays.endpoint[rows]).to(device),
        "target": torch.from_numpy(arrays.target[rows]).to(device),
    }
    if uses_edit:
        batch["fragment_a"] = torch.from_numpy(features.unpack_rows(
            caches.fragment_packed, arrays.fragment_a[rows], caches.fragment_bits)).to(device)
        batch["fragment_b"] = torch.from_numpy(features.unpack_rows(
            caches.fragment_packed, arrays.fragment_b[rows], caches.fragment_bits)).to(device)
    else:
        batch["fragment_a"] = None
        batch["fragment_b"] = None
    protein_rows = arrays.protein[rows]
    if protein_source is None:
        batch["protein"] = None
    elif isinstance(protein_source, np.ndarray):
        batch["protein"] = torch.from_numpy(protein_source[protein_rows]).to(device)
    else:
        batch["protein"] = protein_source.encode(protein_rows)
    return batch


class ProteinEncoderRuntime:
    """ESM2 chạy trong đồ thị tính toán, dùng cho cấu hình finetune.

    Trong một batch, số protein PHÂN BIỆT nhỏ hơn số cặp rất nhiều (một assay thường đo
    nhiều phân tử trên cùng một protein), nên ESM2 chỉ chạy một lần cho mỗi protein phân
    biệt rồi vector được nhân bản theo chỉ số. Nếu chạy một lần cho mỗi dòng thì phần lớn
    công việc là tính lại cùng một thứ.
    """

    def __init__(self, settings: Mapping[str, Any], sequences: Sequence[str],
                 device: str) -> None:
        import torch

        self.tokenizer, self.model, self.dimension = features.load_protein_encoder(
            settings, device)
        self.sequences = list(sequences)
        self.device = device
        self.max_residues = int(settings["max_residues"])
        self.torch = torch
        # Token không phải acid amin. Pool chỉ trên acid amin để giống hệt cache frozen
        # (features.embed_one_sequence bỏ CLS/EOS) — xem CHUẨN POOLING ở encode().
        self.special_ids = {token for token in (self.tokenizer.cls_token_id,
                                                self.tokenizer.eos_token_id,
                                                self.tokenizer.pad_token_id)
                            if token is not None}

    def parameters(self):
        """Tham số của ESM2, để đưa vào nhóm optimizer riêng với learning rate nhỏ."""
        return self.model.parameters()

    def train(self, flag: bool) -> None:
        self.model.train(flag)

    def encode(self, protein_rows: np.ndarray):
        """Vector protein cho từng dòng của batch, có gradient."""
        unique_rows, inverse = np.unique(protein_rows, return_inverse=True)
        texts = [self.sequences[int(index)][:self.max_residues] for index in unique_rows]
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_residues + 2)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.model(**encoded).last_hidden_state
        # CHUẨN POOLING của cả dự án: mean-pool CHỈ trên token acid amin, bỏ CLS/EOS/PAD.
        # CLS và EOS không phải acid amin, nên đại lượng cần lấy là biểu diễn trung bình
        # của các acid amin. Cache frozen đã tính đúng như vậy từ đầu, nên chọn chuẩn này
        # thì hai cấu hình frozen và finetune so sánh được, mà cache không phải dựng lại.
        mask = encoded["attention_mask"]
        for token_id in self.special_ids:
            mask = mask.masked_fill(encoded["input_ids"] == token_id, 0)
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        index = self.torch.from_numpy(inverse.astype(np.int64)).to(self.device)
        return pooled.index_select(0, index)


# --------------------------------------------------------------------------- đánh giá


def predict(model: Any, arrays: data.PairArrays, caches: Caches, protein_source: Any,
            device: str, batch_size: int) -> np.ndarray:
    """Dự đoán trên toàn bộ một partition, theo từng batch, không tính gradient."""
    import torch

    model.eval()
    if isinstance(protein_source, ProteinEncoderRuntime):
        protein_source.train(False)
    outputs = np.empty(len(arrays), dtype=np.float32)
    with torch.no_grad():
        for rows in data.iter_batches(len(arrays), batch_size):
            batch = batch_tensors(arrays, rows, caches, protein_source, device,
                                  model.uses_edit_branch)
            prediction = model(batch["ligand_a"], batch["ligand_b"], batch["fragment_a"],
                               batch["fragment_b"], batch["protein"], batch["endpoint"])
            outputs[rows] = prediction.detach().to("cpu").numpy()
    return outputs


def evaluate(arrays: data.PairArrays, prediction: np.ndarray, keys: Mapping[str, np.ndarray],
             config: Mapping[str, Any]) -> dict[str, Any]:
    """Chỉ số regression và ranking trên từng lát cắt, cộng chỉ số theo endpoint.

    Ranking được tính trên các query dựng từ (phân tử tham chiếu, protein, endpoint) sau
    khi nhân đôi hai chiều (xem metrics.expand_directions).
    """
    settings = config["stage3"]["metrics"]
    endpoints = list(config["endpoints"])
    truth = arrays.target.astype(np.float64)
    predicted = prediction.astype(np.float64)
    result: dict[str, Any] = {"by_endpoint": metrics.metrics_by_endpoint(
        arrays.endpoint, truth, predicted, endpoints)}
    for name, mask in arrays.slices.items():
        rows = np.flatnonzero(mask)
        slice_result: dict[str, Any] = {
            "rows": int(rows.size),
            "regression": metrics.regression_metrics(truth[rows], predicted[rows]),
        }
        if rows.size:
            if settings["expand_both_directions"]:
                reference, _, slice_truth, slice_prediction = metrics.expand_directions(
                    keys["molecule_a"][rows], keys["molecule_b"][rows], truth[rows],
                    predicted[rows])
                protein = np.concatenate([keys["protein"][rows]] * 2)
                endpoint = np.concatenate([keys["endpoint"][rows]] * 2)
            else:
                reference = keys["molecule_a"][rows]
                protein = keys["protein"][rows]
                endpoint = keys["endpoint"][rows]
                slice_truth, slice_prediction = truth[rows], predicted[rows]
            groups = metrics.query_groups([reference, protein, endpoint])
            slice_result["ranking"] = metrics.ranking_metrics(
                groups, slice_truth, slice_prediction, settings["top_k"],
                int(settings["min_candidates"]), float(settings["direction_tolerance"]))
        result[name] = slice_result
    return result


# ------------------------------------------------------------------ bất đối xứng


def asymmetry_report(model: Any, arrays: data.PairArrays, caches: Caches, protein_source: Any,
                     device: str, config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Đo mức bất đối xứng của mô hình đã train, trên một mẫu dòng validation.

    Dự án CHỌN KHÔNG cưỡng chế đối xứng cho đầu ra `direct` (cưỡng chế sẽ làm thay đổi
    bản chất của ablation direct-vs-twin), nên thay vào đó đo mức lệch và ghi vào báo cáo:
        f(A,B) + f(B,A) phải bằng 0 nếu mô hình đối xứng
        f(A,A)          phải bằng 0 vì "phép sửa" không sửa gì
    `twin` cho đúng 0 ở cả hai do kiến trúc, nên số đo này cũng là một phép kiểm tra chéo.
    Chạy trên mẫu ngẫu nhiên có seed cố định (metrics.asymmetry_rows) vì đây là số liệu
    chẩn đoán: ba lượt forward trên toàn bộ validation đắt gấp ba một lượt đánh giá.
    """
    settings = config["stage3"]["metrics"]
    if not settings.get("measure_asymmetry", True):
        return None
    train_settings = config["stage3"]["train"]
    cap = int(settings.get("asymmetry_rows", 0)) or len(arrays)
    count = min(cap, len(arrays))
    if count == 0:
        return None
    if count < len(arrays):
        rng = np.random.default_rng(int(train_settings["seed"]))
        rows = np.sort(rng.choice(len(arrays), size=count, replace=False))
    else:
        rows = np.arange(len(arrays), dtype=np.int64)
    sample = data.take_rows(arrays, rows)
    batch_size = int(train_settings["eval_batch_size"])
    forward = predict(model, sample, caches, protein_source, device, batch_size)
    backward = predict(model, data.swap_sides(sample), caches, protein_source, device,
                       batch_size)
    identity = predict(model, data.identity_sides(sample), caches, protein_source, device,
                       batch_size)
    spread = float(np.std(sample.target.astype(np.float64)))
    return metrics.asymmetry_metrics(forward.astype(np.float64), backward.astype(np.float64),
                                     identity.astype(np.float64), spread)


# ------------------------------------------------------------------ baseline tầm thường


BASELINE_ZERO = "zero"
BASELINE_ENDPOINT_MEAN = "endpoint_mean"
BASELINE_RANDOM = "random"
BASELINES = (BASELINE_ZERO, BASELINE_ENDPOINT_MEAN, BASELINE_RANDOM)


def endpoint_means(endpoints: np.ndarray, targets: np.ndarray,
                   count: int) -> tuple[np.ndarray, np.ndarray]:
    """Trung bình nhãn theo endpoint, cùng số dòng của mỗi endpoint.

    Nhận: mã endpoint của từng dòng, nhãn, số endpoint.
    Trả về: (trung bình theo mã, số dòng theo mã).
    Chỉ được gọi trên partition TRAIN: một baseline ước lượng tham số từ validation thì
    không còn là mốc dưới trung thực nữa. Endpoint không có dòng train nào nhận 0.
    """
    means = np.zeros(count, dtype=np.float64)
    seen = np.zeros(count, dtype=np.int64)
    for code in range(count):
        mask = endpoints == code
        seen[code] = int(mask.sum())
        if seen[code]:
            means[code] = float(np.mean(targets[mask]))
    return means, seen


def baseline_prediction(name: str, endpoints: np.ndarray, means: np.ndarray,
                        pool: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Dự đoán của một baseline tầm thường.

    Nhận: tên baseline; mã endpoint của từng dòng; trung bình theo endpoint (từ train);
    `pool` = phân bố nhãn train để lấy mẫu (chỉ 'random' cần); seed.

    'zero'          : luôn đoán Δp = 0, tức "phép sửa không làm gì". MAE của nó là mức mà
                      một mô hình phải vượt mới gọi là học được điều gì.
    'endpoint_mean' : đoán trung bình Δp của endpoint, ước lượng trên train.
    'random'        : lấy mẫu CÓ HOÀN LẠI từ phân bố nhãn của train, seed cố định.

    Vì sao cần 'random': hai baseline kia cho dự đoán không đổi trong mỗi query, nên mọi
    cặp ứng viên đều hòa và tỉ lệ đúng chiều của chúng bằng 0 — đó là sàn của quy ước
    hòa-tính-sai, không phải sàn của bài toán. 'random' cho dự đoán liên tục và độc lập
    hoàn toàn với dòng đang xét, nên hầu như không có hòa, và tỉ lệ đúng chiều của nó là
    mốc 50 % thật để so. Lấy mẫu từ phân bố nhãn train (chứ không từ một phân bố tự chọn
    như N(0,1)) để không phải đặt thêm một tham số tùy ý nào.
    """
    if name == BASELINE_ZERO:
        return np.zeros(endpoints.size, dtype=np.float64)
    if name == BASELINE_ENDPOINT_MEAN:
        return means[endpoints]
    if name == BASELINE_RANDOM:
        if pool is None or pool.size == 0:
            raise ValueError("baseline 'random' cần phân bố nhãn train để lấy mẫu")
        rng = np.random.default_rng(seed)
        drawn = pool[rng.integers(0, pool.size, size=endpoints.size)]
        return np.asarray(drawn, dtype=np.float64)
    raise ValueError(f"baseline lạ: {name!r}; chỉ có {BASELINES}")


def run_baselines(config: Mapping[str, Any], root: Path, stamp: str) -> dict[str, Any]:
    """Chỉ số của các baseline tầm thường trên validation, theo từng lát cắt.

    Không cần torch, không cần cache fingerprint và không cần ESM2: từ vựng được dựng
    ngay từ bảng validation, vì baseline không dùng cấu trúc phân tử. Chạy được trước cả
    `--mode prepare`.

    LƯU Ý khi đọc kết quả: 'zero' và 'endpoint_mean' cho dự đoán KHÔNG ĐỔI trong một query
    (query đã gồm endpoint), nên mọi cặp ứng viên đều hòa; theo quy ước của dự án dự đoán
    hòa tính là sai, nên tỉ lệ đúng chiều của hai baseline đó bằng 0 chứ không phải 50 %.
    Phần có nghĩa của chúng là MAE/RMSE/R². Mốc 50 % cho ranking do 'random' cung cấp.
    """
    settings = config["stage3"]["metrics"]
    endpoints = list(config["endpoints"])
    names = list(settings.get("baselines", list(BASELINES)))
    seed = int(config["stage3"]["train"]["seed"])
    endpoint_vocabulary = pa.array(endpoints, pa.string())

    train_table = data.read_columns(config, root, data.TRAIN, ["endpoint", "delta_p_mean"])
    train_codes = data.code_column(train_table.column("endpoint"), endpoint_vocabulary,
                                   "endpoint")
    train_targets = np.asarray(
        train_table.column("delta_p_mean").to_numpy(zero_copy_only=False), dtype=np.float64)
    means, seen = endpoint_means(train_codes, train_targets, len(endpoints))
    for code, name in enumerate(endpoints):
        LOGGER.info("train: endpoint %s có %s dòng, Δp trung bình %.4f", name,
                    f"{int(seen[code]):,}", means[code])
        if not seen[code]:
            LOGGER.warning("endpoint %s không có dòng train nào; baseline đoán 0", name)

    table = data.read_partition(config, root, data.VALIDATION)
    molecules = data.distinct_strings(table, ["smiles_a", "smiles_b"])
    fragments = data.distinct_strings(table, ["variable_a", "variable_b"])
    proteins = data.distinct_strings(table, ["protein_hash"])
    arrays = data.encode_pairs(table, molecules, fragments, proteins, endpoints,
                               settings["slices"])
    keys = data.query_key_columns(table, molecules, proteins, endpoints)
    LOGGER.info("validation %s dòng | %s phân tử phân biệt | %s protein",
                f"{len(arrays):,}", f"{len(molecules):,}", f"{len(proteins):,}")

    results: dict[str, Any] = {}
    for name in names:
        prediction = baseline_prediction(name, arrays.endpoint, means, train_targets, seed)
        scores = evaluate(arrays, prediction.astype(np.float32), keys, config)
        results[name] = scores
        for slice_name, block in scores.items():
            if slice_name == "by_endpoint" or not isinstance(block, dict):
                continue
            for line in metrics.format_metrics(f"baseline {name} / {slice_name}",
                                               block["regression"], block.get("ranking")):
                LOGGER.info("%s", line)

    _, runs_dir = layout(config, root)
    manifest = {
        "stage": "stage3_baseline", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baselines": names, "seed": seed,
        "rows": {"train": train_table.num_rows, "validation": len(arrays)},
        "endpoint_means": {name: means[code] for code, name in enumerate(endpoints)},
        "endpoint_train_rows": {name: int(seen[code]) for code, name in enumerate(endpoints)},
        "validation_metrics": results,
        "note": ("zero và endpoint_mean cho dự đoán không đổi trong mỗi query nên tỉ lệ "
                 "đúng chiều của chúng bằng 0 theo quy ước hòa-tính-sai; phần so sánh "
                 "được của hai baseline đó là MAE/RMSE/R². Mốc 50 % cho ranking do "
                 "random cung cấp, lấy mẫu từ phân bố nhãn train với seed cố định."),
    }
    path = runs_dir / f"baseline_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("manifest baseline: %s", path)
    return manifest


# --------------------------------------------------------------------------- train


def build_optimizer(model: Any, protein_source: Any, settings: Mapping[str, Any]):
    """AdamW; ESM2 (nếu được train cùng) nằm ở nhóm có learning rate riêng, nhỏ hơn.

    Vì sao hai nhóm: ESM2 đã pretrain trên hàng trăm triệu chuỗi, còn phần đầu dự đoán
    khởi tạo ngẫu nhiên. Dùng cùng learning rate sẽ phá trọng số pretrain trước khi phần
    đầu kịp học.
    """
    import torch

    groups = [{"params": list(model.parameters()),
               "lr": float(settings["learning_rate"])}]
    if isinstance(protein_source, ProteinEncoderRuntime):
        groups.append({"params": list(protein_source.parameters()),
                       "lr": float(settings["protein_learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(settings["weight_decay"]))


def loss_function(settings: Mapping[str, Any]):
    """Hàm mất mát theo config: huber (mặc định) hoặc mse."""
    import torch

    name = str(settings["loss"])
    if name == "huber":
        return torch.nn.HuberLoss(delta=float(settings["huber_delta"]))
    if name == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"loss lạ: {name!r}; chỉ có 'huber' hoặc 'mse'")


def train_one_epoch(model: Any, arrays: data.PairArrays, caches: Caches, protein_source: Any,
                    device: str, optimizer: Any, criterion: Any, batch_size: int,
                    grad_clip: float, rng: np.random.Generator) -> float:
    """Một epoch; trả về mất mát trung bình theo dòng."""
    import torch

    model.train()
    if isinstance(protein_source, ProteinEncoderRuntime):
        protein_source.train(True)
    total = 0.0
    seen = 0
    for rows in data.iter_batches(len(arrays), batch_size, rng):
        batch = batch_tensors(arrays, rows, caches, protein_source, device,
                              model.uses_edit_branch)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["ligand_a"], batch["ligand_b"], batch["fragment_a"],
                           batch["fragment_b"], batch["protein"], batch["endpoint"])
        loss = criterion(prediction, batch["target"])
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach().to("cpu")) * rows.size
        seen += rows.size
    return total / max(seen, 1)


def prepare_arrays(config: Mapping[str, Any], root: Path, partition: str, caches: Caches,
                   protein_vocabulary: pa.Array, limit: int | None
                   ) -> tuple[data.PairArrays, dict[str, np.ndarray], pa.Table]:
    """Đọc và mã hóa một partition; trả về (mảng, cột khóa query, bảng gốc)."""
    table = data.read_partition(config, root, partition, limit)
    arrays = data.encode_pairs(table, caches.molecule_keys, caches.fragment_keys,
                               protein_vocabulary, list(config["endpoints"]),
                               config["stage3"]["metrics"]["slices"])
    keys = data.query_key_columns(table, caches.molecule_keys, protein_vocabulary,
                                  list(config["endpoints"]))
    return arrays, keys, table


def run_train(config: Mapping[str, Any], root: Path, configuration: str, head: str,
              smoke: bool, stamp: str) -> dict[str, Any]:
    """Train một cấu hình; chọn checkpoint theo MAE trên validation.

    Nhận: config; thư mục gốc; cấu hình ('frozen' | 'finetune' | 'no_protein'); kiểu đầu
    ra ('direct' | 'twin'); smoke; nhãn thời gian của lần chạy.
    Trả về: manifest của lần chạy (đã ghi ra đĩa).
    """
    from . import models

    settings = config["stage3"]["train"]
    protein_mode = models.protein_mode_for(configuration)
    device = resolve_device(settings["device"])
    set_seeds(int(settings["seed"]))
    _, runs_dir = layout(config, root)
    limit = int(settings["smoke_rows"]) if smoke else None
    epochs = int(settings["smoke_epochs"] if smoke else settings["epochs"])
    batch_size = int(settings["batch_size"])
    if smoke:
        # Smoke chỉ để xác nhận pipeline thông trên CPU, nên batch nhỏ cho nhanh.
        batch_size = min(batch_size, 128)
    elif protein_mode == models.PROTEIN_FINETUNE:
        overrides = settings.get("finetune_overrides", {})
        batch_size = int(overrides.get("batch_size", batch_size))

    caches = Caches(config, root, need_protein=protein_mode == models.PROTEIN_FROZEN)
    sequences: dict[str, str] = {}
    protein_source: Any = None
    if protein_mode == models.PROTEIN_FROZEN:
        protein_vocabulary = caches.protein_keys
        protein_source = caches.protein_vectors
        protein_dim = caches.protein_dim
    else:
        sequences = protein_sequences(config, root)
        protein_vocabulary = empty_protein_vocabulary(sequences)
        protein_dim = None
        if protein_mode == models.PROTEIN_FINETUNE:
            encoder_settings = dict(config["stage3"]["protein_encoder"])
            # finetune bắt buộc cắt chuỗi: bộ nhớ gradient tỉ lệ với số acid amin.
            encoder_settings["long_sequence"] = "truncate"
            overrides = settings.get("finetune_overrides", {})
            if "max_residues" in overrides:
                encoder_settings["max_residues"] = int(overrides["max_residues"])
            ordered = [sequences[key] for key in protein_vocabulary.to_pylist()]
            protein_source = ProteinEncoderRuntime(encoder_settings, ordered, device)
            protein_dim = protein_source.dimension
            LOGGER.info("CHIỀU EMBEDDING ĐO ĐƯỢC: %s (model %s)", protein_dim,
                        encoder_settings["model"])

    train_arrays, train_keys, _ = prepare_arrays(config, root, data.TRAIN, caches,
                                                 protein_vocabulary, limit)
    validation_arrays, validation_keys, _ = prepare_arrays(
        config, root, data.VALIDATION, caches, protein_vocabulary, limit)
    LOGGER.info("train %s dòng | validation %s dòng | thiết bị %s | batch %s",
                f"{len(train_arrays):,}", f"{len(validation_arrays):,}", device, batch_size)

    model = models.EvaluatorModel(config["stage3"]["model"], caches.ligand_bits,
                                  caches.fragment_bits, protein_dim,
                                  len(config["endpoints"]), head, protein_mode)
    model.to(device)
    optimizer = build_optimizer(model, protein_source, settings)
    criterion = loss_function(settings)
    rng = np.random.default_rng(int(settings["seed"]))
    patience = int(settings["early_stopping_patience"])
    checkpoint_path = runs_dir / f"{configuration}_{head}_{stamp}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_mae = math.inf
    best_epoch = -1
    since_best = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        loss = train_one_epoch(model, train_arrays, caches, protein_source, device,
                               optimizer, criterion, batch_size, float(settings["grad_clip"]),
                               rng)
        prediction = predict(model, validation_arrays, caches, protein_source, device,
                             int(settings["eval_batch_size"]))
        scores = evaluate(validation_arrays, prediction, validation_keys, config)
        mae = scores["all"]["regression"]["mae"]
        history.append({"epoch": epoch, "train_loss": loss,
                        "validation": scores["all"],
                        "seconds": round(time.perf_counter() - epoch_started, 1)})
        for line in metrics.format_metrics(f"epoch {epoch} validation",
                                           scores["all"]["regression"],
                                           scores["all"].get("ranking")):
            LOGGER.info("%s", line)
        LOGGER.info("epoch %s: train loss %.4f", epoch, loss)
        if mae is not None and mae < best_mae:
            import torch

            best_mae, best_epoch, since_best = mae, epoch, 0
            state = {"model": model.state_dict(), "epoch": epoch, "configuration":
                     configuration, "head": head, "protein_mode": protein_mode,
                     "protein_dim": protein_dim, "ligand_bits": caches.ligand_bits,
                     "fragment_bits": caches.fragment_bits}
            if isinstance(protein_source, ProteinEncoderRuntime):
                state["protein_encoder"] = protein_source.model.state_dict()
            torch.save(state, checkpoint_path)
            LOGGER.info("checkpoint tốt nhất -> %s", checkpoint_path)
        else:
            since_best += 1
            if since_best >= patience:
                LOGGER.info("dừng sớm: %s epoch không cải thiện MAE validation", since_best)
                break

    final = evaluate(validation_arrays, predict(model, validation_arrays, caches,
                                                protein_source, device,
                                                int(settings["eval_batch_size"])),
                     validation_keys, config)
    asymmetry = asymmetry_report(model, validation_arrays, caches, protein_source, device,
                                 config)
    if asymmetry is not None:
        for line in metrics.format_asymmetry("bất đối xứng (validation)", asymmetry):
            LOGGER.info("%s", line)
    manifest = {
        "stage": "stage3_train", "mode": "smoke" if smoke else "train",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configuration": configuration, "head": head, "protein_mode": protein_mode,
        "device": device, "seed": int(settings["seed"]), "epochs_run": len(history),
        "batch_size": batch_size,
        "rows": {"train": len(train_arrays), "validation": len(validation_arrays)},
        "protein_embedding_dim": protein_dim,
        "parameters": models.count_parameters(model),
        "fingerprint_bits": {"ligand": caches.ligand_bits, "fragment": caches.fragment_bits},
        "uses_edit_branch": model.uses_edit_branch,
        "best_epoch": best_epoch, "best_validation_mae": None if best_epoch < 0 else best_mae,
        "checkpoint": str(checkpoint_path) if best_epoch >= 0 else None,
        "history": history, "validation_metrics": final, "asymmetry": asymmetry,
        "seconds": round(time.perf_counter() - started, 1),
    }
    path = runs_dir / f"{manifest['mode']}_{configuration}_{head}_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("manifest lần chạy: %s", path)
    return manifest


def run_eval(config: Mapping[str, Any], root: Path, checkpoint: Path,
             stamp: str) -> dict[str, Any]:
    """Nạp checkpoint rồi báo cáo chỉ số trên validation theo từng lát cắt.

    Không train, không chạm partition test.
    """
    import torch

    from . import models

    settings = config["stage3"]["train"]
    device = resolve_device(settings["device"])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    configuration = str(state["configuration"])
    protein_mode = str(state["protein_mode"])
    caches = Caches(config, root, need_protein=protein_mode == models.PROTEIN_FROZEN)
    protein_source: Any = None
    if protein_mode == models.PROTEIN_FROZEN:
        protein_vocabulary = caches.protein_keys
        protein_source = caches.protein_vectors
    else:
        sequences = protein_sequences(config, root)
        protein_vocabulary = empty_protein_vocabulary(sequences)
        if protein_mode == models.PROTEIN_FINETUNE:
            encoder_settings = dict(config["stage3"]["protein_encoder"])
            encoder_settings["long_sequence"] = "truncate"
            ordered = [sequences[key] for key in protein_vocabulary.to_pylist()]
            protein_source = ProteinEncoderRuntime(encoder_settings, ordered, device)
            if "protein_encoder" in state:
                protein_source.model.load_state_dict(state["protein_encoder"])
    model = models.EvaluatorModel(config["stage3"]["model"], int(state["ligand_bits"]),
                                  int(state["fragment_bits"]), state["protein_dim"],
                                  len(config["endpoints"]), str(state["head"]), protein_mode)
    model.load_state_dict(state["model"])
    model.to(device)
    arrays, keys, _ = prepare_arrays(config, root, data.VALIDATION, caches,
                                     protein_vocabulary, None)
    prediction = predict(model, arrays, caches, protein_source, device,
                         int(settings["eval_batch_size"]))
    scores = evaluate(arrays, prediction, keys, config)
    for name, block in scores.items():
        if name == "by_endpoint" or not isinstance(block, dict):
            continue
        for line in metrics.format_metrics(f"validation/{name}", block["regression"],
                                           block.get("ranking")):
            LOGGER.info("%s", line)
    asymmetry = asymmetry_report(model, arrays, caches, protein_source, device, config)
    if asymmetry is not None:
        for line in metrics.format_asymmetry("bất đối xứng (validation)", asymmetry):
            LOGGER.info("%s", line)
    _, runs_dir = layout(config, root)
    manifest = {"stage": "stage3_eval", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": str(checkpoint), "configuration": configuration,
                "head": str(state["head"]), "protein_mode": protein_mode, "device": device,
                "rows": len(arrays), "validation_metrics": scores,
                "asymmetry": asymmetry}
    path = runs_dir / f"eval_{configuration}_{state['head']}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("manifest đánh giá: %s", path)
    return manifest
