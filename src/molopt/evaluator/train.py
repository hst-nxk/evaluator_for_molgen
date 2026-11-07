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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import analysis, data, features, metrics, sharpness

LOGGER = logging.getLogger(__name__)
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
BYTES_PER_MIB = 1024 * 1024


# --------------------------------------------------------------------------- tiện ích


def resolve_device(setting: str) -> str:
    """Thiết bị thực dùng: 'auto' chọn cuda nếu có, ngược lại cpu."""
    import torch

    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config yêu cầu cuda nhưng torch không thấy GPU")
    return setting


def limit_gpu_memory(settings: Mapping[str, Any], device: str) -> float | None:
    """Chặn trên bộ nhớ GPU mà PyTorch được cấp phát cho tiến trình này.

    Nhận: stage3.train (đọc khóa gpu_memory_limit_mib; None hoặc thiếu = không chặn);
    thiết bị thực dùng.
    Trả về: tỉ lệ bộ nhớ đã đặt cho GPU số 0, hoặc None nếu không chặn.
    Vượt mức thì PyTorch báo hết bộ nhớ (OutOfMemoryError) thay vì lấn thêm VRAM. Mức này
    không gồm CUDA context của tiến trình (vài trăm MiB, nằm ngoài bộ cấp phát của
    PyTorch), nên nvidia-smi sẽ thấy cao hơn một chút.
    """
    limit = settings.get("gpu_memory_limit_mib")
    if limit is None or not device.startswith("cuda"):
        return None
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, int(limit) * BYTES_PER_MIB / total)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    LOGGER.info("giới hạn VRAM: %s MiB / %s MiB của GPU (tỉ lệ %.3f)", int(limit),
                total // BYTES_PER_MIB, fraction)
    return fraction


def peak_gpu_memory_mib(device: str) -> float | None:
    """VRAM cao nhất PyTorch đã giữ trong tiến trình này (MiB); None nếu không dùng GPU."""
    if not device.startswith("cuda"):
        return None
    import torch

    return round(torch.cuda.max_memory_reserved() / BYTES_PER_MIB, 1)


def load_best_checkpoint(path: Path, model: Any, protein_source: Any, device: str) -> None:
    """Nạp lại trọng số của checkpoint tốt nhất vào model (và ESM2 nếu là finetune)."""
    import torch

    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if "protein_encoder" in state:
        protein_source.model.load_state_dict(state["protein_encoder"])
    LOGGER.info("đã nạp lại checkpoint tốt nhất (epoch %s) để tính số cuối", state["epoch"])


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


def build_molecule_embedding_cache(config: Mapping[str, Any], root: Path,
                                   device: str) -> dict[str, Any]:
    """Dựng cache embedding cho SMILES phân tử (dùng khi ligand_features = embedding).

    Chạy model mã hóa SMILES MỘT LẦN cho mỗi phân tử phân biệt rồi ghi ra parquet, đúng
    cách cache ESM2 đang làm với protein. Nhờ vậy vòng train không chậm đi: nó chỉ tra
    theo chỉ số như khi dùng fingerprint.
    Chiều embedding được ĐO từ model đã nạp và ghi vào metadata của cache.
    """
    settings = config["stage3"]["molecule_encoder"]
    paths = cache_paths(config, root)
    source = root / config["stage3"]["inputs"]["evaluator"]
    molecules = distinct_values(source, ("smiles_a", "smiles_b"))
    LOGGER.info("sẽ embed %s phân tử bằng %s trên %s", f"{len(molecules):,}",
                settings["model"], device)
    started = time.perf_counter()
    keys, matrix, dimension, truncated = features.embed_molecules(molecules, settings,
                                                                  device)
    features.write_vector_cache(paths["molecule_embeddings"], keys, matrix, {
        "model": settings["model"], "embedding_dim": dimension,
        "max_tokens": int(settings["max_tokens"]),
        "pooling": str(settings.get("pooling", "pooler")),
        "rows": len(keys), "truncated_smiles": truncated,
    })
    LOGGER.info("CHIỀU EMBEDDING ĐO ĐƯỢC: %s (model %s)", dimension, settings["model"])
    return {"rows": len(keys), "embedding_dim": dimension,
            "truncated_smiles": truncated, "model": settings["model"], "device": device,
            "seconds": round(time.perf_counter() - started, 1),
            "path": str(paths["molecule_embeddings"])}


def run_prepare(config: Mapping[str, Any], root: Path, skip_protein: bool,
                molecule_embeddings: bool = False) -> dict[str, Any]:
    """Chế độ prepare: dựng cache fingerprint, và cache embedding protein nếu có thể.

    skip_protein = True thì chỉ dựng fingerprint (chạy được khi chưa cài torch và
    transformers), và ghi rõ vào manifest rằng chiều embedding vẫn CHƯA ĐO.
    molecule_embeddings = True thì dựng THÊM cache embedding phân tử; mặc định tắt vì
    chế độ mặc định (ligand_features = fingerprint) không cần tới nó.
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
    if molecule_embeddings:
        device = resolve_device(config["stage3"]["train"]["device"])
        manifest["molecule_embeddings"] = {
            "built": True, **build_molecule_embedding_cache(config, root, device)}
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
        # Import tại chỗ: models.py import torch ở đầu file, còn train.py phải nạp được
        # trên máy chưa cài torch (xem docstring module).
        from . import models

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
        # Nhánh phân tử: fingerprint (mặc định) hoặc embedding đã cache. `ligand_bits` là
        # CHIỀU ĐẦU VÀO của nhánh đó, nên ở chế độ embedding nó là chiều embedding.
        self.ligand_features = str(
            config["stage3"]["model"].get("ligand_features", models.LIGAND_FINGERPRINT))
        if self.ligand_features not in models.LIGAND_FEATURES:
            raise ValueError(f"ligand_features lạ: {self.ligand_features!r}; "
                             f"chỉ có {models.LIGAND_FEATURES}")
        self.molecule_vectors: np.ndarray | None = None
        if self.ligand_features == models.LIGAND_EMBEDDING:
            path = paths["molecule_embeddings"]
            if not path.is_file():
                raise FileNotFoundError(
                    f"chưa có cache embedding phân tử {path}; chạy "
                    f"`--mode prepare --molecule-embeddings` trước")
            embedding_keys, self.molecule_vectors, molecule_embedding_meta = (
                features.read_vector_cache(path))
            if not embedding_keys.equals(self.molecule_keys):
                raise ValueError("cache embedding và cache fingerprint khác danh sách "
                                 "phân tử; dựng lại cả hai")
            expected = config["stage3"]["molecule_encoder"]["model"]
            if molecule_embedding_meta.get("model") != expected:
                raise ValueError(
                    f"cache embedding phân tử dựng bằng "
                    f"{molecule_embedding_meta.get('model')!r} nhưng config ghi "
                    f"{expected!r}; dựng lại cache")
            self.ligand_bits = int(molecule_embedding_meta["embedding_dim"])
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

    if caches.molecule_vectors is None:
        ligand_a = features.unpack_rows(caches.molecule_packed, arrays.molecule_a[rows],
                                        caches.ligand_bits)
        ligand_b = features.unpack_rows(caches.molecule_packed, arrays.molecule_b[rows],
                                        caches.ligand_bits)
    else:
        # Cùng từ vựng với cache fingerprint (Caches đã kiểm), nên dùng chung chỉ số.
        ligand_a = caches.molecule_vectors[arrays.molecule_a[rows]]
        ligand_b = caches.molecule_vectors[arrays.molecule_b[rows]]
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
            "change_classes": metrics.change_class_metrics(
                truth[rows], predicted[rows], float(settings["change_class_threshold"])),
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
                     device: str, config: Mapping[str, Any],
                     eval_batch_size: int) -> dict[str, Any] | None:
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
    batch_size = eval_batch_size
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
                                               block["regression"], block.get("ranking"),
                                               block.get("change_classes")):
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


# Các nhánh của EvaluatorModel có thể đặt learning rate riêng (xem
# branch_parameter_groups). Tên lấy đúng tên thuộc tính trong models.EvaluatorModel; sai
# tên thì báo lỗi chứ không bỏ qua, vì một chữ sai sẽ lặng lẽ thành "không đổi gì".
BRANCH_MODULES = ("ligand_encoder", "edit_encoder", "protein_projection", "projectors",
                  "attention", "endpoint_embedding", "predictor")
OTHER_BRANCH = "còn lại"


def branch_parameter_groups(model: Any, base_rate: float,
                            scales: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Chia tham số của model thành các nhóm learning rate theo từng nhánh.

    Nhận: model; learning rate gốc; {tên nhánh: hệ số nhân} (rỗng = một nhóm như cũ).
    Trả về: danh sách nhóm cho AdamW, mỗi nhóm có thêm khóa 'branch' để ghi log.

    Vì sao cần: Wang W, Tran D, Feiszli M. What Makes Training Multi-Modal Classification
    Networks Hard? CVPR 2020, arXiv:1905.12681 — nguyên nhân thứ hai họ nêu là các mô
    thức QUÁ KHỚP VÀ TỔNG QUÁT HÓA VỚI TỐC ĐỘ KHÁC NHAU, nên train chung bằng một chiến
    lược tối ưu là dưới mức tối ưu. Learning rate riêng từng nhánh là cách rẻ nhất để
    thử điều đó (Gradient Blending trong cùng bài đắt hơn nhiều).
    Hệ số NHÂN chứ không phải learning rate tuyệt đối: để các lần quét learning_rate cũ
    vẫn còn nghĩa, và để hệ số 1,0 đúng bằng hành vi cũ.
    """
    unknown = sorted(set(scales) - set(BRANCH_MODULES))
    if unknown:
        raise ValueError(f"nhánh lạ trong branch_learning_rate_scale: {unknown}; "
                         f"tên hợp lệ: {sorted(BRANCH_MODULES)}")
    for name, value in scales.items():
        if float(value) < 0:
            raise ValueError(f"hệ số của nhánh {name!r} phải từ 0 trở lên, "
                             f"nhận {value!r}")
    # Hệ số 1,0 KHÔNG tách nhóm: config liệt kê sẵn mọi nhánh ở 1,0 để --set chạm được
    # từng cái, nên nếu tách hết thì mọi lần chạy mặc định đều có 7 nhóm vô nghĩa. Bỏ
    # 1,0 ra thì "mặc định" đúng bằng một nhóm duy nhất như trước.
    active = {name: float(value) for name, value in scales.items()
              if float(value) != 1.0}
    buckets: dict[str, list[Any]] = {}
    for name, parameter in model.named_parameters():
        branch = name.split(".", 1)[0]
        buckets.setdefault(branch if branch in active else OTHER_BRANCH,
                           []).append(parameter)
    groups = [{"params": parameters, "branch": branch,
               "lr": base_rate * (1.0 if branch == OTHER_BRANCH
                                  else active[branch])}
              for branch, parameters in buckets.items()]
    # Không được để sót tham số nào: sót thì nhánh đó im lặng không được train.
    grouped = sum(len(group["params"]) for group in groups)
    total = len(list(model.named_parameters()))
    if grouped != total:
        raise ValueError(f"chia nhóm làm mất tham số: {grouped} / {total}")
    return groups


def build_optimizer(model: Any, protein_source: Any, settings: Mapping[str, Any]):
    """AdamW; ESM2 (nếu được train cùng) nằm ở nhóm có learning rate riêng, nhỏ hơn.

    Vì sao tách ESM2: ESM2 đã pretrain trên hàng trăm triệu chuỗi, còn phần đầu dự đoán
    khởi tạo ngẫu nhiên. Dùng cùng learning rate sẽ phá trọng số pretrain trước khi phần
    đầu kịp học. Các nhánh còn lại tách theo branch_learning_rate_scale (mặc định rỗng
    = một nhóm duy nhất, đúng hành vi cũ).
    """
    import torch

    scales = settings.get("branch_learning_rate_scale") or {}
    groups = branch_parameter_groups(model, float(settings["learning_rate"]), scales)
    if isinstance(protein_source, ProteinEncoderRuntime):
        groups.append({"params": list(protein_source.parameters()),
                       "branch": "protein_encoder",
                       "lr": float(settings["protein_learning_rate"])})
    if len(groups) > 1:
        LOGGER.info("learning rate theo nhánh: %s",
                    " | ".join(f"{group['branch']} {group['lr']:.4g}"
                               for group in groups))
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


def forward_mean(model: Any, arrays: data.PairArrays, rows: np.ndarray, caches: Caches,
                 protein_source: Any, device: str) -> Any:
    """Đầu ra trung bình của một batch, dựng lại batch từ đầu.

    Chỉ IGNITE dùng hàm này, cho lượt forward ở tham số đã bị làm nhiễu. Phải dựng lại
    batch (không dùng lại tensor cũ) vì ở cấu hình finetune nhánh protein cũng đang được
    train, nên vector protein phải được tính lại theo tham số mới.
    """
    batch = batch_tensors(arrays, rows, caches, protein_source, device,
                          model.uses_edit_branch)
    return model(batch["ligand_a"], batch["ligand_b"], batch["fragment_a"],
                 batch["fragment_b"], batch["protein"], batch["endpoint"]).mean()


def train_one_epoch(model: Any, arrays: data.PairArrays, caches: Caches, protein_source: Any,
                    device: str, optimizer: Any, criterion: Any, batch_size: int,
                    grad_clip: float, rng: np.random.Generator,
                    penalty: sharpness.SharpnessPenalty | None = None,
                    eval_every: int = 0,
                    on_eval: Callable[[int], bool] | None = None,
                    group_by_protein: bool = False,
                    accumulate_steps: int = 1) -> tuple[float, bool, int]:
    """Một epoch; trả về (mất mát trung bình theo dòng, có dừng sớm không, số bước).

    ``penalty`` khác None thì mỗi batch đi theo Algorithm 1 của IGNITE thay vì
    ``loss.backward()`` (xem molopt.evaluator.sharpness).

    ``group_by_protein`` gom batch theo protein (xem data.iter_protein_batches): bắt
    buộc ở cấu hình finetune, nếu không mỗi batch chạm tới hàng chục protein và ESM2
    phải chạy từng ấy lượt trong đồ thị tính toán.

    ``accumulate_steps`` > 1 thì gộp bấy nhiêu batch con lại rồi mới cập nhật tham số
    một lần. Dùng cùng ``group_by_protein``: gom theo protein làm mỗi batch chỉ còn
    một protein, nên gradient của một bước rất lệch; cộng dồn vài batch rồi mới bước
    thì hướng đi lại trộn nhiều protein như khi lấy mẫu ngẫu nhiên, mà bộ nhớ vẫn thấp.
    "Bước" đếm theo số lần CẬP NHẬT THAM SỐ, không phải số batch con, nên nhịp đo
    validation của ``eval_every`` giữ nguyên ý nghĩa.

    ``eval_every`` > 0 thì cứ sau bấy nhiêu bước lại gọi ``on_eval(step)`` để đo
    validation NGAY GIỮA epoch. Cần thiết vì một epoch dài ~12.900 bước mà lần train
    31/10/2026 cho điểm tốt nhất ngay ở epoch 0: đo mỗi epoch một lần thì điểm tốt nhất
    đã trôi qua từ lâu. ``on_eval`` trả về True nghĩa là dừng sớm.
    """
    import torch

    model.train()
    if isinstance(protein_source, ProteinEncoderRuntime):
        protein_source.train(True)
    trainable = list(model.parameters())
    if isinstance(protein_source, ProteinEncoderRuntime):
        trainable += list(protein_source.parameters())
    total = 0.0
    seen = 0
    step = 0
    stopped = False
    batches = (data.iter_protein_batches(arrays.protein, batch_size, rng)
               if group_by_protein else data.iter_batches(len(arrays), batch_size, rng))
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    for rows in batches:
        batch = batch_tensors(arrays, rows, caches, protein_source, device,
                              model.uses_edit_branch)
        prediction = model(batch["ligand_a"], batch["ligand_b"], batch["fragment_a"],
                           batch["fragment_b"], batch["protein"], batch["endpoint"])
        loss = criterion(prediction, batch["target"])
        share = 1.0 / accumulate_steps
        if penalty is None:
            # PyTorch vốn CỘNG vào .grad, nên chia mất mát là đủ để lấy trung bình.
            (loss * share if accumulate_steps > 1 else loss).backward()
        else:
            penalty.apply(loss, prediction, trainable,
                          lambda rows=rows: forward_mean(model, arrays, rows, caches,
                                                         protein_source, device),
                          share, accumulate_steps > 1)
        total += float(loss.detach().to("cpu")) * rows.size
        seen += rows.size
        pending += 1
        if pending < accumulate_steps:
            continue
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        step += 1
        if eval_every > 0 and on_eval is not None and step % eval_every == 0:
            stopped = on_eval(step)
            # `predict` đã chuyển model sang chế độ eval, phải bật lại chế độ train.
            model.train()
            if isinstance(protein_source, ProteinEncoderRuntime):
                protein_source.train(True)
            if stopped:
                break
    if pending:
        # Batch lẻ ở cuối epoch: vẫn cập nhật, không vứt gradient đã tính.
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    return total / max(seen, 1), stopped, step


def effective_batch_sizes(settings: Mapping[str, Any], protein_mode: str,
                          smoke: bool) -> tuple[int, int]:
    """Kích thước batch cho train và cho dự đoán, đã tính tới cấu hình finetune.

    Ở finetune, ESM2 nằm trong đồ thị tính toán nên bộ nhớ tỉ lệ với số protein PHÂN
    BIỆT trong batch nhân số acid amin — kể cả lượt dự đoán, vì vẫn phải chạy ESM2.
    Dùng ``eval_batch_size`` 4096 của cấu hình frozen ở đây sẽ hết VRAM ngay lượt đo
    validation đầu tiên, nên finetune có giá trị riêng trong ``finetune_overrides``.
    Trả về: (batch train, batch dự đoán).
    """
    from . import models

    batch_size = int(settings["batch_size"])
    eval_batch_size = int(settings["eval_batch_size"])
    if protein_mode == models.PROTEIN_FINETUNE:
        overrides = settings.get("finetune_overrides", {})
        batch_size = int(overrides.get("batch_size", batch_size))
        eval_batch_size = int(overrides.get("eval_batch_size", eval_batch_size))
    if smoke:
        # Smoke chỉ để xác nhận pipeline thông, batch nhỏ cho nhanh — nhưng KHÔNG được
        # lớn hơn batch thật của finetune, nếu không smoke lại tốn VRAM hơn lần train.
        batch_size = min(batch_size, 128)
    return batch_size, eval_batch_size


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


def filter_rare_edits(arrays: data.PairArrays, keys: Mapping[str, np.ndarray],
                      min_edit_rows: int, control: bool = False, seed: int = 0
                      ) -> tuple[data.PairArrays, dict[str, np.ndarray], dict[str, Any]]:
    """Bỏ các dòng mà phép sửa của dòng đó xuất hiện ít hơn ``min_edit_rows`` lần.

    Nhận: mảng đã mã hóa của MỘT partition; các cột khóa query; số lần xuất hiện tối
    thiểu để một phép sửa được giữ (1 = không lọc gì); ``control`` bỏ ngẫu nhiên ĐÚNG
    cùng số dòng thay vì bỏ theo tần suất; ``seed`` cho lần bỏ ngẫu nhiên đó.
    Trả về: (mảng đã lọc, khóa đã lọc, thống kê để ghi vào manifest).

    Vì sao cần ``control``: lọc theo tần suất bỏ đi 25 % train, nên nếu kết quả kém đi
    thì KHÔNG phân biệt được "phép sửa hiếm có ích" với "ít dữ liệu hơn thì kém hơn".
    Đối chứng bỏ ngẫu nhiên cùng số dòng tách được hai nguyên nhân đó. Chỉ khi lần lọc
    theo tần suất khác hẳn lần bỏ ngẫu nhiên thì mới nói được điều gì về phép sửa.

    Vì sao: phân tích ca sai trên train ngày 07/11/2026 (thư mục
    runs/evaluator/errors_frozen_direct_train_20261107_081720) cho thấy MAE theo tần suất
    phép sửa đi NGƯỢC chiều mọi nhóm khác — phép sửa xuất hiện 1 lần có MAE 0,1440 và
    đúng dấu 0,9881, còn phép sửa xuất hiện từ 66 lần trở lên có MAE 0,3216 và đúng dấu
    0,8790. Phép sửa chỉ gặp một lần thì mô hình nhớ thẳng giá trị chứ không rút ra quy
    luật nào, nên 1,68 triệu dòng loại đó có thể chỉ đang dạy mô hình ghi nhớ. Cờ này để
    ĐO điều đó, chưa phải kết luận.
    CHỈ áp cho train. Lọc validation sẽ đổi mẫu số và làm mọi con số cũ không so được
    nữa, nên hàm này không bao giờ được gọi cho validation hay test.
    """
    rows_before = len(arrays)
    unchanged: dict[str, Any] = {
        "min_edit_rows": int(min_edit_rows), "rows_before": rows_before,
        "rows_after": rows_before, "rows_dropped": 0, "edits_dropped": 0,
        "control_random_rows": False}
    if min_edit_rows <= 1:
        return arrays, dict(keys), unchanged
    codes = analysis.edit_codes(arrays.fragment_a, arrays.fragment_b)
    frequency = analysis.code_frequency(codes)
    keep = frequency >= int(min_edit_rows)
    if not keep.any():
        raise ValueError(f"min_edit_rows = {min_edit_rows} bỏ hết {rows_before} dòng; "
                         f"phép sửa nhiều dòng nhất chỉ có {int(frequency.max())} dòng")
    edits_dropped = int(np.unique(codes[~keep]).size)
    if control:
        # Giữ ĐÚNG cùng số dòng, nhưng chọn ngẫu nhiên: đối chứng cỡ mẫu.
        chosen = np.random.default_rng(seed).permutation(rows_before)[:int(keep.sum())]
        keep = np.zeros(rows_before, dtype=bool)
        keep[chosen] = True
    filtered = data.PairArrays(
        molecule_a=arrays.molecule_a[keep], molecule_b=arrays.molecule_b[keep],
        fragment_a=arrays.fragment_a[keep], fragment_b=arrays.fragment_b[keep],
        protein=arrays.protein[keep], endpoint=arrays.endpoint[keep],
        target=arrays.target[keep],
        slices={name: mask[keep] for name, mask in arrays.slices.items()})
    report: dict[str, Any] = {
        "min_edit_rows": int(min_edit_rows), "rows_before": rows_before,
        "rows_after": len(filtered), "rows_dropped": rows_before - len(filtered),
        "edits_dropped": 0 if control else edits_dropped,
        "control_random_rows": bool(control)}
    return filtered, {name: column[keep] for name, column in keys.items()}, report


# Chỉ số chọn checkpoint và dừng sớm (stage3.train.selection_metric). mae: nhỏ hơn là tốt
# (hành vi cũ). pairwise_accuracy_macro: đúng chiều macro, lớn hơn là tốt; đo 06/11/2026
# cho thấy MAE gần như bằng nhau (0,585-0,588) cả ở lần train sụp ra hằng số.
SELECT_MAE = "mae"
SELECT_PAIRWISE_MACRO = "pairwise_accuracy_macro"
SELECTION_METRICS = (SELECT_MAE, SELECT_PAIRWISE_MACRO)


def selection_score(scores: Mapping[str, Any], selection: str) -> float | None:
    """Điểm để so mốc đo, quy về dạng "nhỏ hơn là tốt"; None nếu chưa đo được.

    Nhận: khối chỉ số của lát cắt 'all' (kết quả ``evaluate``); tên chỉ số chọn.
    """
    if selection == SELECT_MAE:
        return scores["regression"]["mae"]
    value = (scores.get("ranking") or {}).get(selection)
    return None if value is None else -float(value)


def run_train(config: Mapping[str, Any], root: Path, configuration: str, head: str,
              smoke: bool, stamp: str, sharpness_mode: str | None = None,
              overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Train một cấu hình; chọn checkpoint theo ``selection_metric`` trên validation.

    Nhận: config; thư mục gốc; cấu hình ('frozen' | 'finetune' | 'no_protein'); kiểu đầu
    ra ('direct' | 'twin'); smoke; nhãn thời gian của lần chạy; ``sharpness_mode``
    ('on' | 'off' | None) để bật/tắt số hạng phạt IGNITE mà không phải sửa config;
    ``overrides`` là các tham số đã bị dòng lệnh ghi đè, CHỈ để ghi vào manifest (giá
    trị đã nằm sẵn trong config lúc gọi — xem run_evaluator.apply_overrides).
    Trả về: manifest của lần chạy (đã ghi ra đĩa).
    """
    from . import models

    settings = config["stage3"]["train"]
    protein_mode = models.protein_mode_for(configuration)
    device = resolve_device(settings["device"])
    limit_gpu_memory(settings, device)
    set_seeds(int(settings["seed"]))
    _, runs_dir = layout(config, root)
    limit = int(settings["smoke_rows"]) if smoke else None
    epochs = int(settings["smoke_epochs"] if smoke else settings["epochs"])
    batch_size, eval_batch_size = effective_batch_sizes(settings, protein_mode, smoke)
    # Gom batch theo protein CHỈ ở finetune: chỉ ở đó ESM2 mới nằm trong đồ thị
    # tính toán nên số protein phân biệt mỗi batch mới quyết định bộ nhớ.
    group_by_protein = protein_mode == models.PROTEIN_FINETUNE
    accumulate_steps = max(1, int(settings.get("accumulate_steps", 1) or 1))

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
            # KHÔNG đặt tên biến này là `overrides`: trùng với tham số của hàm, và
            # manifest ở cuối hàm ghi `overrides` — trùng tên thì mọi lần finetune ghi
            # finetune_overrides thay cho danh sách --set, mất dấu tham số đã quét.
            tuning = settings.get("finetune_overrides", {})
            if "max_residues" in tuning:
                encoder_settings["max_residues"] = int(tuning["max_residues"])
            ordered = [sequences[key] for key in protein_vocabulary.to_pylist()]
            protein_source = ProteinEncoderRuntime(encoder_settings, ordered, device)
            protein_dim = protein_source.dimension
            LOGGER.info("CHIỀU EMBEDDING ĐO ĐƯỢC: %s (model %s)", protein_dim,
                        encoder_settings["model"])

    train_arrays, train_keys, _ = prepare_arrays(config, root, data.TRAIN, caches,
                                                 protein_vocabulary, limit)
    validation_arrays, validation_keys, _ = prepare_arrays(
        config, root, data.VALIDATION, caches, protein_vocabulary, limit)
    # Lọc phép sửa hiếm CHỈ ở train; validation giữ nguyên để số liệu còn so được.
    train_arrays, train_keys, edit_filter = filter_rare_edits(
        train_arrays, train_keys, int(settings.get("min_edit_rows", 1) or 1),
        bool(settings.get("min_edit_rows_control", False)), int(settings["seed"]))
    if edit_filter["rows_dropped"]:
        LOGGER.info("lọc train (%s): bỏ %s dòng (%.1f %%), %s phép sửa; còn %s dòng",
                    "ĐỐI CHỨNG bỏ ngẫu nhiên cùng cỡ" if edit_filter[
                        "control_random_rows"] else
                    f"phép sửa dưới {edit_filter['min_edit_rows']} dòng",
                    f"{edit_filter['rows_dropped']:,}",
                    100.0 * edit_filter["rows_dropped"] / edit_filter["rows_before"],
                    f"{edit_filter['edits_dropped']:,}",
                    f"{edit_filter['rows_after']:,}")
    LOGGER.info("train %s dòng | validation %s dòng | thiết bị %s | batch %s",
                f"{len(train_arrays):,}", f"{len(validation_arrays):,}", device, batch_size)

    model = models.EvaluatorModel(config["stage3"]["model"], caches.ligand_bits,
                                  caches.fragment_bits, protein_dim,
                                  len(config["endpoints"]), head, protein_mode)
    model.to(device)
    optimizer = build_optimizer(model, protein_source, settings)
    criterion = loss_function(settings)
    rng = np.random.default_rng(int(settings["seed"]))
    penalty_settings = settings["sharpness"]
    penalty = (sharpness.SharpnessPenalty(penalty_settings)
               if sharpness.is_enabled(penalty_settings, sharpness_mode) else None)
    if penalty is None:
        LOGGER.info("số hạng phạt IGNITE: TẮT")
    else:
        LOGGER.info("số hạng phạt IGNITE: BẬT (lambda đầu %.4g, rho %.4g, epsilon %.4g)",
                    penalty.lambda_value, penalty.rho, penalty.epsilon)
    eval_every = int(settings.get("eval_every_steps", 0) or 0)
    if smoke:
        # Smoke chỉ chạy vài nghìn dòng nên một epoch ngắn hơn eval_every rất nhiều;
        # đo giữa epoch ở đây chỉ làm chậm mà không nói thêm điều gì.
        eval_every = 0
    patience = int(settings["early_stopping_patience_evals"] if eval_every > 0
                   else settings["early_stopping_patience"])
    checkpoint_path = runs_dir / f"{configuration}_{head}_{stamp}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    selection = str(settings.get("selection_metric", SELECT_MAE))
    if selection not in SELECTION_METRICS:
        raise ValueError(f"selection_metric phải là một trong {SELECTION_METRICS}, "
                         f"nhận {selection!r}")
    LOGGER.info("chọn checkpoint theo: %s", selection)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_point = -1
    since_best = 0
    started = time.perf_counter()

    def consider(epoch: int, step: int | None, train_loss: float | None) -> bool:
        """Đo validation tại một mốc, lưu checkpoint nếu tốt nhất; True = dừng sớm.

        ``step`` bằng None là mốc cuối epoch, khác None là mốc giữa epoch.
        """
        nonlocal best_score, best_point, since_best
        import torch

        measured = time.perf_counter()
        prediction = predict(model, validation_arrays, caches, protein_source, device,
                             eval_batch_size)
        scores = evaluate(validation_arrays, prediction, validation_keys, config)
        score = selection_score(scores["all"], selection)
        label = f"epoch {epoch}" if step is None else f"epoch {epoch} bước {step}"
        record: dict[str, Any] = {"epoch": epoch, "step": step,
                                  "validation": scores["all"],
                                  "seconds": round(time.perf_counter() - measured, 1)}
        if train_loss is not None:
            record["train_loss"] = train_loss
        if penalty is not None:
            record["sharpness"] = penalty.state()
        history.append(record)
        for line in metrics.format_metrics(f"{label} validation",
                                           scores["all"]["regression"],
                                           scores["all"].get("ranking"),
                                           scores["all"].get("change_classes")):
            LOGGER.info("%s", line)
        if train_loss is not None:
            LOGGER.info("%s: train loss %.4f", label, train_loss)
        if penalty is not None:
            penalty_state = penalty.state()
            LOGGER.info("%s: lambda %.4g | ||grad đầu ra|| %.4g | độ sắc ước lượng "
                        "%.4g (ngưỡng epsilon %.4g)", label, penalty_state["lambda"],
                        penalty_state["output_gradient_norm"],
                        penalty_state["sharpness_estimate"], penalty.epsilon)
        if score is None:
            return False
        if score < best_score:
            best_score, since_best = score, 0
            best_point = len(history) - 1
            state = {"model": model.state_dict(), "epoch": epoch, "step": step,
                     "configuration": configuration, "head": head,
                     "protein_mode": protein_mode, "protein_dim": protein_dim,
                     "ligand_bits": caches.ligand_bits,
                     "fragment_bits": caches.fragment_bits,
                     "model_settings": models.architecture_settings(
                         config["stage3"]["model"])}
            if isinstance(protein_source, ProteinEncoderRuntime):
                state["protein_encoder"] = protein_source.model.state_dict()
            torch.save(state, checkpoint_path)
            LOGGER.info("checkpoint tốt nhất (%s) -> %s", label, checkpoint_path)
            return False
        since_best += 1
        if since_best >= patience:
            LOGGER.info("dừng sớm: %s lần đo không cải thiện %s validation", since_best,
                        selection)
            return True
        return False

    for epoch in range(epochs):
        loss, stopped, steps = train_one_epoch(
            model, train_arrays, caches, protein_source, device, optimizer, criterion,
            batch_size, float(settings["grad_clip"]), rng, penalty, eval_every,
            lambda step, current=epoch: consider(current, step, None),
            group_by_protein, accumulate_steps)
        LOGGER.info("epoch %s: %s bước", epoch, f"{steps:,}")
        if stopped or consider(epoch, None, loss):
            break

    if best_point >= 0:
        # Số cuối cùng (theo lát cắt, theo endpoint, bất đối xứng) phải là của checkpoint
        # TỐT NHẤT đã lưu, không phải của epoch cuối: dừng sớm nghĩa là epoch cuối kém hơn.
        load_best_checkpoint(checkpoint_path, model, protein_source, device)
    final = evaluate(validation_arrays, predict(model, validation_arrays, caches,
                                                protein_source, device,
                                                eval_batch_size),
                     validation_keys, config)
    asymmetry = asymmetry_report(model, validation_arrays, caches, protein_source, device,
                                 config, eval_batch_size)
    if asymmetry is not None:
        for line in metrics.format_asymmetry("bất đối xứng (validation)", asymmetry):
            LOGGER.info("%s", line)
    manifest = {
        "stage": "stage3_train", "mode": "smoke" if smoke else "train",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configuration": configuration, "head": head, "protein_mode": protein_mode,
        "model": models.architecture_settings(config["stage3"]["model"]),
        "device": device, "seed": int(settings["seed"]),
        "epochs_run": 1 + max((record["epoch"] for record in history), default=-1),
        "measurements": len(history), "eval_every_steps": eval_every,
        "group_by_protein": group_by_protein, "accumulate_steps": accumulate_steps,
        "overrides": dict(overrides or {}), "batch_size": batch_size,
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "loss": str(settings["loss"]), "huber_delta": float(settings["huber_delta"]),
        "rows": {"train": len(train_arrays), "validation": len(validation_arrays)},
        "protein_embedding_dim": protein_dim,
        "parameters": models.count_parameters(model),
        "fingerprint_bits": {"ligand": caches.ligand_bits, "fragment": caches.fragment_bits},
        "uses_edit_branch": model.uses_edit_branch,
        "edit_filter": edit_filter,
        "branch_learning_rate_scale": dict(
            settings.get("branch_learning_rate_scale") or {}),
        "sharpness_penalty": None if penalty is None else {
            "source": "IGNITE, NeurIPS 2024, arXiv:2503.04242",
            "rho": penalty.rho, "epsilon": penalty.epsilon,
            "finite_difference_radius": penalty.radius,
            "lambda_init": float(penalty_settings["lambda_init"]),
            "lambda_learning_rate": penalty.lambda_learning_rate,
            "lambda_final": penalty.lambda_value,
        },
        "best_epoch": None if best_point < 0 else history[best_point]["epoch"],
        "best_step": None if best_point < 0 else history[best_point]["step"],
        "selection_metric": selection,
        "best_validation_mae": None if best_point < 0 else
        history[best_point]["validation"]["regression"]["mae"],
        "final_metrics_index": best_point if best_point >= 0 else len(history) - 1,
        "checkpoint": str(checkpoint_path) if best_point >= 0 else None,
        "history": history, "validation_metrics": final, "asymmetry": asymmetry,
        "seconds": round(time.perf_counter() - started, 1),
        "gpu_peak_reserved_mib": peak_gpu_memory_mib(device),
    }
    LOGGER.info("VRAM cao nhất PyTorch đã giữ: %s MiB", manifest["gpu_peak_reserved_mib"])
    path = runs_dir / f"{manifest['mode']}_{configuration}_{head}_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("manifest lần chạy: %s", path)
    return manifest


@dataclass
class LoadedCheckpoint:
    """Một checkpoint đã nạp, sẵn sàng chấm điểm (dùng chung cho eval, errors, features)."""

    state: dict[str, Any]
    model: Any
    caches: Caches
    protein_source: Any
    protein_vocabulary: pa.Array
    protein_mode: str
    configuration: str
    head: str
    device: str
    eval_batch_size: int


def load_checkpoint(config: Mapping[str, Any], root: Path,
                    checkpoint: Path) -> LoadedCheckpoint:
    """Nạp checkpoint, cache và nguồn vector protein đúng như lúc train."""
    import torch

    from . import models

    settings = config["stage3"]["train"]
    device = resolve_device(settings["device"])
    limit_gpu_memory(settings, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
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
    # Checkpoint trước 03/11/2026 không lưu kiến trúc; khi đó chỉ có concat + plain và head
    # không chuẩn hóa, nên ép đúng các giá trị đó dù config hiện tại đang đặt gì. Chiều các
    # lớp vẫn lấy từ config: các khóa đó chưa từng đổi, và nếu đổi thì load_state_dict báo
    # lỗi lệch chiều. Checkpoint 03-06/11/2026 có model_settings nhưng chưa có head_norm:
    # EvaluatorModel đọc thiếu khóa thành 'none', đúng kiến trúc lúc đó.
    model_settings = state.get("model_settings") or {
        **config["stage3"]["model"], "fusion": models.FUSION_CONCAT,
        "head_block": models.HEAD_BLOCK_PLAIN, "head_norm": models.HEAD_NORM_NONE}
    model = models.EvaluatorModel(model_settings, int(state["ligand_bits"]),
                                  int(state["fragment_bits"]), state["protein_dim"],
                                  len(config["endpoints"]), str(state["head"]), protein_mode)
    model.load_state_dict(state["model"])
    model.to(device)
    _, eval_batch_size = effective_batch_sizes(settings, protein_mode, False)
    return LoadedCheckpoint(state=state, model=model, caches=caches,
                            protein_source=protein_source,
                            protein_vocabulary=protein_vocabulary, protein_mode=protein_mode,
                            configuration=str(state["configuration"]),
                            head=str(state["head"]), device=device,
                            eval_batch_size=eval_batch_size)


def run_eval(config: Mapping[str, Any], root: Path, checkpoint: Path,
             stamp: str, partition: str = data.VALIDATION) -> dict[str, Any]:
    """Nạp checkpoint rồi báo cáo chỉ số trên một partition theo từng lát cắt.

    ``partition`` là validation (mặc định) hoặc train. Chấm trên train để so với validation:
    hai số gần nhau là mô hình chưa học được cả dữ liệu đã thấy, train cao hẳn là overfit.
    Không train, không chạm partition test (read_partition chặn).
    """
    loaded = load_checkpoint(config, root, checkpoint)
    model, caches, device = loaded.model, loaded.caches, loaded.device
    protein_source, protein_vocabulary = loaded.protein_source, loaded.protein_vocabulary
    state, configuration, protein_mode = loaded.state, loaded.configuration, loaded.protein_mode
    eval_batch_size = loaded.eval_batch_size
    arrays, keys, _ = prepare_arrays(config, root, partition, caches,
                                     protein_vocabulary, None)
    prediction = predict(model, arrays, caches, protein_source, device, eval_batch_size)
    scores = evaluate(arrays, prediction, keys, config)
    for name, block in scores.items():
        if name == "by_endpoint" or not isinstance(block, dict):
            continue
        for line in metrics.format_metrics(f"{partition}/{name}", block["regression"],
                                           block.get("ranking"), block.get("change_classes")):
            LOGGER.info("%s", line)
    asymmetry = asymmetry_report(model, arrays, caches, protein_source, device, config,
                                 eval_batch_size)
    if asymmetry is not None:
        for line in metrics.format_asymmetry(f"bất đối xứng ({partition})", asymmetry):
            LOGGER.info("%s", line)
    _, runs_dir = layout(config, root)
    # Khóa "validation_metrics" giữ nguyên tên cũ vì scripts/summarize_*.py đọc khóa này.
    manifest = {"stage": "stage3_eval", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": str(checkpoint), "configuration": configuration,
                "head": str(state["head"]), "protein_mode": protein_mode, "device": device,
                "partition": partition, "rows": len(arrays),
                f"{partition}_metrics": scores, "asymmetry": asymmetry}
    suffix = "" if partition == data.VALIDATION else f"_{partition}"
    path = runs_dir / f"eval_{configuration}_{state['head']}{suffix}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("manifest đánh giá: %s", path)
    return manifest


# ------------------------------------------------------------------ phân tích ca sai


# Cột chỉ đọc thêm cho phân tích, không có trong data.PAIR_COLUMNS. pair_id để khớp dòng.
ANALYSIS_EXTRA_COLUMNS = ["pair_id", "max_replicate_std", "p_a_range", "mixed_assay_type",
                          "assay_types"]


def flag_groups(column: pa.ChunkedArray) -> tuple[np.ndarray, list[str]]:
    """Cột bool -> nhóm False / True / thiếu."""
    from .analysis import MISSING_LABEL

    missing = np.asarray(pc.is_null(column).to_numpy(zero_copy_only=False), dtype=bool)
    values = np.asarray(pc.fill_null(column, False).to_numpy(zero_copy_only=False), dtype=bool)
    codes = np.where(missing, 2, values.astype(np.int64))
    return codes, ["False", "True", MISSING_LABEL]


def string_groups(column: pa.ChunkedArray) -> tuple[np.ndarray, list[str]]:
    """Cột chuỗi -> một nhóm cho mỗi giá trị khác nhau; null vào nhóm 'thiếu'."""
    from .analysis import MISSING_LABEL

    encoded = pc.dictionary_encode(column.combine_chunks())
    names = [str(value) for value in encoded.dictionary.to_pylist()]
    codes = np.asarray(pc.fill_null(encoded.indices, len(names)).to_numpy(
        zero_copy_only=False), dtype=np.int64)
    return codes, names + [MISSING_LABEL]


def numeric(table: pa.Table, name: str) -> np.ndarray:
    """Một cột số thành float64; null thành NaN."""
    return np.asarray(pc.cast(table.column(name), pa.float64()).to_numpy(
        zero_copy_only=False), dtype=np.float64)


def run_errors(config: Mapping[str, Any], root: Path, checkpoint: Path, stamp: str,
               partition: str = data.VALIDATION) -> dict[str, Any]:
    """Việc 6 (họp 02/11/2026): mô hình sai ở đâu trên một partition (train hoặc validation).

    Ghi vào runs_dir/errors_<cấu hình>_<đầu ra>_<partition>_<stamp>/:
        report.json    : mọi bảng chia nhóm, độ co dự đoán, protein và phép sửa tệ nhất
        rows.parquet   : từng dòng — nhãn, dự đoán, lỗi, đúng dấu, độ hiếm; nối với dataset
                         qua pair_id để xem thêm cột khác
        queries.parquet: từng query — số ứng viên, số cặp đúng chiều, Spearman, top-1
    Cách chia nhóm và ngưỡng: xem ghi chú stage3.analysis trong config và module analysis.
    """

    loaded = load_checkpoint(config, root, checkpoint)
    caches = loaded.caches
    arrays, keys, table = prepare_arrays(config, root, partition, caches,
                                         loaded.protein_vocabulary, None)
    extra = data.read_columns(config, root, partition, ANALYSIS_EXTRA_COLUMNS)
    if not extra.column("pair_id").equals(table.column("pair_id")):
        raise ValueError("hai lần đọc partition cho thứ tự dòng khác nhau; không khớp được")
    prediction = predict(loaded.model, arrays, caches, loaded.protein_source, loaded.device,
                         loaded.eval_batch_size)
    truth = arrays.target.astype(np.float64)
    predicted = prediction.astype(np.float64)
    metric_settings = config["stage3"]["metrics"]
    settings = config["stage3"]["analysis"]
    tolerance = float(metric_settings["direction_tolerance"])
    bins = int(settings["quantile_bins"])
    endpoints = list(config["endpoints"])

    protein_rows = analysis.code_frequency(arrays.protein)
    molecule_a_rows, molecule_b_rows = analysis.molecule_frequency(arrays.molecule_a,
                                                                   arrays.molecule_b)
    edits = analysis.edit_codes(arrays.fragment_a, arrays.fragment_b)
    edit_rows = analysis.code_frequency(edits)
    warning = np.asarray(pc.is_valid(table.column("source_quality_warning")).to_numpy(
        zero_copy_only=False), dtype=np.int64)
    groupings: dict[str, tuple[np.ndarray, list[str]]] = {
        "endpoint": (arrays.endpoint.astype(np.int64), endpoints),
        "abs_delta_p": analysis.magnitude_groups(truth, tolerance, bins),
        "n_assays": analysis.quantile_groups(numeric(table, "n_assays"), bins),
        "n_documents": analysis.quantile_groups(numeric(table, "n_documents"), bins),
        "n_distinct_values": analysis.quantile_groups(numeric(table, "n_distinct_values"),
                                                      bins),
        "max_replicate_std": analysis.quantile_groups(numeric(extra, "max_replicate_std"),
                                                      bins),
        "p_a_range": analysis.quantile_groups(numeric(extra, "p_a_range"), bins),
        "mixed_assay_type": flag_groups(extra.column("mixed_assay_type")),
        "assay_types": string_groups(extra.column("assay_types")),
        "plausible_range": flag_groups(table.column("plausible_range")),
        "source_quality_warning": (warning, ["không cảnh báo", "có cảnh báo"]),
        "unseen_pair": flag_groups(table.column("unseen_pair")),
        "qc_status": string_groups(table.column("qc_status")),
        "protein_rows": analysis.quantile_groups(protein_rows.astype(np.float64), bins),
        "molecule_a_rows": analysis.quantile_groups(molecule_a_rows.astype(np.float64), bins),
        "edit_rows": analysis.quantile_groups(edit_rows.astype(np.float64), bins),
    }
    row_tables = {name: analysis.group_table(codes, names, truth, predicted, tolerance)
                  for name, (codes, names) in groupings.items()}

    queries = analysis.query_table(keys, truth, predicted, tolerance,
                                   int(metric_settings["min_candidates"]),
                                   bool(metric_settings["expand_both_directions"]))
    protein_counts = np.bincount(arrays.protein, minlength=len(loaded.protein_vocabulary))
    query_groupings = {
        "candidates": analysis.quantile_groups(queries["candidates"].astype(np.float64), bins),
        "endpoint": (queries["endpoint"], endpoints),
        "protein_rows": analysis.quantile_groups(
            protein_counts[queries["protein"]].astype(np.float64), bins),
    }
    query_tables = {name: analysis.query_group_table(codes, names, queries)
                    for name, (codes, names) in query_groupings.items()}
    overall = analysis.query_group_table(np.zeros(queries["agree"].size, dtype=np.int64),
                                         ["toàn bộ"], queries)

    worst_count = int(settings["worst_count"])
    proteins = analysis.worst_proteins(queries, int(settings["min_group_queries"]),
                                       worst_count)
    for entry in proteins:
        entry["protein_hash"] = loaded.protein_vocabulary[entry.pop("protein")].as_py()
    edit_list = analysis.worst_edits(edits, arrays.fragment_a, arrays.fragment_b, truth,
                                     predicted, tolerance, int(settings["min_group_rows"]),
                                     worst_count)
    for entry in edit_list:
        entry.pop("edit")
        entry["variable_a"] = caches.fragment_keys[entry.pop("fragment_a")].as_py()
        entry["variable_b"] = caches.fragment_keys[entry.pop("fragment_b")].as_py()

    spread = analysis.spread_report(truth, predicted)
    LOGGER.info("độ co dự đoán (%s): %s", partition, spread)
    for line in analysis.format_group_table(f"đúng chiều theo query ({partition})", overall):
        LOGGER.info("%s", line)
    for name, rows in row_tables.items():
        for line in analysis.format_group_table(f"theo dòng / {name}", rows):
            LOGGER.info("%s", line)
    for name, rows in query_tables.items():
        for line in analysis.format_group_table(f"theo query / {name}", rows):
            LOGGER.info("%s", line)

    _, runs_dir = layout(config, root)
    folder = runs_dir / f"errors_{loaded.configuration}_{loaded.head}_{partition}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    counted, correct = analysis.sign_agreement(truth, predicted, tolerance)
    pq.write_table(pa.table({
        "pair_id": table.column("pair_id"),
        "endpoint": table.column("endpoint"),
        "protein_hash": table.column("protein_hash"),
        "delta_p": pa.array(truth),
        "prediction": pa.array(predicted),
        "error": pa.array(predicted - truth),
        "sign_counted": pa.array(counted),
        "sign_correct": pa.array(correct),
        "protein_rows": pa.array(protein_rows),
        "molecule_a_rows": pa.array(molecule_a_rows),
        "molecule_b_rows": pa.array(molecule_b_rows),
        "edit_rows": pa.array(edit_rows),
    }), folder / "rows.parquet")
    pq.write_table(pa.table({
        "reference_smiles": caches.molecule_keys.take(pa.array(queries["reference"])),
        "protein_hash": loaded.protein_vocabulary.take(pa.array(queries["protein"])),
        "endpoint": pa.array([endpoints[int(code)] for code in queries["endpoint"]],
                             pa.string()),
        **{name: pa.array(queries[name]) for name in ("candidates", "agree", "counted",
                                                      "spearman", "top_1")},
    }), folder / "queries.parquet")
    report = {
        "stage": "stage3_errors", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(checkpoint), "configuration": loaded.configuration,
        "head": loaded.head, "partition": partition, "rows": len(arrays),
        "direction_tolerance": tolerance, "settings": dict(settings),
        "spread": spread, "queries_overall": overall[0] if overall else None,
        "by_row": row_tables, "by_query": query_tables,
        "worst_proteins": proteins, "worst_edits": edit_list,
    }
    path = folder / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("phân tích ca sai: %s", folder)
    return report


# ------------------------------------------------------------------ vector cho t-SNE


def represent(model: Any, arrays: data.PairArrays, caches: Caches, protein_source: Any,
              device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """(vector trước đầu dự đoán, Δp dự đoán) cho mọi dòng; ở twin vector là [A, B] nối lại."""
    import torch

    model.eval()
    if isinstance(protein_source, ProteinEncoderRuntime):
        protein_source.train(False)
    vectors: list[np.ndarray] = []
    outputs = np.empty(len(arrays), dtype=np.float32)
    with torch.no_grad():
        for rows in data.iter_batches(len(arrays), batch_size):
            batch = batch_tensors(arrays, rows, caches, protein_source, device,
                                  model.uses_edit_branch)
            inputs = (batch["ligand_a"], batch["ligand_b"], batch["fragment_a"],
                      batch["fragment_b"], batch["protein"], batch["endpoint"])
            pieces = model.representations(*inputs)
            vectors.append(torch.cat(pieces, dim=1).to("cpu").numpy().astype(np.float32))
            outputs[rows] = model(*inputs).to("cpu").numpy()
    if not vectors:
        raise ValueError("không có dòng nào để lấy vector")
    return np.concatenate(vectors), outputs


def run_features(config: Mapping[str, Any], root: Path, checkpoint: Path,
                 stamp: str) -> dict[str, Any]:
    """Vector sau encoder của một mẫu dòng train và validation, để vẽ t-SNE (việc 4).

    Mỗi partition lấy ngẫu nhiên stage3.analysis.feature_rows_per_partition dòng (seed
    stage3.train.seed). Ghi runs_dir/features_<cấu hình>_<đầu ra>_<stamp>.parquet với cột
    partition, pair_id, endpoint, protein_hash, delta_p, prediction và feature (mảng số
    thực cố định độ dài), kèm file .json mô tả. Không đọc test.
    """
    loaded = load_checkpoint(config, root, checkpoint)
    per_partition = int(config["stage3"]["analysis"]["feature_rows_per_partition"])
    seed = int(config["stage3"]["train"]["seed"])
    rng = np.random.default_rng(seed)
    parts: list[pa.Table] = []
    counts: dict[str, int] = {}
    width = 0
    for partition in (data.TRAIN, data.VALIDATION):
        arrays, _, table = prepare_arrays(config, root, partition, loaded.caches,
                                          loaded.protein_vocabulary, None)
        count = min(per_partition, len(arrays))
        rows = np.sort(rng.choice(len(arrays), size=count, replace=False))
        sample = data.take_rows(arrays, rows)
        vectors, prediction = represent(loaded.model, sample, loaded.caches,
                                        loaded.protein_source, loaded.device,
                                        loaded.eval_batch_size)
        width = int(vectors.shape[1])
        picked = pa.array(rows)
        parts.append(pa.table({
            "partition": pa.array([partition] * count, pa.string()),
            "pair_id": table.column("pair_id").take(picked),
            "endpoint": table.column("endpoint").take(picked),
            "protein_hash": table.column("protein_hash").take(picked),
            "delta_p": pa.array(sample.target),
            "prediction": pa.array(prediction),
            "feature": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), width),
        }))
        counts[partition] = count
        LOGGER.info("vector %s: %s dòng x %s chiều", partition, f"{count:,}", width)
    _, runs_dir = layout(config, root)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"features_{loaded.configuration}_{loaded.head}_{stamp}.parquet"
    pq.write_table(pa.concat_tables(parts), path)
    manifest = {"stage": "stage3_features", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": str(checkpoint), "configuration": loaded.configuration,
                "head": loaded.head, "rows": counts, "feature_dim": width, "seed": seed,
                "file": str(path),
                "feature_note": "vector ngay trước MLP dự đoán (mọi nhánh đã mã hóa và ghép); "
                                "ở twin là [vector A, vector B] nối lại"}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    LOGGER.info("vector cho t-SNE: %s", path)
    return manifest


# ------------------------------------------------------------------ chẩn đoán model sụp


# Thứ tự đối số của EvaluatorModel.forward.
MODEL_INPUTS = ("ligand_a", "ligand_b", "fragment_a", "fragment_b", "protein", "endpoint")

# Đo độ nhạy: tên -> các đầu vào được hoán vị CÙNG một thứ tự trong batch. B đi cùng nhóm
# thế của B, để B được thay bằng một phân tử B khác có thật chứ không phải ghép lệch.
SENSITIVITY_INPUTS = {"ligand_b": ("ligand_b", "fragment_b"),
                      "edit": ("fragment_a", "fragment_b"),
                      "protein": ("protein",),
                      "endpoint": ("endpoint",)}


def diagnose_model(model: Any, arrays: data.PairArrays, caches: Caches, protein_source: Any,
                   device: str, batch_size: int, seed: int) -> dict[str, Any]:
    """Model sụp ở đâu: nơ-ron ReLU chết và độ nhạy của dự đoán với từng đầu vào.

    Hai phép đo, đều ở chế độ eval (dropout tắt), không tính gradient:
        relu_layers : với mỗi lớp nn.ReLU, tỉ lệ nơ-ron KHÔNG BAO GIỜ bật (> 0) trên toàn
                      mẫu (dead_fraction) và tỉ lệ bật trung bình (active_rate). Lớp có
                      dead_fraction = 1 thì mọi thứ phía sau chỉ thấy hằng số. ReLU viết dạng
                      hàm torch.relu (lớp ra của ResidualHead, nhánh protein ở concat) không
                      có hook nên không có trong danh sách.
        sensitivity : trung bình |f(đầu vào bị xáo) - f(gốc)| khi hoán vị một đầu vào giữa
                      các dòng trong batch, kèm tỉ lệ so với độ lệch chuẩn nhãn. Gần 0 nghĩa
                      là model không dùng đầu vào đó.
    Hoán vị lấy từ np.random.default_rng(seed), nên chạy lại cho đúng số cũ.
    """
    import torch

    if len(arrays) == 0:
        raise ValueError("không có dòng nào để chẩn đoán")
    model.eval()
    if isinstance(protein_source, ProteinEncoderRuntime):
        protein_source.train(False)
    relus = [(name, module) for name, module in model.named_modules()
             if isinstance(module, torch.nn.ReLU)]
    ever: dict[str, Any] = {}
    active = {name: 0.0 for name, _ in relus}
    total = {name: 0 for name, _ in relus}
    recording = [False]

    def make_hook(name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not recording[0]:
                return
            on = (output.detach() > 0).reshape(-1, output.shape[-1])
            seen = on.any(dim=0)
            ever[name] = seen if name not in ever else ever[name] | seen
            active[name] += float(on.sum())
            total[name] += int(on.numel())
        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in relus]
    rng = np.random.default_rng(seed)
    outputs = np.empty(len(arrays), dtype=np.float32)
    shifts = {name: 0.0 for name in SENSITIVITY_INPUTS}
    counted = {name: 0 for name in SENSITIVITY_INPUTS}
    try:
        with torch.no_grad():
            for rows in data.iter_batches(len(arrays), batch_size):
                batch = batch_tensors(arrays, rows, caches, protein_source, device,
                                      model.uses_edit_branch)
                recording[0] = True
                base = model(*(batch[name] for name in MODEL_INPUTS))
                recording[0] = False
                outputs[rows] = base.to("cpu").numpy()
                order = torch.from_numpy(rng.permutation(len(rows))).to(device)
                for name, keys in SENSITIVITY_INPUTS.items():
                    if any(batch[key] is None for key in keys):
                        continue
                    shuffled = dict(batch)
                    for key in keys:
                        shuffled[key] = batch[key].index_select(0, order)
                    changed = model(*(shuffled[key] for key in MODEL_INPUTS))
                    shifts[name] += float((changed - base).abs().sum())
                    counted[name] += len(rows)
    finally:
        recording[0] = False
        for handle in handles:
            handle.remove()

    layers = []
    for name, _ in relus:
        if total[name] == 0:
            continue
        units = ever[name]
        layers.append({"layer": name, "units": int(units.numel()),
                       "dead_fraction": float(1.0 - units.float().mean()),
                       "active_rate": active[name] / total[name]})
    prediction = outputs.astype(np.float64)
    label_sd = float(np.std(arrays.target.astype(np.float64)))
    sensitivity = {}
    for name, amount in shifts.items():
        if counted[name] == 0:
            continue
        shift = amount / counted[name]
        sensitivity[name] = {"mean_abs_shift": shift,
                             "relative_to_label_sd": shift / label_sd if label_sd > 0 else None}
    return {"rows": len(arrays),
            "prediction": {"mean": float(prediction.mean()), "std": float(prediction.std()),
                           "min": float(prediction.min()), "max": float(prediction.max()),
                           "distinct_1e-6": int(np.unique(np.round(prediction, 6)).size)},
            "label_sd": label_sd, "relu_layers": layers, "sensitivity": sensitivity}


def run_diagnose(config: Mapping[str, Any], root: Path, checkpoint: Path, stamp: str,
                 partition: str = data.VALIDATION) -> dict[str, Any]:
    """Chẩn đoán một checkpoint trên một mẫu dòng của partition (validation hoặc train).

    Dùng khi một lần train cho dự đoán gần như hằng số (06/11/2026: fusion=projector +
    head plain, đúng chiều 0,168, Spearman không tính được). Mẫu lấy
    stage3.analysis.feature_rows_per_partition dòng, seed stage3.train.seed. Ghi
    runs_dir/diagnose_<cấu hình>_<đầu ra>_<partition>_<stamp>.json. Không đọc test.
    """
    loaded = load_checkpoint(config, root, checkpoint)
    arrays, _, _ = prepare_arrays(config, root, partition, loaded.caches,
                                  loaded.protein_vocabulary, None)
    per_partition = int(config["stage3"]["analysis"]["feature_rows_per_partition"])
    seed = int(config["stage3"]["train"]["seed"])
    rng = np.random.default_rng(seed)
    count = min(per_partition, len(arrays))
    rows = np.sort(rng.choice(len(arrays), size=count, replace=False))
    report = diagnose_model(loaded.model, data.take_rows(arrays, rows), loaded.caches,
                            loaded.protein_source, loaded.device, loaded.eval_batch_size, seed)
    summary = report["prediction"]
    LOGGER.info("dự đoán (%s, %s dòng): trung bình %.4f | sd %.4f | [%.4f, %.4f] | "
                "%s giá trị khác nhau | sd nhãn %.4f", partition, f"{count:,}",
                summary["mean"], summary["std"], summary["min"], summary["max"],
                f"{summary['distinct_1e-6']:,}", report["label_sd"])
    for layer in report["relu_layers"]:
        LOGGER.info("ReLU %-28s %5d nơ-ron | chết %5.1f%% | tỉ lệ bật %.3f", layer["layer"],
                    layer["units"], 100 * layer["dead_fraction"], layer["active_rate"])
    for name, shift in report["sensitivity"].items():
        relative = shift["relative_to_label_sd"]
        LOGGER.info("xáo %-9s -> dự đoán đổi trung bình %.4f (= %s lần sd nhãn)", name,
                    shift["mean_abs_shift"], "-" if relative is None else f"{relative:.4f}")
    _, runs_dir = layout(config, root)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"diagnose_{loaded.configuration}_{loaded.head}_{partition}_{stamp}.json"
    manifest = {"stage": "stage3_diagnose", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": str(checkpoint), "configuration": loaded.configuration,
                "head": loaded.head, "partition": partition, "seed": seed,
                "model_settings": loaded.state.get("model_settings"), **report}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("chẩn đoán: %s", path)
    return manifest
