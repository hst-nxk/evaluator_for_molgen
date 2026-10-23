"""Stage 3: biểu diễn đầu vào — Morgan fingerprint và embedding ESM2, đều có cache.

Hai nhóm hàm, cố ý tách rời:

Fingerprint (RDKit, không cần torch)
    Dataset có 1,3 triệu phân tử phân biệt nhưng 10,5 triệu dòng, nên mỗi phân tử xuất
    hiện trung bình 8 lần. Fingerprint được tính MỘT LẦN cho mỗi SMILES phân biệt rồi
    cache ra đĩa; vòng train chỉ tra theo chỉ số. Bit được đóng gói bằng np.packbits
    (2.048 bit -> 256 byte) để cả bảng vừa RAM.

Embedding protein (ESM2, cần torch + transformers)
    Chỉ có 6.315 protein có cặp, nên cấu hình `frozen` chỉ phải chạy ESM2 6.315 lần, một
    lần duy nhất, rồi cache. Cấu hình `finetune` không dùng cache: ESM2 nằm trong đồ thị
    tính toán và chạy lại mỗi batch.

CHIỀU EMBEDDING KHÔNG ĐƯỢC HARDCODE. Nó được đọc từ model đã nạp (`config.hidden_size`),
ghi vào manifest của cache, và train/eval kiểm tra lại cho khớp.

Cache lưu dưới dạng parquet (không dùng pickle): mỗi dòng là một khóa kèm vector, nên
từ vựng và ma trận luôn đi cùng nhau và không thể lệch nhau.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Khóa metadata trong schema parquet của file cache.
METADATA_KEY = b"molopt_stage3"


# --------------------------------------------------------------------------- cache chung


def write_packed_cache(path: Path, keys: Sequence[str], packed: np.ndarray,
                       metadata: Mapping[str, Any]) -> None:
    """Ghi cache bit đã đóng gói: mỗi dòng là (khóa, các byte bit).

    Nhận: đường dẫn; danh sách khóa; ma trận uint8 (số dòng × số byte); metadata.
    Dùng fixed_size_binary nên mọi dòng cùng độ dài, và khi đọc lại có thể lấy cả ma
    trận từ một vùng nhớ liền mạch thay vì ghép từng dòng.
    """
    if packed.dtype != np.uint8:
        raise ValueError(f"cần uint8, nhận {packed.dtype}")
    if packed.shape[0] != len(keys):
        raise ValueError(f"lệch số dòng: {packed.shape[0]} vector và {len(keys)} khóa")
    width = int(packed.shape[1])
    table = pa.table(
        {"key": pa.array(list(keys), pa.string()),
         "packed": pa.array([row.tobytes() for row in packed], pa.binary(width))},
        metadata={METADATA_KEY: json.dumps(dict(metadata), ensure_ascii=False).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_packed_cache(path: Path) -> tuple[pa.Array, np.ndarray, dict[str, Any]]:
    """Đọc cache bit; trả về (khóa, ma trận uint8, metadata)."""
    if not path.is_file():
        raise FileNotFoundError(f"chưa có cache {path}; chạy --mode prepare")
    table = pq.read_table(path)
    values = table.column("packed").combine_chunks()
    width = values.type.byte_width
    # fixed_size_binary: buffers() = [validity, data]; data là vùng liền mạch.
    buffer = values.buffers()[1]
    matrix = np.frombuffer(buffer, dtype=np.uint8, count=len(values) * width)
    raw = table.schema.metadata.get(METADATA_KEY) if table.schema.metadata else None
    metadata = json.loads(raw.decode()) if raw else {}
    return table.column("key").combine_chunks(), matrix.reshape(len(values), width), metadata


def write_vector_cache(path: Path, keys: Sequence[str], vectors: np.ndarray,
                       metadata: Mapping[str, Any]) -> None:
    """Ghi cache vector float32: mỗi dòng là (khóa, vector độ dài cố định)."""
    if vectors.dtype != np.float32:
        raise ValueError(f"cần float32, nhận {vectors.dtype}")
    if vectors.shape[0] != len(keys):
        raise ValueError(f"lệch số dòng: {vectors.shape[0]} vector và {len(keys)} khóa")
    dimension = int(vectors.shape[1])
    flat = pa.array(np.ascontiguousarray(vectors).reshape(-1), pa.float32())
    table = pa.table(
        {"key": pa.array(list(keys), pa.string()),
         "vector": pa.FixedSizeListArray.from_arrays(flat, dimension)},
        metadata={METADATA_KEY: json.dumps(dict(metadata), ensure_ascii=False).encode()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_vector_cache(path: Path) -> tuple[pa.Array, np.ndarray, dict[str, Any]]:
    """Đọc cache vector; trả về (khóa, ma trận float32, metadata)."""
    if not path.is_file():
        raise FileNotFoundError(f"chưa có cache {path}; chạy --mode prepare")
    table = pq.read_table(path)
    values = table.column("vector").combine_chunks()
    dimension = values.type.list_size
    matrix = np.asarray(values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    raw = table.schema.metadata.get(METADATA_KEY) if table.schema.metadata else None
    metadata = json.loads(raw.decode()) if raw else {}
    return (table.column("key").combine_chunks(),
            matrix.reshape(len(values), dimension), metadata)


# --------------------------------------------------------------------------- fingerprint


def morgan_generator(radius: int, bits: int, use_chirality: bool):
    """Bộ sinh Morgan fingerprint của RDKit.

    Dùng rdFingerprintGenerator (API hiện hành) chứ không dùng AllChem.GetMorganFingerprint
    (đã bị RDKit đánh dấu cũ), để không phụ thuộc vào hàm có thể bị bỏ.
    """
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits, includeChirality=use_chirality)


def pack_fingerprints(smiles: Sequence[str], radius: int, bits: int,
                      use_chirality: bool) -> tuple[np.ndarray, list[int]]:
    """Fingerprint của một danh sách SMILES, đã đóng gói bit.

    Nhận: danh sách SMILES; tham số Morgan.
    Trả về: (ma trận uint8 số dòng × bits/8, danh sách chỉ số SMILES mà RDKit không đọc
    được).
    SMILES không đọc được KHÔNG bị bỏ: dòng tương ứng là toàn số 0 và chỉ số của nó được
    trả về để bên gọi đếm và ghi vào manifest. Bỏ lặng lẽ sẽ làm lệch cache so với dataset.
    Dataset Stage 2b đã kiểm 0 SMILES đọc lỗi, nên danh sách này phải rỗng; nếu không rỗng
    thì có gì đã thay đổi và cần biết.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    generator = morgan_generator(radius, bits, use_chirality)
    matrix = np.zeros((len(smiles), bits // 8), dtype=np.uint8)
    failed: list[int] = []
    for index, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            failed.append(index)
            continue
        matrix[index] = np.packbits(generator.GetFingerprintAsNumPy(molecule))
    return matrix, failed


def unpack_rows(packed: np.ndarray, rows: np.ndarray, bits: int) -> np.ndarray:
    """Giải nén bit của một số dòng thành ma trận float32 0/1.

    Nhận: ma trận đã đóng gói; chỉ số dòng cần lấy; số bit của fingerprint.
    Trả về: ma trận float32 (số dòng × bits).
    Chỉ giải nén đúng các dòng của batch, nên bảng đầy đủ vẫn ở dạng gói trong RAM.
    """
    selected = packed[rows]
    return np.unpackbits(selected, axis=1, count=bits).astype(np.float32)


# --------------------------------------------------------------------------- ESM2


def load_protein_encoder(settings: Mapping[str, Any], device: str = "cpu"):
    """Nạp tokenizer và model ESM2; trả về (tokenizer, model, chiều embedding ĐO ĐƯỢC).

    Nhận: khối config stage3.protein_encoder; thiết bị.
    Trả về: (tokenizer, model, dimension).
    Chiều lấy từ `model.config.hidden_size` của chính model đã nạp — đây là chỗ duy nhất
    trong dự án biết chiều thật, và nó được ghi vào manifest cache để các bước sau kiểm
    lại. Nếu config có `embedding_dim` khác chiều đo được thì báo lỗi, không im lặng dùng
    một trong hai.
    """
    from transformers import AutoTokenizer, EsmModel

    name = settings["model"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name)
    model.to(device)
    dimension = int(model.config.hidden_size)
    declared = settings.get("embedding_dim")
    if declared is not None and int(declared) != dimension:
        raise ValueError(f"config ghi embedding_dim = {declared} nhưng {name} có chiều thật "
                         f"{dimension}; sửa config chứ đừng bỏ qua")
    return tokenizer, model, dimension


def sequence_windows(sequence: str, max_residues: int, strategy: str) -> list[str]:
    """Chia một chuỗi protein thành các đoạn vừa cửa sổ của ESM2.

    Nhận: chuỗi; số acid amin tối đa mỗi đoạn; 'windows' hoặc 'truncate'.
    Trả về: danh sách đoạn.
    'truncate' chỉ giữ đoạn đầu (bắt buộc cho finetune vì bộ nhớ gradient tỉ lệ với độ
    dài); 'windows' giữ toàn bộ chuỗi thành các đoạn liền nhau, không mất phần đuôi.
    """
    if max_residues < 1:
        raise ValueError(f"max_residues phải >= 1, nhận {max_residues}")
    if strategy == "truncate":
        return [sequence[:max_residues]]
    if strategy != "windows":
        raise ValueError(f"long_sequence lạ: {strategy!r}; chỉ có 'windows' hoặc 'truncate'")
    return [sequence[start:start + max_residues]
            for start in range(0, max(len(sequence), 1), max_residues)]


def embed_one_sequence(sequence: str, tokenizer, model, settings: Mapping[str, Any],
                       device: str) -> np.ndarray:
    """Vector của một protein: mean-pool theo acid amin, bỏ token đặc biệt.

    Chuỗi dài hơn cửa sổ được chia đoạn rồi lấy trung bình CÓ TRỌNG SỐ theo độ dài đoạn,
    nên kết quả đúng bằng mean-pool trên toàn chuỗi nếu model không có hiệu ứng biên —
    cách tổng hợp trung thực hơn là lấy trung bình không trọng số của các đoạn dài khác
    nhau.
    """
    import torch

    windows = sequence_windows(sequence, int(settings["max_residues"]),
                               str(settings["long_sequence"]))
    totals = None
    weight_sum = 0.0
    with torch.no_grad():
        for window in windows:
            encoded = tokenizer(window, return_tensors="pt", add_special_tokens=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state[0]
            # Bỏ token đầu (CLS) và token cuối (EOS): chỉ lấy phần ứng với acid amin.
            residues = hidden[1:-1]
            if residues.shape[0] == 0:
                continue
            pooled = residues.mean(dim=0).to("cpu").numpy().astype(np.float64)
            weight = float(residues.shape[0])
            totals = pooled * weight if totals is None else totals + pooled * weight
            weight_sum += weight
    if totals is None or weight_sum == 0.0:
        raise ValueError(f"không lấy được embedding cho chuỗi dài {len(sequence)}")
    return (totals / weight_sum).astype(np.float32)


def embed_proteins(sequences: Mapping[str, str], settings: Mapping[str, Any], device: str,
                   progress_every: int = 500) -> tuple[list[str], np.ndarray, int, int]:
    """Embedding cho nhiều protein.

    Nhận: {hash protein: chuỗi}; config protein_encoder; thiết bị; nhịp ghi log.
    Trả về: (danh sách hash theo thứ tự, ma trận float32, chiều đo được, số chuỗi phải
    chia đoạn hoặc cắt).
    Thứ tự hash được sắp xếp để cache dựng lại hai lần cho cùng một file.
    """
    import logging

    tokenizer, model, dimension = load_protein_encoder(settings, device)
    model.eval()
    keys = sorted(sequences)
    matrix = np.zeros((len(keys), dimension), dtype=np.float32)
    long_sequences = 0
    limit = int(settings["max_residues"])
    for position, key in enumerate(keys):
        sequence = sequences[key]
        if len(sequence) > limit:
            long_sequences += 1
        matrix[position] = embed_one_sequence(sequence, tokenizer, model, settings, device)
        if progress_every and (position + 1) % progress_every == 0:
            logging.info("đã embed %s/%s protein", position + 1, len(keys))
    return keys, matrix, dimension, long_sequences
