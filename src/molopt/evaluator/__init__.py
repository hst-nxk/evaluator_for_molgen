"""Stage 3: evaluator baseline — dự đoán Δp từ cặp phân tử, protein và endpoint.

Các module:
    data      : đọc dataset, lọc theo partition và theo lát cắt, đóng gói batch
    features  : Morgan fingerprint và embedding ESM2, đều có cache ra đĩa
    models    : ligand encoder dùng chung, edit encoder, ba đầu ra (direct,
                direct_no_edit, twin)
    metrics   : regression, ranking trong từng query, số đo bất đối xứng
    train     : dựng cache, baseline tầm thường, vòng train, validation, checkpoint

Chỉ `models` import torch ở cấp module. `metrics`, `data`, `features` và `train` thì
không: torch chỉ được import trong thân những hàm cần nó, nên `--mode baseline` và
`--mode prepare --skip-protein` chạy được trên máy chưa cài torch, và phần lớn test
cũng chạy được ở đó.
"""
