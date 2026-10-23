# Báo cáo dataset MOLOPT — Δp theo cặp MMP

Ngày lập: 22/10/2026. Nguồn số liệu: `data/dataset/full/full_manifest.json`,
`data/dataset/export_manifest.json`, `data/dataset/protein_split_manifest.json`,
`runs/observations/verify_20261018_235759.json`, và các kiểm chứng chỉ-đọc chạy
ngày 22/10/2026 (ghi ở mục 9).

Mọi con số trong báo cáo này đều lấy từ một lần chạy đã hoàn tất hoặc từ một
kiểm chứng đã thực sự chạy. Không có số nào là dự đoán.

---

## 1. Trả lời ngắn

Dataset **đã sẵn sàng cho việc train evaluator**.

- File dùng để train: `data/dataset/evaluator_training.parquet` — 10.522.759 dòng.
- Chia theo protein, không rò rỉ: 0 protein nằm ở hai partition.
- Toàn bộ 13.302.366 Δp theo assay đã được kiểm lại: không có cặp nào lai hai protein.
- Không tìm thấy lỗi nào trong code chuyển đơn vị, dựng định danh phân tử, gộp
  replicate, hay gộp trùng chéo nguồn.

Một việc **đã quyết định và đã viết code, chưa chạy**: thêm cột
`source_quality_warning` mang nguyên văn cảnh báo chất lượng của chính ChEMBL
(mục 10.1). Cột này có trong code bước export nhưng **chưa có trong ba file
trên đĩa**, vì các file đó được xuất ngày 22/10/2026 trước khi thêm cột. Cần
chạy lại `python run_dataset.py --mode export` để cột xuất hiện; không cần chạy
lại `--mode full`.

---

## 2. File dataset hiện có

| File | Dòng | Dùng để |
| --- | --- | --- |
| `data/dataset/evaluator_training.parquet` | 10.522.759 | Train evaluator (dự đoán Δp) |
| `data/dataset/generator_training.parquet` | 6.414.841 | Train generator (A yếu sang B mạnh, gain > 0,3) |
| `data/dataset/evidence_records.parquet` | 10.546.330 | Toàn bộ bản ghi, kể cả phần bị cách ly |
| `data/dataset/protein_split.parquet` | 6.315 | protein_hash sang train/validation/test |
| `data/dataset/full/proteins.parquet` | 11.835 | Chuỗi protein, accession, tên |
| `data/dataset/full/molecules.parquet` | 1.306.158 | molecule_key sang SMILES |

Cột của `evaluator_training.parquet`: `pair_id`, `endpoint`, `protein_hash`,
`partition`, `smiles_a`, `smiles_b`, `constant_smiles`, `variable_a`,
`variable_b`, `delta_p_mean`, `n_assays`, `qc_status`, `unseen_pair`,
`plausible_range`, `short_protein`, và `source_quality_warning` sau khi chạy lại
`--mode export` (mục 10.1).

Nhãn cần học là `delta_p_mean` = Δp = p(B) − p(A), có dấu, cùng một protein và
cùng một endpoint.

Phân bố `evaluator_training.parquet` theo partition và endpoint:

| Partition | Ki | Kd | IC50 | EC50 | Tổng |
| --- | --- | --- | --- | --- | --- |
| train | 1.372.512 | 142.152 | 6.328.210 | 574.130 | 8.417.004 |
| validation | 127.312 | 10.450 | 816.104 | 98.925 | 1.052.791 |
| test | 137.852 | 23.153 | 798.249 | 93.710 | 1.052.964 |

---

## 3. Nguồn dữ liệu và xác minh tải về

| Nguồn | Phiên bản | File | Thuật toán | Trạng thái |
| --- | --- | --- | --- | --- |
| ChEMBL | 37 | `chembl_37_sqlite.tar.gz` | SHA-256 | Khớp checksum chính thức |
| ChEMBL | 37 | `chembl_37.sdf.gz` | SHA-256 | Khớp (không đọc trong pipeline) |
| BindingDB | 202610 | `BindingDB_All_202610_tsv.zip` | MD5 | Khớp checksum chính thức |
| BindingDB | 202610 | `BindingDB_Assays_202610_tsv.zip` | MD5 | Khớp |
| BindingDB | 202610 | `BindingDB_rsid_eaids_202610_tsv.zip` | MD5 | Khớp |
| BindingDB | 202610 | `BindingDB_All_2D_202610_sdf.zip` | MD5 | Khớp (không đọc trong pipeline) |

Ngày 22/10/2026 đã băm lại cả 6 file trên đĩa và tải mới checksum chính thức từ
`ftp.ebi.ac.uk` (ChEMBL) và `bindingdb.org` (BindingDB): **6/6 khớp**.

Ghi chú: `raw/OFFICIAL_REVERIFICATION.json` ghi `status: INCOMPLETE_OR_FAILED`
từ ngày 18/10/2026. Lý do duy nhất là lúc đó bindingdb.org trả HTTP 503 nên
không tải lại được 4 file .md5 để so lần hai. Lần tải checksum ngày 22/10/2026
đã lấp đúng chỗ đó; cả 4 giá trị khớp. Trạng thái trong file JSON cũ chưa được
cập nhật lại (file nằm trong `raw/`, không sửa).

---

## 4. Stage 1 — Raw sang Observations

Mỗi phép đo thành một dòng, kèm cột `*_status` và `*_reason` giải thích vì sao
một trường không xác định được. Không dòng nào bị xóa ở bước này.

### 4.1 Chọn dòng

- **ChEMBL**: `SELECT ... FROM activities WHERE standard_type IN ('Ki','Kd','IC50','EC50')`.
  Mọi bảng khác nối bằng `LEFT JOIN` theo khóa chính, nên không mất dòng vì bảng
  bên kia thiếu dữ liệu và không nhân đôi dòng. Kết quả: 5.338.213 dòng
  (IC50 3.623.879, Ki 887.151, EC50 613.608, Kd 213.575).
- **BindingDB**: đọc 4 cột `Ki (nM)`, `Kd (nM)`, `IC50 (nM)`, `EC50 (nM)` từ
  `BindingDB_All.tsv`. Mỗi ô có giá trị thành một dòng, nên một dòng thô có hai
  endpoint sẽ thành hai observation. Kết quả: 3.246.509 dòng
  (IC50 2.220.490, Ki 623.701, EC50 270.787, Kd 131.531).

### 4.2 Chuyển giá trị thành pActivity

- Đơn vị tra trong một bảng khai báo sẵn: M 1, mM 1e-3, uM/µM/μM 1e-6,
  nM 1e-9, pM 1e-12, fM 1e-15. Đơn vị không có trong bảng thì dòng đó
  `activity_status = unresolved`, lý do `unit_not_in_table`. **Không đoán đơn vị
  ở bất kỳ chỗ nào** (ví dụ ChEMBL có 92.929 dòng `ug.mL-1` bị để lại unresolved
  vì đổi sang mol/L cần khối lượng phân tử).
- `p_activity = -log10(nồng độ mol/L)`.
- Qualifier: chỉ `=` là exact. Dòng bị chặn (`<`, `>`, `<=`, `>=`, `<<`, `>>`)
  vẫn giữ nồng độ nhưng **không** có pActivity, nên không thể lọt vào Δp.
  ChEMBL có cột qualifier riêng (`standard_relation`); BindingDB không có nên
  qualifier được tách từ tiền tố của ô giá trị, không có tiền tố thì hiểu là `=`.

### 4.3 Chọn chuỗi protein

- **ChEMBL**: chỉ nhận target có **đúng 1** protein component. Nếu assay khai
  một biến thể (`variant_id`) thì dùng chuỗi của chính biến thể đó
  (`variant_sequences`, 45.752 dòng), còn lại dùng `component_sequences`
  (3.232.432 dòng). Target nhiều component hoặc không có protein component thì
  unresolved: 1.823.185 dòng không có protein component, 231.297 dòng có nhiều
  hơn 1 component, 5.097 dòng khai biến thể nhưng nguồn thiếu chuỗi, và 450 dòng
  thiếu chuỗi của component.
- **BindingDB**: chỉ nhận mục tiêu có **đúng 1** chuỗi protein
  (`Number of Protein Chains in Target = 1`), 3.059.262 dòng. Phức nhiều chuỗi
  thì unresolved (187.247 dòng). Tức là không có dòng nào gán phép đo trên phức
  nhiều chuỗi cho chuỗi số 1.
- Định danh protein là `protein_sequence_hash` = SHA-256 của chuỗi sau khi bỏ
  khoảng trắng và viết hoa. Dùng chuỗi chứ không dùng accession vì hai construct
  khác nhau của cùng một gene là hai mục tiêu đo khác nhau.

### 4.4 Chuẩn hóa cấu trúc phân tử

SMILES nguồn qua RDKit thành canonical SMILES (giữ lập thể) và InChIKey. **Định
danh phân tử luôn do RDKit tính lại, không dùng InChIKey của nguồn** (lý do ở
mục 9.3). ChEMBL 5.318.702 dòng resolved, BindingDB 3.240.453 dòng resolved.

Cờ lập thể tương đối: dòng có khối CXSMILES chứa cờ `r` (mọi tâm lập thể là
tương đối) được gắn hậu tố `-REL` vào khóa phân tử, nên không bao giờ ghép cặp
với dòng không mang cờ.

### 4.5 Khớp assay

- ChEMBL: `assays.assay_id` — 5.338.213 dòng đều resolved.
- BindingDB: (entry_id, assay_id) lấy từ `BindingDB_rsid_eaids`. 58.159 dòng
  không có liên kết nên unresolved (`no_rsid_assay_link`).

---

## 5. Stage 2a — Gộp trùng chéo nguồn

Cùng một phép đo có thể xuất hiện ở cả hai nguồn. Hai dòng chỉ bị coi là trùng
khi **đồng thời**: cùng tài liệu (PMID, DOI hoặc số patent đã chuẩn hóa), cùng
InChIKey, cùng endpoint, cùng qualifier, nồng độ trùng khít, và protein khớp
theo chuỗi hoặc accession. Thiếu bất kỳ điều kiện nào thì ghi `uncertain` và
**giữ lại cả hai dòng** — không xóa gì theo phỏng đoán.

Kết quả: 2.074.718 nhóm trùng. Giữ bản ChEMBL, bỏ bản BindingDB
(2.127.500 dòng BindingDB có `dedup_keep = False`). 1.000.717 dòng được gắn
`uncertain` và vẫn giữ.

Trùng **trong cùng một nguồn** thì Stage 2a không xử lý. Chúng được giữ nguyên
và trở thành replicate ở bước 1 của Stage 2b.

Độ nhiễu thực nghiệm đo được từ chính dữ liệu này (σ cho một phép đo, đơn vị log):
**Ki 0,50, Kd 0,57, IC50 và EC50 0,73**. Xem mục 10 để so với tài liệu.

---

## 6. Stage 2b — Observations sang cặp MMP và Δp

### 6.1 Bộ lọc vào

Giữ dòng thỏa **tất cả**: `dedup_keep = True`, qualifier `=`,
`protein_status = resolved`, `assay_status = resolved`, có pActivity, có
InChIKey, có SMILES.

| Bước lọc | Ki | Kd | IC50 | EC50 |
| --- | --- | --- | --- | --- |
| tất cả | 1.510.852 | 345.106 | 5.844.369 | 884.395 |
| dedup_keep | 1.091.930 | 251.794 | 4.390.860 | 722.638 |
| qualifier `=` | 789.416 | 125.593 | 3.251.805 | 518.819 |
| protein resolved | 685.797 | 106.180 | 1.965.052 | 274.778 |
| assay resolved | 663.219 | 106.120 | 1.945.221 | 273.723 |
| có pActivity | 662.356 | 105.368 | 1.939.206 | 272.996 |
| có InChIKey và SMILES | 659.231 | 105.212 | 1.935.645 | 272.419 |

Tổng vào: **2.972.507 dòng**.

Dòng ChEMBL có cờ `potential_duplicate = 1` **vẫn được giữ**, nhưng không tính
là phép đo độc lập và không tính là assay độc lập.

### 6.2 Năm bước

1. **Gộp replicate** trong cùng `(assay, protein, phân tử)`: 2.972.507 dòng
   thành **2.876.220 nhóm**. Các pActivity lệch nhau không quá 0,01 được coi là
   một giá trị rồi lấy trung bình. 3.555 nhóm có nhiều `molecule_source_id`
   khác nhau (cùng InChIKey, khác ID nguồn) được giữ và gắn cờ `multi_source_id`.
2. **Tìm cặp MMP** trong từng `(assay, protein)`: RDKit `rdMMPA.FragmentMol` với
   một lần cắt (`minCuts = maxCuts = 1`, `maxCutBonds = 20`). Giữ cặp khi lõi
   chung chiếm ít nhất 60 % số nguyên tử nặng của **cả hai** phía, phần biến đổi
   không quá 10 nguyên tử nặng, và đúng một điểm gắn bậc 1. 2.775.152 phân tử đã
   phân mảnh, 0 SMILES đọc lỗi.
3. **Δp trong từng assay**: Δp = p(B) − p(A), có dấu, A là phân tử có khóa nhỏ
   hơn để chiều cặp luôn xác định. Kết quả **13.302.366 dòng**. 164.156 cặp bị
   loại vì một phía mang cờ `-REL` và phía kia không (1,2 %).
4. **Gộp theo bản ghi** `(A, B, protein, endpoint)`: lấy trung bình các Δp theo
   assay, được **10.546.330 bản ghi**; 1.306.158 phân tử, 11.835 protein.
5. **Xuất** ba file ở mục 2.

### 6.3 QC của bước 4

| qc_status | Số bản ghi | Nghĩa |
| --- | --- | --- |
| `pass` | 1.836.651 | Từ 2 assay trở lên và độ phân tán dưới ngưỡng |
| `single_assay` | 8.686.108 | Chỉ có 1 assay, không đo được độ phân tán |
| `quarantine` | 23.571 | Từ 2 assay trở lên nhưng độ phân tán vượt ngưỡng |

Cách đo phân tán: 2 assay thì dùng |ΔX − ΔY|; từ 3 assay trở lên dùng độ lệch
chuẩn mẫu (ddof = 1). Ngưỡng: Ki và Kd 1,0; IC50 và EC50 1,4.

1.860.222 bản ghi (17,6 %) có từ 2 assay trở lên. Trong số đó, 86,4 % có phân
tán không quá 0,3 và 2,6 % có phân tán trên 1,0.

`evaluator_training.parquet` lấy `pass` và `single_assay`, tức bỏ đúng 23.571
bản ghi `quarantine`; `evidence_records.parquet` giữ cả ba.

---

## 7. Chia train / validation / test

Chia **theo protein**, không theo dòng, để model không được học rồi kiểm tra
trên cùng một protein.

1. 6.315 protein có ít nhất một cặp, ghi ra FASTA (`protein_split_input.fasta`).
2. Gom bằng MMseqs2 15-6f452+ds-2 (WSL Ubuntu 24.04):
   `mmseqs easy-cluster <fasta> <prefix> <tmp> --min-seq-id 0.5 -c 0.8
   --cov-mode 2 --cluster-mode 1 --single-step-clustering --max-seqs 6315
   --alignment-mode 3 --seq-id-mode 0 -s 7.5`, được **2.782 cụm**.
3. Rải cụm vào ba partition theo số cặp: **80,0 % / 10,0 % / 10,0 %** (lệch 0,00).

Chia được đúng tỉ lệ vì cụm lớn nhất chỉ chiếm 4,55 % số cặp (10 cụm lớn nhất
chiếm 17,89 %); có 1.722 cụm chỉ một protein. Train giữ 80 % số cặp nhưng chỉ
49 % số protein, vì các cụm lớn mang nhiều cặp.

Lý do dùng `-c 0.8` (ràng buộc coverage): nếu chỉ xét identity, một construct
ngắn và protein đầy đủ của cùng một gene có thể rơi vào hai partition khác nhau,
với dữ liệu theo construct của dự án này thì đó là rò rỉ thật. Chi tiết và lịch
sử đổi tham số ở `docs/STAGE2B_DATASET.md` mục 8.1.

---

## 8. Ba cờ mô tả (gắn cờ, không lọc)

Nguyên tắc của dự án: không lọc thầm. Ba cờ dưới đây có ở cả ba file xuất và
không loại dòng nào.

| Cờ | Số dòng True | Nghĩa |
| --- | --- | --- |
| `plausible_range` | 10.531.299 trong evidence (False: 15.031) | Mọi phép đo đóng góp có 2 ≤ p ≤ 12 |
| `unseen_pair` | 1.711.573 | Cặp (A, B) không xuất hiện ở partition train |
| `short_protein` | 1.638 | Chuỗi protein ngắn hơn 50 acid amin (21 chuỗi, ngắn nhất 9 aa) |

`unseen_pair` luôn False ở train. Ở validation là 894.182 dòng (84,8 %), ở test
là 817.391 dòng (77,5 %). Cờ này cần vì split chia theo protein, nên cùng một
cặp phân tử có thể gặp lại ở một protein khác; muốn đánh giá nghiêm ngặt thì lọc
`unseen_pair = True`.

Cột thứ tư, `source_quality_warning`, cũng không lọc gì và được mô tả ở mục 10.1.

---

## 9. Kiểm chứng đã chạy

### 9.1 Stage 1 — bảo toàn và bất biến

| Kiểm chứng | Kết quả |
| --- | --- |
| ChEMBL: đếm độc lập từ SQLite so với số dòng observations | 5.338.213 bằng 5.338.213, khớp từng endpoint |
| BindingDB: đếm độc lập ô endpoint từ file zip so với observations | 3.246.509 bằng 3.246.509, khớp từng endpoint |
| BindingDB: phân bố qualifier đếm lại từ file thô | 2.591.987 `=`, 427.486 `>`, 227.036 `<` — khớp tuyệt đối |
| `observation_id` trùng lặp | 0 trên cả hai file |
| Số dòng `data/observations` so với `data/dedup` | Bằng nhau; dedup chỉ thêm 5 cột |
| `p = -log10(nồng độ)` tính lại trên mọi dòng có pActivity | 6.303.070 dòng, **0 lệch** |
| Dòng bị chặn (censored) mang pActivity | 0 |
| Dòng exact thiếu pActivity | 0 |
| `protein_sequence_hash` băm lại từ chuỗi đã lưu | **0 lệch** (sau chuẩn hóa viết hoa đúng như docstring) |
| Dòng protein resolved mà thiếu chuỗi | 0 |
| Dòng structure resolved mà thiếu InChIKey hoặc SMILES | 0 |
| Dòng BindingDB đa chuỗi bị gán sang chuỗi 1 | 0 — mọi dòng resolved đều có chain_count = 1 |

Ngoài ra `runs/observations/verify_20261018_235759.json` (chạy 18/10/2026) ghi:
số dòng chọn từ raw bằng số dòng ghi ra ở cả hai nguồn, 0 `observation_id`
trùng, đối chiếu lại 5.000 dòng với nguồn cho 0 lệch, và hai bài kiểm tra hồi
quy (construct P03367 với hai độ dài 99 aa và 1.447 aa; lập thể enhanced của
monomer 13590) đều đạt.

### 9.2 Stage 1 — cấu trúc phân tử

Tính lại bằng RDKit từ SMILES nguồn trên 40.000 dòng lấy mẫu (20.000 mỗi nguồn,
seed 20261022):

| Kiểm chứng | Kết quả |
| --- | --- |
| Canonical SMILES tính lại so với đã lưu | **0 lệch** |
| InChIKey tính lại so với đã lưu | **0 lệch** |
| Canonical SMILES đã lưu có phải điểm bất động của RDKit | **0 dòng không phải** |

### 9.3 Vì sao không dùng InChIKey của nguồn

So InChIKey của nguồn với InChIKey do RDKit tính từ chính SMILES của nguồn đó,
trên toàn bộ hai file:

| Nguồn | Số dòng lệch | Trong đó lệch cả phần khung phân tử (14 ký tự đầu) |
| --- | --- | --- |
| ChEMBL 37 | 5.751 | 42 |
| BindingDB 202610 | 1.222.863 (37,7 %) | 5.960 |

Phần lớn chênh lệch của BindingDB chỉ ở khối lập thể hoặc proton. Nhưng 5.960
dòng lệch ở cả phần khung, tức **chính nguồn tự mâu thuẫn giữa cột SMILES và
cột InChIKey của nó**. Đã loại giả thuyết muối hoặc ion đối: 0 trong 5.960 dòng
đó có SMILES nhiều mảnh.

Hệ quả cho dataset: không có. `molecule_source_inchikey` chỉ được lưu làm dấu
vết nguồn và **không được dùng ở bất kỳ chỗ nào ngoài Stage 1** (đã kiểm bằng
tìm toàn bộ mã nguồn). Định danh phân tử ở mọi bước là InChIKey do RDKit tính
từ SMILES, nên mâu thuẫn nội bộ của nguồn không lan vào việc ghép cặp.

### 9.4 Stage 2a — quy tắc gộp trùng

| Kiểm chứng | Kết quả |
| --- | --- |
| Nhóm trùng không có dòng ChEMBL | 0 |
| Nhóm trùng không có dòng BindingDB | 0 |
| Nhóm có dòng ChEMBL bị bỏ | 0 |
| Nhóm có dòng BindingDB được giữ | 0 |
| Nhóm có nhiều hơn một dòng ChEMBL | 48.253 — đúng thiết kế: Stage 2a chỉ bỏ trùng **chéo nguồn** |

### 9.5 Stage 2b — không có cặp lai protein

BindingDB có 3.850 khóa assay (1,8 %) trải trên nhiều chuỗi protein (ChEMBL: 0).
Nếu Δp gộp theo assay mà không kèm protein thì đây sẽ là lỗi thật. Mã nguồn gộp
theo `(assay_key, protein_hash)`, và đã kiểm lại trên dữ liệu đã sinh:

| Kiểm chứng | Kết quả |
| --- | --- |
| Dòng Δp mà phía A không thuộc đúng nhóm (assay, protein, phân tử) | **0 / 13.302.366** |
| Dòng Δp mà phía B không thuộc đúng nhóm | **0 / 13.302.366** |

### 9.6 Stage 2b — các kiểm chứng đã chạy trước đó

Giữ lại ở đây cho đầy đủ; chi tiết ở `docs/STAGE2B_DATASET.md`.

- `run_dataset.py --mode verify` với mẫu 50.000 dòng Δp: dựng lại 90.336 nhóm từ
  observations gốc, 0 lệch; **7/7 PASS**.
- Tính xác định: hai lần chạy `--mode smoke` độc lập cho **322/322 file dữ liệu
  giống nhau theo SHA-256**; chỉ `full_manifest.json` khác ở thời gian và đường dẫn.
- 278 test tự động đạt.
- `pair_id` không trùng; 0 cặp tự ghép với chính nó; 0 cặp sai chiều; 0 cặp có
  SMILES hoặc phần biến đổi giống nhau; 0 lõi rỗng.
- 0 cặp lẫn `-REL` với không-`-REL`; 0 null ở 14 cột khóa; 0 protein ở hai
  partition; một `molecule_key` ứng đúng một SMILES (973.003 khóa).
- `evaluator` là tập con của `evidence`, 0 trùng, loại đúng 23.571 bản ghi
  `quarantine`; `generator` so với `evaluator` cho 0 lỗi trên 6.414.841 dòng.
- Lắp lại phân tử từ lõi và phần biến đổi bằng RDKit trên 2.880 bản ghi: 0 lệch.
  Lưu ý cho người dùng sau: `Chem.molzip` trả về nhãn lập thể chưa chuẩn hóa,
  phải gọi `Chem.AssignStereochemistry(mol, cleanIt=True, force=True)` sau khi lắp.
- Tính lại độc lập ba cờ, khớp đến từng đơn vị.

---

## 10. Ngưỡng, lý do và tài liệu tham khảo

| Ngưỡng trong dự án | Giá trị | Căn cứ |
| --- | --- | --- |
| Gộp replicate trong một assay | lệch pActivity không quá 0,01 | Quy ước nội bộ: 0,01 log nhỏ hơn hai bậc so với σ thực nghiệm (0,5 đến 0,7), nên chỉ gộp các giá trị khác nhau do làm tròn. Ghi ở `docs/STAGE2A_DEDUP_AUDIT.md` mục 5.6 |
| Cách tổng hợp replicate và Δp nhiều assay | Trung bình trên thang log (p), không trên thang nồng độ | pActivity là thang log; Ki và IC50 phân bố lệch nên trung bình trên thang log là cách tổng hợp chuẩn — [1], [2] |
| σ thực nghiệm cho một phép đo | Ki 0,50, Kd 0,57, IC50 và EC50 0,73 (đo từ chính dữ liệu dự án) | Khớp với tài liệu: σ(pKi) = 0,54 trên ChEMBL [1]; σ(pIC50) = 0,68, MUE 0,55, MedUE 0,43, tức nhiễu IC50 lớn hơn Ki khoảng 25 % [2] |
| Ngưỡng cách ly bản ghi phân tán | Ki và Kd 1,0; IC50 và EC50 1,4 | Khoảng 2σ của chính endpoint đó (2 nhân 0,50 bằng 1,0; 2 nhân 0,73 xấp xỉ 1,4). Chênh lệch dưới 2σ không phân biệt được với nhiễu thực nghiệm [1], [2] |
| Δp tính **trong cùng một assay**, không trộn assay | bắt buộc | [3] cho thấy gộp IC50 hoặc Ki từ các assay và nguồn khác nhau tạo nhiễu lớn: với mức curation tối thiểu, khoảng 65 % cặp lệch trên 0,3 log và 27 % lệch trên 1 log; siết metadata assay thì còn 48 % và 13 %. Vì vậy Δp chỉ tính trong một assay, rồi mới lấy trung bình giữa các assay và báo độ phân tán |
| Cặp MMP một lần cắt | `minCuts = maxCuts = 1`, `maxCutBonds = 20` | Thuật toán phân mảnh và chỉ mục của [4]; RDKit `rdMMPA` hiện thực thuật toán này. Cách dùng MMP cho dữ liệu nhiều thuộc tính theo [5], [6] |
| Lõi chung từ 60 % nguyên tử nặng, phần biến đổi không quá 10 nguyên tử nặng | ngưỡng dự án | Giữ nghĩa "một thay đổi nhỏ trên cùng một khung" của MMP; [5] dùng giới hạn tương tự để cặp còn ý nghĩa hóa dược |
| Chỉ dùng giá trị qualifier `=` | bắt buộc | Giá trị bị chặn (`>`, `<`) không cho biết hoạt tính thật, nên không thể trừ nhau thành Δp |
| Định danh phân tử tính lại bằng RDKit | bắt buộc | [7]: lỗi cấu trúc trong dữ liệu công khai lan xuống descriptor và làm giảm chất lượng model, nên phải tự chuẩn hóa thay vì tin định danh của nguồn. Số đo của dự án ở mục 9.3 |
| Chia theo protein, gom ở identity 50 % | `--min-seq-id 0.5` | Tiền lệ: LP-PDBBind giới hạn tối đa 50 % identity giữa train và các split khác [8]. Công cụ gom: MMseqs2 [9] |
| Thêm ràng buộc coverage 0,8 | `-c 0.8 --cov-mode 2` | Điểm cố ý khác tiền lệ. Không có coverage thì construct ngắn và protein đầy đủ của cùng một gene rơi vào hai partition — rò rỉ thật với dữ liệu theo construct. Lý do đầy đủ ở `docs/STAGE2B_DATASET.md` mục 8.1 |
| Khoảng pActivity hợp lý 2 ≤ p ≤ 12 (cờ, không lọc) | cờ `plausible_range` | p nhỏ hơn 0 nghĩa là nồng độ trên 1 M, bất khả thi về mặt vật lý. Khoảng 2 đến 12 tương ứng 10 mM đến 1 pM, bao trọn vùng đo được của các assay sinh hóa. Xem mục 10.1 về sự trùng khớp với cảnh báo của chính ChEMBL |
| Nguồn dữ liệu | ChEMBL 37, BindingDB 202610 | [10], [11] |

### 10.1 Cột `source_quality_warning`: cảnh báo chất lượng của chính ChEMBL

ChEMBL có cột `data_validity_comment` để tự đánh dấu dòng đáng nghi. Trước ngày
22/10/2026 pipeline lưu cột này trong `activity_context` nhưng không đưa ra file
xuất.

Trong 2.272.131 dòng ChEMBL đã qua bộ lọc của Stage 2b:

| `data_validity_comment` | Số dòng | Tỉ lệ |
| --- | --- | --- |
| (không có) | 2.219.814 | 97,697 % |
| Outside typical range | 48.258 | 2,124 % |
| Potential transcription error | 3.633 | 0,160 % |
| Manually validated | 372 | 0,016 % |
| Potential author error | 54 | 0,002 % |

Tổng số dòng mang cảnh báo đáng nghi: 51.945 (ba nhóm đầu tiên không kể
"Manually validated", vì nhãn đó là dấu hiệu tốt chứ không phải cảnh báo).

Phát hiện quan trọng: trong 5.073 dòng ChEMBL có pActivity ngoài khoảng 2 đến
12, **không một dòng nào thuộc nhóm "không có cảnh báo"** — tất cả đều đã được
chính ChEMBL gắn cảnh báo. Hai dấu hiệu độc lập (cờ `plausible_range` của dự án
và cột curation của ChEMBL) chỉ về cùng một tập dòng.

Quyết định ngày 22/10/2026: thêm cột `source_quality_warning` ở bước xuất, lấy
**nguyên văn** `data_validity_comment`, **không lọc dòng nào**, đúng nguyên tắc
hiện hành.

- Null nghĩa là **không có cảnh báo**, hoặc nguồn không có cột tương đương —
  không phải "chưa biết". BindingDB 202610 không có cột này nên bản ghi chỉ từ
  BindingDB luôn null.
- Một bản ghi gộp nhiều phép đo, nên giá trị là các cảnh báo khác nhau của mọi
  phép đo đóng góp, bỏ trùng, sắp xếp rồi nối bằng `"; "`. Sắp xếp để giá trị
  không phụ thuộc thứ tự dòng.
- Bước export đọc lại hai file nguồn một lượt để lấy cảnh báo, nên **không phải
  chạy lại `--mode full`**. Chỉ parse JSON ở dòng mà regex đã thấy khóa có giá
  trị chuỗi.
- `export_manifest.json` ghi thêm số observation có cảnh báo, số bản ghi mang
  cảnh báo, và số bản ghi theo từng giá trị.

**Đã chạy ngày 22/10/2026** (`--mode export`, 278 test pass). Số dòng ba file
không đổi, đúng như mong đợi vì cột này không lọc gì. Đo được:

| | |
| --- | --- |
| Observation mang cảnh báo | 237.660 — ChEMBL 237.660, BindingDB 0 |
| Bản ghi mang cảnh báo | 146.464 trong evidence (1,39 %), 142.331 trong evaluator, 94.142 trong generator |

| Giá trị trong evidence | Số bản ghi |
| --- | --- |
| `Outside typical range` | 118.357 |
| `Potential transcription error` | 15.630 |
| `Outside typical range; Potential transcription error` | 10.063 |
| `Manually validated` | 2.074 |
| `Potential author error` | 291 |
| `Manually validated; Outside typical range` | 49 |

`Potential missing data` có 553 observation trong ChEMBL nhưng **0 bản ghi**: các
dòng đó không dòng nào qua được bộ lọc của Stage 2b, khớp với số đo ở mục 10.1.

**Audit cột này** (chạy 22/10/2026, dựng lại bằng đường đi độc lập: parse JSON
mọi dòng thay vì chọn trước bằng regex, và gom bằng vòng lặp Python thay vì theo
offset):

| Kiểm chứng | evidence | evaluator | generator |
| --- | --- | --- | --- |
| Số bản ghi mang cảnh báo, tính độc lập | 146.464 (khớp) | 142.331 (khớp) | 94.142 (khớp) |
| Null mà đáng ra phải có cảnh báo | 0 | 0 | 0 |
| Có cảnh báo mà không được phép | 0 | 0 | 0 |
| Giá trị sai | 0 | 0 | 0 |

**Đối chiếu chéo với cờ `plausible_range`.** Trong 15.031 bản ghi có
`plausible_range = False`: 11.928 bản ghi (79,4 %) mang cảnh báo của nguồn, còn
3.103 bản ghi (20,6 %) thì không. Đã truy nguồn 3.103 bản ghi đó: **toàn bộ
3.113 phép đo thực sự ngoài khoảng của chúng đến từ BindingDB, 0 từ ChEMBL** —
và BindingDB không có cột chất lượng tương đương. Nói cách khác, hai dấu hiệu
không hề mâu thuẫn: giá trị ChEMBL bất khả thi thì luôn đã bị ChEMBL gắn cảnh
báo, và phần còn lại là giá trị BindingDB mà không nguồn nào gắn cảnh báo được.

---

## 11. Hạn chế đã biết

Tất cả đều đã đo, đã gắn cờ hoặc đã ghi lại; không có cái nào bị lọc thầm.

1. **Giá trị bất khả thi từ nguồn.** ChEMBL có 2.076 dòng p nhỏ hơn 0 và 46 dòng
   p lớn hơn 15; BindingDB có 178 dòng và 1 dòng tương ứng. Đã truy về từng dòng
   gốc: **tất cả đều có đơn vị `nM` và giá trị nguồn tự nó vô lý** (ví dụ ChEMBL
   ghi `1e+28` nM cho một Ki và `1.427e-13` nM cho một IC50; BindingDB ghi
   `4527050000000` nM cho một IC50). Code chuyển đơn vị đúng; lỗi ở nguồn.
   Pipeline chuyển tiếp nguyên trạng và gắn cờ `plausible_range`.
2. **Trùng hóa học giữa các partition.** Split chia theo protein nên cùng một
   cặp phân tử có thể xuất hiện ở hai partition khác nhau. Đã gắn cờ
   `unseen_pair` để định nghĩa tập đánh giá nghiêm ngặt.
3. **Dữ liệu tập trung vào một ít protein.** 100 protein nhiều cặp nhất chiếm
   44,7 % tổng số cặp. Chỉ báo cáo, không can thiệp.
4. **21 chuỗi protein ngắn hơn 50 aa** (ngắn nhất 9 aa). Gom theo identity 50 %
   với coverage 0,8 ít có nghĩa với chuỗi ngắn như vậy. Đã gắn cờ
   `short_protein` (1.638 dòng).
5. **`single_assay` chiếm 82,4 % bản ghi.** Những bản ghi này chỉ có một assay
   nên không đo được độ phân tán; chúng vẫn vào tập train và được đánh dấu bằng
   `qc_status = single_assay`.
6. **Dòng `pKi` và `pIC50` của ChEMBL không được dùng.** Pipeline chỉ lấy
   `standard_type` thuộc Ki/Kd/IC50/EC50. Đã đo mức mất mát: chỉ có **3.849**
   tổ hợp (assay, phân tử) tồn tại duy nhất dưới dạng tiền tố `p`, nên phần
   không phủ là không đáng kể.
7. **Target ChEMBL không phải "SINGLE PROTEIN".** Có 47 chuỗi protein chỉ xuất
   hiện dưới các nhãn target khác (PROTEIN COMPLEX, CHIMERIC PROTEIN, ORGANISM
   và tương tự) dù target chỉ có 1 component. Ảnh hưởng xuống dataset cuối:
   **1 dòng trong 10.522.759**.
8. **Mâu thuẫn nội bộ của BindingDB** giữa cột SMILES và cột InChIKey ở 5.960
   dòng (mục 9.3). Không ảnh hưởng vì định danh luôn tính lại từ SMILES.
9. **BindingDB mất 58.159 dòng** vì không có liên kết assay trong
   `BindingDB_rsid_eaids`, nên không xác định được "cùng một assay".

---

## 12. Tài liệu tham khảo

[1] Kramer, C.; Kalliokoski, T.; Gedeck, P.; Vulpetti, A. *The Experimental
Uncertainty of Heterogeneous Public Ki Data.* J. Med. Chem. **2012**, 55 (11),
5165–5173. DOI 10.1021/jm300131x.

[2] Kalliokoski, T.; Kramer, C.; Vulpetti, A.; Gedeck, P. *Comparability of
Mixed IC50 Data – A Statistical Analysis.* PLoS ONE **2013**, 8 (4), e61007.
DOI 10.1371/journal.pone.0061007. (σ(pIC50) = 0,68; MUE = 0,55; MedUE = 0,43;
nhiễu pIC50 lớn hơn pKi khoảng 25 %; σ cho một phép đo lấy bằng σ của cặp chia
căn hai.)

[3] Landrum, G. A.; Riniker, S. *Combining IC50 or Ki Values from Different
Sources Is a Source of Significant Noise.* J. Chem. Inf. Model. **2024**,
64 (5), 1560–1567. DOI 10.1021/acs.jcim.4c00049.

[4] Hussain, J.; Rea, C. *Computationally Efficient Algorithm to Identify
Matched Molecular Pairs (MMPs) in Large Data Sets.* J. Chem. Inf. Model.
**2010**, 50 (3), 339–348. DOI 10.1021/ci900450m.

[5] Griffen, E.; Leach, A. G.; Robb, G. R.; Warner, D. J. *Matched Molecular
Pairs as a Medicinal Chemistry Tool.* J. Med. Chem. **2011**, 54 (22),
7739–7750. DOI 10.1021/jm200452d.

[6] Dalke, A.; Hert, J.; Kramer, C. *mmpdb: An Open-Source Matched Molecular
Pair Platform for Large Multiproperty Data Sets.* J. Chem. Inf. Model. **2018**,
58 (5), 902–910. DOI 10.1021/acs.jcim.8b00173.

[7] Fourches, D.; Muratov, E.; Tropsha, A. *Trust, But Verify: On the Importance
of Chemical Structure Curation in Cheminformatics and QSAR Modeling Research.*
J. Chem. Inf. Model. **2010**, 50 (7), 1189–1204. DOI 10.1021/ci100176x.

[8] Li, J.; Guan, X.; Zhang, O.; Sun, K.; Wang, Y.; Bagni, D.; Head-Gordon, T.
*Leak Proof PDBBind: A Reorganized Dataset of Protein-Ligand Complexes for More
Generalizable Binding Affinity Prediction.* arXiv:2308.09639. (Tiền bản; giới
hạn tối đa 50 % identity chuỗi protein giữa train và các split khác.)

[9] Steinegger, M.; Söding, J. *MMseqs2 Enables Sensitive Protein Sequence
Searching for the Analysis of Massive Data Sets.* Nat. Biotechnol. **2017**.
DOI 10.1038/nbt.3988.

[10] Zdrazil, B.; et al. *The ChEMBL Database in 2023: A Drug Discovery Platform
Spanning Multiple Bioactivity Data Types and Time Periods.* Nucleic Acids Res.
**2024**, 52 (D1), D1180–D1192. DOI 10.1093/nar/gkad1004.

[11] Gilson, M. K.; Liu, T.; Baitaluk, M.; Nicola, G.; Hwang, L.; Chong, J.
*BindingDB in 2015: A Public Database for Medicinal Chemistry, Computational
Chemistry and Systems Pharmacology.* Nucleic Acids Res. **2016**, 44 (D1),
D1045–D1053. DOI 10.1093/nar/gkv1072.

Ghi chú về cách dẫn: các số liệu định lượng ở [2] được đọc trực tiếp từ bản
toàn văn mở của bài báo ngày 22/10/2026. Số σ(pKi) = 0,54 của [1] được dẫn lại
qua [2], vì bài [1] nằm sau tường phí. Số tập, số trang và DOI của [1], [4],
[5], [6], [7], [10], [11] lấy từ hồ sơ nhà xuất bản hoặc PubMed Central. Với
[9] chưa xác nhận được số tập và số trang nên chỉ ghi DOI. [8] là tiền bản
arXiv, chưa qua phản biện ở thời điểm lập báo cáo.
