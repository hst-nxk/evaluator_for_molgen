# MOLOPT: dataset Δp theo cặp MMP và evaluator

## Đọc theo thứ tự nào

| # | Muốn biết | Đọc | Thời gian |
|---|---|---|---|
| 1 | Toàn cảnh: dữ liệu từ đâu, qua những bước nào, kiểm chứng ra sao | [`docs/DATASET_REPORT.md`](docs/DATASET_REPORT.md), mục 1–9 | 20 phút |
| 2 | **Ngưỡng nào, vì sao, dựa trên tài liệu nào** | [`docs/DATASET_REPORT.md`](docs/DATASET_REPORT.md), mục 10 (bảng ngưỡng) và mục 12 (11 tài liệu tham khảo có DOI); hạn chế đã biết ở mục 11 | 15 phút |
| 3 | Audit ngày 24/10/2026: tìm ra gì, sửa gì | [`docs/AUDIT_20261024.md`](docs/AUDIT_20261024.md), mục 0 (tóm tắt) và mục 6 (quyết định) | 10 phút |
| 4 | **Dataset hiện dùng (v2)**: thay đổi so với v1 và số đo thật | [`docs/STAGE2B_DATASET.md`](docs/STAGE2B_DATASET.md), mục 0 (bảng v1/v2 ở 0.4) | 10 phút |
| 5 | **Kiến trúc evaluator dự kiến** | [`docs/STAGE3_EVALUATOR.md`](docs/STAGE3_EVALUATOR.md), mục 1–6 (phạm vi, đầu vào, kiến trúc, chỉ số, ba cấu hình, ablation ba đầu ra) | 20 phút |

Chi tiết kỹ thuật của từng giai đoạn, khi cần tra:
- Stage 1, Raw → Observations: [`docs/OBSERVATIONS_PIPELINE.md`](docs/OBSERVATIONS_PIPELINE.md)
- Stage 2a, gộp trùng giữa hai nguồn và đo nhiễu: [`docs/STAGE2A_DEDUP_AUDIT.md`](docs/STAGE2A_DEDUP_AUDIT.md)
- Stage 2b, cặp MMP và Δp: [`docs/STAGE2B_DATASET.md`](docs/STAGE2B_DATASET.md)
- Mọi tham số đều nằm trong `config.json`, mỗi khối có trường `note` ghi lý do.

**Lưu ý phiên bản.** `DATASET_REPORT.md` từ mục 1 trở đi mô tả dataset **v1** (21–22/10/2026).
Dataset đang dùng là **v2** (24/10/2026): 10.712.915 bản ghi, verify 7/7 PASS. Số đo của v2
nằm ở `STAGE2B_DATASET.md` mục 0.4. Dữ liệu không nằm trong repo (`data/` bị
`.gitignore`); repo chỉ có code, config và tài liệu.

## Stage 1: Raw → Observations

### Mục tiêu

Repo này trích xuất các *observation* về hoạt tính phân tử từ hai cơ sở dữ liệu công
khai là ChEMBL 37 và BindingDB 202610. Mục đích là chuẩn bị dữ liệu đầu vào để xây
dataset cho Evaluator của dự án MOLOPT.

Mỗi observation là đúng một phép đo: một phân tử, đo trên một protein, trong một assay,
theo một endpoint (Ki, Kd, IC50 hoặc EC50). Mỗi dòng giữ nguyên giá trị gốc của nguồn,
cộng thêm các cột suy ra: nồng độ quy về mol/L, pActivity, canonical SMILES, InChIKey,
và bốn cột trạng thái cho biết phần protein, assay, cấu trúc và giá trị đo đã được xác
định chắc chắn hay chưa.

Bước này chưa ghép cặp phân tử, chưa tính Δp, chưa gộp phép đo lặp lại và chưa khử
trùng lặp giữa hai nguồn. Không dòng nào bị bỏ đi. Dòng thiếu thông tin vẫn được ghi ra,
kèm lý do cụ thể.

Phần giải thích chi tiết (cách đọc từng nguồn, cách lấy protein và assay, quy tắc chuẩn
hóa, schema đầy đủ, ví dụ lần theo từng dòng) nằm trong
[`docs/OBSERVATIONS_PIPELINE.md`](docs/OBSERVATIONS_PIPELINE.md).

## Cấu trúc thư mục

```
.
├── run_observations.py          # điểm vào Stage 1: --mode inspect | smoke | full | verify
├── run_dataset.py               # điểm vào Stage 2b: --mode smoke | full | split | export | verify
├── run_evaluator.py             # điểm vào Stage 3: --mode prepare | smoke | train | eval
├── config.json                  # đường dẫn, tên cột, bảng đơn vị, kích thước lô
├── requirements.txt
├── src/molopt/observations/
│   ├── chembl.py                # đọc ChEMBL (SQLite chỉ đọc)
│   ├── bindingdb.py             # đọc BindingDB (TSV đọc thẳng từ ZIP)
│   ├── normalize.py             # chuẩn hóa dùng chung: đơn vị, pActivity, RDKit, ID
│   └── pipeline.py              # ghép cột, ghi Parquet, verify
├── src/molopt/dataset/          # Stage 2b: cặp MMP và Δp
│   ├── aggregate.py             # gộp replicate trong assay; gộp Δp giữa assay + QC
│   ├── mmp.py                   # khóa định danh, cắt phân tử (RDKit), tạo cặp
│   ├── delta.py                 # pair_id, chiều của cặp, Δp trong assay
│   ├── split.py                 # split protein theo nhóm chuỗi, 80/10/10 theo số cặp
│   └── pipeline.py              # I/O, song song hóa, báo cáo, năm chế độ chạy
├── src/molopt/evaluator/        # Stage 3: evaluator dự đoán Δp của một cặp
│   ├── data.py                  # đọc partition, lát cắt, mã hóa thành chỉ số, chia batch
│   ├── features.py              # Morgan fingerprint, embedding ESM2, cache trên đĩa
│   ├── models.py                # ligand encoder dùng chung, nhánh edit, đầu direct/twin
│   ├── metrics.py               # regression + ranking trong từng query
│   └── train.py                 # prepare, vòng train, đánh giá, checkpoint, manifest
├── tests/                       # unit test (pytest), không cần dữ liệu thật
├── scripts/
│   ├── check_*.py, summarize_smoke.py   # script chẩn đoán chỉ đọc của Stage 1
│   ├── dedup_cross_source.py    # Stage 2a: đánh dấu observation trùng giữa hai nguồn
│   ├── audit_noise.py           # Stage 2a: audit độ nhiễu (chỉ đọc, chỉ báo cáo)
│   └── diagnose_noise.py        # Stage 2a: chẩn đoán thêm cho audit (chỉ đọc, chỉ báo cáo)
├── docs/OBSERVATIONS_PIPELINE.md
├── docs/STAGE2A_DEDUP_AUDIT.md
├── docs/STAGE2B_DATASET.md
├── docs/DATASET_REPORT.md       # báo cáo số liệu, các bước, kiểm chứng và tài liệu tham khảo
├── docs/STAGE3_EVALUATOR.md     # Stage 3: kiến trúc v1, chỉ số, ba cấu hình, chỗ cần lưu ý
│
├── raw/      (không có trong repo)  dữ liệu gốc tải về
├── work/     (không có trong repo)  ChEMBL SQLite đã giải nén
├── data/     (không có trong repo)  output Parquet: observations/ (Stage 1), dedup/ (Stage 2a),
│                                   dataset/ (Stage 2b)
├── work/evaluator_cache/ (không có trong repo)  cache fingerprint và embedding của Stage 3
└── runs/     (không có trong repo)  log và báo cáo: observations/, stage2a/, dataset/,
                                    evaluator/ (Stage 3)
```

Bốn thư mục `raw/`, `work/`, `data/` và `runs/` được loại khỏi repo bằng `.gitignore`,
vì chúng là dữ liệu chứ không phải code, và quá lớn để đưa lên GitHub:
- `raw/` chứa các file tải từ ChEMBL và BindingDB, khoảng 8 GB;
- `work/` chứa file ChEMBL `.db` sau khi giải nén, khoảng 30 GB;
- `data/observations/` chứa hai file Parquet output, tổng khoảng 1 GB;
- `runs/` chỉ chứa log và kết quả của từng lần chạy.

Ai cần chạy lại có thể tự tạo các thư mục này theo hướng dẫn bên dưới.

Các script chẩn đoán của Stage 1 trong `scripts/` đang ghi cứng đường dẫn tuyệt đối
`E:\eval_training\...`. Muốn chạy chúng trên máy khác thì phải sửa đường dẫn ở đầu mỗi
file. Hai script của Stage 2a lấy đường dẫn từ `config.json`, tính từ thư mục gốc của
repo.

## Yêu cầu môi trường

Code dùng Python 3.10 (lần chạy dưới đây dùng Python 3.10.11). Các thư viện cần cài nằm
trong `requirements.txt`:
- `rdkit==2026.9.1`
- `pyarrow==25.0.1`
- `pytest==8.4.2`
- `numpy==2.2.6` (hai script Stage 2a dùng trực tiếp; cài kèm rdkit nhưng vẫn ghi rõ phiên bản)

Cài bằng lệnh:

```powershell
python -m pip install -r requirements.txt
```

Hai module `chembl.py` và `bindingdb.py` chỉ dùng thư viện chuẩn của Python, nên mode
`inspect` chạy được ngay cả khi chưa cài RDKit và pyarrow.

## Lấy dữ liệu gốc

**ChEMBL 37.** Tải file `chembl_37_sqlite.tar.gz` từ trang phát hành chính thức của
EBI và đặt vào `raw/chembl37/`:
- file: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_sqlite.tar.gz
- checksum: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/checksums.txt

Pipeline không tự giải nén. Bạn cần giải nén một lần vào `work/` bằng lệnh sau,
chạy từ thư mục gốc của repo:

```powershell
New-Item -ItemType Directory -Force work | Out-Null
tar -xzf raw/chembl37/chembl_37_sqlite.tar.gz -C work
```

Sau khi giải nén phải có file `work/chembl_37/chembl_37_sqlite/chembl_37.db`. Đường
dẫn này được khai báo trong `config.json`. Code mở file ở chế độ chỉ đọc, và dừng với
thông báo rõ ràng nếu không tìm thấy nó.

**BindingDB 202610.** Tải bốn file sau từ trang tải về của BindingDB và đặt vào
`raw/bindingdb202610/`:
- `BindingDB_All_202610_tsv.zip`
- `BindingDB_Assays_202610_tsv.zip`
- `BindingDB_rsid_eaids_202610_tsv.zip`
- `BindingDB_All_2D_202610_sdf.zip`

Địa chỉ tải có dạng `https://www.bindingdb.org/rwd/bind/downloads/<tên file>`. Mỗi
file có một file `.md5` cùng tên để kiểm tra.

Các file BindingDB **giữ nguyên dạng ZIP, không giải nén**. Code đọc TSV thẳng từ ZIP.
Hai file SDF (`chembl_37.sdf.gz` và `BindingDB_All_2D_202610_sdf.zip`) không được đọc ở
bước này. Bạn không cần tải chúng để chạy pipeline.

## Cách chạy

Chạy từ thư mục gốc của repo, theo thứ tự sau.

```powershell
# 0. Unit test (không cần dữ liệu thật)
python -m pytest tests -q

# 1. inspect: in danh sách file raw, schema và vị trí các cột cần dùng; không đọc dòng dữ liệu
python run_observations.py --mode inspect

# 2. smoke: chạy thử 5.000 dòng mỗi nguồn vào runs/observations/smoke_<thời điểm>/
python run_observations.py --mode smoke

# 3. full: chạy toàn bộ, ghi vào data/observations/
python run_observations.py --mode full

# 4. verify: đối chiếu output của full ngược về raw, ghi runs/observations/verify_<thời điểm>.json
python run_observations.py --mode verify
```

Ở lần chạy gần nhất:
- `smoke` mất khoảng 20 giây.
- `full` mất khoảng **1 giờ 53 phút** cho 8.584.722 dòng: ChEMBL khoảng 1 giờ 4 phút,
  BindingDB khoảng 49 phút.
- `verify` mất khoảng 4 phút.

Thời gian của `inspect` không được đo riêng. Mode này chỉ đọc header và schema.

`full` ghi mỗi file vào `<tên>.parquet.partial` và chỉ đổi tên khi đã ghi xong, nên một
file `.parquet` đã tồn tại luôn là file hoàn chỉnh. Nếu chạy lại, `full` bỏ qua nguồn đã
có file hoàn chỉnh. Muốn làm lại từ đầu thì thêm `--force`. `verify` trả mã thoát 1 nếu
có phép kiểm tra không đạt.

## Kết quả lần chạy ngày 18–19/10/2026

| Nguồn | File output | Số observation |
|---|---|---|
| ChEMBL 37 | `data/observations/chembl37_observations.parquet` | 5.338.213 |
| BindingDB 202610 | `data/observations/bindingdb202610_observations.parquet` | 3.246.509 |
| Tổng | | 8.584.722 |

`verify` đạt **5/5 phép kiểm tra**:
1. **Bảo toàn số dòng:** số dòng ghi ra bằng số dòng chọn từ raw ở cả hai nguồn, và
   không có `observation_id` trùng.
2. **Đối chiếu ngẫu nhiên:** 5.000 dòng đối chiếu ngược về raw, 0 sai lệch.
3. **Construct P03367:** 1.040 dòng, trong đó 854 dòng giữ chuỗi 99 aa và 186 dòng giữ
   chuỗi 1.447 aa, 0 sai lệch.
4. **Dấu lập thể `|r|` của monomer 13590:** được giữ nguyên trong cột SMILES nguồn.
5. **PMID:** không còn dạng `.0`.

Hai nguồn được ghi ra hai file riêng, cùng schema, không gộp. Lý do: BindingDB chứa
nhiều dữ liệu nhập lại từ ChEMBL, nên việc khử trùng lặp để dành cho bước sau. Số liệu
chi tiết theo trạng thái và các giới hạn đã biết có trong mục 8 và mục 9 của
[`docs/OBSERVATIONS_PIPELINE.md`](docs/OBSERVATIONS_PIPELINE.md).

## Stage 2a: Dedup giữa hai nguồn và audit độ nhiễu

Stage 2a đọc hai file observations ở trên, nhưng không sửa chúng. Stage này làm hai
việc:
- đánh dấu những observation của ChEMBL và BindingDB thực chất là cùng một phép đo
  (đánh dấu, không xóa);
- đo độ nhiễu của dữ liệu, để làm căn cứ chốt ngưỡng QC ở bước sau.

Stage 2a chưa ghép cặp MMP, chưa tính Δp và chưa train.

Chạy từ thư mục gốc của repo, **sau khi** Stage 1 đã có hai file trong
`data/observations/`:

```powershell
# 1. Dedup -> data/dedup/ (file ánh xạ + hai bản sao có cột dedup_*) và runs/stage2a/dedup_*
python scripts\dedup_cross_source.py            # thêm --force để ghi đè output cũ

# 2. Audit độ nhiễu, đọc hai bản sao của bước 1 -> runs/stage2a/noise_audit_*
python scripts\audit_noise.py

# 3. Chẩn đoán thêm (bản sao trong cùng nguồn, publication lớn, cờ |r|) -> runs/stage2a/noise_diagnose_*
python scripts\diagnose_noise.py
```

Hai script đầu in báo cáo ra màn hình, có phần tóm tắt ở cuối; script thứ ba chỉ in
đường dẫn báo cáo. Báo cáo đầy đủ (`.txt`) và
số liệu (`.json`) nằm trong `runs/stage2a/`. Thời gian chạy (21/10/2026): dedup 5,8 phút, audit 0,7 phút,
chẩn đoán 0,6 phút. Cách hoạt
động, cách đọc kết quả và các quyết định thiết kế được mô tả trong
[`docs/STAGE2A_DEDUP_AUDIT.md`](docs/STAGE2A_DEDUP_AUDIT.md).

## Stage 2b: Dataset cặp MMP có Δp

Stage 2b đọc hai bản sao dedup của Stage 2a (chỉ đọc) và dựng dataset cặp matched
molecular pair: gộp replicate trong assay → ghép cặp MMP một lần cắt → Δp trong từng assay
→ gộp Δp giữa các assay kèm QC → dựng split protein → xuất hai file training và một file
evidence.

Trạng thái 22/10/2026: **đã chạy xong cả năm chế độ, verify 7/7 PASS.** Dataset cuối:
2.972.507 dòng đã lọc → 13.302.366 Δp trong assay → **10.546.330 bản ghi cặp MMP có Δp**
(17,6 % có ≥ 2 assay nên đo được phân tán), split protein 80/10/10 theo số cặp trên 2.782
nhóm MMseqs2 ở ngưỡng 50 % identity, và ba file xuất: evidence 10.546.330 dòng, evaluator
10.522.759, generator 6.414.841 — mỗi file kèm ba cờ mô tả (`unseen_pair`,
`plausible_range`, `short_protein`), không lọc dòng nào. Pipeline đã chứng minh tính xác
định: hai lần `smoke` độc lập cho 322/322 file dữ liệu trùng SHA-256. Số đo chi tiết ở
`docs/STAGE2B_DATASET.md` mục 11.

Cột thứ tư, `source_quality_warning`, mang nguyên văn cảnh báo chất lượng của chính
ChEMBL và cũng không lọc dòng nào: 146.464 bản ghi evidence mang cảnh báo, và cột đã được
audit độc lập với 0 lệch trên cả ba file. Xem `docs/STAGE2B_DATASET.md` mục 9.2.

Các số trên là của **v1** (`data/dataset/`). Ngày 24/10/2026 audit
(`docs/AUDIT_20261024.md`) dẫn tới **dataset v2** (`data/dataset_v2/`, v1 giữ nguyên):
`max_cut_bonds` 20 → 400, Δp chép giữa các assay chỉ tính một lần (cờ QC mới
`identical_values`), thêm cột `n_documents`, `n_distinct_values`, `p_a_range`,
`max_replicate_std`, `min_target_confidence`, `inactive_comment`, `nonstandard_sequence`.
v2 đã dựng xong ngày 24/10/2026, verify 7/7 PASS: 13.496.766 Δp theo assay → 10.712.915
bản ghi (pass 550.328; identical_values 1.303.052; single_assay 8.830.918; quarantine
28.617), 6.492 protein trên 2.846 nhóm MMseqs2, evaluator 10.684.298 dòng, generator
6.508.628 dòng. Thay đổi, lý do, số đo và thứ tự chạy: `docs/STAGE2B_DATASET.md` mục 0.

```powershell
python run_dataset.py --mode smoke     # ~1/200 số assay -> runs/dataset/smoke_<thời điểm>/
python run_dataset.py --mode full      # toàn bộ -> data/dataset_v2/full/ (thêm --force để ghi đè)
python run_dataset.py --mode split     # ghi FASTA đầu vào; cần --clusters để dựng split
python run_dataset.py --mode export    # training + evidence -> data/dataset_v2/
python run_dataset.py --mode verify    # kiểm tra -> runs/dataset/verify_<thời điểm>.json
```

Tham số, quy tắc QC, giả định và những điểm chưa làm: [`docs/STAGE2B_DATASET.md`](docs/STAGE2B_DATASET.md).

Báo cáo tổng hợp số liệu hiện có, các bước đã làm từ dữ liệu thô đến dataset,
toàn bộ kiểm chứng đã chạy và tài liệu tham khảo cho từng ngưỡng:
[`docs/DATASET_REPORT.md`](docs/DATASET_REPORT.md).

## Stage 3: evaluator baseline

Đã chạy: `pytest`, `--mode baseline`, `--mode prepare --skip-protein` và `--mode smoke`
cho cả ba đầu ra trên GPU. **Chưa train mô hình nào** — smoke chỉ 20.000 dòng và 2 epoch.
Phần ESM2 còn chờ `transformers`.

Evaluator nhận một cặp `(A, B, protein, endpoint)` và dự đoán
Δp = p(B) − p(A). Phiên bản v1 cố tình dùng biểu diễn đơn giản nhất — Morgan fingerprint cho
phân tử, chuỗi acid amin qua ESM2 cho protein, không 3D, không pocket — vì mục tiêu của v1 là
ra được một thứ chạy được từ đầu đến cuối, không phải kiến trúc cuối.

Ba cấu hình: `frozen` (ESM2 đóng băng, embedding tính sẵn), `finetune` (ESM2 train cùng) và
`no_protein` (đối chứng, bỏ hẳn nhánh protein). Ba đầu dự đoán để ablation tách được hai yếu
tố: `direct` (dự đoán thẳng Δp, có nhánh edit), `direct_no_edit` (như direct nhưng bỏ nhánh
edit) và `twin` (chấm điểm từng phân tử rồi lấy hiệu).

Mốc dưới đã đo được trên validation bằng `--mode baseline`: MAE của "luôn đoán 0" là
**0,582** (0,528 trên lát cắt nhiều assay), còn đoán bừa cho **49,7 %** đúng chiều và top-1
**0,23** — mọi mô hình phải vượt rõ các số này mới gọi là học được gì. Ba baseline: `zero` và
`endpoint_mean` làm mốc cho MAE/RMSE/R², `random` (lấy mẫu từ phân bố nhãn train, seed cố
định) làm mốc 50 % cho xếp hạng. Chi tiết và cách đọc: `docs/STAGE3_EVALUATOR.md` mục 4.5.

Chỉ số xếp hạng **chỉ tính trong cùng một query** `(phân tử tham chiếu, protein, endpoint)`,
không bao giờ gộp ứng viên của các query khác nhau vào một bảng. Mọi mẫu số được in riêng.
Partition `test` bị chặn ở mức code: `evaluator/data.py` ném `ValueError` nếu bị yêu cầu đọc.

```powershell
python run_evaluator.py --mode baseline                        # ba mốc dưới (không cần torch)
python run_evaluator.py --mode prepare --skip-protein          # cache fingerprint (không cần torch)
python run_evaluator.py --mode smoke --config no_protein       # thử pipeline trên CPU
python run_evaluator.py --mode prepare                         # cache embedding; in chiều thật của ESM2
python run_evaluator.py --mode train --config frozen --head direct
python run_evaluator.py --mode eval --checkpoint runs\evaluator\<tên>.pt
```

Môi trường đã kiểm chứng: torch 2.5.1+cu121 (GPU hoạt động), transformers 4.44.2.
**Chiều embedding ESM2 đo được là 480**, không phải 512 như nhắc trong buổi họp — code đọc
chiều từ chính `model.config.hidden_size` lúc `--mode prepare` chứ không hardcode, và giá
trị trong config giờ là chốt kiểm để đổi model mà quên sửa thì chương trình dừng.

Cache đã dựng xong trong `work/evaluator_cache/` (~104 MB): 958.926 fingerprint phân tử,
70.455 fingerprint nhóm thế (0 SMILES không đọc được) và 6.313 embedding protein.
Hai chế độ `baseline` và `prepare --skip-protein` không cần torch.

Kiến trúc, bốn quyết định về chỉ số, ước lượng tài nguyên và danh sách chỗ đáng lưu ý nhưng
chưa sửa: [`docs/STAGE3_EVALUATOR.md`](docs/STAGE3_EVALUATOR.md).
