# Stage 2b: Observations → dataset cặp MMP có Δp

Trạng thái ngày 22/10/2026: **đã chạy hết năm chế độ (`smoke`, `full`, `split`,
`export`, `verify`); verify 7/7 PASS; 278 unit test pass.** Dataset cuối: 10.546.330 bản
ghi cặp MMP có Δp, split protein 80/10/10 theo số cặp, ba file xuất trong `data/dataset/`
kèm ba cờ mô tả (mục 9.1). Pipeline đã chứng minh tính xác định (hai lần smoke trùng
SHA-256 từng byte). Mọi số đo thật ở mục 11.

Cập nhật cùng ngày: thêm cột `source_quality_warning` (mục 9.2) và chạy lại `--mode
export`; 278 unit test pass, số dòng ba file không đổi, cột mới đã được audit độc lập với
0 lệch.

> **Mọi số đo ở mục 11 là của v1** (`data/dataset/`, dựng 21–22/10/2026). Mục 0 dưới đây
> mô tả **v2**, ghi vào `data/dataset_v2/`; v1 giữ nguyên, chỉ đọc, để đối chiếu.

## 0. Dataset v2 (24/10/2026)

Trạng thái: **code và unit test đã viết, CHƯA chạy** (chưa chạy pytest, chưa dựng dataset).
Chưa có số đo nào của v2. Lý do từng thay đổi và số đo trên v1 nằm ở
`docs/AUDIT_20261024.md`.

### 0.1 Những gì đổi và vì sao

| # | Thay đổi | Vì sao (đo trên v1, 24/10/2026) | Code |
|---|---|---|---|
| 1 | `mmp.max_cut_bonds` 20 → 400; `cut_smarts` ghi rõ trong config; phân tử vượt giới hạn thì **báo lỗi** | `maxCutBonds` của RDKit không giới hạn kích thước phép sửa: vượt giới hạn thì `FragmentMol` trả về rỗng cho **cả phân tử**. v1 mất 194.400 Δp assay và 166.585 bản ghi (+1,58 %). Phân tử nhiều liên kết cắt nhất có 340. Kích thước phép sửa vẫn do `max_variable_heavy_atoms = 10` và `min_core_fraction = 0,6` khống chế, không đổi. | `mmp.check_cut_bonds`, `mmp.fragment_molecule` |
| 2 | Bước 4: Δp của các assay lệch nhau ≤ 0,01 được gộp làm **một giá trị** trước khi tính thống kê (cùng quy tắc neo và cùng ngưỡng với bước 1) | 69,3 % bản ghi ≥ 2 assay chỉ có một giá trị Δp, phần lớn là số liệu chép giữa các tài liệu. Gộp như vậy để bản sao không được tính là bằng chứng thứ hai và không kéo mean về phía nó. | `aggregate.collapse_values`, `summarize_values` |
| 3 | Cờ QC mới `identical_values` (≥ 2 assay, một giá trị). `pass` chỉ còn nghĩa "≥ 2 giá trị khác nhau, phân tán ≤ ngưỡng" | Ở v1 các bản ghi này mang `pass`, tức bị đọc nhầm là "được nhiều assay xác nhận". Vẫn đủ điều kiện training (`export.eligible_qc`). | `aggregate.qc_status` |
| 4 | Cột mới ở bản ghi: `n_distinct_values`, `n_documents`, `p_a_range`, `max_replicate_std` | `n_documents` = số tài liệu độc lập (assay nối với nhau khi chung PMID, DOI, patent đã chuẩn hóa, hoặc mã tài liệu của nguồn). `p_a_range` lộ ra bản sao sai đơn vị (Δp trùng mà p(A) lệch đúng 3 hay 6 log: 30.391 bản ghi ở v1). `max_replicate_std`: 6.602 nhóm replicate trong assay có STD > 1. | `aggregate.document_count`, `pipeline.assay_documents` |
| 5 | Cột mới ở file xuất: `min_target_confidence`, `inactive_comment`, `nonstandard_sequence` | Confidence 8 của ChEMBL = target đồng đẳng (24,4 % dòng evaluator v1 có ≥ 1 assay mức 8). `activity_comment` kiểu "inactive" trên dòng `=` (2.699 dòng v1). Chuỗi protein có chữ số hay tên nhóm hóa học (2 chuỗi, 136 dòng). **Chỉ gắn cờ, không lọc.** | `pipeline.chembl_observation_info`, `observation_flag_columns`, `nonstandard_proteins` |
| 6 | `split.clustering.max_seqs` 6315 → 11835 và `run_split` báo lỗi nếu nhỏ hơn số chuỗi FASTA | v2 có protein mới nên FASTA đổi; 11835 = tổng số chuỗi của Stage 2b nên luôn đủ. | `pipeline.run_split` |
| 7 | `verify` tính lại `n_distinct_values`, `n_documents` và mọi thống kê theo quy tắc mới, trên **mọi** bản ghi | Phép kiểm tra cũ tính trên giá trị chưa gộp nên sẽ báo sai với v2. | `pipeline.check_record_stats`, `recompute_mismatch` |

**Không đổi:** bộ lọc đầu vào, định nghĩa cặp MMP (single cut, lõi ≥ 60 %, phần biến đổi
≤ 10 nguyên tử nặng), khóa `-REL`, `pair_id`, ngưỡng phân tán (Ki/Kd 1,0; IC50/EC50 1,4),
tỷ lệ split, seed.

**Chưa làm (phạm vi):** cặp "thay H" (≈ 3,4 % Δp theo ước lượng ở audit mục 3.5) không được
thêm vào v2; phải nêu là giới hạn trong paper.

### 0.2 Ngưỡng phân tán: lập luận đã sửa

Ngưỡng giữ nguyên, nhưng lập luận "≈ 2σ" ở v1 không đúng về thống kê. Ngưỡng được chọn trước
khi xem nhãn, từ σ mỗi phép đo ở Stage 2a (Ki 0,50; Kd 0,57; IC50/EC50 0,73). Đo lại trên các
cặp Δp đến từ **hai tài liệu khác nhau và có giá trị khác nhau** (tức không phải bản sao), SD
của hiệu là Ki ≈ 0,27; IC50 ≈ 0,38; EC50 ≈ 0,49. Nghĩa là ngưỡng ≈ 3–4 SD của nhiễu giữa hai
tài liệu độc lập: một bản ghi bị cách ly là bất đồng thật, không phải nhiễu đo thông thường.

### 0.3 Thứ tự chạy v2

Mọi lệnh ghi vào `data/dataset_v2/` (config `stage2b.paths`) và `work/evaluator_cache_v2/`;
không lệnh nào ghi vào `data/dataset/` của v1.

```powershell
python -m pytest -q
python run_dataset.py --mode smoke
python run_dataset.py --mode full
python run_dataset.py --mode split          # ghi FASTA rồi thoát mã 2 (bình thường)
```

```bash
# trong WSL — thư mục mmseqs_v2 để không ghi đè kết quả gom của v1
cd /mnt/e/eval_training
mkdir -p runs/dataset/mmseqs_v2 ~/mmseqs_tmp_v2
mmseqs easy-cluster data/dataset_v2/protein_split_input.fasta \
    runs/dataset/mmseqs_v2/p50 ~/mmseqs_tmp_v2 \
    --min-seq-id 0.5 -c 0.8 --cov-mode 2 --cluster-mode 1 \
    --single-step-clustering --max-seqs 11835 \
    --alignment-mode 3 --seq-id-mode 0 -s 7.5
```

```powershell
python run_dataset.py --mode split --clusters runs\dataset\mmseqs_v2\p50_cluster.tsv `
    --cluster-tool mmseqs2 --cluster-tool-version <đúng dòng `mmseqs version` in ra>
python run_dataset.py --mode export
python run_dataset.py --mode verify
python run_evaluator.py --mode prepare      # cache Stage 3 mới, chuỗi protein đã viết hoa
```

## 1. Mục tiêu và phạm vi

Stage 2b biến các phép đo lẻ của Stage 1 (đã được Stage 2a đánh dấu trùng nguồn) thành
dataset cặp matched molecular pair (MMP) kèm Δp:

- mỗi bản ghi là một bộ (phân tử A, phân tử B, protein, endpoint), với Δp trung bình trên
  các assay cùng đo cặp đó;
- hai file dùng để train (evaluator và generator) tách khỏi một file evidence giữ nguyên
  mọi bằng chứng, nối với nhau bằng `pair_id`.

Stage 2b **chỉ đọc** `data/dedup/*.parquet` và không sửa gì trong `raw/`,
`data/observations/` hay `data/dedup/`. Không train model.

## 2. Đầu vào và bộ lọc

Đầu vào: hai bản sao dedup của Stage 2a
(`data/dedup/chembl37_observations_dedup.parquet`,
`data/dedup/bindingdb202610_observations_dedup.parquet`).

Một dòng vào dataset khi qua đủ các bước lọc sau (báo cáo in số dòng còn lại sau từng
bước, theo endpoint — "phễu lọc"):

| bước | điều kiện |
|---|---|
| `dedup_keep` | dòng được Stage 2a giữ lại làm bản chính |
| `qualifier '='` | `activity_qualifier` thuộc `qualifiers.exact` (chỉ `=`) |
| `protein resolved` | `protein_status = resolved` |
| `assay resolved` | `assay_status = resolved` |
| `có pActivity` | `activity_p_activity` không null |
| `có InChIKey` | `molecule_inchikey` không rỗng |
| `có SMILES` | `molecule_canonical_smiles` không rỗng |

Dòng ChEMBL mang cờ `potential_duplicate` trong `activity_context` **được giữ**, nhưng
không được tính là phép đo độc lập và không được tính là assay độc lập: cột
`n_independent` (bước 1) và `n_independent_assays` (bước 4) đếm riêng phần không mang cờ.
Mọi giá trị trung bình vẫn tính trên toàn bộ dòng được giữ.

Hai bước lọc cuối (InChIKey, SMILES) là bổ sung so với đề bài: không có InChIKey thì không
có khóa định danh phân tử, không có SMILES thì không cắt được mảnh. Số dòng bị loại vì hai
bước này hiện ra trong phễu lọc.

## 3. Luồng và vị trí code

```
data/dedup/*.parquet
   │  lượt gom (pipeline.collect_rows): lọc, dựng assay_key + khóa phân tử,
   │  ghi shard theo (endpoint, bucket protein); dựng molecules.parquet, proteins.parquet
   ▼
bước 1  aggregate.collapse_replicates   gộp replicate trong assay      -> replicates/
   ▼
bước 2  mmp.find_pairs                  cặp MMP trong từng assay
   ▼
bước 3  delta.assay_deltas              Δp = p(B) − p(A) trong assay   -> assay_deltas/
   ▼
bước 4  aggregate.aggregate_deltas      gộp Δp giữa assay + QC         -> records/
   ▼
split   split.py                        gom protein rồi chia 80/10/10  -> protein_split.parquet
   ▼
bước 5  pipeline.run_export             training + evidence            -> *.parquet
```

| file | nội dung |
|---|---|
| `src/molopt/dataset/aggregate.py` | bước 1 và bước 4 (logic thuần trên bảng Arrow) |
| `src/molopt/dataset/mmp.py` | khóa định danh, cắt phân tử bằng RDKit, tạo cặp |
| `src/molopt/dataset/delta.py` | `pair_id`, chiều của cặp, bảng Δp trong assay |
| `src/molopt/dataset/split.py` | đếm cặp theo protein, FASTA, xếp partition, kiểm tra |
| `src/molopt/dataset/pipeline.py` | I/O, song song hóa, báo cáo, năm chế độ chạy |
| `run_dataset.py` | CLI: `--mode smoke \| full \| split \| export \| verify` |

### Vì sao chia shard theo bucket protein

Mọi bước đều nhóm trong một protein: bước 1 nhóm theo (assay, protein, phân tử), cặp MMP
chỉ tìm trong một assay của một protein, bản ghi bước 4 khóa theo protein. Vì vậy dữ liệu
được chia thành `delta.protein_buckets = 16` phần theo hai chữ số hex đầu của
`protein_sequence_hash`, và mỗi (endpoint, bucket) được xử lý độc lập. Bộ nhớ vì thế chỉ
phụ thuộc phần lớn nhất, không phụ thuộc tổng kích thước — đây là điều kiện để chạy trên
máy 16 GB.

## 4. Bước 1 — gộp replicate trong assay

Nhóm: (nguồn, assay, protein, endpoint, phân tử). `assay_key = <nguồn>:<ENTRYID>:<assay>`
như Stage 2a (ASSAYID của BindingDB chỉ duy nhất trong một ENTRYID).

Khóa phân tử = InChIKey, cộng hậu tố `-REL` nếu dòng mang cờ lập thể tương đối `r` trong
`molecule_cx_extension` (regex `stage2a.dedup.relative_stereo_pattern`). Lý do: cờ `r`
nghĩa là nguồn chỉ biết quan hệ giữa các tâm lập thể chứ không biết cấu hình tuyệt đối,
còn InChIKey do RDKit sinh ra lại khẳng định một cấu hình cụ thể.

Trong một nhóm, các pActivity lệch nhau ≤ `replicate.merge_tolerance` (0,01) được coi là
**một** phép đo và gộp làm một (quyết định 1, `STAGE2A_DEDUP_AUDIT.md` mục 5.6). Cụm được
neo vào giá trị nhỏ nhất chưa xếp, nên mọi giá trị trong một cụm lệch nhau ≤ 0,01; không
nối chuỗi (6,000 → 6,008 → 6,016 cho hai cụm, không phải một).

Giá trị đại diện của nhóm = mean các phép đo sau gộp. Cột lưu: `p_mean`, `p_std` (null khi
chỉ có một phép đo), `n_merged`, `n_independent`, `n_raw`, `n_source_ids`,
`multi_source_id`, `observation_ids`.

`multi_source_id` = nhóm có nhiều `molecule_source_id` khác nhau (cùng InChIKey, nguồn ghi
nhiều ID phân tử). Theo quyết định 2 của audit, nhóm như vậy **được giữ** và chỉ gắn cờ;
cờ được mang lên bảng Δp trong assay (`multi_source_id` = ít nhất một phía có cờ) để tra
lại được, và không dùng để lọc.

## 5. Bước 2 — cặp MMP trong cùng một assay

Dùng `rdkit.Chem.rdMMPA.FragmentMol` với `min_cuts = max_cuts = 1`, `max_cut_bonds = 20`
(single cut, như mmpdb). Mỗi lần cắt cho hai mảnh; mỗi mảnh lần lượt đóng vai lõi chung
(constant) và phần biến đổi (variable).

Điều kiện giữ một cặp (A, B):

- A và B nằm trong **đúng một assay** (nhóm tìm cặp là một assay của một protein);
- cùng `constant` (SMILES chính tắc, gồm vị trí nối) và khác `variable`;
- `constant` ≥ `min_core_fraction` (0,6) số nguyên tử nặng của **cả A và B** — kiểm tra
  được trên từng phân tử khi cắt, vì constant giống hệt nhau ở hai phía;
- `variable` ≤ `max_variable_heavy_atoms` (10) nguyên tử nặng mỗi phía;
- mỗi mảnh đúng một điểm nối (`attachment_points = 1`) và nguyên tử nối có bậc 1
  (`attachment_degree = 1`).

Khi một cặp có nhiều cách cắt hợp lệ, giữ cách cắt có phần biến đổi nhỏ nhất; hòa thì so
tổng kích thước rồi so chuỗi, nên kết quả không đổi giữa các lần chạy.

**Quy tắc `|r|`:** phân tử có cờ lập thể tương đối mang khóa khác (hậu tố `-REL`), và cặp
giữa một phân tử có cờ với một phân tử không cờ bị loại. Số cặp mất vì quy tắc này được
ghi thành file riêng (`lost_relative_pairs/`) và đếm trong báo cáo, theo yêu cầu "báo cáo
bao nhiêu cặp mất vì quy tắc này".

Song song hóa: `mmp.workers` tiến trình con, mỗi tiến trình nhớ kết quả cắt của
`fragment_cache_size` phân tử (LRU), mỗi lần gửi `groups_per_task` nhóm assay. Kết quả
được sắp lại theo chỉ số dòng trước khi dùng, nên output không phụ thuộc thứ tự hoàn thành.

## 6. Bước 3 — Δp trong từng assay

`delta_p = p(B) − p(A)`, tính riêng trong từng assay, **giữ dấu**. Chiều cố định: A là phân
tử có khóa định danh nhỏ hơn theo thứ tự chuỗi. Nhờ vậy cùng một cặp ở mọi assay luôn có
cùng chiều, bước 4 gộp được mà không phải đảo dấu, và không bao giờ lưu cả A→B lẫn B→A.

`pair_id` = 24 ký tự hex đầu của `SHA-256("endpoint|protein|A|B")`. Đây là ID ổn định nối
phần training với phần evidence.

Bảng `assay_deltas` giữ cả `observation_ids_a` và `observation_ids_b`, nên từ một dòng Δp
luôn truy được về các dòng observation gốc (verify dùng đúng đường này).

## 7. Bước 4 — gộp Δp giữa các assay và QC

Mỗi (A, B, protein, endpoint) thành một bản ghi; mỗi assay góp **đúng một** Δp. Lưu: mean,
median, STD, min, max, `n_assays`, `n_independent_assays`, danh sách Δp từng assay, danh
sách `assay_key`, độ phân tán và cờ QC.

Độ phân tán:

- ≥ 3 assay: STD của các Δp (ddof = 1);
- đúng 2 assay: `|Δ_X − Δ_Y|` (= max − min);
- 1 assay: không tính được → cờ `single_assay`.

Ngưỡng theo endpoint (`qc.dispersion_thresholds`): Ki 1,0; Kd 1,0; IC50 1,4; EC50 1,4 —
xấp xỉ 2σ với σ mỗi phép đo đo được ở Stage 2a (Ki 0,50; Kd 0,57; IC50/EC50 0,73). Vượt
ngưỡng → cờ `quarantine`; bản ghi **vẫn được giữ**, chỉ không vào phần training.

Lưu ý về quy tắc hai assay: `|Δ_X − Δ_Y|` lớn hơn STD (ddof = 1) của hai giá trị đúng √2
lần, nên so với cùng ngưỡng thì trường hợp hai assay bị xét chặt hơn trường hợp ba assay.
Đây là cách đề bài yêu cầu, ghi lại ở đây để không bị hiểu là lỗi.

Báo cáo in tỷ lệ bản ghi có phân tán ≤ 0,3 và > 1,0 cho từng endpoint
(`qc.report_levels`), để đối chiếu với Nelen et al. 2025. Hai mức này **chỉ để báo cáo**,
không lọc gì.

## 8. Dựng split protein (sau bước 4)

Split được dựng **sau** khi đã có bản ghi bước 4, trên đúng những protein có mặt trong
dataset cuối, và tỷ lệ tính theo **số cặp** (không theo số protein). Thứ tự: gộp replicate
→ MMP → Δp trong assay → gộp Δp → dựng split → xuất dataset. Không dùng lại split cũ.

Quy tắc: 80/10/10 theo số cặp; mọi protein cùng nhóm vào cùng một partition; nhóm lớn xếp
trước (xếp sau thì không còn chỗ bù, tỷ lệ cuối sẽ lệch); seed `split.seed` phá thế hòa và
được ghi vào manifest.

Output: `data/dataset/protein_split.parquet` (`protein_sequence_hash`,
`protein_cluster_id`, `partition`, `cluster_proteins`, `cluster_pairs`, `pairs`) và
`protein_split_manifest.json` (tham số, công cụ và phiên bản, seed, số nhóm, số protein,
phân bố cặp theo partition, SHA-256 của FASTA đầu vào và của file nhóm).

### 8.1 Công cụ gom chuỗi: MMseqs2 trong WSL (apt)

Chốt ngày 21/10/2026. Máy chưa có sẵn công cụ nào (kiểm tra cùng ngày: không có `mmseqs`
hay `cd-hit` trên PATH của Windows; Miniconda ở `C:\ProgramData\miniconda3` không có env
nào và không nằm trên PATH; WSL Ubuntu-24.04 không có hai công cụ đó lẫn conda; Docker
Desktop đã cài nhưng daemon không chạy), nên phải cài trong WSL trước:

```bash
sudo apt-get update && sudo apt-get install -y mmseqs2
mmseqs version        # đã cài: 15-6f452+ds-2
```

Tham số nằm trong `split.clustering` của config và được ghi nguyên vào
`protein_split_manifest.json`:

| tham số | cờ MMseqs2 | giá trị | so với split cũ |
|---|---|---|---|
| ngưỡng identity | `--min-seq-id` | 0,5 | giữ nguyên |
| mẫu số của identity | `--alignment-mode 3`, `--seq-id-mode 0` | số cột alignment | giữ nguyên |
| thuật toán gom | `--cluster-mode 1` | connected component | giữ nguyên (single-linkage) |
| coverage | `-c` | 0,8 | **cố ý khác: split cũ không có** |
| coverage đo trên | `--cov-mode 2` | chuỗi query, tức chuỗi ngắn trong cặp | **cố ý khác** |
| số vòng gom | `--single-step-clustering` | một vòng, không cascade | giữ tinh thần single-linkage |
| láng giềng mỗi chuỗi | `--max-seqs` | 6315 = số chuỗi (bỏ giới hạn) | split cũ là all-vs-all |
| độ nhạy prefilter | `-s` | 7,5 | split cũ là all-vs-all global NW |

**Chỗ cố ý khác split cũ và lý do.** Split cũ chỉ có ngưỡng identity, không có ràng buộc
coverage. Dữ liệu của ta lấy `protein_sequence` theo đúng construct của từng phép đo, nên
cùng một gene xuất hiện dưới nhiều độ dài khác nhau: một construct 150 aa và protein đầy
đủ 800 aa giống hệt nhau trên phần chồng nhau vẫn có identity toàn cục thấp, và hai dòng
dữ liệu của cùng một protein rơi vào hai partition — rò rỉ thật, không phải rủi ro lý
thuyết. Thêm `-c 0.8` với coverage đo trên chuỗi ngắn hơn khiến cặp construct/full-length
nối được với nhau.

Các mode coverage của MMseqs2 (wiki, mục *Clustering criteria*): mode 0 =
`alnRes / max(qLen, tLen)`, mode 1 = `alnRes / tLen`, mode 2 = `alnRes / qLen`. **Mode 0
không dùng được** vì nó chia cho chuỗi **dài hơn**, nên loại đúng loại cặp
construct/full-length cần gom, dù tên gọi là "bidirectional". Mode 1 và mode 2 đều thỏa ý
định "coverage trên chuỗi ngắn hơn", chỉ khác vai: mỗi cặp được xét ở cả hai chiều, và
chiều đạt ngưỡng là chiều chuỗi ngắn nằm ở mẫu số — mode 1 là chiều chuỗi ngắn làm target,
mode 2 là chiều chuỗi ngắn làm query. Vai query/target không đi vào dataset vì ta chỉ dùng
**thành viên** của nhóm, không dùng chuỗi đại diện.

#### Ba tham số đổi ngày 22/10/2026 sau lần gom thử đầu tiên

Lần đầu chạy với `--cov-mode 1`, gom cascade mặc định và `--max-seqs` mặc định, ra 2.839
nhóm từ 6.315 chuỗi. Ba bằng chứng trong log của lần đó buộc phải sửa tham số — nhóm bị
tách vụn hơn mức đúng, mà nhóm vụn chính là đường rò rỉ giữa các partition:

1. **`--cov-mode 1` → `--cov-mode 2`.** MMseqs2 tự in cảnh báo ngay trước bước gom:

   ```
   Combining cluster mode 1 in combination with coverage mode 1 can produce wrong results.
   Please use --cov-mode 2
   ```

   Đổi theo khuyến nghị của chính công cụ. Ý định khoa học không đổi: như đoạn trên, mode 2
   vẫn đo coverage trên chuỗi ngắn hơn (chiều chuỗi ngắn làm query). Đây cũng là chỗ tài
   liệu này trước đó nói sai rằng mode 1 là cách duy nhất đạt được điều đó.
2. **Thêm `--single-step-clustering`.** Log cho `Cascaded clustering steps 3` và dãy
   6.315 → 3.583 → 3.214 → 2.860 → 2.839: mỗi vòng chỉ xét chuỗi đại diện của vòng trước,
   nên các họ protein bị tách. MMseqs2 cũng nói thẳng:

   ```
   Connected component clustering produces less clusters in a single step clustering.
   Please use --single-step-clustering
   ```

   Ít nhóm hơn = gộp nhiều hơn = an toàn hơn trước rò rỉ, nên chọn gom một vòng.
3. **Thêm `--max-seqs 6315`.** Mặc định `mmseqs cluster` chỉ giữ 20 láng giềng mỗi chuỗi
   (`Max results per query 20`), và log chứng minh mức chặn đã bị chạm ở vòng cuối:

   ```
   19 sequences passed prefiltering per query sequence
   20 median result list length
   ```

   Median bằng đúng mức chặn nghĩa là có chuỗi còn láng giềng ≥ 50 % identity bị cắt mất —
   với dữ liệu nhiều kinase/protease cùng họ thì chắc chắn xảy ra, và mỗi cạnh mất là một
   đường rò rỉ tiềm năng. Split cũ dùng all-vs-all global NW nên không có giới hạn này.
   Đặt `--max-seqs` bằng số chuỗi trong FASTA là bỏ hẳn giới hạn; **nếu số protein có cặp
   thay đổi thì phải sửa `max_seqs` trong config cho ≥ số chuỗi mới**.

#### Kết quả gom chuỗi thật (22/10/2026, bộ tham số đã sửa)

| mục | lần đầu (đã loại bỏ) | lần dùng |
|---|---|---|
| nhóm | 2.839 | **2.782** |
| chuỗi vào | 6.315 | 6.315 |
| láng giềng mỗi query qua prefilter | 19, median = 20 = **đúng mức chặn** | 802, median 705 (cap 6315, không chạm) |
| alignment tính / cặp đạt ngưỡng | — | 2.848.894 / 27.424 (4,64 láng giềng mỗi chuỗi) |
| số vòng gom | cascade 3 vòng (`clu_step0..2`) | một vòng (`clu_step0`) |
| thời gian | ~15 s | ~1 phút (prefilter 21 s, align 35 s) |

Ba thay đổi chỉ làm giảm **57 nhóm (2 %)**, nên lo ngại "cascade và mức chặn 20 làm tách
vụn các họ protein" là **nói quá**; tác động thật nhỏ. Nhưng bộ tham số mới mới là bộ bảo
vệ được: không còn cạnh nào bị cắt vì mức chặn, gom đúng một vòng connected component, và
cov-mode theo khuyến nghị của công cụ.

**Cảnh báo của MMseqs2 không thể thỏa mãn — đừng dựa vào nó.** Với `--cov-mode 2` nó vẫn
in `Combining cluster mode 1 in combination with coverage mode 2 can produce wrong
results. Please use --cov-mode 2`, tức khuyên dùng đúng cái đang dùng; lần đầu (cov-mode 1)
nó in cùng câu đó. Điều kiện kiểm tra trong bản 15-6f452 bị lỗi, cảnh báo bắn ra với mọi
cov-mode. Thứ dựa vào được là ba số đo trong bảng trên, không phải việc cảnh báo có mất đi
hay không.

`cluster` còn chạy một bước phụ mặc định trước khi gom: `clusthash --min-seq-id 0.99` gộp
6.315 → 5.904 chuỗi, tức 411 chuỗi trùng gần như hoàn toàn. Bước này vô hại và không thay
đổi ngưỡng 50 %.

Còn một điều phải xem sau khi dựng split: phân bố `cluster_proteins` trong
`protein_split.parquet` — nếu construct ngắn vẫn tách riêng thì sẽ thấy rất nhiều nhóm một
protein.

Một điểm wiki nói mà ta **cố ý không theo**: wiki khuyến nghị `--cluster-mode 2` kèm
`--cov-mode 1` để fragment không trở thành chuỗi đại diện. Ta giữ `--cluster-mode 1` vì
connected component mới là thứ khớp với single-linkage của split cũ, và chuỗi đại diện
không đi vào dataset. Cũng theo wiki, `--cov-mode` "automatically sets the
`--cluster-mode`", nên luôn phải đọc lại dòng `Cluster mode` trong log tham số MMseqs2 in
ra; lần đầu đã kiểm tra, nó đúng là 1.

`--seq-id-mode` không xuất hiện trong wiki, nhưng có trong bản đã cài — `mmseqs cluster
-h` của 15-6f452+ds-2 in ra `--seq-id-mode INT 0: alignment length 1: shorter, 2: longer
sequence [0]`, tức mặc định đã là 0; truyền thêm chỉ để ghi rõ ý định. `--alignment-mode 3`
mới là cờ bắt buộc: nó cho identity = số residue giống nhau chia cho số cột alignment, đúng
mẫu số của split cũ. Mặc định của MMseqs2 **không** phải vậy — nó ước lượng identity từ
score mỗi cột, nên không so được với ngưỡng 50 % của split cũ.

Thứ tự chạy:

```bash
# trong WSL; ổ E: của Windows là /mnt/e
cd /mnt/e/eval_training
mkdir -p runs/dataset/mmseqs ~/mmseqs_tmp
mmseqs easy-cluster data/dataset/protein_split_input.fasta \
    runs/dataset/mmseqs/p50 ~/mmseqs_tmp \
    --min-seq-id 0.5 -c 0.8 --cov-mode 2 --cluster-mode 1 \
    --single-step-clustering --max-seqs 6315 \
    --alignment-mode 3 --seq-id-mode 0 -s 7.5
```

Thư mục tmp đặt **ngoài** cây dự án (`~/mmseqs_tmp` trong WSL) là có lý do: `easy-cluster` để
lại trong tmp một symlink kiểu Linux (`latest`) mà Python trên Windows không stat được
(WinError 1920), và pytest quét cả cây dự án nên gặp symlink đó là dừng ngay lúc collect. Đặt
tmp ngoài dự án thì không còn vấn đề; nếu đã chạy với tmp trong dự án thì xóa bằng `wsl rm -rf
/mnt/e/eval_training/runs/dataset/mmseqs/tmp`.

`easy-cluster` ghi `runs/dataset/mmseqs/p50_cluster.tsv` đúng dạng hai cột
`<đại diện>\t<thành viên>` mà `--mode split` cần, nên không phải chuyển định dạng:

```powershell
python run_dataset.py --mode split --clusters runs\dataset\mmseqs\p50_cluster.tsv `
    --cluster-tool mmseqs2 --cluster-tool-version <đúng dòng `mmseqs version` in ra>
```

Chạy `--mode split` khi chưa có `--clusters` vẫn hợp lệ: nó chỉ ghi
`data/dataset/protein_split_input.fasta` (đúng những protein có cặp, tiêu đề là
`protein_sequence_hash`) kèm SHA-256, in hướng dẫn rồi thoát với mã 2. Đó là bước phải
chạy trước để có file FASTA cho MMseqs2.

### 8.2 Tham số của split cũ (P50 V2) — đối chiếu

Đọc từ `split_manifest.json` của
`E:\data\molopt_benchmark_split_candidate_v2\P50_GLOBAL_NW_50_CANDIDATE_20260910_V2`
(mục `protein_similarity_contract`):

| tham số | giá trị của split cũ |
|---|---|
| alignment | global Needleman-Wunsch, BLOSUM62, gap open 10, gap extend 1 |
| identity đo trên | số khớp chính xác **chia cho số cột alignment** |
| ràng buộc coverage | **không có**; chỉ có mẹo bỏ qua cặp khi `min_len/max_len < 0,30` |
| ngưỡng | 50 % (P50) |
| thuật toán gom | single-linkage connected components |
| chuẩn hóa chuỗi | ký tự ngoài bảng chữ cái → `X` |
| đơn vị chia | cả nhóm protein, không tách nhóm; 80/10/10 theo "logical queries" |
| kiểm tra của họ | 0 nhóm nằm ở hai partition; identity cao nhất giữa hai partition 0,4986 |

Đánh giá (chỉ nêu, không tự đổi):

- **Giữ được**: ngưỡng 50 %; single-linkage (bảo thủ: hai protein chỉ cần liên hệ qua một
  chuỗi trung gian là vào cùng partition, nên khó rò rỉ hơn); bỏ qua cặp theo tỷ lệ độ dài
  là mẹo tăng tốc chính xác, không làm sai kết quả ở ngưỡng 0,5.
- **Có vấn đề 1 — identity chia cho số cột alignment mà không có coverage.** Một construct
  ngắn (vd. một domain 150 aa) so với protein đầy đủ 800 aa có identity toàn cục rất thấp
  dù domain giống hệt, nên hai dòng dữ liệu của cùng một protein có thể rơi vào hai
  partition. Dữ liệu của ta lấy `protein_sequence` theo construct của từng phép đo, nên
  đây là đường rò rỉ thật. **Đã quyết ngày 21/10/2026**: thêm ràng buộc coverage
  `-c 0.8` đo trên chuỗi ngắn hơn (`--cov-mode 1`); chi tiết và lý do ở mục 8.1. Ngưỡng
  identity và thuật toán gom giữ nguyên như split cũ.
- **Có vấn đề 2 — không so sánh 1:1 được với công cụ mới.** MMseqs2 và CD-HIT đều là
  heuristic trên alignment cục bộ với định nghĩa identity riêng, nên nhóm sinh ra sẽ khác
  nhóm global-NW của split cũ. Kiểm tra số 5 của verify sẽ cho thấy mức khác nhau đó dưới
  dạng "% protein chung được xếp cùng partition", chứ không chứng minh được hai split
  tương đương.
- **Có vấn đề 3 — phân bố kích thước nhóm lệch nặng.** Audit cũ: 2.587 nhóm, median 27
  query, p99 4.942, max 17.682. Với single-linkage, một nhóm lớn có thể chiếm hơn 10 % số
  cặp, khi đó validation/test không thể đạt đúng 10 % — cách xếp tham lam của ta sẽ báo
  mức lệch ở kiểm tra số 4 thay vì tách nhóm.
- **Khác biệt có chủ ý so với split cũ**: split cũ tối ưu một hàm mục tiêu nhiều thành
  phần (tổng query 0,40; query theo endpoint 0,35; số protein 0,10; số nhóm 0,10; mức tập
  trung 0,05) kèm local moves và swap. Ta chỉ xếp tham lam theo **số cặp**, đúng như đề
  bài; hệ quả: tỷ lệ theo từng endpoint có thể lệch nhiều hơn, chỉ tổng số cặp là gần
  80/10/10.

## 9. Bước 5 — xuất dataset

Ba file trong `data/dataset/`, nối với nhau bằng `pair_id`:

| file | nội dung |
|---|---|
| `evidence_records.parquet` | **mọi** bản ghi bước 4, kể cả `quarantine`, kèm Δp từng assay, danh sách assay, phân tán, cờ QC, partition |
| `evaluator_training.parquet` | bản ghi đủ điều kiện, Δp **có dấu** (`delta_p_mean`) |
| `generator_training.parquet` | bản ghi đủ điều kiện, hướng **yếu → mạnh**, `gain > 0,3` (nghiêm ngặt) |

"Đủ điều kiện" = `qc_status` thuộc `export.eligible_qc` = `{pass, single_assay}`. Bộ lọc
chỉ quyết định tính đủ điều kiện; phần evidence không bị xóa dòng nào.

Hướng của generator lấy theo dấu của Δp trung bình: Δp > 0 thì B mạnh hơn nên giữ nguyên
(A yếu → B mạnh), Δp < 0 thì đảo hai phía; `gain = |Δp trung bình|`.

`--mode export` **đọc** `protein_split.parquet`, không tự dựng split. Nếu một protein trong
bản ghi không có trong file split (vd. dataset đã chạy lại), export dừng với thông báo
phải dựng lại split.

### 9.1 Ba cờ mô tả (thêm ngày 22/10/2026, không lọc dòng nào)

Cả ba file xuất đều có thêm ba cột bool. Chúng **chỉ mô tả**: không dòng nào bị bỏ vì
chúng, đúng nguyên tắc của dự án là không lọc thầm. Số dòng `True` của từng cờ được ghi
vào `export_manifest.json` và in ở cuối `--mode export`.

| cờ | nghĩa | tham số |
|---|---|---|
| `unseen_pair` | cặp (A, B) **không** xuất hiện trong partition train; dòng train luôn `False` | — |
| `plausible_range` | **mọi** phép đo đóng góp cho bản ghi có pActivity trong khoảng hợp lý | `export.plausible_p_range` = [2,0; 12,0] |
| `short_protein` | chuỗi protein ngắn hơn ngưỡng | `export.min_protein_length` = 50 aa |

**Vì sao `unseen_pair`.** Split chia theo protein, nên cùng một cặp phân tử có thể gặp lại
ở protein khác trong partition khác: đo được 22,5 % dòng test và 15,2 % dòng validation có
cặp (A, B) cũng nằm trong train. Đó là đúng thiết kế, nhưng người đánh giá cần phân biệt
được "protein mới" với "protein mới **và** cặp mới". Cờ này định nghĩa tập nghiêm ngặt
"cặp chưa từng thấy" mà không phải dựng lại dataset. Cách tính: quét một lượt mọi bản ghi,
thu tập cặp (A, B) của train, rồi so từng bản ghi với tập đó; cặp được gói thành một int64
(hai mã phân tử, mỗi mã 31 bit) nên không phải giữ hàng triệu chuỗi trong bộ nhớ.

**Vì sao `plausible_range`.** pActivity nhỏ hơn 0 nghĩa là nồng độ trên 1 M — không thể có
thật, nên là lỗi của nguồn, không phải quan điểm. Nhưng dự án không lọc thầm, nên dataset
giữ nguyên dòng và chỉ gắn cờ; phần Methods của bài báo báo cáo con số, và ai dùng lại có
thể lọc theo cờ. Biên tính từ Δp assay của chính bản ghi đó: lấy min/max của `p_a`, `p_b`
trên mọi assay đóng góp, nên cờ phản ánh toàn bộ dẫn chứng chứ không chỉ giá trị trung
bình. Nguồn gốc con số: xem mục "Audit bổ sung ngoài verify".

**Vì sao `short_protein`.** Có 21 chuỗi ngắn hơn 50 aa (ngắn nhất 9 aa) mang 0,02 % số
cặp; gom chuỗi ở identity 50 % với coverage 0,8 gần như vô nghĩa với chuỗi vài acid amin,
nên nhóm của chúng không đáng tin bằng nhóm của protein đầy đủ.

Đo được ở lần export 22/10/2026 (76,8 s, trong đó lượt quét trước để thu 6.048.225 cặp
(A, B) của train):

| partition | dòng | `unseen_pair` | `plausible_range` = False | `short_protein` |
|---|---|---|---|---|
| train | 8.437.064 | 0 (0,0 %) | 11.387 (0,13 %) | 1.097 |
| validation | 1.054.633 | 894.182 (**84,8 %**) | 1.388 (0,13 %) | 307 |
| test | 1.054.633 | 817.391 (**77,5 %**) | 2.256 (0,21 %) | 234 |
| tổng | 10.546.330 | 1.711.573 (16,2 %) | 15.031 (0,14 %) | 1.638 |

Tập đánh giá nghiêm ngặt cho bài báo vì thế là 817.391 dòng test và 894.182 dòng
validation — protein mới **và** cặp phân tử chưa từng thấy trong train.

### 9.2 Cột `source_quality_warning` (thêm ngày 22/10/2026, không lọc dòng nào)

Ngoài ba cờ bool trên, cả ba file xuất có thêm **một cột chuỗi**:
`source_quality_warning` — cảnh báo chất lượng do **chính nguồn** gắn, lấy nguyên văn.

| | |
|---|---|
| Nguồn giá trị | `data_validity_comment` trong `activity_context` của ChEMBL |
| Giá trị thật trong ChEMBL 37 | `Outside typical range`, `Potential transcription error`, `Potential missing data`, `Manually validated`, `Potential author error` |
| BindingDB 202610 | không có cột tương đương, nên bản ghi chỉ từ BindingDB luôn null |
| Null nghĩa là | **không có cảnh báo**, hoặc nguồn không có cột tương đương — không phải "chưa biết" |
| Nhiều cảnh báo trên một bản ghi | gom cảnh báo của mọi phép đo đóng góp, bỏ trùng, sắp xếp, nối bằng `export.source_quality_separator` (`"; "`) |
| Lọc dòng | **không**, như ba cờ kia |

**Vì sao cần.** ChEMBL tự đánh dấu dòng đáng nghi, và 51.945 dòng mang cảnh báo đã vào
dataset (48.258 `Outside typical range`, 3.633 `Potential transcription error`,
54 `Potential author error`, cộng 372 dòng `Manually validated` là dấu hiệu tốt chứ không
phải cảnh báo). Quan trọng hơn: **toàn bộ 5.073 dòng ChEMBL có pActivity ngoài khoảng
[2; 12] đều nằm trong số dòng đã bị ChEMBL gắn cảnh báo** — không một dòng nào thuộc nhóm
"không cảnh báo" (đo ngày 22/10/2026). Vậy cờ `plausible_range` của dự án và cột curation
của ChEMBL là hai dấu hiệu độc lập chỉ về cùng một tập dòng. Giữ nguyên văn đánh giá của
nguồn để người đọc tự quyết, và để trả lời được câu hỏi "ChEMBL có cờ chất lượng, sao
không dùng?".

**Cách tính.** Cảnh báo nằm trong `activity_context`, mà các file trung gian của bước 1–4
không mang cột đó, nên bước 5 đọc lại hai file nguồn một lượt — không phải chạy lại
`--mode full`. Chỉ parse JSON ở dòng mà `export.source_quality_regex` đã thấy khóa có giá
trị chuỗi (`null` không khớp), đúng cách làm của `independent_flags`, nên không phải đọc
JSON của 8,6 triệu dòng. Nối bản ghi với phép đo qua `observation_ids_a` và
`observation_ids_b` của file Δp assay; việc tra được làm theo offset của cột danh sách
thay vì vòng lặp Python.

`export_manifest.json` ghi thêm khối `source_quality_warning` gồm số observation có cảnh
báo, số bản ghi mang cảnh báo, và số bản ghi theo từng giá trị; `--mode export` in cùng
bảng đó.

Đo được ở lần export 22/10/2026 (237.660 observation có cảnh báo — ChEMBL 237.660,
BindingDB 0; đúng bằng tổng đếm trực tiếp từ ChEMBL SQLite):

| giá trị | evidence | evaluator | generator |
|---|---|---|---|
| `Outside typical range` | 118.357 | 115.449 | 78.978 |
| `Potential transcription error` | 15.630 | 14.507 | 7.857 |
| `Outside typical range; Potential transcription error` | 10.063 | 9.965 | 5.737 |
| `Manually validated` | 2.074 | 2.070 | 1.331 |
| `Potential author error` | 291 | 291 | 220 |
| `Manually validated; Outside typical range` | 49 | 49 | 19 |
| **tổng** | **146.464** (1,39 %) | **142.331** | **94.142** |

`Potential missing data` có 553 observation trong ChEMBL nhưng 0 bản ghi: không dòng nào
trong số đó qua được bộ lọc của mục 2.

**Audit cột này (22/10/2026).** Dựng lại bằng đường đi độc lập — parse JSON mọi dòng thay
vì chọn trước bằng regex, và gom bằng vòng lặp Python thay vì theo offset — rồi so từng
dòng với cả ba file: số bản ghi mang cảnh báo khớp tuyệt đối (146.464 / 142.331 / 94.142),
**0 dòng null mà đáng ra phải có cảnh báo, 0 dòng có cảnh báo mà không được phép, 0 giá
trị sai.**

Đối chiếu chéo với `plausible_range`: trong 15.031 bản ghi `plausible_range = False`,
11.928 (79,4 %) mang cảnh báo của nguồn; 3.103 (20,6 %) không mang, và toàn bộ 3.113 phép
đo ngoài khoảng của chúng **đến từ BindingDB, 0 từ ChEMBL**. Hai dấu hiệu vì thế không mâu
thuẫn: giá trị ChEMBL bất khả thi thì luôn đã bị ChEMBL gắn cảnh báo, phần còn lại là giá
trị BindingDB mà nguồn đó không có cột chất lượng để gắn.

## 10. Output

```
data/dataset/
├── full/                                 (bước 1 → 4)
│   ├── molecules.parquet                 khóa phân tử -> SMILES đại diện, số dòng, cờ -REL
│   ├── proteins.parquet                  protein_hash -> chuỗi, accession, tên, độ dài
│   ├── shards/<endpoint>/bNN.parquet     dòng đã lọc (trung gian)
│   ├── replicates/<endpoint>/bNN.parquet bước 1
│   ├── assay_deltas/<endpoint>/bNN.parquet   bước 3
│   ├── lost_relative_pairs/<endpoint>/bNN.parquet  cặp mất vì quy tắc |r|
│   ├── records/<endpoint>/bNN.parquet    bước 4
│   └── full_manifest.json                tham số, phễu lọc, số đếm, thời gian từng chặng
├── protein_split_input.fasta             đầu vào cho công cụ gom chuỗi
├── protein_split.parquet                 split protein (sau khi có kết quả gom)
├── protein_split_manifest.json
├── evidence_records.parquet
├── evaluator_training.parquet
├── generator_training.parquet
└── export_manifest.json

runs/dataset/
├── smoke_<thời điểm>/                    output của một lần smoke (cùng cấu trúc full/)
├── <mode>_<thời điểm>.log
└── verify_<thời điểm>.json
```

Thư mục `full/` được dựng dưới tên `full.partial` rồi mới đổi tên, nên một lần chạy bị ngắt
không để lại thư mục trông như đã hoàn chỉnh. `--mode full` từ chối chạy nếu `full/` đã có,
trừ khi thêm `--force`.

## 11. Cách chạy

```powershell
# 0. Unit test (không cần dữ liệu thật)
python -m pytest -q

# 1. smoke: ~1/200 số assay -> runs/dataset/smoke_<thời điểm>/ và ước lượng thời gian full
python run_dataset.py --mode smoke

# 2. full: toàn bộ bước 1 -> 4 vào data/dataset/full/
python run_dataset.py --mode full            # thêm --force để ghi đè output cũ

# 3a. split: ghi FASTA đầu vào rồi thoát mã 2 (bước bắt buộc để có FASTA cho MMseqs2)
python run_dataset.py --mode split
# 3b. gom chuỗi bằng MMseqs2 trong WSL (xem mục 8.1 cho câu lệnh đầy đủ), rồi:
python run_dataset.py --mode split --clusters runs\dataset\mmseqs\p50_cluster.tsv `
    --cluster-tool mmseqs2 --cluster-tool-version <dòng `mmseqs version` in ra>

# 4. export: hai file training + file evidence vào data/dataset/
python run_dataset.py --mode export

# 5. verify: các phép kiểm tra -> runs/dataset/verify_<thời điểm>.json
python run_dataset.py --mode verify
```

### Đo được ở lần smoke (`runs/dataset/smoke_20261021_184545`, 21/10/2026)

| mục | giá trị |
|---|---|
| lọc xong (trước khi lấy mẫu) | 2.972.507 dòng (Ki 659.231; Kd 105.212; IC50 1.935.645; EC50 272.419) |
| mẫu smoke (1/200 số assay) | 16.273 dòng, 822 protein, 14.888 khóa phân tử |
| nhóm bước 1 | 15.725 (15 nhóm `multi_source_id`) |
| Δp trong assay | 59.407 (mất 288 cặp vì lệch cờ `\|r\|`, 0,48 %) |
| bản ghi bước 4 | 59.406, trong đó 59.405 là `single_assay` |
| SMILES không đọc được | 0 |
| cắt bị loại vì điểm nối | 2.533 |
| thời gian | gom dòng 7,4 s; bước 1→4 113,2 s; tổng 2,0 phút |

Lưu ý khi đọc bảng trên: smoke lấy mẫu **theo assay** (giữ nguyên toàn bộ assay được
chọn), nên những đại lượng tính trong một assay (số nhóm, số Δp, số cặp mất) là mẫu không
chệch và nhân 200 ra được cỡ của full: khoảng **11–12 triệu Δp assay**. Nhưng mọi đại lượng
**giữa các assay** thì vô nghĩa ở smoke: một cặp đo ở k assay chỉ vào được mẫu khi ≥ 2
trong k assay đó cùng rơi vào 1/200, xác suất ~ C(k,2)/40.000. Vì thế 99,998 % bản ghi là
`single_assay` và bảng QC/phân tán (mục 7) chỉ có 1 điểm đo — **không** kết luận được gì về
QC, ngưỡng quarantine hay tỷ lệ so với Nelen et al. 2025 trước khi chạy full.

### Đo được ở lần full (`data/dataset/full`, 21/10/2026)

| mục | giá trị |
|---|---|
| dòng vào / nhóm bước 1 | 2.972.507 / 2.876.220 (3.555 nhóm `multi_source_id`) |
| phân tử / protein (bảng tra) | 1.306.158 khóa / 11.835 chuỗi |
| phân tử / protein có mặt trong bản ghi | 959.239 / 6.315 (đo 24/10/2026) |
| protein thực sự có cặp (đầu vào của split) | 6.315 / 11.835 (53,4 %); 5.520 protein không tạo được cặp MMP nào |
| Δp trong assay | 13.302.366 |
| **bản ghi bước 4** | **10.546.330** |
| cặp mất vì lệch cờ `\|r\|` | 164.156 (1,2 % số Δp assay) |
| SMILES không đọc được | 0; cắt bị loại vì điểm nối 427.382 |
| thời gian | gom dòng 16,2 s; bước 1→4 4.343 s; **tổng 1 giờ 13 phút** |
| đĩa | 1,8 GB (`records` 813 MB, `assay_deltas` 696 MB, `shards` 112 MB) |

QC và phân tán (tỷ lệ tính trên số bản ghi đo được phân tán):

| endpoint | pass | single_assay | quarantine | đo được phân tán | ≤ 0,3 | > 1,0 |
|---|---|---|---|---|---|---|
| Ki | 301.738 | 1.335.938 | 4.477 | 306.215 (18,6 %) | 94,0 % | 1,5 % |
| Kd | 20.848 | 154.907 | 1.535 | 22.383 (12,6 %) | 76,6 % | 6,9 % |
| IC50 | 1.417.398 | 6.525.165 | 15.664 | 1.433.062 (18,0 %) | 85,0 % | 2,7 % |
| EC50 | 96.667 | 670.098 | 1.895 | 98.562 (12,8 %) | 86,2 % | 3,5 % |
| tổng | 1.836.651 | 8.686.108 | 23.571 | 1.860.222 (17,6 %) | 86,4 % | 2,6 % |

Ba điều đọc được từ bảng này:

1. **17,6 % bản ghi có ≥ 2 assay**, nên QC tầng 2 có việc thật để làm. Con số gần 100 %
   `single_assay` của smoke là ảo giác lấy mẫu như mục trên đã nói, không phải tính chất
   của dữ liệu.
2. **Kd tái lập kém nhất** (76,6 % ≤ 0,3; 6,9 % > 1,0) dù ngưỡng quarantine của nó là 1,0 —
   khớp với σ mỗi phép đo đo ở Stage 2a (Kd 0,57, cao nhất trong Ki/Kd).
3. Quarantine rất ít (23.571, tức 1,27 % số đo được phân tán) vì ngưỡng ≈ 2σ. Kiểm tra
   chéo: Ki ngưỡng 1,0 → "> 1,0" là 1,5 % × 306.215 ≈ 4.593, so với quarantine 4.477; IC50
   ngưỡng 1,4 nên "> 1,0" (38.692) nhiều hơn quarantine (15.664), đúng như mong đợi.

Về ước lượng thời gian: con số **6,3 giờ** suy từ smoke (113,2 s × 200) sai 5,2 lần và sai
cả chiều — nó được gọi là cận dưới nhưng thực tế là cận trên. Lý do: ở smoke mỗi assay
được chọn gần như không dùng lại phân tử nào nên cache cắt mảnh vô dụng, và 64 bucket nhỏ
đều phải trả phí khởi động process pool. Rút ra: phép nhân thẳng từ smoke chỉ dùng được
cho **số lượng**, không dùng được cho **thời gian**. Để so sánh, Stage 2a trên cùng dữ
liệu: dedup 5,8 phút, audit 0,7 phút.

### Đo được ở split, export và verify (22/10/2026)

Split (`data/dataset/protein_split.parquet`, `protein_split_manifest.json`): 6.315 protein,
**2.782 nhóm**, 10.546.330 cặp.

| partition | nhóm | protein | cặp | mục tiêu | thực tế | median cặp/protein |
|---|---|---|---|---|---|---|
| train | 1.024 | 3.091 | 8.437.064 | 80,0 % | 80,0 % | 61 |
| validation | 879 | 1.655 | 1.054.633 | 10,0 % | 10,0 % | 46 |
| test | 879 | 1.569 | 1.054.633 | 10,0 % | 10,0 % | 42 |

Bốn điều đọc được:

1. **Tỷ lệ khớp đúng 80/10/10, lệch 0,00 điểm %.** Đạt được vì nhóm lớn nhất chỉ chiếm
   **4,55 %** số cặp (top 10 nhóm cộng lại 17,89 %), tức không có nhóm nào vượt 10 % để
   phá được mục tiêu. Nỗi lo "một nhóm chiếm hơn 10 % nên validation/test không đạt 10 %"
   nêu ở mục 8 **không xảy ra với dữ liệu này**; nó vẫn đúng về nguyên tắc nếu sau này có
   nhóm lớn hơn.
2. **Phân bố nhóm lệch mạnh nhưng không có nhóm khổng lồ**: 1.722/2.782 nhóm chỉ có một
   protein, nhóm lớn nhất 117 protein; median 118 cặp/nhóm, p90 7.232, p99 65.550.
3. **Train giữ 80 % số cặp nhưng chỉ 49 % số protein** (3.091/6.315), vì xếp nhóm lớn
   trước nên các nhóm nhiều cặp dồn về train. Hệ quả cần biết khi đánh giá:
   validation/test gồm phần lớn protein ít dữ liệu (median 42–46 cặp/protein), nên chỉ số
   tính theo từng protein ở hai tập đó sẽ nhiễu hơn ở train.
4. **Đối chiếu split cũ**: 4.398 protein có ở cả hai, **43,0 %** cùng partition. Ma trận
   chuyển: old-train 3.490 protein → 1.641 train, 958 validation, 891 test. Mức khớp thấp
   là điều đã lường trước ở mục 8.2 (đổi global-NW sang MMseqs2 có coverage, cộng việc xếp
   partition theo số cặp của dataset mới); con số này chỉ để báo cáo, không dùng để kết
   luận split nào đúng hơn.

Export (`data/dataset/`): evidence 10.546.330 dòng; evaluator 10.522.759 (= toàn bộ trừ
23.571 bản ghi `quarantine`); generator 6.414.841 (60,9 % số bản ghi đủ điều kiện, tức
phần có |Δp trung bình| > 0,3).

Verify (`runs/dataset/verify_20261022_012955.json`): **7/7 PASS**, với quy mô kiểm tra:

| phép kiểm tra | quy mô đã kiểm | kết quả |
|---|---|---|
| `bao_toan_cap` | mọi (endpoint, bucket) | PASS |
| `thong_ke_tu_evidence` | 10.546.330 bản ghi tính lại từ danh sách Δp | 0 lệch |
| `khoa_endpoint_protein` | 23.848.696 dòng (Δp assay + bản ghi) | PASS |
| `mot_delta_moi_assay` | 10.546.330 bản ghi | PASS |
| `doi_chieu_observations` | **50.000** dòng Δp mẫu → **90.336** nhóm dựng lại từ observation gốc | 0 lệch, 0 nhóm không tìm thấy |

(`run.verify_sample_rows` nâng từ 2.000 lên 50.000 ngày 22/10/2026; lần chạy
`verify_20261022_021936` dùng mức mới, cả bảy phép kiểm vẫn PASS.)
| `split` | 5 kiểm tra của mục 8 | PASS |
| `export` | 3 file so với bản ghi | PASS, 0 dòng generator dưới ngưỡng gain |

Phân bố Δp trung bình mỗi bản ghi (median đúng 0,00 ở cả bốn endpoint là **kiểm tra tỉnh
táo**, không phải phát hiện: chiều của cặp do thứ tự khóa phân tử quyết định chứ không do
hoạt tính, nên phân bố buộc phải đối xứng):

| endpoint | min | P1 | P25 | median | P75 | P99 | max |
|---|---|---|---|---|---|---|---|
| Ki | −26,59 | −2,34 | −0,45 | 0,00 | 0,45 | 2,36 | 26,32 |
| Kd | −7,40 | −2,42 | −0,44 | 0,00 | 0,43 | 2,34 | 9,10 |
| IC50 | −12,85 | −2,20 | −0,41 | 0,00 | 0,41 | 2,20 | 13,49 |
| EC50 | −10,68 | −2,21 | −0,45 | 0,00 | 0,45 | 2,19 | 10,66 |

Đuôi cực đoan rất mỏng nhưng có thật: trong 10,5 triệu bản ghi có 50.764 bản ghi
|Δp| > 3, 7.888 bản ghi > 4, 681 bản ghi > 6 và **58 bản ghi > 10**. Chênh 10 đơn vị log
giữa hai phân tử khác nhau một nhóm thế gần như chắc chắn là lỗi đơn vị ở nguồn, không
phải SAR. Chưa lọc (đề bài không yêu cầu) và cũng chưa truy từng dòng về nguồn.

Số assay mỗi bản ghi trong phần đủ điều kiện (10.522.759 bản ghi):

| endpoint | 1 assay | 2 assay | ≥ 3 assay |
|---|---|---|---|
| Ki | 1.335.938 | 247.153 | 54.585 |
| Kd | 154.907 | 16.023 | 4.825 |
| IC50 | 6.525.165 | 1.069.588 | 347.810 |
| EC50 | 670.098 | 63.634 | 33.033 |
| tổng | 8.686.108 | 1.396.398 | 440.253 |

Nghĩa là trong số bản ghi đo được phân tán, **76 %** dùng quy tắc hai assay
(|Δ_X − Δ_Y|) và chỉ 24 % dùng STD của ≥ 3 assay. Lưu ý √2 ở mục 7 vì thế áp dụng cho đa
số bản ghi đã qua QC, không phải cho thiểu số.

Bộ nhớ: không có bucket nào gây thiếu bộ nhớ ở `protein_buckets = 16`. Nặng nhất là
IC50/b12 (201.182 dòng → 1.191.161 Δp assay → 952.889 bản ghi, 201,5 s). Dự đoán từ smoke
là IC50/b06 — sai, vì mẫu 1/200 không chỉ đúng bucket nặng nhất. Nếu sau này thiếu bộ nhớ
thì tăng `delta.protein_buckets` và chạy lại từ `--mode full`.

## 12. verify kiểm tra gì

| phép kiểm tra | nội dung |
|---|---|
| `bao_toan_cap` | mỗi (endpoint, bucket): số dòng Δp assay = tổng `n_assays` của các bản ghi; số bản ghi = số `pair_id` khác nhau |
| `thong_ke_tu_evidence` | tính lại mean, median, STD, min, max, phân tán, cờ QC của **mọi** bản ghi từ danh sách Δp đã lưu |
| `khoa_endpoint_protein` | không bản ghi nào trộn endpoint hay protein; protein thuộc đúng bucket; khóa A < khóa B; `pair_id` tính lại khớp |
| `mot_delta_moi_assay` | bảng Δp assay không có `(pair_id, assay_key)` lặp; danh sách `assay_keys` của bản ghi không lặp và dài đúng `n_assays` |
| `doi_chieu_observations` | lấy mẫu `run.verify_sample_rows` dòng Δp, quét lại hai bản sao dedup với đúng bộ lọc, gộp replicate lại từ đầu, so `p_a`, `p_b`, `Δp` và danh sách `observation_id` |
| `split` | 1) không nhóm nào ở hai partition; 2) không protein nào ở hai partition; 3) tổng số cặp theo partition = tổng của dataset (đếm lại từ bản ghi); 4) tỷ lệ thực tế so với mục tiêu và mức lệch; 5) đối chiếu split cũ: chỉ báo % protein chung được xếp cùng partition |
| `export` | số dòng evidence = số bản ghi; số dòng evaluator = số bản ghi đủ điều kiện; generator không có dòng nào `gain ≤ 0,3` |

Ngoài ra verify in phân bố Δp theo endpoint (phân vị của Δp trung bình mỗi bản ghi và của
Δp từng assay) và số bản ghi theo cờ QC. `--mode verify` trả mã thoát 1 nếu có phép kiểm
tra không đạt.

### Tính xác định của pipeline (22/10/2026)

Chạy `--mode smoke` hai lần độc lập (`smoke_20261021_184545` và `smoke_20261022_015945`,
cách nhau 7 giờ, cùng máy, cùng config): **322/322 file dữ liệu trùng SHA-256 từng byte**.
File duy nhất khác là `full_manifest.json`, và khác đúng ở `generated_at`, `output_dir` và
các trường đo thời gian; mọi con số dữ liệu trong manifest giống hệt. Smoke đi qua đúng
cùng đường code với full (chỉ khác bộ lọc mẫu), gồm cả phần ghép cặp chạy song song 8 tiến
trình, nên đây là bằng chứng cho việc kết quả không phụ thuộc thứ tự hoàn thành của các
tiến trình con. Chưa kiểm tính xác định của `full` (sẽ mất thêm ~1,2 giờ và 1,8 GB).

### Audit bổ sung ngoài verify (22/10/2026)

Những phép kiểm tra sau **không** nằm trong `--mode verify`; chúng chỉ đọc file output và
được chạy một lần để chuẩn bị nộp bài. Tất cả đều sạch.

Cấu trúc, trên **toàn bộ** 10.546.330 bản ghi:

| kiểm tra | kết quả |
|---|---|
| `pair_id` phân biệt | 10.546.330 = số dòng, 0 trùng |
| cặp tự ghép (A = B), thứ tự A < B, SMILES hai bên trùng nhau | 0 / 0 / 0 |
| lõi rỗng, `variable_a` = `variable_b` | 0 / 0 |
| cặp lệch cờ lập thể tương đối (một bên `-REL`) | 0 (320.454 bản ghi có **cả hai** bên `-REL`, đúng quy tắc) |
| null trong 14 cột khóa | 0 |
| protein nằm ở hai partition | 0 |

Logic QC và danh sách dẫn chứng, trên toàn bộ bản ghi: độ dài `delta_p_values` và
`assay_keys` đều bằng `n_assays` (0 lệch); không bản ghi nào có `assay_key` trùng; `n = 1`
luôn là `single_assay` với phân tán NaN; cờ `pass`/`quarantine` khớp đúng ngưỡng endpoint;
`n = 2` có phân tán bằng |Δ_X − Δ_Y|; `n ≥ 3` bằng STD (ddof = 1); `n_independent_assays`
không bao giờ lớn hơn `n_assays`; Δp trung bình luôn nằm trong [min, max].

Liên kết giữa các file:

| kiểm tra | kết quả |
|---|---|
| header FASTA = SHA-256 của chính chuỗi dưới nó | 6.315/6.315 khớp, 0 chuỗi rỗng |
| tập protein FASTA = tập thành viên TSV của MMseqs2 = `protein_split` | bằng nhau |
| dòng có `partition` khác `protein_split` | 0 |
| số cặp mỗi protein trong split = số bản ghi thật | khớp |
| một `molecule_key` đi với nhiều SMILES | 0 (973.003 khóa) |
| evaluator ⊂ evidence, không trùng `pair_id` | 0 dòng lạ, 0 trùng; 23.571 dòng bị loại đúng là `quarantine` |
| generator so với evaluator (6.414.841 dòng) | 0 `pair_id` lạ, 0 dòng `gain ≠ |Δp|`, 0 dòng `gain ≤ 0,3`, 0 dòng sai chiều yếu→mạnh |

Ba cờ mô tả của mục 9.1 (audit sau lần export 22/10/2026):

| kiểm tra | kết quả |
|---|---|
| `unseen_pair` của dòng train | 0/8.437.064 — đúng bất biến |
| `unseen_pair` của validation và test | 894.182 và 817.391, **khớp từng đơn vị** với con số tính độc lập trước khi viết cờ (1.054.633 − 160.451 và − 237.242) |
| `plausible_range` tính lại từ Δp assay cho **toàn bộ** endpoint Ki | 1.642.153/1.642.153 bản ghi, 0 lệch, 0 bản ghi thiếu biên |
| `short_protein` tính lại từ bảng độ dài protein | 10.546.330 dòng, 0 lệch; tổng 1.638 khớp bảng split |
| ba cờ giống nhau giữa evidence, evaluator, generator (endpoint Ki) | 1.637.676 và 1.029.625 dòng, 0 khác, 0 `pair_id` lạ |

Hóa học, mẫu 2.880 bản ghi rải khắp file (RDKit): ghép `constant_smiles` với `variable_a`
dựng lại đúng `smiles_a`, và với `variable_b` dựng lại đúng `smiles_b` — **0 lệch**; lõi
luôn ≥ 60 % số nguyên tử nặng của cả hai phía và phần biến đổi luôn ≤ 10 nguyên tử nặng —
0 vi phạm. Lưu ý kỹ thuật cho ai dùng lại dataset: `Chem.molzip` trả về phân tử có nhãn
lập thể chưa chuẩn hóa, **phải** gọi `Chem.AssignStereochemistry(mol, cleanIt=True,
force=True)` sau khi ghép; không gọi thì ~2 % trường hợp ra SMILES chính tắc khác dù
InChIKey vẫn giống (đã kiểm: cùng InChIKey).

#### Bốn điểm dữ liệu cần quyết định, không phải lỗi pipeline

1. **pActivity bất khả thi, đến từ nguồn chứ không từ Stage 2b.** Nhóm bước 1 có Ki nhỏ
   nhất −19,00 và lớn nhất 18,22; IC50 lớn nhất 19,69. Có 442 nhóm p > 12, 25 nhóm p > 15
   và ~6.500 nhóm p < 2. Trong chính file dedup đầu vào: 2.076 dòng ChEMBL và 178 dòng
   BindingDB có p < 0; 46 và 1 dòng có p > 15 — tức Stage 2b chỉ chuyển tiếp nguyên trạng.
   Đây là nguồn của đuôi |Δp| > 10 (58 bản ghi). **Đã làm ngày 22/10/2026**: cờ
   `plausible_range` ở mục 9.1, vẫn không lọc dòng nào.
2. **Trùng hóa học giữa các partition — theo thiết kế, nhưng phải báo cáo.** Split chia
   theo protein, nên cùng một cặp (A, B) có thể xuất hiện ở hai partition với protein khác
   nhau: **22,5 %** dòng test và 15,2 % dòng validation có cặp (A, B) cũng xuất hiện trong
   train; 25,6 % phân tử của test và 21,1 % của validation cũng có trong train. Tập đánh
   giá nghiêm ngặt "cặp chưa từng thấy" là 77,5 % số dòng test. **Đã làm ngày
   22/10/2026**: cờ `unseen_pair` ở mục 9.1; phần **phân tử** lẻ thì chưa có cờ riêng.
3. **Mức tập trung theo protein.** Protein nhiều cặp nhất chiếm 1,49 % số cặp; top 10
   chiếm 11,1 %, top 50 chiếm 31,6 %, top 100 chiếm 44,7 %. Theo assay thì phân tán hơn
   nhiều: 104.312 assay, assay lớn nhất 0,51 %, top 10 chỉ 3,34 %, median 11 Δp mỗi assay.
4. **21 chuỗi protein ngắn dưới 50 aa** (11 chuỗi dưới 30 aa, ngắn nhất 9 aa), mang 1.638
   cặp tức 0,02 %. Gần như chắc là peptide hoặc mẩu chuỗi lỗi; gom chuỗi ở ngưỡng
   identity 50 % với coverage 0,8 không có nghĩa nhiều với chuỗi dài 9 aa. **Đã làm ngày
   22/10/2026**: cờ `short_protein` ở mục 9.1.

Thêm hai con số để viết bài: trong các bản ghi đo được phân tán, **76 %** chỉ có 2 assay
(dùng |Δ_X − Δ_Y|) và 24 % có ≥ 3 assay (dùng STD). Và 176.219 bộ ba (A, B, protein) xuất
hiện dưới từ 2 endpoint trở lên (173.249 bộ ở 2 endpoint, 2.960 ở 3, 10 ở 4); chúng luôn
cùng partition vì cùng protein, nên không rò rỉ, nhưng các dòng đó tương quan với nhau khi
tính chỉ số.

## 13. Giả định đã chọn, cần xác nhận

1. **SMILES đại diện của một phân tử**: một khóa định danh có thể đi với nhiều SMILES chính
   tắc khác nhau (vd. khác dạng muối đã bị tách khác nhau). Lấy SMILES nhỏ nhất theo thứ tự
   chuỗi trong toàn bộ dữ liệu đã lọc, để một khóa chỉ có một cấu trúc khi cắt mảnh.
2. **Hậu tố `-REL` dùng cả ở bước 1**, không chỉ khi ghép cặp: hai dạng (tương đối và tuyệt
   đối) là hai nhóm replicate khác nhau, nên giá trị của chúng không bị trộn.
3. **Không sinh transform thay thế hydro**: single cut không cắt được liên kết C–H ẩn, nên
   cặp kiểu "thêm một nhóm vào vị trí trống" không có trong dataset này.
4. **QC chọn theo `n_assays` (mọi assay)**, không theo `n_independent_assays`; số assay độc
   lập được lưu kèm để lọc lại sau nếu cần.
5. **Một Δp có thể trộn hai nguồn dẫn chứng**: một phía của cặp có thể là nhóm chỉ gồm dòng
   mang cờ `potential_duplicate` (`n_independent = 0`). Cặp như vậy được giữ nhưng
   `independent = false` ở bảng Δp assay. Nếu muốn loại, lọc theo cột này — hiện chưa loại.
6. **`--mode export` là chế độ thêm** so với bốn chế độ đề bài yêu cầu
   (`smoke|full|verify` cộng `split`): vì split phải dựng sau bước 4 và export phải đọc
   split, bước 5 không thể nằm trong `full`.
7. **`max_cut_bonds = 20`** là giới hạn số liên kết được thử cắt mỗi phân tử (tham số của
   RDKit); phân tử nhiều liên kết cắt được hơn 20 sẽ bị cắt thiếu. Giữ đúng con số đề bài.

## 14. Điểm chưa làm / chưa đo

- Đuôi |Δp| cực đoan (58 bản ghi > 10 đơn vị log) chưa được truy về dòng observation
  gốc; dataset chỉ gắn cờ `plausible_range`, không lọc.
- Chưa có cờ cho phân tử lẻ gặp lại giữa các partition (25,6 % phân tử của test); chỉ có
  cờ cho **cặp** (`unseen_pair`).
- Ba cờ của mục 9.1 không nằm trong `--mode verify`; chúng đã được audit riêng bằng vòng
  chỉ đọc (xem dưới), nhưng chạy lại export thì phải audit lại.
- Chưa đánh giá mô hình nào trên dataset này; chưa train.
- Chưa đọc được định dạng `.clstr` của CD-HIT; chỉ đọc TSV hai cột, đủ cho
  `mmseqs easy-cluster`.
- Lần gom chuỗi thử đầu tiên ngày 22/10/2026 (cov-mode 1, cascade, max-seqs mặc định) ra
  2.839 nhóm và **đã bị loại bỏ**; split dùng lần chạy thứ hai, 2.782 nhóm (mục 8.1).
- Chưa đo bộ nhớ thực tế của bucket lớn nhất. Nếu một (endpoint, bucket) quá lớn, tăng
  `delta.protein_buckets` và chạy lại từ `--mode full`.
- Số cặp của một assay nhiều phân tử tăng theo bình phương; chưa có giới hạn số cặp mỗi
  assay, vì đề bài không yêu cầu.
