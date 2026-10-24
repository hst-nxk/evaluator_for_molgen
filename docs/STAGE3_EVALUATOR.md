# Stage 3: Evaluator baseline (dự đoán Δp của một cặp chỉnh sửa)

Trạng thái ngày 23/10/2026:

- **Đã chạy**: `pytest` — **374 passed** sau khi có torch (350 trước đó cộng 24 test của `models.py` vốn bị bỏ qua khi thiếu torch), `--mode baseline` (số
  đo ở mục 4.5), `--mode prepare --skip-protein` (cache fingerprint, mục 8) và
  `--mode smoke --config no_protein` cho cả ba đầu ra (mục 6.2). torch 2.5.1+cu121 đã cài,
  chạy được trên GPU.
  `--mode prepare` đầy đủ cũng đã chạy: **chiều embedding ESM2 đo được là 480** (mục 7),
  cache cho 6.313 protein.
- **Chưa chạy**: `--mode train`, `--mode eval`, và cấu hình `finetune`. **Chưa có mô hình
  nào được train thật** — smoke chỉ dùng 20.000 dòng và 2 epoch, nên mọi con số của nó nói
  về *pipeline*, không nói về chất lượng mô hình.

## 1. Phạm vi v1

Mục tiêu duy nhất: **ra được một thứ chạy được từ đầu đến cuối**, để biết pipeline có khả
thi không. Không phải kiến trúc cuối.

Vì vậy v1 dùng biểu diễn đơn giản nhất:

| thành phần | v1 | KHÔNG thuộc v1 |
|---|---|---|
| phân tử | Morgan fingerprint (SMILES, 2D) | GNN, transformer trên SMILES |
| protein | chuỗi acid amin qua ESM2 | cấu trúc 3D, pocket, apo/holo |
| phép sửa | fingerprint của `variable_a`, `variable_b` | graph edit, reaction template |

Không thêm thành phần nào nếu chưa có lý do.

## 2. Đầu vào

```
data/dataset/evaluator_training.parquet   10.522.759 dòng, 16 cột (chỉ đọc)
data/dataset/full/proteins.parquet        11.835 chuỗi protein (chỉ đọc)
```

**Cập nhật 24/10/2026:** config giờ trỏ tới dataset v2 (`data/dataset_v2/…`, cache
`work/evaluator_cache_v2`). v2 đã dựng xong (evaluator 10.684.298 dòng), và `--mode prepare`
đã chạy trên v2: 986.011 SMILES phân tử, 71.806 SMILES nhóm thế, 6.490 protein, chiều 480.
Các số khác trong tài liệu này vẫn là của v1. Hai thay đổi code của Stage 3:

- `train.canonical_sequence`: chuỗi protein được bỏ khoảng trắng và viết hoa trước khi vào
  ESM2. Trước đó, 11 protein viết thường của BindingDB (1.387 dòng) bị tokenizer biến thành
  một token `<unk>` (`docs/AUDIT_20261024.md` mục 5).
- Lát đánh giá mới `independent_agreement` (`n_documents ≥ 2` và `n_distinct_values ≥ 2`):
  chỉ gồm bản ghi được hai tài liệu độc lập đo ra hai giá trị khác nhau. Lát `multi_assay`
  cũ vẫn giữ để so với v1, nhưng ở v1 phần lớn bản ghi trong lát đó là bản sao.

Mỗi dòng: `(smiles_a, smiles_b, protein_hash, endpoint) -> delta_p_mean`, kèm
`constant_smiles`, `variable_a`, `variable_b`, `partition`, `n_assays`, `qc_status` và bốn
cờ mô tả. Nhãn cần học là `delta_p_mean` = Δp = p(B) − p(A), có dấu.

**Partition `test` bị chặn ở mức code.** Mọi lần đọc dữ liệu đi qua
`data.read_partition`, hàm này gọi `data.check_partition` và hàm đó ném `ValueError` khi
partition là `test`. Đường duy nhất để đọc test là truyền `allow_test=True`, và hiện không
chỗ nào trong dự án truyền giá trị đó. Một lời nhắc trong tài liệu dễ bị bỏ qua hơn một
ngoại lệ.

## 3. Kiến trúc

```
fingerprint(smiles_a)   -> ligand encoder ─┐
fingerprint(smiles_b)   -> ligand encoder ─┤  CÙNG một bộ trọng số
fingerprint(variable_a) -> edit encoder   ─┼─> concat -> MLP -> Δp
fingerprint(variable_b) -> edit encoder   ─┤
embedding protein       -> MLP -> 128     ─┤
endpoint                -> embedding (16) ─┘
```

Bốn điểm thiết kế:

1. **Một ligand encoder dùng chung cho A và B.** Hai phía cùng là SMILES nên không có lý
   do để hai bộ trọng số khác nhau; dùng chung thì cùng một phân tử luôn được biểu diễn
   như nhau dù nó nằm ở phía nào của cặp. Có test kiểm tra không tồn tại encoder thứ hai.
2. **Edit encoder là nhánh riêng** cho chính phép sửa, lấy từ `variable_a` và
   `variable_b` (dataset đã có sẵn hai cột này từ Stage 2b).
3. **ESM2 có hai chế độ**: đóng băng (`frozen`) hoặc train cùng (`finetune`). Xem mục 5 —
   `frozen` chạy trước để có mốc so sánh, vì fine-tune chưa được chứng minh là tốt hơn.
4. **Giảm chiều protein xuống 128 trước khi concat**, để đầu dự đoán không bị một nhánh
   áp đảo. Chiều vào của phép giảm này là chiều thật của ESM2, **được đo lúc chạy**, không
   hardcode (mục 7).

Ligand encoder v1 là fingerprint → MLP. Chọn fingerprint thay vì GNN vì train nhanh và cho
baseline thật trong vài giờ. **Đây là lựa chọn của v1, không phải kết luận rằng fingerprint
tốt hơn GNN.** Đổi sang GNN là việc của v2.

## 4. Chỉ số

### 4.1 Regression

MAE, RMSE, R², Spearman toàn cục. Báo cáo thêm riêng từng endpoint (Ki, Kd, IC50, EC50).
R² tính quanh trung bình của nhãn thật nên có thể âm. Khi nhãn không có phương sai thì R²
và Spearman trả về `None` và in ra `-`, **không trả về 0.0** — một chỉ số không xác định
không được trộn vào trung bình như thể là kết quả thật.

### 4.2 Ranking, chỉ trong cùng một query

Một query là `(phân tử tham chiếu, protein, endpoint)` với nhiều ứng viên. Không bao giờ
xếp chung ứng viên của hai query khác nhau vào một bảng.

**Nhân đôi hai chiều.** Dataset lưu cặp theo chiều khóa A < B, nên nếu chỉ nhóm theo phía
A thì query bị cắt vụn theo thứ tự khóa phân tử chứ không theo hóa học. Mỗi dòng
`(A, B, Δp)` vì thế sinh hai hướng: `A->B` giữ dấu và `B->A` đổi dấu. Với một phân tử tham
chiếu cố định, xếp hạng ứng viên theo Δp đúng bằng xếp hạng theo hoạt tính của ứng viên,
nên đổi dấu là phép biến đổi hợp lệ chứ không phải bịa thêm dữ liệu. Tham số:
`metrics.expand_both_directions`.

Đo trên partition validation (1.052.791 dòng) để chọn tham số của chỉ số:

| | chỉ nhóm theo `smiles_a` | nhân đôi hai chiều |
|---|---|---|
| số query | 157.048 | **198.197** |
| query có ≥ 2 ứng viên | 116.080 (73,9 %) | **164.996 (83,2 %)** |
| phần cặp nằm trong query xếp hạng được | 96,1 % | **98,4 %** |
| trung vị số ứng viên/query | 3 | **5** |
| p90 / p99 số ứng viên | 16 / 52 | **25 / 79** |
| nhiều nhất | 166 | **178** |

### 4.3 Bốn điều buổi họp chưa chốt, nay chốt như sau

1. **Macro hay micro?** **Báo cáo cả hai**, vì hai con số trả lời hai câu hỏi khác nhau:
   macro cho mỗi query một phiếu như nhau, micro gộp mọi cặp nên query dài chi phối. Với
   tỷ lệ đoán đúng chiều, hai con số có thể lệch nhau nhiều.
   Ngoại lệ có lý do: **không có `spearman_micro`** — Spearman chỉ có nghĩa trong một
   query, gộp hạng của nhiều query vào một bảng chính là điều phải tránh. Và với top-k,
   macro và micro trùng nhau (mỗi query cho một giá trị 0/1), nên chỉ có một con số.
2. **Mẫu số là gì?** Ghi rõ từng cái, không gộp: `queries_total` (mọi query),
   `queries_ranked` (query có ≥ `min_candidates` ứng viên), `pairs_counted` (mẫu số của
   micro), `spearman_queries` (query tính được Spearman), và `top_k_queries` riêng cho
   **từng** k.
3. **Query chỉ có 1 ứng viên?** Không xếp hạng được, nên **loại khỏi mẫu số của ranking**
   và đếm riêng ở `queries_skipped_too_few_candidates`. Dòng của nó **vẫn tính regression**.
4. **k bằng bao nhiêu?** `top_k = [1, 3, 10]`. Lý do lấy từ số đo ở mục 4.2: trung vị 5
   ứng viên nên k = 1 và k = 3 có nghĩa với phần lớn query; k = 10 chỉ áp dụng cho 64.869
   query (32,8 %) nên mẫu số của nó được in riêng để không bị đọc lẫn.

Hai quy ước nhỏ, ghi để khỏi tranh luận lại:
- Cặp ứng viên có hai nhãn lệch nhau không quá `direction_tolerance` (mặc định 0) bị loại
  khỏi mẫu số đúng chiều: không có chiều đúng để so. Cặp mà mô hình dự đoán bằng nhau
  tính là **sai**, vì mô hình không chọn được.
- top-k: nhãn trùng ở đỉnh thì tính là trúng nếu **bất kỳ** ứng viên đạt nhãn lớn nhất nằm
  trong top-k dự đoán.

### 4.4 Mốc dưới: ba baseline tầm thường

Không có mốc dưới thì mọi con số đều không đọc được. `--mode baseline` tính chỉ số của ba
mô hình không học gì, trên đúng các lát cắt và đúng các chỉ số như mô hình thật:

| baseline | dự đoán | đọc thế nào |
|---|---|---|
| `zero` | Δp = 0 với mọi cặp | "phép sửa không làm gì". MAE của nó là mức mà mô hình phải vượt mới gọi là học được điều gì |
| `endpoint_mean` | trung bình Δp của endpoint, **ước lượng chỉ trên train** | mô hình chỉ biết endpoint, không biết phân tử hay protein |
| `random` | lấy mẫu có hoàn lại từ phân bố nhãn train, **seed cố định** | mốc 50 % cho xếp hạng: dự đoán độc lập hoàn toàn với dòng đang xét |

Tham số của `endpoint_mean` lấy từ partition train, không từ validation: một baseline ước
lượng tham số trên tập mà nó bị đánh giá thì không còn là mốc dưới trung thực. `random`
cũng lấy phân bố từ train, và lấy đúng phân bố nhãn thật chứ không phải một phân bố tự
chọn như N(0,1) — để không phải đặt thêm một tham số tùy ý nào.

**Hai mốc khác nhau cho hai loại chỉ số, không trộn lẫn:**

- **Regression**: `zero` và `endpoint_mean` là mốc. `random` sẽ có MAE *tệ hơn* cả hai, nên
  đừng đọc nó như mốc regression.
- **Ranking**: chỉ `random` là mốc. `zero` cho dự đoán bằng nhau với mọi ứng viên trong
  một query nên mọi cặp đều hòa, và theo quy ước ở mục 4.3 dự đoán hòa tính là sai — tỉ lệ
  đúng chiều của nó bằng **0 %**. Đó là sàn của quy ước, không phải sàn của bài toán.
  `endpoint_mean` cho **16,6 %** chứ không phải 0 %, vì một lý do tinh tế được giải thích ở
  mục 4.5 — con số đó cũng không phải tín hiệu. `random` cho dự đoán liên tục và độc lập
  nên hầu như không có hòa, và tỉ lệ đúng chiều của nó là mốc **50 %** thật để so.

Seed của `random` là `stage3.train.seed`, nên hai lần chạy `--mode baseline` cho đúng cùng
một bảng số.

Chế độ này không cần torch, không cần cache fingerprint và không cần ESM2: từ vựng được
dựng ngay từ bảng validation, vì baseline không dùng cấu trúc phân tử. Chạy được trước cả
`--mode prepare`.

### 4.5 Số đo mốc dưới, đã chạy 23/10/2026

`python run_evaluator.py --mode baseline`, toàn bộ partition validation (1.052.791 dòng),
seed 20261023. Khoảng 6 phút cho cả ba baseline, chỉ CPU.

Regression (macro/micro không áp dụng; đây là chỉ số trên từng dòng):

| baseline | lát cắt | n | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| `zero` | all | 1.052.791 | **0,5820** | 0,8314 | −0,0000 |
| `zero` | strict | 892.742 | 0,5873 | 0,8416 | −0,0000 |
| `zero` | multi_assay | 139.640 | **0,5281** | 0,7465 | −0,0003 |
| `zero` | clean | 1.035.364 | 0,5763 | 0,8203 | −0,0000 |
| `endpoint_mean` | all | 1.052.791 | 0,5821 | 0,8314 | −0,0000 |
| `endpoint_mean` | multi_assay | 139.640 | 0,5281 | 0,7465 | −0,0003 |
| `random` | all | 1.052.791 | 0,8823 | 1,1739 | −0,9936 |
| `random` | multi_assay | 139.640 | 0,8428 | 1,1171 | −1,2401 |

Ranking (`all`; `strict` và `clean` lệch không quá 0,002 nên không in lại):

| baseline | đúng chiều macro | micro | Spearman macro | top-1 | top-3 | top-10 |
|---|---:|---:|---:|---:|---:|---:|
| `zero` | 0,0000 | 0,0000 | – | 0,2294 | 0,4701 | 0,6263 |
| `endpoint_mean` | 0,1658 | 0,1664 | −0,0026 | 0,2298 | 0,4696 | 0,6283 |
| `random` | **0,4971** | 0,4973 | 0,0009 | **0,2308** | **0,4715** | **0,6240** |

Mẫu số: 164.996 query xếp hạng được trên 198.197 query (83,2 %); top-3 trên 142.097 query;
top-10 trên 64.869 query. Ba con số này trùng khít với phép đo độc lập làm trước khi viết
code (mục 4.2) bằng một script khác, nên hai cách dựng query cho cùng kết quả.

**Những gì các con số này nói:**

1. **Ngưỡng MAE phải vượt là 0,582** (lát cắt `all`) và **0,528** trên `multi_assay`. Một
   mô hình cho MAE 0,57 thì gần như chưa học được gì, dù nghe như một con số nhỏ.
2. **Độ trải của nhãn ≈ 0,83 log** (RMSE của `zero` khi trung bình nhãn ≈ 0). Trên
   `multi_assay` là 0,75 — nhãn gộp từ ≥ 2 assay trải hẹp hơn khoảng 10 %, phù hợp với kỳ
   vọng rằng gộp nhiều assay thì bớt nhiễu.
3. **Endpoint không mang thông tin biên nào về Δp.** `endpoint_mean` cho MAE 0,5821, tức
   *tệ hơn* `zero` ở chữ số thứ tư. Nguyên nhân: trung bình Δp của từng endpoint chỉ từ
   −0,0131 đến +0,0040, và MAE đạt cực tiểu tại **trung vị** chứ không tại trung bình. Hệ
   quả: đừng mong nhánh endpoint đóng góp gì nếu mô hình không học được tương tác giữa
   endpoint và cấu trúc.
4. **`random` cho R² ≈ −1 đúng như lý thuyết.** Hai mẫu độc lập cùng phân bố có
   Var(y − ŷ) = 2·Var(y), nên R² = 1 − 2 = −1; đo được −0,9936. Đây là một kiểm chứng chéo
   cho công thức R² trong `metrics.py`.
5. **`random` cho đúng chiều 49,71 %**, Spearman macro 0,0009. Chỉ số ranking vì thế không
   lệch, và mốc 50 % đã có thật để so.
6. **Sàn của top-k không nhỏ: 0,23 / 0,47 / 0,62.** Trung vị 5 ứng viên một query nên đoán
   bừa cũng trúng top-1 gần một phần tư số lần. Một mô hình đạt top-1 = 0,30 là hơn sàn
   rất ít, không phải "trúng 30 %".

**Vì sao `endpoint_mean` cho 16,6 % chứ không phải 0 %** — và vì sao con số đó cũng không
phải tín hiệu. Khi nhân đôi hai chiều (mục 4.2), dự đoán của hướng ngược được lấy bằng
−f(A, B). Với một hằng số m ≠ 0, dự đoán trong một query vì thế nhận **hai** giá trị +m và
−m tùy ứng viên đó đến từ hướng thuận hay hướng ngược, chứ không còn là hằng số. Hướng của
một dòng lại do thứ tự khóa phân tử A < B của Stage 2b quyết định, hoàn toàn không liên
quan tới nhãn — nên các cặp "lẫn hướng" được đoán đúng khoảng 50 %, còn các cặp cùng hướng
vẫn hòa và tính là sai. 0,1658 ≈ 0,5 × 0,33 khớp với việc khoảng một phần ba số cặp so sánh
được là cặp lẫn hướng. `zero` không bị hiện tượng này vì −0 = 0.

Đây là **hiện vật của bộ dự đoán hằng số**, không phải lỗi của chỉ số: với một mô hình thật
thì dự đoán thay đổi theo từng ứng viên nên không có kênh nào để thứ tự khóa lọt vào. Giải
thích trên suy ra từ cơ chế và khớp với số liệu, **chưa kiểm chứng trực tiếp**; cách kiểm
là chạy lại với `expand_both_directions = false`, khi đó `endpoint_mean` phải cho đúng 0 %.

## 5. Ba cấu hình

| cấu hình | nhánh protein | mục đích |
|---|---|---|
| `frozen` | ESM2 đóng băng, embedding tính sẵn và cache | baseline rẻ, chạy được trên máy cá nhân |
| `finetune` | ESM2 train cùng, trong đồ thị tính toán | theo gợi ý trong buổi họp |
| `no_protein` | bỏ hẳn nhánh protein | đối chứng: protein có đóng góp gì không |

### 5.1 Đọc kết quả `no_protein` thế nào — và không được đọc thế nào

`no_protein` là đối chứng, nên dễ bị đọc quá mạnh. Quy tắc:

- **Được phép kết luận**: nếu `frozen` không hơn `no_protein` một cách rõ ràng thì *biểu
  diễn protein này, trong kiến trúc này, ở ngân sách train này* không thêm gì đo được.
- **KHÔNG được kết luận**: "protein không đóng góp gì cho bài toán". Có ít nhất năm cách để
  protein thật sự quan trọng mà vẫn ra kết quả trên:
  1. mean-pool toàn chuỗi thành 480 chiều quá thô để phân biệt túi gắn kết — v1 cố tình
     không dùng 3D hay pocket (mục 1);
  2. MLP ghép có thể học cách bỏ qua một nhánh bị các nhánh khác áp đảo;
  3. tham số chưa được tinh chỉnh riêng cho từng cấu hình, nên so sánh là ở ngân sách
     chung chứ không ở điểm tốt nhất của mỗi cấu hình;
  4. 82,4 % nhãn chỉ dựa trên một assay (mục 11.5), nên nhiễu nhãn có thể nhấn chìm một
     hiệu ứng protein yếu;
  5. nhiều cặp MMP có SAR chuyển được giữa các protein, nên Δp của chúng vốn ít phụ thuộc
     protein — điều này là tính chất của bài toán, không phải của mô hình.

Để kết luận mạnh hơn thì phải so hai cấu hình trên **cùng dữ liệu, cùng ngân sách và cùng
protocol**, và nếu muốn công bằng tuyệt đối thì phải tinh chỉnh riêng từng cấu hình.

Chạy `frozen` trước để có mốc, rồi mới so với `finetune`. Ở `finetune`, ESM2 nằm ở một
nhóm optimizer riêng với learning rate nhỏ hơn hai bậc (`protein_learning_rate` = 1e-5 so
với 1e-3): ESM2 đã pretrain còn đầu dự đoán khởi tạo ngẫu nhiên, dùng cùng learning rate
sẽ phá trọng số pretrain trước khi đầu dự đoán kịp học.

## 6. Ablation: ba đầu ra

| đầu ra | thấy A và B cùng lúc | có nhánh edit | cách tính |
|---|---|---|---|
| `direct` | có | có | MLP trên cả hai phía cộng hai nhóm thế |
| `direct_no_edit` | có | **không** | MLP trên cả hai phía, bỏ nhánh edit |
| `twin` | không (tách rời) | không | chấm điểm riêng từng phân tử: Δp = s(B) − s(A) |

Ba đầu ra để tách được hai yếu tố khác nhau, thay vì trộn chúng vào một so sánh:

- `direct` so với `direct_no_edit` → tác dụng riêng của **nhánh edit**;
- `direct_no_edit` so với `twin` → tác dụng riêng của việc **tính Δ thẳng** thay vì lấy
  hiệu hai điểm độc lập.

**Ô thứ tư không tồn tại được.** Không có "twin có nhánh edit": một phép sửa là thuộc tính
của *cặp*, không phải của một phân tử, nên hàm chấm điểm một phân tử không có chỗ đặt nó.
Ablation vì thế là ba ô của một bảng 2×2 thiếu một ô, và đó là giới hạn của thiết kế chứ
không phải thiếu sót của lần chạy.

`twin` có hai tính chất đúng do kiến trúc, đã có test: Δp = 0 khi hai phía là cùng một
phân tử, và đổi chỗ A với B thì Δp đổi dấu. Hai đầu ra `direct` **không** có hai tính chất
đó; dự án chọn **đo** mức lệch thay vì cưỡng chế (mục 6.1).

### 6.1 Đo mức bất đối xứng, không cưỡng chế

Cưỡng chế đối xứng (ví dụ lấy `f(A,B) − f(B,A)`) sẽ làm thay đổi chính thứ mà ablation ở
trên muốn đo, nên lựa chọn là giữ nguyên kiến trúc và **đo** mức lệch, rồi ghi vào phần
giới hạn của báo cáo. `--mode train` và `--mode eval` đều tính, trên một mẫu ngẫu nhiên có
seed cố định (`metrics.asymmetry_rows`, mặc định 200.000 dòng validation):

| số đo | ý nghĩa | giá trị lý tưởng |
|---|---|---|
| `antisymmetry_mean_abs` | trung bình \|f(A,B) + f(B,A)\|/2, tức phần đối xứng của dự đoán | 0 |
| `antisymmetry_p90`, `_max` | phân vị 90 và giá trị lớn nhất của cùng đại lượng | 0 |
| `identity_mean_abs` | trung bình \|f(A,A)\| khi phía B được thay bằng chính phía A | 0 |
| `antisymmetry_over_label_spread` | hai số trên chia cho độ lệch chuẩn của nhãn | 0 |
| `antisymmetry_over_prediction` | chia cho biên độ dự đoán trung bình | 0 |

Hai tỉ lệ cuối có mặt vì một con số tuyệt đối không đọc được nếu không có thước đo: lệch
0,1 log là nhỏ khi nhãn trải 1,5 log và lớn khi nhãn trải 0,2 log.

Hai điều phải nói rõ khi đọc số này:

1. **Cặp (A, A) không tồn tại trong dataset** — cặp MMP luôn gồm hai phân tử khác nhau.
   Đầu vào đó là nhân tạo, nên `identity_mean_abs` nói về khả năng ngoại suy ra ngoài phân
   bố huấn luyện, **không** nói về chất lượng khớp dữ liệu.
2. Với đầu ra `twin`, cả hai số phải ra đúng 0. Nếu không, đó là lỗi của code chứ không
   phải tính chất của mô hình — nên phép đo này cũng là một kiểm tra chéo.

Bài SMER báo cáo trên QM9 rằng mô hình dự đoán giá trị tuyệt đối đạt R² 0,997 mà dẫn hướng
tìm kiếm chỉ 16–46 %, còn dự đoán thẳng Δ đạt 62–86 %. **Nhãn QM9 tính bằng DFT nên không
có nhiễu thực nghiệm**, khác hẳn dữ liệu MOLOPT (σ mỗi phép đo 0,50–0,73 log, xem
`docs/DATASET_REPORT.md` mục 10). Con số đó vì thế **không được dẫn như thể đã đúng cho bài
toán này**; ablation ở đây để tự kiểm chứng.

### 6.2 Smoke đã chạy 23/10/2026 — pipeline thông, chưa nói gì về chất lượng

`--mode smoke --config no_protein`, cả ba đầu ra, trên GPU. 20.000 dòng train, 20.000 dòng
validation, 2 epoch, batch 128. **5 giây một epoch.**

**Bẫy phải biết trước khi đọc bảng dưới:** `limit` lấy các dòng **đầu tiên** của partition,
không lấy mẫu ngẫu nhiên (xem `data.read_partition`). Dataset được ghi theo thứ tự
(endpoint, bucket protein), nên 20.000 dòng đầu của validation là **100 % Ki và chỉ 101
protein**. Mốc dưới trên đúng tập đó là MAE **0,6372** và sd nhãn **0,8815** — khác hẳn
0,5820 và 0,8314 của toàn bộ validation ở mục 4.5. **Không được so số smoke với số ở mục
4.5.**

| đầu ra | train loss | MAE | so với sàn 0,6372 | R² | đúng chiều macro | Spearman macro |
|---|---:|---:|---|---:|---:|---:|
| `direct` | 0,0537 | 0,6623 | **tệ hơn** | −0,056 | 0,5256 | 0,0704 |
| `direct_no_edit` | 0,0558 | 0,6486 | **tệ hơn** | −0,020 | 0,5268 | 0,0741 |
| `twin` | 0,0529 | 0,6471 | **tệ hơn** | −0,016 | 0,5322 | 0,0876 |

Ba điều đọc được, và một điều **không** đọc được:

1. **Pipeline thông từ đầu đến cuối** trên cả ba đường forward khác nhau, có GPU, có
   checkpoint, có manifest. Đó là toàn bộ mục đích của smoke và nó đạt.
2. **Cả ba đầu ra đều tệ hơn "luôn đoán 0" về sai số tuyệt đối.** Không bất ngờ: train
   loss 0,053 ứng với RMSE train ≈ 0,33 so với RMSE validation 0,89 — mô hình ~2,7 triệu
   tham số học thuộc 20.000 dòng. Đây là lý do mốc dưới phải có trước khi train: nó biến
   "MAE 0,66" từ một con số vô nghĩa thành một kết luận dứt khoát.
3. **Nhưng xếp hạng thì vượt sàn.** Đúng chiều 0,526–0,532 so với sàn cấu trúc 0,50, và
   Spearman macro 0,070–0,088 so với sàn 0. Nhỏ nhưng cùng dấu ở cả ba đầu ra. Đây đúng là
   căng thẳng mà bài SMER nêu — xếp hạng được không kéo theo sai số tuyệt đối tốt, và
   ngược lại — xuất hiện ngay trên dữ liệu MOLOPT ở lần chạy đầu.
4. **Không đọc được: ablation.** Thứ tự twin < direct_no_edit < direct theo MAE là trên
   20.000 dòng và 2 epoch, trong chế độ học thuộc. Nó **không** là kết quả ablation. Kết
   quả ablation chỉ có sau `--mode train` đầy đủ.

**Số đo bất đối xứng, lần đầu có số thật** (mục 6.1):

| đầu ra | \|f(A,B)+f(B,A)\|/2 | so với sd nhãn | \|f(A,A)\| | so với sd nhãn |
|---|---:|---:|---:|---:|
| `direct` | 0,0430 | 4,9 % | 0,0421 | 4,8 % |
| `direct_no_edit` | 0,0206 | 2,3 % | 0,0190 | 2,2 % |
| `twin` | **0,0000** | **0,0 %** | **0,0000** | **0,0 %** |

`twin` ra đúng 0 ở cả bốn số — **phép kiểm tra chéo đã đạt**: kiến trúc twin bảo đảm hai
tính chất đó, nên 0,0000 nghĩa là `swap_sides`, `identity_sides` và `asymmetry_metrics`
tính đúng. Và nhánh edit **làm mức bất đối xứng tăng gấp đôi** (2,3 % → 4,9 %), hợp lý vì
nó là đầu vào phụ thuộc chiều mạnh nhất. Hai con số này đo trên một mô hình đang học thuộc
20.000 dòng nên là **tạm thời**; số dùng cho paper phải lấy từ `--mode train` đầy đủ.

## 7. Chiều embedding ESM2 = 480, đã đo

Đo ngày 24/10/2026 bằng `--mode prepare`, đọc từ `model.config.hidden_size` của chính
`facebook/esm2_t12_35M_UR50D` sau khi nạp:

```
CHIỀU EMBEDDING ĐO ĐƯỢC: 480 (model facebook/esm2_t12_35M_UR50D)
```

**Con số 512 nhắc trong buổi họp là sai.** Nó là lời nhớ lúc họp, và vì code không bao giờ
hardcode chiều nên cái sai đó không lọt vào đâu được: nếu tôi đã viết 512 vào kiến trúc thì
`protein_projection` sẽ nhận sai chiều đầu vào và lỗi chỉ lộ ra ở lần train đầu.

480 giờ nằm trong ba chỗ, với ba vai khác nhau:

| chỗ | vai |
|---|---|
| `cache_manifest.json` → `protein_embeddings.embedding_dim` | **nguồn sự thật**, ghi lúc đo |
| metadata của `protein_embeddings.parquet` | để `train`/`eval` kiểm lại cache |
| `config.json` → `protein_encoder.embedding_dim` | **chốt kiểm**: `features.load_protein_encoder` báo lỗi nếu model có chiều khác |

Vai thứ ba đáng nói: nó không phải nguồn mà là cái phanh. Đổi `protein_encoder.model` sang
bản ESM2 khác mà quên sửa 480 thì chương trình **dừng** chứ không âm thầm chạy với cache
sai. Cùng lý do, `Caches` từ chối một cache được dựng bằng model khác với model trong config.

**Môi trường đã kiểm chứng:** torch 2.5.1+cu121 (CUDA build 12.1,
`torch.cuda.is_available() = True` trên RTX 3060 Laptop 6144 MiB), transformers 4.44.2,
tokenizers 0.19.1, safetensors 0.8.0. `pip check`: no broken requirements.

**Một cảnh báo của Hugging Face xuất hiện và là vô hại — nhưng chỉ vì thiết kế:**

```
Some weights of EsmModel were not initialized ... ['esm.pooler.dense.bias',
'esm.pooler.dense.weight']. You should probably TRAIN this model ...
```

ESM2 không có trọng số pooler trong checkpoint nên `transformers` khởi tạo **ngẫu nhiên**
lớp đó. Dự án không bao giờ đọc `pooler_output`: cả `features.embed_one_sequence` và
`train.ProteinEncoderRuntime.encode` đều lấy `last_hidden_state` rồi tự mean-pool trên
token acid amin (mục 8). Nếu code dùng `pooler_output` thì toàn bộ 6.313 embedding sẽ là
nhiễu ngẫu nhiên — và không có gì trong log báo cho ta biết. Đây là chỗ im lặng nguy hiểm
nhất của cả Stage 3, nên ghi lại rõ.

## 8. Thực tế tài nguyên

Đo ngày 23/10/2026: GPU là **NVIDIA RTX 3060 Laptop, 6 GB VRAM**, driver 617.42. RAM 16 GB.
Python 3.10.11.

6 GB VRAM là ràng buộc chính và nó ảnh hưởng tới lựa chọn model:

- `config` mặc định dùng `facebook/esm2_t12_35M_UR50D` (bản nhỏ nhất của ESM2). Lý do:
  `finetune` giữ ESM2 trong đồ thị nên bộ nhớ tỉ lệ với (số protein phân biệt trong batch ×
  số acid amin); bản 650M gần như chắc chắn không vừa 6 GB với batch có nghĩa.
- `frozen` thì model nào cũng chạy được vì ESM2 chỉ chạy một lượt rồi cache. **Chỉ có
  6.315 protein có cặp**, nên cache rất nhỏ: 6.315 × chiều × 4 byte (ví dụ chiều 480 là
  khoảng 12 MB). Đây là lý do `frozen` rẻ đến mức chạy được trên CPU.
- `finetune_overrides` trong config đặt `batch_size` 64 và `max_residues` 512. **Đây là
  phỏng đoán để tránh hết VRAM, chưa đo.** Nếu vẫn hết bộ nhớ: giảm `batch_size` trước,
  rồi mới giảm `max_residues`.

Chuỗi dài: trong **6.313 protein thực sự được dùng, 797 chuỗi dài hơn cửa sổ 1.022 aa**
nên phải chia cửa sổ (đo ngày 24/10/2026). Trên toàn bộ 11.835 chuỗi của Stage 2b thì có
1.409 chuỗi dài hơn 1.024 aa, dài nhất 34.350 aa (p50 = 455, p90 = 1.104). Hai cách xử lý,
chọn theo cấu hình:
- `frozen` dùng `long_sequence = "windows"`: chia chuỗi thành các cửa sổ liền nhau rồi lấy
  trung bình **có trọng số theo độ dài cửa sổ**, nên không mất phần đuôi chuỗi và kết quả
  bằng mean-pool trên toàn chuỗi nếu model không có hiệu ứng biên.
- `finetune` buộc dùng `"truncate"` (chỉ lấy `max_residues` acid amin đầu), vì bộ nhớ
  gradient tỉ lệ với độ dài. Mất phần đuôi của 1.409 protein là cái giá của cấu hình này
  và cần ghi vào báo cáo khi so sánh hai cấu hình.

**Chuẩn pooling, dùng chung cho cả hai cấu hình:** mean-pool **chỉ trên token acid amin**,
bỏ CLS, EOS và PAD. Chọn chuẩn này vì CLS và EOS không phải acid amin, nên đại lượng cần
lấy là biểu diễn trung bình của các acid amin; và vì cache `frozen` đã tính đúng như vậy
từ đầu, nên chọn nó thì **cache không phải dựng lại**. Trước ngày 23/10 đường `finetune`
pool theo `attention_mask` nên gồm cả CLS/EOS, làm hai cấu hình không so sánh được hoàn
toàn công bằng; `train.ProteinEncoderRuntime.encode` nay che cả ba token đặc biệt, giống
hệt `features.embed_one_sequence`.

**Thời gian đã đo được (23/10/2026, chỉ CPU):**

| bước | khối lượng | thời gian | kết quả trên đĩa |
|---|---|---:|---|
| `prepare`, fingerprint phân tử | 958.926 SMILES phân biệt | 374–388 s | 83,7 MiB parquet |
| `prepare`, fingerprint nhóm thế | 70.455 SMILES phân biệt | 10,5–11 s | 3,1 MiB parquet |
| `prepare`, embedding ESM2 | 6.313 protein trên GPU | 457 s | ~12 MiB parquet |
| tải trọng số ESM2 (một lần) | 136 MB | 140 s | trong `HF_HOME` |
| `--mode baseline` | 3 baseline × 4 lát cắt trên 1.052.791 dòng | ~360 s | manifest JSON |
| `--mode smoke` một đầu ra | 20.000 dòng, 2 epoch, GPU | ~16 s | checkpoint + manifest |

**0 SMILES không đọc được** trên cả hai cache, nên không có dòng fingerprint toàn 0 nào.

Con số phân tử ở đây là **958.926**, không phải 1.306.158. 1.306.158 là số dòng của **bảng
tra** `data/dataset/full/molecules.parquet` (1.285.061 SMILES phân biệt; phần chênh là các
khóa `-REL`), tức mọi phân tử đi vào Stage 2b. Số phân tử thật sự có trong bản ghi là
959.239 ở `evidence_records` và **958.926** ở `evaluator_training` (lệch 313 là phần chỉ
xuất hiện trong 23.571 bản ghi `quarantine`). Kế toán đầy đủ cho 1.306.158 khóa nằm ở
`docs/DATASET_REPORT.md` mục 6.2: 0 phân tử mất sau bước ghép cặp, 11.693 bị loại vì quy tắc
đã ghi, và **321.462 phân tử không tạo được MMP hợp lệ nào** dù nằm trong khối có phân tử
khác — tính đầy đủ của nhóm cuối này **chưa được kiểm chứng** (`DATASET_REPORT` mục 11.10).
Nó không ảnh hưởng tới tính đúng của dữ liệu train hiện có, chỉ tới việc dữ liệu đó có đủ
hay không.

**Thời gian train: số học từ số đo, không phải số đo.** Smoke xử lý 40.000 dòng (20.000
train + 20.000 validation) cộng chỉ số trên 6.089 query trong **5,4 giây** trên GPU ở batch
128. Lần train đầy đủ có 8.417.004 dòng train và 1.052.791 dòng validation, tức ~237 lần
khối lượng đó, và chạy ở batch 512 nên phần train nhanh hơn trên mỗi dòng; chỉ số trên
198.197 query mất khoảng 2 phút một lượt (đo ở mục 4.5). Cộng lại: **cỡ 15–25 phút một
epoch** cho cấu hình `frozen`, tức 1,5–4 giờ cho 10 epoch có dừng sớm.

Đây là phép nhân, không phải phép đo. Bài học từ Stage 2b là ước lượng thời gian của tôi
sai 5,2 lần và sai cả chiều, nên đừng lập kế hoạch dựa vào con số này; số thật sẽ có sau
epoch đầu tiên của lần train thật.

## 9. Cấu trúc file

```
src/molopt/evaluator/
├── __init__.py
├── data.py       # đọc partition, lọc lát cắt, mã hóa thành chỉ số, chia batch
├── features.py   # Morgan fingerprint, embedding ESM2, đọc/ghi cache
├── models.py     # ligand encoder dùng chung, edit encoder, ba đầu ra
├── metrics.py    # regression, ranking trong query, số đo bất đối xứng
└── train.py      # prepare, baseline, vòng train, đánh giá, checkpoint, manifest
run_evaluator.py  # CLI
work/evaluator_cache/   # cache fingerprint và embedding (không nằm trong repo)
runs/evaluator/         # log, checkpoint .pt, manifest JSON
```

Dữ liệu đi qua như sau:

1. `prepare` đọc dataset một lượt, lấy SMILES phân biệt (đo được: 958.926 phân tử và
   70.455 nhóm thế), tính Morgan fingerprint **một lần cho mỗi SMILES**, đóng gói bit
   (`np.packbits`, 2.048 bit → 256 byte) và ghi parquet. Rồi chạy ESM2 cho 6.315 protein
   và ghi parquet thứ hai.
2. `data.read_partition` đọc một partition, `data.encode_pairs` đổi mỗi dòng thành **chỉ
   số** vào ba bảng cache cộng nhãn. Một batch vì thế chỉ là một mảng số nguyên.
3. `train.batch_tensors` giải nén bit của đúng các dòng trong batch rồi dựng tensor. Bảng
   đầy đủ vẫn ở dạng gói trong RAM; RDKit **không** được gọi trong vòng train.
4. `models.EvaluatorModel` cho ra Δp; `metrics` tính chỉ số trên validation theo từng lát
   cắt; `train` ghi checkpoint tốt nhất theo MAE validation và một manifest JSON.

Cache được dựng trên **toàn bộ** dataset, kể cả partition test. Đây không phải rò rỉ:
fingerprint là hàm thuần của cấu trúc phân tử và embedding ESM2 frozen là hàm thuần của
chuỗi — không tham số nào được ước lượng từ dữ liệu của ta, nên biết trước fingerprint của
một phân tử test không cho mô hình thêm thông tin nào về nhãn. Nếu muốn chặt hơn thì dựng
cache riêng cho train+validation, nhưng khi đó phải dựng lại cache trước khi đánh giá trên
test. **Đây là giả định của tôi, cần bạn xác nhận.**

## 10. Lệnh chạy, theo thứ tự

Bước 0 — cài thư viện (chưa có trên máy). CUDA 12.1 cho RTX 3060:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
python -m pip install transformers
python -m pip install pytest   # nếu chạy test
```

Bước 1 — test (không cần GPU, không cần ESM2; file test của `models` tự bỏ qua khi chưa
có torch):

```powershell
python -m pytest -q
```

Bước 2 — ba mốc dưới. Chạy được **ngay bây giờ**, không cần torch và không cần cache:

```powershell
python run_evaluator.py --mode baseline
```

Bước 3 — cache fingerprint. Cũng chạy được ngay bây giờ, không cần torch:

```powershell
python run_evaluator.py --mode prepare --skip-protein
```

Bước 4 — smoke không cần ESM2, xác nhận pipeline thông trên CPU:

```powershell
python run_evaluator.py --mode smoke --config no_protein --head direct
```

Bước 5 — cache embedding protein (tải trọng số ESM2 lần đầu; in ra chiều thật):

```powershell
python run_evaluator.py --mode prepare
```

Bước 6 — baseline `frozen`, rồi các đối chứng và ablation:

```powershell
python run_evaluator.py --mode train --config frozen     --head direct
python run_evaluator.py --mode train --config no_protein --head direct
python run_evaluator.py --mode train --config frozen     --head direct_no_edit
python run_evaluator.py --mode train --config frozen     --head twin
```

Bước 7 — `finetune` (cần GPU; xem mục 8 về VRAM):

```powershell
python run_evaluator.py --mode train --config finetune --head direct
```

Bước 8 — báo cáo lại chỉ số của một checkpoint theo từng lát cắt (kèm số đo bất đối
xứng):

```powershell
python run_evaluator.py --mode eval --checkpoint runs\evaluator\frozen_direct_<thời điểm>.pt
```

## 11. Chỗ đáng lưu ý, chưa tự sửa

1. **Hai đầu ra `direct` không đối xứng** — đổi chỗ A với B không bảo đảm Δp đổi dấu, và
   cặp (X, X) không bảo đảm cho 0. Đây là **lựa chọn đã chốt**: đo chứ không cưỡng chế
   (mục 6.1), vì cưỡng chế sẽ làm thay đổi chính thứ mà ablation muốn đo. Số đo đầu tiên
   (smoke, nên tạm thời): mức lệch bằng **4,9 % sd nhãn** với `direct` và **2,3 %** với
   `direct_no_edit` — nhánh edit làm tăng gấp đôi. Hệ quả phải ghi vào phần giới hạn của
   paper, không được bỏ qua.
2. **Ablation chỉ có ba ô.** Ô "twin có nhánh edit" không định nghĩa được (mục 6), nên
   không thể tách hoàn toàn hai yếu tố trong một bảng 2×2 đầy đủ.
3. **`endpoint_mean` cho 16,6 % đúng chiều do thứ tự khóa A < B lọt vào qua phép nhân đôi
   hai chiều** (giải thích ở mục 4.5). Chỉ ảnh hưởng tới bộ dự đoán hằng số, không ảnh
   hưởng tới mô hình thật, và chưa kiểm chứng trực tiếp — cách kiểm là chạy lại với
   `expand_both_directions = false`.

4. **Lát cắt `strict` trên validation là 892.742 dòng, không phải 894.182.** Con số
   894.182 trong đề bài là của `evidence_records.parquet`; `evaluator_training.parquet` đã
   bỏ 23.571 bản ghi `quarantine` nên nhỏ hơn. Đo được: strict 892.742, nhiều assay
   139.640, sạch 1.035.364, strict ∧ sạch 877.852.
5. **82,4 % bản ghi có `n_assays = 1`**, nên nhãn của phần lớn dữ liệu chỉ dựa trên một
   assay. Lát cắt `multi_assay` chính là chỗ để thấy điều đó ảnh hưởng thế nào.
6. **`weight_decay` áp cho cả bias và LayerNorm.** Thông lệ là loại hai nhóm đó ra. Ảnh
   hưởng nhỏ ở v1, nhưng là chỗ cần sửa nếu so sánh cấu hình một cách chặt chẽ.
7. **Seed chỉ cố định ở mức numpy/torch**, không bật `cudnn.deterministic` vì nó làm chậm
   đáng kể. Hai lần chạy trên GPU có thể lệch nhau chút ít ở chữ số cuối.
8. **`--mode prepare` không kiểm cache đã có, nên dựng lại từ đầu mỗi lần.** Lần chạy
   24/10 tốn 374 s dựng lại hai cache fingerprint y hệt cái đã có từ hôm trước, chỉ để
   thêm phần ESM2. Cần một chốt "bỏ qua nếu đã có, trừ khi `--force`" như Stage 2b. Chưa
   làm; mỗi lần `prepare` sau này mất thêm ~6 phút vô ích.
9. **`frozen` chia cửa sổ còn `finetune` cắt chuỗi**, nên 797 protein dài (mục 8) được
   hai cấu hình thấy khác nhau. Đây là **yếu tố gây nhiễu cho ablation frozen-vs-finetune**:
   chênh lệch kết quả sẽ lẫn giữa "train cùng ESM2 có lợi không" và "mất đuôi chuỗi có hại
   không". Cách sửa sạch: dựng thêm **một biến thể cache thứ ba** — `frozen` nhưng
   `long_sequence = "truncate"` ở đúng `max_residues` của `finetune` — rồi so `finetune`
   với biến thể đó, giữ `frozen`+cửa sổ làm bản tốt nhất của đường đóng băng. Tốn thêm một
   lần `prepare` (~8 phút) và một lần train. **Chưa làm**; không chắn đường train `frozen`.

## 12. Nhật ký quyết định

Ngày 23/10/2026, sau rà soát danh sách ở mục 11 của bản trước:

| quyết định | kết quả |
|---|---|
| thêm `direct_no_edit` để tách nhánh edit khỏi cách tính Δ | đã làm (mục 6) |
| thêm hai baseline tầm thường `zero` và `endpoint_mean` | đã làm (mục 4.4) |
| thêm baseline `random` có seed cố định, để ranking có mốc 50 % | đã làm (mục 4.4) |
| bất đối xứng: **không** cưỡng chế, chỉ đo và ghi vào giới hạn | đã làm (mục 6.1) |
| chọn một chuẩn pooling dùng chung: chỉ acid amin, bỏ CLS/EOS/PAD | đã làm (mục 8) |
| ghi `torch` và `transformers` vào `requirements.txt` | đã làm; nay pin `torch==2.5.1` |
| giữ `esm2_t12_35M_UR50D` cho v1 | giữ nguyên |

Chạy trong ngày: `pytest` 350 passed / 1 skipped; `--mode baseline` (mục 4.5);
`--mode prepare --skip-protein` (mục 8); `--mode smoke` ba đầu ra (mục 6.2). Phép kiểm tra
chéo quan trọng nhất đã đạt: `twin` cho bất đối xứng đúng 0,0000. Hai số tôi từng nói sai
và đã sửa: số phân tử cần fingerprint (958.926, không phải 1.306.158) và tỉ lệ đúng chiều
của `endpoint_mean` (16,6 %, không phải 0 %).

Ngày 24/10/2026: cài transformers 4.44.2, chạy `--mode prepare` đầy đủ. **Chiều embedding
ESM2 = 480**, không phải 512 như nhắc trong buổi họp (mục 7). Số protein dùng thật là
6.313, không phải 6.315 — hai protein của `protein_split.parquet` chỉ còn cặp trong các bản
ghi `quarantine` đã bị loại. Thêm `.cache/` vào `.gitignore` vì `HF_HOME` được trỏ vào trong
repo. Kế toán đầy đủ chênh lệch phân tử (mục 8 và `DATASET_REPORT` mục 6.2): nhóm "có cặp
nhưng mất bản ghi" bằng 0, nhưng 321.462 phân tử không tạo được MMP nào và tính đầy đủ của
nhóm đó chưa kiểm chứng. Sửa lại hai con số sai trong `docs/DATASET_REPORT.md` mục 6.2 cùng
`docs/STAGE2B_DATASET.md` mục 11, và rút lại câu "không phải mất dữ liệu" đã viết vội ở lượt
trước — phép kiểm tập con không chứng minh được điều đó.

Tất cả là thay đổi code và tài liệu. **Chưa có lần chạy nào**, nên chưa có số đo nào của
mô hình trong tài liệu này.
