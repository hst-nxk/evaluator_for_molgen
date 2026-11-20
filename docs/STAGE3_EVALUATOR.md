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

**Nhưng điều đó KHÔNG có nghĩa là tập test của split 50 % còn nguyên.** `check_partition`
chỉ chặn việc *gọi tên* partition `test`. `data.read_columns` áp bảng đổi partition
**trước** bước lọc, nên các thí nghiệm đổi ngưỡng 30 %/80 % đã gán một phần dòng test của
split 50 % sang train và validation của chính chúng — những dòng đó đã tham gia huấn
luyện. Cách nói đúng: chưa có checkpoint nào được chấm trên partition test của split 50 %,
nhưng một phần tập đó đã bị dùng trong các thí nghiệm độ nhạy, nên nó không còn là tập
đánh giá độc lập hoàn toàn. Muốn một ước lượng độc lập cho cấu hình được chọn trên
validation thì phải tách một tập mới, chưa chạm bởi bất kỳ bảng đổi partition nào.

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

### 4.5.1 Mốc dưới đo lại trên dataset_v4 (31/10/2026)

Validation của v4 có 500.704 cặp; bỏ các dòng thiếu nhãn còn n = 493.852.

| baseline | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|
| `endpoint_mean` (trung bình theo endpoint) | **0,5880** | **0,8115** | −0,0000 | 0,0039 |
| `random` | 0,8760 | 1,1484 | −1,0026 | −0,0006 |

`endpoint_mean` là mốc phải vượt. Hai lần train thử ở máy cá nhân (`frozen direct`) **chưa
vượt được mốc này về MAE**: 0,6138 ở learning rate 1e-3 và 0,6247 ở 2e-4, cả hai đều dừng
sớm với MAE validation tăng dần. Nhưng cả hai **vượt rõ về xếp hạng** (Spearman 0,12 so với
0,004 của baseline; tỉ lệ đúng chiều 0,55 so với 0,50). Đọc đúng số này: mô hình học được
*thứ tự* tốt hơn hẳn mức ngẫu nhiên nhưng *giá trị tuyệt đối* còn kém cách đoán trung bình,
và nó overfit trong epoch đầu. Đó là lý do có mục 4.6.

## 4.6 Số hạng phạt chuẩn gradient đầu ra (IGNITE), tùy chọn

Nguồn: Dao MC, Nguyen PL, Truong TN, Hoang TN. *Incorporating Surrogate Gradient Norm to
Improve Offline Optimization Techniques*. NeurIPS 2024, DOI 10.52202/079017-0258,
arXiv:2503.04242. Code tham chiếu của tác giả: `github.com/cuong-dm/IGNITE`.

**Vì sao liên quan.** Evaluator của dự án đúng là một *surrogate offline*: nó học từ dữ
liệu đã có rồi sẽ bị generator hỏi ở những cặp chưa từng thấy. Bài báo chỉ ra chỗ hỏng của
kiểu mô hình này là *độ sắc* — đổi tham số một chút thì đầu ra đổi nhiều — và phạt chuẩn
gradient đầu ra làm nó bớt sắc, nhờ đó đáng tin hơn ngoài vùng dữ liệu. Đo trên v4 ngày
31/10/2026, vòng train thường **overfit ngay trong epoch đầu** (train loss 0,107 → 0,052
trong khi MAE validation tăng 0,614 → 0,627), và hạ learning rate từ 1e-3 xuống 2e-4
**không** cứu được (MAE 0,625). Nói cách khác vấn đề là tổng quát hóa, đúng chỗ bài báo
nhắm tới — nên phương pháp được cài vào code để **đo bằng ablation**, không phải để tin
sẵn.

**Công thức.** Bài báo ước lượng độ sắc bằng `rho * ||grad_omega h(omega)||` với
`h(omega)` là đầu ra trung bình của surrogate trên dữ liệu offline (công thức 10, 18), rồi
giải bài toán có ràng buộc `rho * ||grad_omega h|| <= epsilon` (công thức 11) bằng nhân tử
Lagrange `lambda` cập nhật theo dual ascent (công thức 12-14). Gradient của số hạng phạt
cần Hessian, nên bài báo xấp xỉ bằng sai phân hữu hạn bán kính `r` (công thức 83-85,
Algorithm 1) — chỉ cần gradient bậc nhất, nhưng thêm một lượt forward.

Một bước, đúng theo Algorithm 1 (code: `src/molopt/evaluator/sharpness.py`):

```
g1 = grad (mất mát trung bình)
g2 = grad (đầu ra trung bình)                  # = grad_omega h
omega_hat = omega + r * g2 / ||g2||
g3 = grad (đầu ra trung bình, tại omega_hat)
g  = g1 + lambda * (rho / r) * (g3 - g2)
lambda <- lambda + eta_lambda * (rho * ||g2|| - epsilon)
```

Có test ghim rằng cách xấp xỉ này được cài đúng: trên mô hình **tuyến tính** (Hessian bằng
0) số hạng phạt phải triệt tiêu đúng bằng 0 dù `lambda` lớn, còn trên MLP thì không.

**Cách chạy.** Mặc định **tắt** (`stage3.train.sharpness.enabled = false`) để các lần train
cũ vẫn so sánh được. Bật bằng dòng lệnh, không phải sửa config:

```
python run_evaluator.py --mode train --config frozen --head direct --sharpness on
```

`scripts/server_run_all.sh` chạy ablation: bốn lần train như cũ (`sharpness=off`) rồi
`frozen direct` và `no_protein direct` với `sharpness=on`, mọi tham số khác giữ nguyên.

**Tham số, lấy từ bài báo** (`stage3.train.sharpness`): `lambda_init` 0,01 (grid
{1e-4, 1e-3, 1e-2}, mục 5.1); `epsilon` 0,1 (mục 5.1, ablation {0,01 … 0,5});
`lambda_learning_rate` 1e-3 (hình 2b); `rho` = `r` = 0,05 (mục 5.1, grid
{0,01, 0,05, 0,1, 0,2}). **Bài báo tự mâu thuẫn ở chỗ này**: phụ lục G.2 ghi `rho = r = 0,2`
trên grid rộng hơn. Config lấy số của mục 5.1 và ghi lại mâu thuẫn chứ không chọn bừa.

**Bốn chỗ phải nói rõ khi viết paper:**

1. Bài báo cập nhật tham số bằng gradient descent thuần; dự án dùng **AdamW** cho khớp các
   lần train trước, nên đây không phải bản sao y nguyên.
2. Giả thiết 1 của bài báo cần **đầu ra bị chặn trong [0, 1]**; nhãn Δp của dự án không bị
   chặn (khoảng −10 đến 10). Vì vậy định lý 2 **không áp dụng trực tiếp**; chỉ còn lại lý
   do trực giác cộng số đo ablation.
3. Kéo theo điều trên: `epsilon` 0,1 của bài báo đo trên thang [0, 1], còn độ sắc ở đây đo
   trên thang Δp. Smoke ngày 31/10/2026 đo được độ sắc 0,11–0,14 nên `lambda` tăng dần.
   Nếu ablation cho kết quả xấu thì phải quét lại `epsilon` theo thang Δp **trước khi** kết
   luận phương pháp không hiệu quả.
4. Bài báo không nói `lambda` có bị chặn dưới 0 hay không; code chặn tại 0 vì dạng Lagrange
   ở công thức 12 yêu cầu `lambda > 0`.

**Chi phí.** Mỗi batch cần 3 lượt forward thay vì 1 (lượt gốc, lượt lấy gradient đầu ra,
lượt tại điểm bị làm nhiễu), nên chậm khoảng 3 lần. VRAM đo được ở smoke: 328 MiB, gần như
không đổi so với 314–320 MiB của vòng train thường.

## 4.7 Lọc phép sửa hiếm khỏi train (`min_edit_rows`), tùy chọn

**Vì sao.** Phân tích ca sai trên train ngày 07/11/2026
(`runs/evaluator/errors_frozen_direct_train_20261107_081720`) cho một bảng đi **ngược
chiều** mọi nhóm khác. MAE theo số dòng train dùng chung một phép sửa:

| `edit_rows` | dòng | MAE | đúng dấu | Spearman |
|---|---|---|---|---|
| [1, 2) | 1.676.782 | 0,1440 | 0,9881 | 0,9677 |
| [2, 5) | 2.034.961 | 0,1987 | 0,9644 | 0,9226 |
| [5, 66) | 1.588.541 | 0,2523 | 0,9374 | 0,8666 |
| [66, 27.770] | 1.327.275 | 0,3216 | 0,8790 | 0,7536 |

Mọi nhóm khác đi đúng chiều mật độ dữ liệu — càng nhiều dòng càng tốt (`protein_rows`
0,3045 → 0,1582; `molecule_a_rows` 0,3678 → 0,1329). Riêng phép sửa thì ngược: phép sửa
gặp **một lần** được khớp gần như hoàn hảo, phép sửa gặp **27.000 lần** sai gấp đôi. Đó là
dấu hiệu nhánh edit đang được dùng như **bảng tra**: gặp một lần thì nhớ thẳng giá trị,
gặp nhiều lần thì buộc phải trung bình qua nhiều ngữ cảnh protein khác nhau. Trên
validation, phép sửa hữu ích lại chính là loại thường gặp.

**Cách dùng.** `stage3.train.min_edit_rows` (mặc định 1 = không lọc gì). Đặt 2 để bỏ các
dòng mà phép sửa của dòng đó chỉ xuất hiện đúng một lần trong train.

```
python run_evaluator.py --mode train --set min_edit_rows=2
```

**Ba điều phải nhớ khi đọc kết quả.**

1. Chỉ lọc **train**. Validation và test không bao giờ bị lọc (`filter_rare_edits` chỉ
   được gọi cho train trong `run_train`), nên mẫu số của mọi chỉ số không đổi và số liệu
   vẫn so trực tiếp được với các lần chạy trước.
2. Ngưỡng 2 **chưa có nguồn** từ bài báo nào. Nó đến từ chính phép đo ở trên, nên đây là
   một ablation, không phải mặc định mới.
3. Lọc đi 1,68 triệu dòng là bỏ 25 % train, nên **phải chạy kèm đối chứng**:
   `min_edit_rows_control=true` bỏ **ngẫu nhiên đúng cùng số dòng** (seed = `seed`).
   Không có đối chứng thì "kém đi" không phân biệt được với "ít dữ liệu hơn thì kém
   hơn". Chỉ khi hai lần chạy khác hẳn nhau thì mới nói được điều gì về phép sửa.

```
python run_evaluator.py --mode train --set min_edit_rows=2 min_edit_rows_control=true
```

Số dòng bỏ đi được ghi vào log và vào khóa `edit_filter` của manifest.

## 4.8 Learning rate riêng từng nhánh (`branch_learning_rate_scale`), tùy chọn

**Vì sao.** Wang W, Tran D, Feiszli M. *What Makes Training Multi-Modal Classification
Networks Hard?* CVPR 2020, arXiv:1905.12681. Họ đo thấy mạng **đơn** mô thức tốt nhất
**luôn** thắng mạng đa mô thức, trên mọi tổ hợp mô thức và mọi bộ chuẩn họ thử, và nêu hai
nguyên nhân: (a) dung lượng lớn hơn nên quá khớp dễ hơn; (b) các mô thức **quá khớp và
tổng quát hóa với tốc độ khác nhau**, nên train chung bằng một chiến lược tối ưu là dưới
mức tối ưu. Họ giải bằng Gradient Blending (trộn theo tỉ lệ quá khớp trên tổng quát hóa).

Learning rate riêng từng nhánh là cách rẻ nhất để thử nguyên nhân (b): chỉ cần thêm param
group trong AdamW, không đổi kiến trúc, không thêm lượt forward nào.

Việc này khớp với ba quan sát của dự án: bỏ nhánh protein gần như không đổi kết quả;
train 0,868 so với validation 0,564; `projector` 0,5455 và `attention` 0,5491 đều **kém**
`concat` 0,5586 (xem thêm Rheude T, Eils R, Wild B. *Fusion or Confusion? Multimodal
Complexity Is Not All You Need*, arXiv:2512.22991).

**Cách dùng.** `stage3.train.branch_learning_rate_scale` là **hệ số nhân** của
`learning_rate`, theo từng nhánh. Bảy nhánh được liệt kê sẵn ở 1,0 để dòng lệnh chạm được
từng cái một; nhánh nào còn hệ số 1,0 thì **không** tách thành nhóm riêng, nên để nguyên
mặc định là đúng hành vi cũ (một nhóm duy nhất).

```
python run_evaluator.py --mode train --set branch_learning_rate_scale.edit_encoder=0.1
```

Tên nhánh hợp lệ (`train.BRANCH_MODULES`, lấy đúng tên thuộc tính trong
`models.EvaluatorModel`): `ligand_encoder`, `edit_encoder`, `protein_projection`,
`projectors`, `attention`, `endpoint_embedding`, `predictor`. Gõ sai tên thì báo lỗi ngay,
vì một chữ sai sẽ lặng lẽ thành "không đổi gì".

**Ba điều phải nhớ.**

1. Hệ số **nhân**, không phải learning rate tuyệt đối: mọi lần quét `learning_rate` cũ
   vẫn còn nghĩa.
2. Hệ số 0 là hợp lệ và có nghĩa **đóng băng** nhánh đó: AdamW với `lr = 0` không đổi
   tham số nào, kể cả weight decay (decoupled weight decay cũng nhân với `lr`).
3. Bài báo **không** cho con số hệ số nào, nên mọi giá trị đặt ở đây là tham số cần quét,
   **chưa có nguồn**.

Hệ số đã dùng được ghi vào khóa `branch_learning_rate_scale` của manifest, và learning
rate từng nhóm được ghi vào log khi có nhánh nào khác 1,0.

## 4.9 The three diagnostics of the 2026-11-18 meeting

The three modes below are **not** meant to report results, but to answer three questions
raised at the meeting. Each one reuses the data reading, the splitting and the metric set
of `--mode train` untouched, so their numbers compare directly with the batches already
run.

### 4.9.1 Classical baselines — `--mode classical`

From the meeting: *"if this complicated 3.7-million-parameter deep learning model of yours
cannot even beat a linear regression, then it is not solving anything."*

```powershell
python run_evaluator.py --mode classical --config frozen --baseline all `
  --set inputs.train_subset=work/splits/train_subset_c200_r500.parquet
```

* Features: fingerprint A, fingerprint B, substituent A, substituent B, protein embedding,
  one-hot endpoint — **exactly** the five inputs of the MLP, read from the very caches it
  uses. Each block's column range is recorded in the manifest (`feature_blocks`).
* `--config no_protein` drops the protein block; that is also a feature control.
* Models: `linear` (`LinearRegression`), `ridge`, `lasso`, `xgboost`. Every
  regularisation level is a separate fit, reported separately.
* **The alpha grid is per model.** `stage3.classical.alphas` is the default grid and
  `alphas_by_model` overrides it for one model (currently ridge `[0.1, 1, 10, 100]`, lasso
  `[0.0001, 0.001, 0.01]`). One shared grid cannot serve both: the features are not
  standardised and the two penalties are on different scales, so a level that barely
  regularises ridge drives every lasso coefficient to zero. Until 2026-11-20 the grid held
  the single value `1.0`, which for lasso is almost certainly that degenerate case.
* **The level that gets quoted is chosen by the run, not by hand.** The manifest's
  `selected_alpha[model]` holds `{alpha, label, value, alphas_compared}`, picked on
  **validation** through `train.selection_score` with `stage3.train.selection_metric` and
  `selection_slice` — the same code path `--mode train` uses to choose a checkpoint, which
  is the only way the two sides compare. A tie goes to the smaller alpha. The manifest also
  records `selection_metric` and `selection_slice` at the top level, so a table never has
  to guess which number was compared.
* **Two flags per fit, so a non-baseline cannot be quoted as one.**
  `results[label].coefficients` = `{total, nonzero, constant_predictor}`: a fit whose alpha
  zeroed the whole vector predicts its intercept for every row, which gives an MAE equal to
  the labels' mean absolute deviation and a direction accuracy decided by the tie-breaking
  rule — numbers that look like a baseline and are not one. `results[label].convergence` =
  `{iterations, max_iter, hit_the_cap}`: lasso's coordinate descent and ridge's `sparse_cg`
  both stop at `max_iter` whatever state the solution is in, and scikit-learn says so only
  through a warning that no batch log keeps. Both are `null` for a model that reports
  neither (xgboost) — `null` and not a zero, because "not reported" is not "none".
  **Both flags only work in one direction.** All-zero coefficients prove a constant
  prediction; non-zero coefficients do not prove the opposite, since a fit can still be
  (near-)constant on a particular set of rows. `hit_the_cap` says the iteration budget was
  spent, not that the answer is bad; not hitting it does not prove the answer is good.
* **Both sides have to select the same way.** The alpha is chosen on validation, and so is
  the MLP's checkpoint, so `selection_metric` and `selection_slice` have to match across
  `--mode classical` and `--mode train`, or the table compares two different questions.
  `scripts/server_run_diagnostics_1911.sh` keeps the two flags in one variable
  (`$SELECTION`) that both kinds of task pass, and
  `tests/test_diagnostics_script.py::test_the_alpha_and_the_checkpoint_are_chosen_by_the_same_metric`
  fails if any command loses them. Note what the resulting number then is: the validation
  score of a configuration that was CHOSEN on validation. An independent estimate of that
  configuration needs a separate held-out set.
* The selection slice is resolved **before** the feature matrix is built, with the
  validation row count of every slice, exactly as `--mode train` does it: a slice with no
  validation rows stops the run immediately instead of paying for nine fits and then
  writing a manifest with no winner in it. If a slice does have rows but the metric is
  still undefined for every level (no query in it reaches `min_candidates`), the model is
  named in `selected_alpha_unavailable` and logged as a warning — an absent key would read
  as "not implemented".
* Cost: `--baseline all` is now 9 fits (1 linear + 4 ridge + 3 lasso + 1 xgboost) instead
  of 4, and the lasso levels are the slow part. To trim them without editing the config:
  `--set classical.alphas_by_model.lasso=[0.001]`.
* Scored on **both train (the very rows that were fitted) and validation**, slice by slice
  — exactly the "you have to look at the metrics on the training set as well" requirement.
* Features are **not** standardised: fingerprints are 0/1 bits while protein embeddings are
  real numbers, so one `alpha` lands on the two blocks with two different effective
  strengths. The reason: the usual standardisation **subtracts the mean**, and subtracting
  the mean turns every zero into a non-zero, destroying sparsity and making the matrix
  fully dense. **Scaling alone** (`with_mean=False`, or scaling the protein block
  separately) does keep sparsity — a legitimate option the project has **not tried**, not
  something that cannot be done. This has to be stated when reporting.
* Needs `scikit-learn`, `scipy`, `xgboost` (see `requirements.txt`); all three are imported
  inside function bodies, so the other modes keep working without them.
* Manifest: `runs/evaluator/classical_<configuration>_<stamp>.json`.

### 4.9.2 The capacity check — `--mode capacity`

One reading at the meeting: *"an architecture this size, with the tiny amount of data you
feed it, cannot possibly train to a good result."* The other: *"but the point is that it is
not even overfitting."* Only a direct check can reconcile the two.

```powershell
# The pair that isolates dropout: SAME hyper-parameters, one keeps dropout.
python run_evaluator.py --mode capacity --config frozen --head direct `
  --set learning_rate=9.86e-05 --set weight_decay=0.0
python run_evaluator.py --mode capacity --config frozen --head direct --regularized `
  --set learning_rate=9.86e-05 --set weight_decay=0.0
# The 100k question: fit the WHOLE SUB100 subset, no sampling window left.
python run_evaluator.py --mode capacity --config frozen --head direct `
  --set inputs.train_subset=work/splits/train_subset_c200_r500.parquet `
  --set capacity.rows=100000 --set capacity.read_rows=100000
```

**The two runs of a pair must differ in ONE thing.** Until 2026-11-19 the batch script
gave the plain run the config default learning rate (1e-3) and the `--regularized` run the
reference 9.86e-05, so their difference measured dropout *and* a tenfold change of learning
rate at once, which is to say it measured nothing. Both now take the same `--set` flags.

Hand the model `stage3.capacity.rows` rows (1,024 by default), **switch off** dropout and
weight decay, run up to `epochs` (200 by default) with no early stopping on validation,
then measure on **that same sample**. `--regularized` keeps the configured dropout and
weight decay, to separate "the model lacks capacity" from "those two anti-overfitting knobs
are blocking it".

* `verdict` in the manifest: `fits` when train R² ≥ `fit_train_r2` (0.95);
  `does_not_fit` when it is lower; `undecided` when the labels have no variance. `fits`
  establishes exactly one thing — that for **this** architecture, **these** hyper-parameters
  and **that** sample the fit is attainable within the epoch budget — and `verdict_note`
  says so, printing the dropout and weight-decay VALUES actually used rather than a word
  like "on" or "off" (the 2026-11-19 batch passes `weight_decay=0.0` to its `--regularized`
  run as well, so that pair isolates dropout alone and "the knobs are on" would be false).
  This is
  also the **early-stopping condition** — deliberately one threshold for both jobs, because
  two different thresholds once produced a run that both logged "sample fitted" and
  recorded `does_not_fit`.
* Fits ⇒ **a lack of capacity is not the explanation** for the model failing to overfit
  real data; what is left to examine is the input representation and the noise in the
  labels. This check does **not** inspect per-branch gradients, so it does not prove that
  every branch learns — for that, use `--mode diagnose`.
* Does not fit ⇒ what the check measured is that the target R² was **not reached on this
  sample within the epochs that ran**, and nothing more. Why is a separate question with at
  least four candidate answers, to be told apart afterwards: the gradient path (`--mode
  diagnose --partition train`), a representation that cannot tell two rows with different
  labels apart, self-contradictory labels, and the optimisation schedule plus the number of
  update steps.
* **Every** dropout key of the `stage3.model` block goes off, not just `model.dropout`: the
  residual block has its own `residual_hidden_dropout` and `residual_dropout`, and the
  `attention` fusion has `attention_dropout`. The manifest records the whole set in
  `dropout_keys`.
* The verdict applies only to the **exact architecture of that run**. The config default is
  the `plain` 512-256 head; a conclusion about the reference batches' residual 7x102
  requires passing that batch's `--set` flags (see
  `scripts/server_run_diagnostics_1911.sh`).
* Under the `finetune` configuration, the batch size and batch grouping come through
  `--mode train`'s own path (`finetune_overrides`, grouping by protein); otherwise VRAM
  runs out on the first batch.
* This mode saves **no** checkpoint and touches **no** validation data. It says nothing
  about prediction quality on unseen data.
* The small sample is drawn at random from the first `read_rows` rows of train. **Measured
  on 2026-11-19: the first 200,000 train rows of dataset_v4 are all `Ki`.** The default
  window therefore holds a single endpoint and only part of the proteins, so the verdict of
  a default run reads "this configuration can (or cannot) fit 1,024 Ki rows" — the question
  "can it fit **some** small sample", and nothing wider. The manifest's `sample` block
  records `distinct_endpoints` and `distinct_proteins`, and `verdict_note` repeats them, so
  the scope travels with the number.
* To ask the **100k** question instead, point the run at SUB100 with
  `--set inputs.train_subset=... --set capacity.rows=100000 --set capacity.read_rows=100000`:
  it then fits every row it reads and no sampling window is left to skew it. This is task 2d
  of `scripts/server_run_diagnostics_1911.sh`.
* `does_not_fit` after a fixed epoch budget leaves the **optimisation schedule and the
  number of update steps** among the candidate causes, next to the gradient path, the
  representation and the labels. `verdict_note` says so.

### 4.9.3 The protein control — `stage3.train.protein_permute_seed`

From the meeting: *"why would you think that feeding in the protein branch does not make
sense when we are not splitting by protein?"* The old evidence was "dropping the protein
branch barely changes the result", but dropping the branch **changes** the model (fewer
parameters, fewer inputs), so it cannot separate "the model does not use the protein" from
"the model compensates for what is missing".

```powershell
python run_evaluator.py --mode train --config frozen --head direct `
  --set inputs.partition_override_pairs=work/splits/pair_split_1411.parquet `
  --set protein_permute_seed=20261119
```

**The scope has to be the right one.** The meeting's question is about the case where the
split is *not* by protein, i.e. the PAIR75 scope (`inputs.partition_override_pairs`:
proteins seen, pairs unseen). A run that only sets `inputs.train_subset` is still split by
protein cluster, so it does **not** answer that question — it answers the corresponding
question for the SUB100 scope. The 2026-11-19 batch runs both and labels each one.

The model, the parameter count, the input size and even the **multiset** of proteins all
stay as they are; only the row → protein link is shuffled (a permutation of that very
column, with its own permutation per partition). The query key columns are **not** shuffled,
so queries are still grouped by the real protein and the denominator matches the control run
exactly.

* The result drops ⇒ the real protein link **does** help under the conditions being tested.
* The result does not move ⇒ under those conditions, the link does not help. Note: this is
  **not yet** proof that the original model ignores the protein entirely — the model in the
  shuffled run is retrained, so it can learn to compensate for what it lost. A stronger
  conclusion needs `--mode diagnose` (per-input sensitivity) on the checkpoint of the
  **unshuffled** run.
* Read it as a **difference** against a run on the same seed without the shuffle; this
  project's seed-noise floor is 0.0021 on macro direction accuracy, and one seed is not
  enough to conclude anything when the difference is small.
* The two runs share `queries_ranked`, the number of counted pairs and the denominator of
  `pairwise_accuracy_macro`, because the labels, the slices and the query key columns are
  all left unshuffled. One exception: `spearman_queries` can change, since it drops queries
  where the model predicts a constant.
* The shuffle lives in `train.prepare_arrays`, the single chokepoint every mode goes
  through, so `eval`, `errors`, `features` and `diagnose` obey the same setting. The seed is
  written **into the `.pt` checkpoint as well**, and `load_checkpoint` **refuses** to
  re-score when the current setting differs (otherwise "train shuffled, score clean"
  produces an ordinary-looking table of numbers that is higher than the truth).
  `scripts/group_runs.py` warns when a group mixes the two kinds of run.
* **The split is locked the same way, since 2026-11-19** (`train.check_input_split`). The
  shuffle seed alone was not enough: a PAIR75 run re-scored with the right
  `protein_permute_seed` but *without* `--set inputs.partition_override_pairs=...` passed
  every check and then read the default validation partition (493,852 rows) instead of the
  split's own 24,946 — a different experiment on different rows. The checkpoint now carries
  `train_subset`, `partition_override` and `partition_override_pairs`, and the loader
  refuses a mismatch and prints the `--set` flags that fix it. Checkpoints from before that
  date carry no split; for those the loader **warns every time**, including when the
  command sets no flag at all — that is the dangerous case, not the safe one (an old PAIR75
  checkpoint scored by a bare command reads the default validation partition), so silence
  there would clear exactly what nobody can catch by eye.
* The validation column of a table comes from the training run itself (the selection
  point), but the **train column does not**: it needs
  `--mode eval --checkpoint <file> --partition train` with the same split and shuffle
  flags. Tasks 3a'–3d' of the batch script do exactly that, which is what makes the
  train-and-validation comparison the meeting asked for possible for the MLP as well as for
  the classical baselines.
* **Which checkpoint those steps use** is decided by a **run id**, not by a timestamp or
  by file age. Three ways of naming it were tried and are wrong:
  * *the newest `.pt`* — belongs to the previous task when this one died before writing
    anything, or to an unfinished run, since `--mode train` saves its best checkpoint
    DURING the run;
  * *the newest manifest written since I started* — loses a race between two batches
    sharing `runs/evaluator/`: job A writes its manifest and exits 0, job B writes its own
    a moment later, and A's next step then opens B's manifest. With the same split and
    shuffle seed every check inside the loader passes, so nothing announces it;
  * *the timestamp in the file name* — accurate only to the second, so two runs started in
    the same second write the SAME paths and one overwrites the other.

  So `--run-id` (see `run_stamp` in `run_evaluator.py`) is added to the name of every file
  a run writes — its checkpoint **and** its manifest — the batch gives EVERY command an id
  of `<task>-<pid>` (not only the train commands: a `classical`, `capacity`, `eval` or
  `diagnose` manifest is named the same way, so two batches could overwrite each other's
  results although each command had loaded the right checkpoint), and
  `scripts/pick_train_checkpoint.py` opens the manifest carrying that id — refusing, rather
  than guessing, when none or more than one matches. A manifest is
  written only when the run reaches the end, so its existence is also the proof that the
  run finished; the batch still checks the train command's exit code first
  (`LAST_STATUS`), because `run` returns 0 on purpose so one dead task cannot stop the
  batch.

### 4.9.5 Per-block gradient norms — `--mode diagnose --partition train`

From the meeting: does every one of the 7 residual blocks actually learn? `--mode diagnose`
already counted ReLU units that never fire and measured how much the prediction moves when
one input is shuffled, but **both of those run under `torch.no_grad()`**, and a block with
perfectly ordinary activations can still receive a gradient of exactly zero (a detached
path, a gate stuck at zero, a head that ignores the branch). Since 2026-11-19 the mode also
reports, per block, the norm of the gradient of that block's own parameters after
`loss.backward()` on train rows:

```powershell
python run_evaluator.py --mode diagnose --checkpoint runs/evaluator/<file>.pt `
  --partition train --set inputs.partition_override_pairs=work/splits/pair_split_1411.parquet
```

* Groups are cut at the first numbered component of the parameter name
  (`head.blocks.3.linear.weight` → `head.blocks.3`), so an `nn.Sequential` of residual
  blocks gives one group per block — the granularity the question is asked at.
* Per group: `parameters`, `grad_norm`, `grad_norm_per_parameter` (divided by the square
  root of the parameter count, so a 7,000-parameter block can be compared with a
  600,000-parameter one), `share_of_total` and `zero_fraction`.
* Averaged over `stage3.analysis.gradient_batches` batches (5 by default) and measured in
  **eval mode**, so repeating the command gives the same numbers. Report it as what it is:
  *the gradient measured on train batches, at this checkpoint, with dropout off*. A zero is
  strong evidence that nothing reaches the block, but eval-mode gradients are **not** the
  training gradients in general — with dropout active the expectation is taken over masks,
  and the two need not agree.
* The labels are put back into the space the checkpoint was **trained** in, using the
  `LabelScaler` stored in the checkpoint itself: `run_train` fits on
  `LabelScaler.forward(target)`, so a model trained with `zscore`/`minmax` learned on z,
  and feeding raw Δp here would be the gradient of a different loss — wrong scale, and with
  a non-zero shift the wrong sign too. The manifest records `loss`, `huber_delta` and
  `labels` so a number is never read without them. The loss comes from the **current**
  command's config (a checkpoint does not store it), so reproducing the gradients of an
  older train run means passing that run's `loss` and `huber_delta` — both are in its own
  manifest — with `--set`.
* Only the evaluator's own parameters, not ESM2's under `finetune` — and for that reason
  the batch is built inside `torch.no_grad()`, which keeps ESM2 out of the graph. Otherwise
  the backward pass would run through it and hold its activations for every batch, spending
  VRAM on numbers this report never shows. Nothing is updated: the gradients are cleared
  again and no optimizer is involved, so the checkpoint on disk is untouched.
* `--partition train` is the meaningful one: a gradient measured on validation rows is not
  the gradient the model was trained by.

### 4.9.4 Hyper-parameters, and why not the config defaults

The `config.json` defaults are the `plain` 512-256 head, learning rate 1e-3, and checkpoint
selection on MAE over the `all` slice. Every result table this project has reported was
measured with the 2026-11-14 batch's configuration: residual 7x102 with factor 1.849,
dropout 0.5 / hidden 0.3936, learning rate 9.86e-05, weight decay 0, selection on
`pairwise_accuracy_macro` over the `strict` slice. Running the three diagnostics on the
defaults would produce numbers that compare with **no existing table**, so
`scripts/server_run_diagnostics_1911.sh` passes the reference batch's `--set` flags and
keeps the defaults for a single, clearly labelled capacity run.

#### Before a table is built: `scripts/check_diagnostics_manifests.py`

```bash
python scripts/check_diagnostics_manifests.py runs/evaluator runs/diagnostics_1911_gpu1.log
```

Exit 0 means the seven manifests the three tables need are there and were all selected the
same way; exit 1 prints one `FAIL` line per task and means the output is **not fit to
quote**. The script reads the batch's pid back out of its own log (the pid file is gone by
then, removed by the trap) and looks up each task by the run-id suffix of its filename,
`_<task>-<pid>.json`. It checks that each of `t1a`–`t1c` and `t3a`–`t3d` wrote **exactly
one** manifest, that all seven carry `pairwise_accuracy_macro` / `strict`, that the three
classical manifests have `complete: true` and a winner for **both** ridge and lasso, and
that the four train manifests name a checkpoint that is on disk.

It exists because the obvious check — glob `runs/evaluator/*.json` and print the two
selection fields — is wrong in three ways, each of which lets a bad batch look fine: a glob
sees the whole history (two train manifests from 2026-10-31 predate the selection fields
and print `None None`, which reads as a broken new batch, or makes a directory look
populated when this batch wrote nothing); a task that died leaves **no file**, and a glob
says nothing about it; and `--mode classical` rewrites its manifest after every fit, so a
run that died mid-grid already carries both selection fields and a provisional winner with
`complete: false`.

#### Building the tables: `scripts/diagnostics_tables.py`

```bash
python scripts/diagnostics_tables.py runs/evaluator runs/diagnostics_1911_gpu1.log
```

Prints the four tables as Markdown, every number read out of a manifest at a fixed key.
The baseline table carries **train and validation** side by side, which is what the meeting
asked for: the classical rows are scored on the very rows they were fitted on, from their
own manifest, and the MLP's train half comes from the separate `--mode eval --partition
train` run of the SAME checkpoint (`t3a-eval`, `t3c-eval`) - a `--mode train` manifest holds
no pass over the training rows, so without that eval run the train columns are `-` and a
footnote says which run is missing.

Two things the table will not do. It will not pair halves from different models: both
manifests carry the checkpoint they used, and the train half is refused (with the reason in
the row) unless the two paths agree and the eval manifest says `partition: "train"`. And it
will not read both halves at one slice. The train columns are the `--train-slice` (default
`all`), the validation columns the `--slice` (default `strict`), and **the header names
both**, because on a pair-keyed split they cannot be the same: `build_pair_split.py` writes
`unseen_pair = is_validation`, so every TRAIN row carries false and the `strict` slice of
the train partition is empty by construction. Reading `strict` on both sides printed `-`
for the train half of a batch that had finished perfectly - that was the state until
2026-11-20. `--slice all --train-slice all` measures the gap over the two whole
partitions.
Carrying numbers by hand was the one step of this pipeline with no test behind it, and it
has three failure modes the script is built against: a regularised model quoted at a level
nobody selected (it shows only the level the run recorded in `selected_alpha`, and names
the alpha in the row), a column headed `strict` holding `all` numbers (asking for a slice
the manifests do not have empties the columns instead of falling back), and a task with no
manifest filled in from another batch (it prints `-` and names the task in the footnotes).
Degenerate and unconverged fits keep their flags in a notes column — including the non-zero
coefficient count, which is what makes a fit held up by a single coefficient visible when
`constant_predictor` is false. The footnotes are generated, not written: a slice that holds
every validation row is called out as not being a second measurement, the share of
rankable queries is stated, and every table ends on the reminder that each score belongs to
a configuration selected on validation.

#### Is the PAIR75 score carried by molecules already seen? `scripts/seen_molecules.py`

```bash
setsid nohup bash scripts/server_run_seen_molecules_2011.sh 1 t3a-<batch pid> \
    > /dev/null 2>&1 &
```

The second argument is the training task's **run id**, which the script resolves through
`scripts/pick_train_checkpoint.py` (two positional arguments, no flags:
`python scripts/pick_train_checkpoint.py runs/evaluator t3a-4174688`); a `.pt` path is
accepted too.

PAIR75 is keyed on the **pair**, so no validation pair was in training - that is what
`unseen_pair` and the `strict` slice guarantee - but a single **molecule** of a validation
pair may well have been, inside another pair. If the 0.7778 lives on those rows it is not a
generalisation number, and the gap to SUB100's 0.53 is a property of the split rather than
of the model. The batch writes the two row dumps the analysis needs (`--mode errors` on
train and on validation, same checkpoint, same split) and runs the analysis on them; the
protein-shuffle flag is deliberately absent, so a `t3b` checkpoint cannot be analysed here
by accident - `--mode errors` refuses a checkpoint whose recorded `protein_permute_seed`
does not match the command's.

The report is written in three units, because they answer different questions. **Per row**
(one MMP pair): the rows split into three groups by how many of their two molecules appear
in the train dump - 0, 1 or 2 - each with its row count, MAE and direction accuracy over
rows (the *micro* quantity, from the dump's own `sign_counted`/`sign_correct`), plus the
label statistics of the group. **Per group as a slice**: the same three groups with
`pairwise_accuracy_macro` measured inside each - the rows of the group are selected first
and the queries built afterwards, which is the order every slice of this project is
measured in. That order is what makes a macro over a 0/1/2 group well defined at all: one
comparison inside a query involves THREE molecules (a reference and two candidates), so the
count of seen molecules belongs to a row, and a slice is a set of rows. The three groups do
**not** add up to the whole set - restricting the rows cuts a query into smaller queries, so
`ranked` can be far below the whole set's - and the report says so rather than inviting the
reader to average them. **Per query of the whole set** (reference molecule, protein,
endpoint): each query placed by whether its *reference* was seen and by how many of its
candidates were, which answers whether it matters which SIDE of a comparison the model had
already met.

The query side is a reconstruction from the row dump - pairs doubled into both directions,
grouped by (reference, protein, endpoint), scored with `metrics.pairwise_direction` at the
run's own tolerance - so it is **checked**: the overall macro and the number of queries
**with at least `min_candidates` candidates** must reproduce the dump's own
`queries.parquet`, and the script exits non-zero when they do not. That bar is part of the
check and not a detail: `analysis.query_table` writes down only the queries that clear it,
so comparing raw totals calls a correct run wrong - two rows A-B and A-C rebuild three
queries (A, B, C) while the file holds one (A). The full rebuilt total is printed beside it
as coverage, and every table carries both counts.

A query below the bar is **skipped**, not merely left out of the count:
`metrics.ranking_report` skips it before scoring anything, so scoring it here would make
this macro a different quantity from the one the manifests report. At `min_candidates = 2`
the two behaviours agree (a single candidate yields no comparison), which is why a run at 3
was needed to show the difference.

Where the threshold comes from is printed with it. `report.json` carries `metrics_settings`
(added 2026-11-20) for exactly this; its `settings` field is the stage3 **analysis** block
and is never read here, since a metric threshold taken from it would silently pick up a
different setting of the same name. For a dump written before that field existed the
fallback is `config.json`, and the report says so - today's config.json is not necessarily
the one the dump was written with.
A breakdown that cannot reproduce the number it breaks down is not evidence, and the batch
carries that exit code through instead of printing DONE.

What it does **not** establish: that the model memorises. A higher score on seen molecules
has at least two readings - the model recalls them, or those rows are simply easier - so
each row group also carries its label mean, label sd and mean |Δp|, which is what lets the
second reading be checked rather than assumed away.

#### Giving the protein null a noise scale: `scripts/server_run_protein_seeds_2011.sh`

The 2026-11-19 batch measured `pairwise_accuracy_macro` 0.7778 with the real protein column
and 0.7805 with it shuffled across rows: **+0.0027**, i.e. destroying every row's protein
identity made the model slightly better. One run per side cannot tell that from
run-to-run noise, and "no difference" and "a difference below our noise" are different
claims. This batch runs three pairs on PAIR75 (about 20 minutes, six train runs): pair *i*
is the same configuration twice, differing only in whether the protein column was shuffled,
while across pairs both the training seed and the permutation draw change — so the spread
of the three differences covers both sources of noise. The first seed is the 2026-11-19
batch's own, so pair 1 reproduces t3a/t3b and a mismatch there means something else
changed.

```bash
setsid nohup bash scripts/server_run_protein_seeds_2011.sh 1 > /dev/null 2>&1 &
python scripts/protein_control_seeds.py runs/evaluator runs/protein_seeds_2011_gpu1.log
```

The summary also checks what the batch IS before putting a name on it. `pair_problems`
only asks whether the two sides of a pair agree with each other, and two manifests on the
default split with the same seed agree perfectly while being a different experiment
altogether - so the split named in the manifests has to be PAIR75, the complete pairs have
to carry one training seed each (otherwise the spread across pairs is not a spread across
seeds), and pair 1 has to repeat the 2026-11-19 seed and permutation before it is called a
reproducibility check of t3a/t3b. A sentence whose condition does not hold is withheld
rather than reworded, and the exit code is non-zero.

The summary refuses to print a mean difference without the spread next to it, and it says
in so many words that what it prints is descriptive statistics: a mean of this size next to
a spread of that size. It is **not** a bound on the effect - the spread of three differences
is a property of this sample, not a limit on anything - and not a significance test. It also
refuses to subtract two manifests that are not the two sides of one pair: the seed, the
split, the subset, the configuration and the head all have to match and exactly one side has
to carry `protein_permute_seed >= 0`, or the row prints both scores and no difference. And
because pair 1 repeats the 2026-11-19 configuration, the output says it is a reproducibility
check of t3a/t3b rather than a fourth independent pair.

All of them run in one go with `scripts/server_run_diagnostics_1911.sh <GPU index>`,
whose log, pid file and every run id are named after that GPU index so two batches on
two GPUs can run side by side without overwriting each other's files. One batch per GPU is
enforced with a `flock` lock on `.cache/gpu<N>.lock` (`scripts/gpu_batch_lock.sh`), not by
reading a pid file:
reading and then writing is two steps, so two batches launched in the same second would both
read "nobody there" before either wrote. The lock lives on a descriptor the batch keeps
open, so the kernel releases it however the batch ends — `kill -9` included — and it also
owns the pid file, which is written under the lock and removed by a trap on EXIT (the early
exit for a missing split file used to leave it behind, and a leftover pid eventually names
an unrelated process). Both files are named after the GPU and nothing else - `.cache/gpu<N>.lock` and
`.cache/gpu<N>.pid`. Until 2026-11-20 the script's own prefix was part of both names, which
made the claim per SCRIPT rather than per GPU: `server_run_diagnostics_1911.sh` took
`diagnostics_1911_gpu1.lock` while `server_run_protein_seeds_2011.sh` took
`protein_seeds_2011_gpu1.lock`, both claims succeeded, and two batches shared one GPU with
each believing it held it. A GPU is one resource, so one lock name; the prefix is still
recorded inside the pid file, where it names the batch without keying the lock, and
`stop_gpu_batch.sh <GPU> [cache dir] [prefix]` checks the record against that name and
refuses to signal a batch that is not the one asked for. (A batch started before this
change holds the old file name, so it must be stopped with the version of the script that
started it.)

To stop a batch, use `scripts/stop_gpu_batch.sh <GPU index>`, never a `kill` one-liner: a
pid alone cannot be checked, and a pid left behind by a SIGKILLed batch can later belong to
a batch of the SAME script on another GPU, where even `ps | grep
server_run_diagnostics_1911` matches. The record `claim_gpu` writes therefore holds the GPU,
the process group and the process start time (which a reused pid cannot inherit), and the
stop script compares all three before it signals anything. It does not trust the group
number either: the record can be out of date (a newer batch takes the lock a moment before it
replaces the record), so before signalling it requires that the process leads its own group
(`pgid == pid`, which is what `setsid` gives) and that **some member of that group is holding
this GPU's lock file open**, checked through `/proc/<pid>/fd`. That last fact is the only one
that ties a process group to a GPU; pid, group number and start time can all go stale. The
same check covers the case where only the shell died while the python run still holds the
lock: the script lists what is alive in the group, stops it, and then waits to confirm the
group is really gone — it returns 5, not 0, if the signal was refused or the group survives.
"Alive" excludes zombies: a process that has exited but whose parent has not reaped it is
still listed by `ps`, and counting it would make a correctly stopped batch report failure
(tasks 1a–1c classical, 2a–2d capacity, 3a–3d' the protein control with both halves of the
table, 4 the gradient norms). The
script counts failed commands and **exits non-zero** if there were any, so it cannot finish
with the word "DONE" while one task died part-way.

### 4.9.6 The sequence window of a finetune checkpoint

`--mode train` narrows the protein window through `stage3.train.finetune_overrides`
(`max_residues` 512 instead of the shared 1022), because finetune keeps ESM2 in the backward
graph and its memory grows with the number of residues. Until 2026-11-19 the re-scoring path
rebuilt the encoder from the shared config instead, so for every protein longer than 512
residues the same weights were handed a longer sequence: the same model, a different input,
and therefore predictions that differ from the ones that selected the checkpoint — with
nothing in the output to announce it.

A finetune checkpoint now records the encoder settings its own run used
(`protein_encoder_settings`), and `finetune_encoder_settings` prefers them over the config.
Checkpoints trained before 2026-11-19 do not carry them; for those the window **cannot** be
checked, so loading one logs a warning that names both numbers instead of letting the reader
assume they matched. The frozen tasks of 2026-11-19 never go through this branch.

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

> **CẬP NHẬT 10/11/2026 — lựa chọn ở mục này KHÔNG còn là lựa chọn duy nhất.** Ở họp
> 09/11/2026 anh Tiến yêu cầu kiến trúc phải **tự thỏa** `f(A,B,P) = −f(B,A,P)` ngay từ
> thiết kế, với mốc sai số hai chiều `1e-8` (dòng 141 biên bản). Mục này vẫn đúng cho hai
> đầu ra joint (`direct`, `direct_no_edit`): ở đó dự án vẫn chỉ ĐO. Nhưng từ nay có thêm
> **hai đầu ra phản đối xứng đúng theo cấu tạo**, xem mục 6.3:
> `twin` (có thể bật `twin_edit`) và `antisymmetric`. Với hai đầu ra đó, con số trong bảng
> dưới đây phải bằng 0 — nếu khác 0 thì là **lỗi code**, không phải tính chất của mô hình.

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

### 6.3 Hai đầu ra phản đối xứng đúng theo cấu tạo (thêm 10/11/2026)

Yêu cầu: `f(A,B,P) = −f(B,A,P)` phải đúng **với mọi bộ trọng số**, chưa cần train. Hai
đường đạt được điều đó, cả hai đều có trong `src/molopt/evaluator/models.py`:

| đầu ra | công thức | học được tương tác trong cặp |
|---|---|---|
| `twin` (+ `twin_edit`) | `s(B, F_B, P) − s(A, F_A, P)` | không — `s` chỉ thấy một phân tử |
| `antisymmetric` | `⟨g(P, e), h(d)⟩`, `d = [enc(A) − enc(B), enc(F_A) − enc(F_B)]` | có — `h` phi tuyến trên hiệu |

`h` là **MLP lẻ**: mọi `Linear` không bias, kích hoạt `tanh`, **không** dropout và
**không** LayerNorm. Ba điều kiện, bỏ một cái là mất tính chất:

- có bias thì `W(−x) + b ≠ −(Wx + b)`;
- `ReLU` không lẻ — đúng chỗ anh Tiến phản biện ở dòng 123 biên bản;
- LayerNorm trừ trung bình và cộng tham số dịch, hai phép đó phá tính lẻ.

Ngữ cảnh `P` vào bằng phép **nhân** (cổng `g`), không bằng phép cộng: cộng `P` vào trong
`h` sẽ phá tính lẻ, còn nhân thì không vì `g` không phụ thuộc `d`. Nhờ vậy `P` vẫn đổi
được **cả độ lớn và dấu** của dự đoán — đúng chỗ ablation 07/11/2026 thấy mô hình đang
phớt lờ `P`. Đây là điểm khác so với "mô hình tuyến tính không bias trên vector hiệu" mà
anh Tiến nói ở dòng 135: ở đó `P` chỉ cộng thêm một số hạng không đổi dấu nên **buộc phải
bỏ** nhánh `P` đi.

**Một giới hạn phải ghi vào paper.** Tính chất đúng **tuyệt đối ở chế độ eval**, tức ở mọi
con số được báo cáo (`asymmetry_report` chạy qua `predict`, luôn ở chế độ eval). Trong lúc
**train** với `dropout > 0` nó chỉ đúng theo kỳ vọng: encoder dùng chung được gọi hai lần
trong một lượt forward và mỗi lần lấy một mặt nạ dropout độc lập, nên
`enc_{m1}(A) − enc_{m2}(B) ≠ −(enc_{m1}(B) − enc_{m2}(A))`. Đặt `model.dropout = 0` thì
đúng tuyệt đối ở cả hai chế độ. Ghim bằng
`test_dropout_breaks_antisymmetry_during_training_but_not_at_eval`.

Ô thứ tư của bảng ablation ở mục 6 (twin **có** nhánh edit) bật bằng `model.twin_edit`.
Trước 09/11/2026 `models.py` ghi rằng ô đó "không tồn tại được" vì một phép sửa là thuộc
tính của **cặp**. Lập luận đó **sai**, và anh Tiến đã chỉ ra ở dòng 163: `F_A` là phần bị
cắt khỏi `A` nên nó thuộc về `A`, `F_B` là phần lắp vào tạo `B` nên nó thuộc về `B`.

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

### 7.1 Swapping an encoder: the flags that are actually required (2026-11-19)

Two items of the post-2026-11-18 list are "try ESM-2 150M" and "try ChemBERTa". Neither
needs new code, but neither is a single flag either:

| item | the complete command |
|---|---|
| ESM-2 150M | `--set protein_encoder.model=facebook/esm2_t30_150M_UR50D --set protein_encoder.embedding_dim=640 --set paths.cache_dir=work/evaluator_cache_esm150` |
| ESM-2 650M | the same with `embedding_dim=1280` and `batch_sequences=4` (see `scripts/server_run_esm650.sh`) |
| ChemBERTa | `--set molecule_encoder.model=DeepChem/ChemBERTa-77M-MLM --set paths.cache_dir=work/evaluator_cache_chemberta --set model.ligand_features=embedding`, after `--mode prepare --molecule-embeddings` |

**Leaving out `embedding_dim` stops the program**, exactly as section 7 describes:
`features.load_protein_encoder` compares the model's real dimension with the one declared
in the config. That is the brake working as intended — but it does mean "just change
`protein_encoder.model` and run" is wrong.

`molecule_encoder` has only been routed through `--set` **since 2026-11-19**. ChemBERTa was
already in `molecule_encoder.allowed_models` before that, but reaching it meant editing
`config.json` by hand — and a hand edit changes **every** run of a batch script at once, so
the ChemBERTa experiment could not be separated from the run beside it. The same reason
`antisymmetric_hidden` got a route on 2026-11-11 (see `parse_number_list` in
`run_evaluator.py`).

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
