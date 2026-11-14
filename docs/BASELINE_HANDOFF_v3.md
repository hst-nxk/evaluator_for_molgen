# Gói bàn giao cho nhóm baseline agent

Dựng tự động bằng `scripts/export_baseline_handoff.py`, ngày 15/11/2026.
Việc 16 biên bản họp 09/11/2026 — điều kiện cứng: baseline agent sẽ được so với
mô hình của paper, nên hai bên phải chạy trên dữ liệu TRÙNG TỪNG DÒNG.

## 1. Hai file, và cách kiểm trùng từng dòng

| file | đường dẫn | dòng | byte | sha256 |
|---|---|---|---|---|
| evaluator | `data/dataset_v3/evaluator_training.parquet` | 11,938,241 | 569,744,016 | `96851129419c2c71a0933e540dc5bfe3e212ac191991ecdcde237abe25e72e93` |
| proteins | `data/dataset_v4/full/proteins.parquet` | 10,573 | 7,417,174 | `f4942ac2f5a554d4e1ddf85a2154d232dca81826c6d128d442fc68e3807060e4` |

Bên nhận chạy `sha256sum <file>` (Windows: `certutil -hashfile <file> SHA256`).
Khớp thì chạy tiếp; KHÔNG khớp thì dừng và báo lại, đừng chạy rồi so kết quả —
hai tập lệch nhau một ít dòng vẫn ra số trông hợp lý, và đó là kiểu sai khó phát
hiện nhất.

## 2. Stage 3 có lọc bớt dòng nào nữa không: KHÔNG

Chỗ lọc dòng duy nhất của toàn Stage 3 là `read_columns`
(`src/molopt/evaluator/data.py`), và nó chỉ lọc theo cột `partition`. Hai đường
chia khác (`train_subset`, `partition_override`, `partition_override_pairs`) đều
TẮT trong `config.json` mặc định; lần chạy nào bật thì ghi rõ trong manifest của
lần chạy đó. Nên hai file trên một mình đã định nghĩa đủ tập dữ liệu: không có bộ
lọc ngầm nào nằm ngoài chúng.

Chia tập đọc thẳng từ cột `partition` có sẵn trong file. KHÔNG tự chia lại.
KHÔNG đụng vào `partition == 'test'`.

## 3. Một "query" là gì — chốt bằng số, không bằng lời

Biên bản ghi "669.000 vs 636.000 — đã đo là khác cách gom nhóm, không phải khác
dữ liệu" nhưng chưa ghi cách gom nào ra số nào. Bảng dưới đếm trên chính dữ liệu
này; con số nào khớp thì tự nó chỉ ra hai bên đang nói cách gom nào.

Định nghĩa của dự án (`metrics.query_groups`, gọi từ `train.py`):
**một query = (phân tử tham chiếu, protein, endpoint)**. Hai thứ làm số đổi:

* `expand_both_directions` — mỗi dòng (A, B) tính hai lần, A làm tham chiếu rồi B
  làm tham chiếu. Giá trị hiện tại: `True`.
* `min_candidates` = 2 — query ít ứng viên hơn ngần ấy bị bỏ khỏi
  mẫu số của `pairwise_accuracy_macro` (`queries_ranked`), nhưng vẫn tồn tại
  (`queries_total`).

### partition `train`

| cách đếm | số |
|---|---|
| rows | 10,147,566 |
| protein phân biệt | 3,178 |
| pair_id phân biệt | 10,147,566 |
| query · một chiều · mọi query | 929,967 |
| query · một chiều · >= 2 ứng viên | 720,678 |
| query · hai chiều · mọi query | 1,138,518 |
| query · hai chiều · >= 2 ứng viên | 984,087 |

## 4. Thước đo phải khớp

Số chính của paper là `pairwise_accuracy_macro`: tỉ lệ xếp đúng chiều của các cặp
ứng viên TRONG một query, rồi lấy trung bình THEO QUERY (mỗi query một phiếu).
Không phải micro (gộp mọi cặp rồi mới chia) — query dài sẽ chi phối micro.

Ngưỡng bỏ cặp quá gần nhau: `direction_tolerance` = 0.3.
Cặp có hai nhãn lệch nhau không quá ngần ấy bị loại khỏi MẪU SỐ. Bên kia không áp
ngưỡng này thì con số không so được, dù dữ liệu trùng từng dòng.

Lát cắt: `all` (mọi dòng) và `strict` (chỉ `unseen_pair = True`). Số báo cáo của
paper là lát `strict`. Lát `all` KHÔNG dùng để so hai mô hình — đã đo được là nó
đảo cả thứ tự xếp hạng.

## 5. Việc của bên nhận

1. Kiểm `sha256` hai file.
2. Chia tập theo cột `partition`, không tự chia lại; không đụng `test`.
3. Báo cáo `pairwise_accuracy_macro` lát `strict`, kèm mẫu số (`queries_ranked`).
4. Chạy ít nhất 3 hạt giống và báo cả 3, không báo mỗi số tốt nhất. Sàn nhiễu hạt
   giống đo được của dự án trên tập đầy đủ là 0,0021; chênh lệch nhỏ hơn ngần ấy
   không kết luận được gì.

