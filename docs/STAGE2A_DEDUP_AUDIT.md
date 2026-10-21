# Stage 2a: Dedup giữa hai nguồn và audit độ nhiễu

## 1. Mục đích

Stage 1 tạo ra hai file observation riêng cho ChEMBL 37 và BindingDB 202610. Hai file
này chưa được gộp và chưa được khử trùng lặp. Stage 2a làm hai việc:

1. **Dedup giữa hai nguồn** (`scripts/dedup_cross_source.py`). Script tìm những
   observation của ChEMBL và BindingDB thực chất là cùng một phép đo, để sau này không
   đếm một kết quả hai lần như hai bằng chứng độc lập. Script chỉ đánh dấu, không xóa
   dòng nào.
2. **Audit độ nhiễu** (`scripts/audit_noise.py`). Script chỉ đọc dữ liệu và đo mức
   nhiễu: độ lệch giữa các replicate, số assay trên mỗi cặp phân tử–protein, độ lệch
   giữa các assay, phân bố pActivity, các protein có nhiều nhiễu, và mức ChEMBL đã sửa
   giá trị gốc. Kết quả chỉ là số liệu: script không lọc gì và không đề xuất ngưỡng.

Stage 2a chưa ghép cặp phân tử (MMP), chưa tính Δp và chưa train. Các bước đó thuộc
Stage 2b trở đi.

## 2. Đầu vào và đầu ra

| Script | Đọc | Ghi |
|---|---|---|
| `dedup_cross_source.py` | `data/observations/chembl37_observations.parquet`, `data/observations/bindingdb202610_observations.parquet` | `data/dedup/cross_source_duplicates.parquet`, `data/dedup/chembl37_observations_dedup.parquet`, `data/dedup/bindingdb202610_observations_dedup.parquet`, `runs/stage2a/dedup_<thời điểm>.{log,txt,json}` |
| `audit_noise.py` | hai file `data/dedup/*_observations_dedup.parquet` | `runs/stage2a/noise_audit_<thời điểm>.{log,txt,json}` |
| `diagnose_noise.py` | hai file `data/dedup/*_observations_dedup.parquet` | `runs/stage2a/noise_diagnose_<thời điểm>.{log,txt,json}` |

Hai file observations của Stage 1 chỉ được đọc, không bao giờ bị ghi đè. Mọi đường dẫn,
tên cột, ngưỡng và tham số nằm trong mục `stage2a` của `config.json`.

## 3. Cách chạy

Chạy từ thư mục gốc của repo, theo đúng thứ tự sau:

```powershell
# 0. Unit test (dữ liệu tự tạo, không đọc dữ liệu thật)
python -m pytest tests -q

# 1. Dedup: ghi file ánh xạ, hai bản sao có cột dedup_*, và báo cáo
python scripts\dedup_cross_source.py

# 2. Audit: đọc hai bản sao do bước 1 ghi
python scripts\audit_noise.py

# 3. Chẩn đoán thêm (mục 5.5): cũng chỉ đọc hai bản sao của bước 1
python scripts\diagnose_noise.py
```

Nếu output của bước 1 đã tồn tại, `dedup_cross_source.py` dừng lại mà không ghi đè.
Muốn chạy lại thì thêm `--force`. Mọi file của bước 1 được ghi vào tên `.partial` trước
và chỉ đổi tên khi cả ba file đã ghi xong. Vì vậy một file tên chính thức luôn là file
hoàn chỉnh. `audit_noise.py` không ghi gì vào `data/`. Mỗi lần chạy, script tạo một bộ
báo cáo mới mang mốc thời gian trong `runs/stage2a/`.

Thời gian chạy đo ngày 21/10/2026: dedup 5,8 phút, audit 0,7 phút, chẩn đoán 0,6 phút
(xem mục 7).

## 4. Dedup giữa hai nguồn

### 4.1. Luồng xử lý

1. **Lượt tổng quan.** Script đọc mỗi file một lần để đếm mọi giá trị của cột
   `source_curation`. Ở BindingDB, cột này lấy từ `Curation/DataSource`; ở ChEMBL, từ
   `source.src_short_name`. Lượt này cũng lấy riêng các dòng của ca kiểm chứng.
2. **Chia phân vùng theo ký tự đầu của InChIKey.** Hai observation chỉ có thể trùng
   khi có cùng InChIKey, nên xử lý từng phân vùng riêng cho kết quả giống hệt xử lý cả
   khối một lần. Mỗi lần chỉ cần giữ một phần nhỏ dữ liệu trong bộ nhớ. Phân vùng
   `other` nhận các InChIKey bắt đầu bằng ký tự ngoài bảng chữ cái đã khai báo, còn
   phân vùng `null` nhận các dòng không có InChIKey. Nhờ vậy không dòng nào bị bỏ sót.
3. **Tìm cặp ứng viên trong từng phân vùng.** Có hai phép join:
   - cùng InChIKey, cùng endpoint, cùng tài liệu và cùng protein (theo chuỗi hoặc
     accession);
   - cùng InChIKey, cùng endpoint, cùng tài liệu, cùng qualifier và cùng giá trị. Phép
     join này chỉ áp dụng khi ít nhất một bên thiếu protein.

   "Cùng tài liệu" nghĩa là có chung PMID, DOI hoặc số patent.
4. **Chấm từng cặp** theo protein và giá trị, rồi gán mức căn cứ (mục 4.2).
5. **Gom nhóm.** Các cặp duplicate nối với nhau được gom thành nhóm liên thông, và mỗi
   nhóm nhận một `group_id`.
6. **Ghi file ánh xạ** sau từng phân vùng, rồi ghi **hai bản sao** của file
   observations có thêm 5 cột.

### 4.2. Mức căn cứ

Mỗi cặp ứng viên đã có chung phân tử (InChIKey), endpoint và tài liệu. Kết luận phụ
thuộc vào ba yếu tố còn lại:

| Mức | Trạng thái | Điều kiện |
|---|---|---|
| 1 | `duplicate` | Cột nguồn nhập liệu của một bên chỉ ra bên kia; protein khớp; giá trị bằng nhau |
| 2 | `duplicate` | Không có dấu hiệu nguồn nhập liệu; protein khớp; giá trị bằng nhau |
| 3 | `uncertain` | Giá trị khác hoặc chỉ gần bằng (`value_differs`, `value_close`, `value_qualifier_differs`, `value_missing`) |
| 3 | `uncertain` | Giá trị bằng nhau nhưng một bên thiếu protein (`protein_missing`) |
| 3 | `uncertain` | Dòng có dấu hiệu nguồn nhập liệu nhưng không tìm được bản tương ứng (`provenance_without_counterpart`), hoặc không có InChIKey (`provenance_without_inchikey`) |

Các định nghĩa dùng trong bảng:

- **Protein khớp** khi hai bên có cùng hash chuỗi, hoặc cùng accession UniProt.
- **Giá trị bằng nhau** khi qualifier giống nhau và nồng độ mol/L lệch nhau không quá
  `value_exact_relative_tolerance` (mặc định 1e-9, tức gần như bằng tuyệt đối).
- **Gần bằng** khi |Δp| ≤ `value_close_p_tolerance` (mặc định 0,05 đơn vị log, khoảng
  12 % về nồng độ). Cặp gần bằng chỉ được ghi `uncertain`, để bạn xem lại ảnh hưởng của
  việc hai nguồn làm tròn số khác nhau.
- **Dấu hiệu nguồn nhập liệu** là một chuỗi con trong `source_curation`, không phân
  biệt hoa thường, được khai báo ở `provenance_markers`. Mặc định là `"chembl"` cho dòng
  BindingDB và `"bindingdb"` cho dòng ChEMBL. Dấu hiệu được xét theo cả hai chiều, vì
  ChEMBL cũng có thể nhập dữ liệu từ BindingDB.

Khi không đủ căn cứ, script ghi `uncertain` kèm lý do cụ thể, không đoán.

### 4.3. Quy tắc "mỗi nhóm chỉ tính một lần"

Trong mỗi nhóm duplicate, mọi dòng ChEMBL được giữ (`keep_source = "chembl"`). Các dòng
BindingDB của nhóm có `dedup_keep = False`. Dòng `uncertain` và dòng không liên quan đến
dedup đều có `dedup_keep = True`. Không dòng nào bị xóa khỏi bất kỳ file nào.

Giữ ChEMBL là quyết định có chủ đích, vì hai lý do kiểm chứng được:
- **Với cặp mức 1, ChEMBL là bản gốc.** Cột nguồn nhập liệu của BindingDB cho biết dòng
  đó được nhập lại từ ChEMBL.
- **Dòng ChEMBL giữ giá trị gốc trước chuẩn hóa** (giá trị, đơn vị, quan hệ do tác giả
  công bố, trong `activity_context`). Mục 6 của audit cần các giá trị này. Dòng BindingDB
  chỉ có một giá trị đã quy về nM.

Với cặp mức 2 (không có dấu hiệu nguồn nhập liệu), chưa biết bên nào là bản gốc. Ở đây,
giữ ChEMBL chỉ là quy ước để kết quả nhất quán, không dựa trên nhận định rằng nguồn này
curate kỹ hơn.

### 4.4. File ánh xạ `cross_source_duplicates.parquet`

Mỗi dòng của file ánh xạ là một cặp ứng viên (`record_type = "pair"`), hoặc một dòng có
dấu hiệu nguồn nhập liệu mà không có cặp nào (`record_type = "unpaired"`).

| Cột | Ý nghĩa |
|---|---|
| `record_type` | `pair` hoặc `unpaired` |
| `evidence_level` | 1, 2 hoặc 3 |
| `dedup_status` | `duplicate` hoặc `uncertain` |
| `reason` | lý do (xem mục 4.2) |
| `group_id` | nhóm duplicate, dạng `dup:<observation_id nhỏ nhất của nhóm>`; null nếu `uncertain` |
| `endpoint`, `molecule_inchikey` | khóa chung của cặp |
| `document_match` | loại tài liệu chung, vd. `pmid+doi` |
| `protein_match` | `sequence`, `accession`, `missing` hoặc `differs` |
| `value_match` | `exact`, `close`, `differs`, `qualifier_differs` hoặc `missing` |
| `p_activity_difference` | \|log10(c₁) − log10(c₂)\| |
| `chembl_*`, `bindingdb_*` | observation_id, ID phép đo gốc (activity_id hoặc Reactant_set_id), source_curation, cờ nguồn nhập liệu, PMID, DOI, patent, assay, accession, hash protein, qualifier, giá trị và đơn vị nguyên văn |

### 4.5. Năm cột mới trên bản sao

Tên cột lấy từ `stage2a.dedup_columns` trong config.

| Cột | Ý nghĩa |
|---|---|
| `dedup_group_id` | nhóm duplicate chứa observation; null nếu không thuộc nhóm nào |
| `dedup_evidence_level` | mức mạnh nhất (số nhỏ nhất) trong các bản ghi có mặt observation |
| `dedup_status` | `duplicate`, `uncertain` hoặc null |
| `dedup_keep` | `False` chỉ với dòng BindingDB trong nhóm duplicate |
| `dedup_reason` | `counterparts:<n>` với duplicate; các lý do uncertain nối bằng `;` |

### 4.6. Đọc báo cáo `dedup_<thời điểm>.txt`

Báo cáo vừa được in ra màn hình, vừa được ghi vào file, và gồm các phần sau:

1. **Giá trị `source_curation`** phổ biến nhất của mỗi nguồn, có cột cho biết giá trị
   nào khớp dấu hiệu. Hãy xem phần này trước tiên. Nếu giá trị thật của BindingDB cho
   dữ liệu nhập từ ChEMBL không chứa chữ "chembl", mức 1 sẽ trống và các cặp đó rơi
   xuống mức 2.
2. **Nhóm duplicate theo mức**: số nhóm, số nhóm có nhiều hơn một dòng ChEMBL, và số
   observation của mỗi nguồn.
3. **Bản ghi uncertain theo lý do**, cùng số cặp uncertain đã bị bỏ (xem mục 6, ý 6).
4. **Cặp ứng viên theo loại tài liệu chung.** Các cặp được chia theo `document_match`
   thành ba nhánh: chỉ PMID/DOI, chỉ patent, và patent + PMID/DOI. Mỗi nhánh có số cặp
   duplicate mức 1, mức 2, số cặp uncertain, và bảng lý do uncertain. Dùng patent làm
   khóa tài liệu là giả định thêm vào (đề bài chỉ nói PMID/DOI). Nếu nhánh "chỉ patent"
   có tỷ lệ uncertain, nhất là `value_differs` hay `protein_*`, cao hơn hẳn nhánh
   PMID/DOI, các cặp tìm qua patent đáng nghi.
5. **Dòng BindingDB nhập từ ChEMBL, theo cờ `|r|`.** Các dòng này được chia thành: có
   cặp, không có cặp (`provenance_without_counterpart`) và không có InChIKey. Mỗi nhóm
   có số dòng mang cờ lập thể tương đối `|r|` trong `molecule_cx_extension` (regex
   `relative_stereo_pattern`). Nếu tỷ lệ `|r|` ở nhóm không có cặp cao hơn hẳn nhóm có
   cặp, `|r|` là một nguyên nhân gây sót. Số dòng `|r|` không có cặp là giới hạn trên
   của phần sót do `|r|`, không phải số chính xác (xem mục 7). Phần này chỉ đếm, không
   đổi cách ghép.
6. **Observation bị ảnh hưởng ở mỗi nguồn**: duplicate, `keep=False`, uncertain, kèm
   tỷ lệ trên tổng.
7. **Theo endpoint.**
8. **5 ví dụ.** Mặc định gồm 2 ví dụ mức 1, 2 ví dụ mức 2 và 1 ví dụ mức 3, chọn ngẫu
   nhiên với seed cố định. Mỗi ví dụ kèm câu SQL để tra ChEMBL và cột để tra trong TSV
   BindingDB.
9. **Ca kiểm chứng**: BindingDB reactant set 50126247–50126294 và ChEMBL activity
   1423853–1423971. Bảng `KẾT QUẢ` chia các observation BindingDB trong khoảng thành ba
   nhóm: duplicate với activity ChEMBL trong khoảng, duplicate chỉ với activity ChEMBL
   ngoài khoảng, và không thành duplicate. Báo cáo không in một nhãn "bắt được" chung,
   vì hai nhóm đầu dẫn tới hai kết luận khác nhau. Bảng chẩn đoán bên dưới đếm số PMID,
   DOI, InChIKey, khóa protein và giá trị mà hai tập dòng có chung.

   Lần chạy ngày 21/10/2026: 3/48 observation khớp với activity trong khoảng; 36 bản ghi
   mức 1 khớp với 33 activity ChEMBL ngoài khoảng (474281–511386 và 1423972); hai tập ID
   chỉ chung 1 PMID và 3 InChIKey. Nghĩa là các dòng BindingDB này đúng là nhập từ
   ChEMBL và dedup tìm được bản gốc, nhưng giả thuyết "hai khoảng ID là cùng các phép đo"
   là sai.

Cuối báo cáo là phần `TÓM TẮT`. Toàn bộ số liệu nằm trong file `.json` cùng tên.

## 5. Audit độ nhiễu

### 5.1. Tập audit

Với mỗi endpoint (Ki, Kd, IC50, EC50), script đọc các dòng của endpoint đó từ hai bản
sao, rồi giữ lại những dòng thỏa **tất cả** điều kiện sau:

- `dedup_keep = True`, nên mỗi nhóm duplicate chỉ được tính một lần;
- qualifier là `=`;
- `protein_status` và `assay_status` đều là `resolved`;
- có pActivity.

Báo cáo in số dòng còn lại sau từng bước lọc, để thấy bước nào loại nhiều dữ liệu nhất.
Các mục 1, 2, 3 và 5 cần biết "cùng phân tử", nên chúng chỉ dùng thêm những dòng có
InChIKey.

### 5.2. Phạm vi

Mọi mục được tính cho ba phạm vi:

- `chembl`;
- `bindingdb`: phần BindingDB còn lại sau dedup, tức là các phép đo không trùng với
  ChEMBL;
- `all`: hai nguồn gộp lại sau dedup.

Phạm vi `all` gần nhất với dữ liệu sẽ dùng ở các bước sau.

### 5.3. Các mục

| Mục | Nhóm theo | In ra |
|---|---|---|
| 1. Replicate trong cùng assay | (nguồn, assay, protein, endpoint, phân tử) | số nhóm ≥ 2 phép đo; STD pActivity: min, P25, median, P75, P90, P99, max; tỷ lệ STD ≤ 0,3 và ≤ 0,5 |
| 2. Số assay trên mỗi cặp | (protein, endpoint, phân tử) | số cặp có 1, 2 và ≥ 3 assay |
| 3. Offset giữa các assay | (protein, endpoint, phân tử), mỗi cặp assay | \|Δ pActivity\|: min, P1, P25, median, P75, P99, max; tỷ lệ > 1,0, > 2,0, > 3,0 |
| 4. Phân bố pActivity | endpoint | min, P1, median, P99, max; số dòng < 0 và > 15 |
| 5. Protein tập trung nhiễu | protein (hash chuỗi) | top 20 theo số dòng, kèm số nhóm replicate và STD replicate trung vị |
| 6. ChEMBL gốc so với curation | endpoint | số dòng có value, units hoặc relation gốc khác `standard_*`, và các cặp đổi đơn vị phổ biến nhất |
| 7.1. STD replicate theo khoảng | như mục 1 | tỷ lệ nhóm có STD = 0 (mọi giá trị giống hệt), (0; 0,01), [0,01; 0,1), [0,1; 0,3), [0,3; 0,5), ≥ 0,5 |
| 7.2. \|Δ\| giữa assay theo khoảng | như mục 3 | các khoảng như 7.1; tỷ lệ chính xác |
| 7.3. Cờ `potential_duplicate` của ChEMBL | nhóm replicate và cặp assay | % nhóm STD ≈ 0 có cờ so với nhóm còn lại; số cặp assay có cờ ở ≥ 1 bên và tỷ lệ \|Δ\| = 0 khi có và không có cờ |
| 7.4. Cùng hay khác publication | cặp assay | với cặp cùng publication, khác publication, và khác publication không cờ: số cặp, % \|Δ\| = 0, % < 0,01, median, P75, % > 1,0 |

Báo cáo đầy đủ nằm ở `noise_audit_<thời điểm>.txt`. Màn hình chỉ in phần `TÓM TẮT` của
phạm vi `all`.

### 5.4. Cách đọc từng mục

- **Mục 1** đo nhiễu thực nghiệm khi cùng một phòng thí nghiệm, cùng một assay, đo lại
  cùng một phân tử. Đây là sàn nhiễu thấp nhất có thể kỳ vọng.
- **Mục 2** cho biết sau này có bao nhiêu cặp phân tử–protein đo được độ phân tán giữa
  các assay. Chỉ cặp có từ 2 assay trở lên mới làm được việc đó.
- **Mục 3** đo nhiễu hệ thống giữa các assay. Nhiễu này triệt tiêu khi lấy Δp trong
  cùng một assay, nên QC dựa trên Δp không nhìn thấy nó. Một |Δ| quanh 3,0 là dấu hiệu
  nhầm đơn vị µM với nM (Nelen et al. 2025).
- **Mục 4.** pActivity < 0 nghĩa là nồng độ lớn hơn 1 mol/L, còn > 15 nghĩa là nhỏ hơn
  1 fM. Cả hai gần như chắc chắn là lỗi đơn vị ở nguồn.
- **Mục 5** dùng để xem có protein hay họ protein nào vừa nhiều dữ liệu vừa có STD
  replicate cao bất thường không. Nelen et al. 2025 đã thấy hiện tượng này với họ
  Carbonic Anhydrase trong dữ liệu Ki.
- **Mục 6** cho biết bao nhiêu giá trị ChEMBL là kết quả quy đổi của ChEMBL chứ không
  phải con số tác giả công bố.
- **Mục 7** phân biệt phép đo lặp thật với cùng một con số được ghi nhiều lần. Lý do
  thêm mục này: lần chạy ngày 21/10/2026 cho STD replicate trung vị chỉ 0,006–0,14, và
  ngay trong phạm vi `chembl` |Δ| trung vị giữa hai assay Ki là 0,001. Hai phép đo độc
  lập không trùng nhau tới mức đó.
  - **7.1, 7.2:** nếu khoảng `= 0` chiếm phần lớn thì phần lớn "replicate" hay "assay thứ
    hai" là bản ghi lặp, không phải đo lại.
  - **7.3:** ChEMBL đặt `potential_duplicate` khi giá trị có vẻ là trích dẫn lại một giá
    trị đã có ở nguồn ChEMBL trước đó. Cờ này chủ yếu nói về cặp assay khác tài liệu.
  - **7.4:** hai assay cùng publication thường là cùng một thí nghiệm. Dòng "khác
    publication, không cờ" là ước lượng gần nhất cho độ lệch giữa các phép đo độc lập,
    và chỉ được in cho phạm vi `chembl`. Cờ `potential_duplicate` chỉ có ở ChEMBL, nên ở
    phạm vi `bindingdb` dòng đó trùng hẳn với "khác publication", còn ở phạm vi `all` nó
    vẫn chứa các cặp BindingDB chưa lọc được. Báo cáo ngày 21/10/2026 (02:42) còn in dòng
    này cho cả ba phạm vi; con số của `bindingdb` và `all` ở dòng đó không phải mức nhiễu.
    File JSON vẫn giữ đủ mọi nhóm.
    Publication được gom khi hai assay chung PMID, DOI (đã chuẩn hóa như ở dedup),
    patent, hoặc cùng tài liệu nguồn; bảng kiểm tra in số assay của publication lớn nhất
    để phát hiện gom nhầm.

### 5.5. Chẩn đoán thêm: `diagnose_noise.py`

Script này trả lời ba câu hỏi còn mở sau báo cáo audit ngày 21/10/2026. Nó chỉ đọc, và
dùng lại tập audit, `assay_key`, publication, nhóm replicate và cách dựng cặp assay của
`audit_noise.py`. Vì vậy số cặp và số nhóm phải khớp với mục 7.1 và 7.4 của báo cáo
audit; không khớp nghĩa là có lỗi.

- **A. Cặp assay khác publication có \|Δ\| = 0 (cùng nguồn).** Phạm vi chính là
  BindingDB. ChEMBL (bỏ cặp có cờ `potential_duplicate`) chạy cùng cách để làm đối
  chứng. Mỗi cặp assay được xếp theo cặp publication chứa nó, theo tỷ lệ \|Δ\| = 0 trên
  mọi cặp chung của hai publication đó. Hai tài liệu chép cùng một bảng số liệu cho tỷ
  lệ gần 1. Trùng hợp (hai lab cùng ra đúng 10 µM) cho tỷ lệ thấp, và thường là giá trị
  tròn (pActivity nguyên). Báo cáo in thêm loại tài liệu hai bên (patent / bài báo), số
  nhóm và phép đo liên quan, và các cặp publication nhiều \|Δ\| = 0 nhất để tra tay.
  Đây là căn cứ để quyết định có mở rộng dedup sang trong cùng nguồn hay không.
- **B. Publication nhiều assay nhất mỗi endpoint.** In số assay theo nguồn, số protein,
  phân tử, số khóa theo loại và các khóa phủ nhiều assay nhất. Một tài liệu thật có
  không quá 1 PMID và 1 DOI. Nhiều PMID/DOI trong một publication nghĩa là đã gom bắc
  cầu nhầm.
- **C. Nhóm replicate theo khoảng STD.** Với từng khoảng: % nhóm có dòng mang cờ `|r|`;
  % nhóm "trộn", tức vừa có dòng mang `|r|` vừa có dòng không mang (hai dạng lập thể bị
  gộp vì InChIKey bỏ cờ `|r|`, xem mục 7); % nhóm có nhiều mã phân tử nguồn. Bảng cuối
  cho tỷ lệ nhóm STD ≥ 0,5 khi bỏ các nhóm trộn, tức phần đóng góp của lỗi `|r|` vào
  phần đuôi STD lớn.
- **D. Nhóm nhiều mã phân tử nguồn.** Kiểm tra giả thuyết ở mục 5.6 (C): chia nhóm theo
  số mã phân tử nguồn (1 hay > 1) và theo việc phân tử có tâm bất đối chưa ghi cấu hình
  hay không (cột `molecule_unspecified_stereo_centers` của Stage 1; không tính đồng phân
  E/Z). Giả thuyết đúng nếu nhóm > 1 mã có tỷ lệ "có tâm chưa xác định" cao hơn hẳn nhóm
  1 mã, và trong nhóm > 1 mã, STD lớn tập trung ở phần có tâm chưa xác định.

Không ngưỡng nào ở đây được dùng để lọc dữ liệu.

### 5.6. Kết quả chẩn đoán ngày 21/10/2026 và quyết định

Nguồn: `runs/stage2a/noise_audit_20261021_024216.txt` và
`runs/stage2a/noise_diagnose_20261021_030543.txt`. Số cặp và số nhóm của báo cáo chẩn
đoán khớp với mục 7.1 và 7.4 của báo cáo audit (ví dụ Ki BindingDB 47.371 cặp; Ki ChEMBL
không cờ 44.873 cặp; 14.784 và 4.107 nhóm replicate Ki).

**Mức nhiễu giữa các phép đo độc lập** (ChEMBL, khác publication, không cờ
`potential_duplicate`; σ mỗi phép đo ≈ trung vị \|Δ\| / 0,954, giả định phân phối chuẩn):

| endpoint | cặp assay | \|Δ\| trung vị | P75 | \|Δ\| > 1,0 | σ mỗi phép đo |
|---|---|---|---|---|---|
| Ki | 44.873 | 0,480 | 0,981 | 24,0 % | ≈ 0,50 |
| Kd | 6.681 | 0,539 | 1,200 | 31,1 % | ≈ 0,57 |
| IC50 | 185.160 | 0,699 | 1,341 | 35,3 % | ≈ 0,73 |
| EC50 | 30.159 | 0,699 | 1,322 | 35,7 % | ≈ 0,73 |

Trong nhóm này chỉ 0,2–0,8 % cặp có \|Δ\| = 0 (Kd 5,6 %), nên cờ của ChEMBL đã bắt gần
hết bản sao. Các dòng tương ứng ở phạm vi `bindingdb` và `all` không dùng làm mức nhiễu
(mục 5.4, 7.4).

**Replicate trong cùng assay phần lớn không phải đo lặp.** Ki ChEMBL: 48 % nhóm có
STD < 0,01; EC50 ChEMBL: 54 %. Cờ `potential_duplicate` chỉ giải thích 18 % các nhóm
này. Vì vậy phân bố STD trong cùng assay không dùng làm mức nhiễu thực nghiệm.

**A. BindingDB có bản sao trong nội bộ nguồn, không phải trùng hợp.**
- Cặp assay khác publication có \|Δ\| = 0: BindingDB Ki 48,9 %, Kd 23,9 %, IC50 14,5 %,
  EC50 11,1 %; đối chứng ChEMBL (bỏ cặp có cờ) 0,2–0,8 % (Kd 5,6 %).
- Chỉ 0,5–11 % các cặp \|Δ\| = 0 đó là giá trị tròn (pActivity nguyên), nên không phải
  hai lab tình cờ cùng ra 10 µM.
- Hai kiểu sao chép:
  - chép cả bảng (cặp publication có ≥ 90 % cặp chung \|Δ\| = 0): Ki 32,8 %, IC50 55,2 %
    các cặp \|Δ\| = 0; phần lớn là patent–patent, ví dụ US12195427 ↔ US12351648 với
    273/274 giá trị giống hệt;
  - trích lại từng giá trị (hai tài liệu chỉ chung 1–2 cặp): Ki 54,0 %, IC50 20,6 %;
    chủ yếu bài báo–bài báo. Đây là kiểu ChEMBL đánh cờ `potential_duplicate`, nhưng
    BindingDB không có cờ đó. Đây là suy luận từ số liệu; chưa tra tay từng bài.
- Đây là đặc điểm của dữ liệu gốc. Dedup giữa hai nguồn (mục 4) không xử lý được, vì
  nó chỉ ghép ChEMBL với BindingDB.

**Quyết định 1 (cho Stage 2b): gộp giá trị lệch ≤ 0,01 log trước khi lấy mean.** Với
mỗi (phân tử, protein, endpoint), các giá trị lệch nhau ≤ 0,01 được tính là một phép
đo, rồi mới tính mean, STD và n. Lý do:
- gộp hai giá trị bằng nhau không làm đổi mean của chúng; gộp nhầm hai phép đo độc lập
  tình cờ bằng nhau chỉ làm giảm n;
- không gộp thì bản sao kéo mean về phía giá trị bị chép (6,0 chép ba lần cùng 7,0 cho
  6,25 thay vì 6,5) và làm STD nhỏ đi giả tạo;
- một quy tắc xử lý được cả hai kiểu sao chép ở trên và 168.693 cặp `value_close` giữa
  hai nguồn mà dedup để uncertain, không cần dựng thêm dedup theo publication trong
  cùng nguồn.

**B. Cách gom publication không gom nhầm.** Publication Kd có 4.242 assay là một bài
(PMID 29928781, DOI 10.1002/cmdc.201800398): 1 PMID, 1 DOI, khóa phủ 100 % assay; 3 phân
tử trên 2.142 protein, có ở cả hai nguồn. Trường hợp duy nhất nối hai tài liệu là IC50
hạng 2: hai bộ dữ liệu nộp trực tiếp vào ChEMBL (CHEMBL4507327, CHEMBL4507326) nối với
nhau vì đúng một assay chứa hoạt tính của cả hai (629 + 292 = 921 so với 920 assay).
Đây là đặc điểm của dữ liệu gốc, ảnh hưởng nhỏ.

**C. Cờ `|r|` không gây ra phần đuôi STD lớn.**
- Nhóm "trộn" (dạng `|r|` và dạng không `|r|` chung một InChIKey trong cùng assay): 0
  nhóm ở ChEMBL, 2–18 nhóm mỗi endpoint ở BindingDB. Bỏ chúng thì tỷ lệ nhóm STD ≥ 0,5
  đổi tối đa 0,2 điểm phần trăm.
- 0,0 % nhóm replicate ChEMBL có dòng mang `|r|`, khớp với việc ChEMBL không có
  CXSMILES (mục 7). **Đính chính:** trong trao đổi trước đó (không có trong tài liệu
  này) đã nói lỗi `|r|` "có ở cả hai phía". Câu đó sai. Đếm trên file observations
  (lệnh ở mục 7): ChEMBL 0 dòng, BindingDB 429.320 dòng. Lỗi chỉ nằm ở phía BindingDB.
- 17–31 % nhóm BindingDB có STD > 0 chứa nhiều MonomerID cho cùng một InChIKey (ChEMBL
  gần 0 %). Giả thuyết ban đầu: patent tách riêng các đồng phân ('isomer 1/2') nhưng
  SMILES không ghi cấu hình.

**D. Giả thuyết đồng phân không được xác nhận** (`noise_diagnose_20261021_122930.txt`,
mục D). Nhóm BindingDB > 1 mã phân tử:

| endpoint | % có tâm chưa xác định (> 1 mã / 1 mã) | STD ≥ 0,5: có tâm chưa xác định | STD ≥ 0,5: không có tâm chưa xác định |
|---|---|---|---|
| Ki | 36,3 % / 40,9 % | 40,6 % | 46,9 % |
| Kd | 35,2 % / 17,7 % | 32,4 % | 33,8 % |
| IC50 | 49,1 % / 33,3 % | 32,0 % | 29,4 % |
| EC50 | 50,3 % / 42,8 % | 37,2 % | 21,2 % |

- Điều kiện 1 (nhóm > 1 mã có nhiều tâm chưa xác định hơn hẳn) không đúng với Ki, chỉ
  đúng một phần với Kd và IC50.
- Điều kiện 2 không đúng: nhóm > 1 mã mà lập thể đầy đủ (hoặc không có tâm bất đối) vẫn
  có STD ≥ 0,5 ngang hoặc cao hơn, và chỉ 5–22 % có STD = 0. Tức là có hai MonomerID
  cùng InChIKey, cấu hình đầy đủ, mà giá trị khác nhau.
- Nguyên nhân của hiện tượng nhiều MonomerID **chưa xác định**. Các khả năng chưa kiểm
  tra: đồng phân E/Z (Stage 1 không đếm), dạng muối bị bỏ phần ion đối trong SMILES
  nguồn, hai đồng phân đối quang vẽ cùng một cấu hình. Đây là cách ghi của nguồn; việc
  của Stage 2a là gộp theo InChIKey nên gộp chúng vào một nhóm.
- Một quan sát phụ, chỉ là tương quan, chưa phải nguyên nhân: ở nhóm 1 mã, phân tử có
  tâm chưa xác định có tỷ lệ STD ≥ 0,5 gần gấp đôi phân tử lập thể đầy đủ, ở cả hai nguồn
  (ChEMBL Ki 24,1 % so với 11,5 %; IC50 31,2 % so với 15,9 %; BindingDB IC50 26,6 % so
  với 14,3 %). Cột `molecule_unspecified_stereo_centers` của Stage 1 đã có sẵn nếu
  Stage 2b muốn dùng.

**Quyết định 2 (cho Stage 2b): giữ và gắn cờ `multi_source_id`.** Nhóm (assay, protein,
phân tử) có nhiều hơn một mã phân tử nguồn được giữ, kèm cờ `multi_source_id`, để Stage
2b tự quyết và vẫn truy ngược được. Không bỏ, vì nguyên nhân chưa xác định: giả thuyết
đồng phân đã được kiểm tra và không đứng (mục D ở trên).

Phần audit kết thúc ở đây; không thêm chẩn đoán nào khác.

## 6. Quyết định thiết kế và lý do

1. **Đánh dấu, không xóa.** Dedup là một kết luận có thể sai. Giữ mọi dòng cùng lý do
   cho phép xem lại và đổi quy tắc mà không phải chạy lại Stage 1.
2. **Trùng giá trị không đủ để kết luận trùng phép đo.** Mọi cặp ứng viên phải có chung
   tài liệu. Script không tìm các cặp chỉ trùng phân tử, protein và giá trị mà không
   chung tài liệu. Đó không phải bằng chứng provenance, và trên dữ liệu thật phép join
   đó sinh ra rất nhiều trùng hợp ngẫu nhiên (ví dụ nhiều bài cùng báo 10.000 nM cho
   một chất tham chiếu).
3. **Số patent được dùng như PMID/DOI.** Dữ liệu patent không có PMID hay DOI. Nếu bỏ
   patent, mọi phép đo lấy từ patent sẽ không bao giờ ghép được, kể cả khi một nguồn
   nhập trực tiếp từ nguồn kia. Muốn chỉ dùng PMID/DOI như đề bài thì bỏ `"patent"` khỏi
   `document_keys`. Báo cáo ghi rõ loại tài liệu nào khớp trong `document_match`.
4. **Protein khớp theo chuỗi hoặc accession.** Hai nguồn có thể lưu hai construct khác
   nhau cho cùng một phép đo (ví dụ P03367 có bản 99 aa và bản 1.447 aa). Khi đã chung
   tài liệu, phân tử, endpoint và giá trị, khác biệt về construct là do cách ghi của
   nguồn. Cột `protein_match = accession` giúp tách riêng các ca này.
5. **Thiếu protein thì không kết luận duplicate.** Một bài báo có thể đo cùng một phân
   tử trên cả panel protein và cho cùng giá trị (ví dụ 10.000 nM cho mọi kinase không
   bị ức chế). Khi một bên thiếu protein, không phân biệt được các phép đo đó với nhau.
6. **Bỏ các cặp uncertain đã được giải quyết.** Khi một bài đo cùng phân tử hai lần
   (giá trị A và B), ghép chéo sinh ra hai cặp "khác giá trị". Nếu cả bốn dòng đều đã
   nằm trong cặp duplicate đúng, hai cặp chéo đó không mang thông tin. Script bỏ chúng
   và in số lượng.
7. **Giữ phía ChEMBL trong mỗi nhóm.** ChEMBL có giá trị đã curation (`standard_*`),
   khóa assay rõ ràng và là nguồn gốc của dữ liệu BindingDB nhập lại. Khi một dòng
   BindingDB khớp nhiều dòng ChEMBL giống hệt nhau, mọi dòng ChEMBL đều được giữ, vì
   dedup giữa hai nguồn không được gộp các dòng trong cùng một nguồn.
8. **Khóa assay của BindingDB gồm cả ENTRYID và ASSAYID.** ASSAYID chỉ duy nhất trong
   một entry, nên nếu chỉ dùng ASSAYID thì các assay của những bài khác nhau sẽ bị gộp
   làm một.
9. **STD replicate dùng ddof = 1** (`replicate_std_ddof`). Đây là độ lệch chuẩn mẫu,
   cách thường dùng khi ước lượng nhiễu từ vài lần đo lặp. Nhóm chỉ có một phép đo
   không có STD và không được tính vào mục 1.
10. **Offset giữa assay dùng pActivity trung bình của từng assay**, để nhiễu replicate
    của mục 1 không bị tính lại. Script lấy trị tuyệt đối vì dấu của hiệu chỉ phụ thuộc
    thứ tự đặt tên assay.
11. **Phân vị offset lấy từ histogram.** Số cặp assay tăng theo bình phương số assay
    của mỗi phân tử, nên không giữ hết các hiệu trong bộ nhớ. min, max và tỷ lệ vượt
    ngưỡng vẫn chính xác. Các phân vị còn lại có sai số không quá nửa độ rộng ô (mặc
    định 0,0005).
12. **Bộ nhớ có giới hạn.** Dedup xử lý từng phân vùng InChIKey. Audit xử lý từng
    endpoint, và đọc mục 6 theo batch.

## 7. Giới hạn đã biết và điểm cần xác nhận

- **Dấu hiệu nguồn nhập liệu chưa được đối chiếu với dữ liệu thật.** Script in mọi giá
  trị `source_curation` để bạn kiểm tra. Nếu cần, sửa `provenance_markers` rồi chạy lại
  với `--force`.
- **Định dạng số patent** của hai nguồn chưa được đối chiếu. Phần chuẩn hóa giả định
  dạng như `US-9718838-B2` hoặc `US9718838`.
- **Dòng không có InChIKey** (RDKit không xử lý được cấu trúc) không thể ghép cặp. Chúng
  chỉ xuất hiện trong file ánh xạ khi có dấu hiệu nguồn nhập liệu.
- **Hai nguồn cho InChIKey khác nhau** cho cùng một phép đo (ví dụ một bên ghi lập thể,
  bên kia không) thì cặp đó bị bỏ sót. InChIKey được tính lại bằng RDKit cho cả hai
  nguồn, nhưng SMILES đầu vào vẫn có thể khác nhau. Có một nguyên nhân cụ thể đã biết:
  - 429.320 dòng BindingDB mang cờ `r` (mọi tâm lập thể là tương đối); ChEMBL 0 dòng
    (đếm ngày 21/10/2026 bằng lệnh ở cuối mục này; bản trước ghi 421.184 vì chỉ đếm
    khối CX đúng bằng `|r|`, bỏ sót 8.136 dòng có `r` đi kèm trường khác);
  - RDKit 2026.09.1 bỏ qua cờ này, nên InChIKey phía BindingDB mã hóa lập thể như thể
    là tuyệt đối;
  - ChEMBL chỉ có SMILES thường, không có CXSMILES.

  Vì vậy cùng một chất có thể mang hai InChIKey khác nhau ở hai nguồn, và dedup có thể
  đếm thiếu. 429.320 là số dòng có `|r|`, không phải số cặp bị sót; số cặp bị sót chưa
  biết. Báo cáo dedup (mục 4.6, phần 5) đếm số dòng `|r|` trong các dòng BindingDB nhập
  từ ChEMBL mà không có cặp, làm giới hạn trên cho phần sót này.

  Phạm vi ảnh hưởng: mỗi bản trùng bị sót được audit tính như một assay thứ hai với
  |Δ| ≈ 0. Hệ quả là mục 3 ở phạm vi `all` trông ít nhiễu hơn thực tế, còn mục 2 có
  nhiều cặp ≥ 2 assay hơn thực tế. Phạm vi `chembl` và `bindingdb` riêng không bị ảnh
  hưởng. STD replicate (mục 1) cũng không bị ảnh hưởng, vì khóa assay có tiền tố nguồn
  nên hai bản của cùng một phép đo không bao giờ rơi vào cùng nhóm replicate.

  Trong cùng một nguồn, việc bỏ cờ `|r|` hiếm khi gộp hai dạng lập thể vào một nhóm
  replicate: 2–18 nhóm mỗi endpoint, chỉ ở BindingDB (mục 5.6, C). Lỗi này vẫn phải sửa
  trước bước MMP của Stage 2. Lệnh đếm dòng `|r|` theo nguồn để chốt phạm vi:

  ```powershell
  python -c "import pyarrow.parquet as pq, pyarrow.compute as pc; [print(f, pc.sum(pc.fill_null(pc.match_substring_regex(pq.read_table('data/observations/'+f, columns=['molecule_cx_extension'])['molecule_cx_extension'], '[|,]r[|,]'), False))) for f in ['chembl37_observations.parquet','bindingdb202610_observations.parquet']]"
  ```
- **Cặp có protein khác nhau ở cả hai bên** được coi là hai phép đo khác nhau và không
  được ghi lại. Nếu hai nguồn ánh xạ cùng một construct sang hai accession khác nhau,
  cặp đó bị bỏ sót.
- **Phân vị trong mục 3 là xấp xỉ** (xem mục 6, ý 11).
- **Trong mục 3, mỗi cặp assay góp một giá trị.** Một chất tham chiếu đo ở hàng trăm
  assay đóng góp rất nhiều cặp, nên phân bố bị lệch về các chất đó.
- **Thời gian chạy** (đo ngày 21/10/2026): dedup 5,8 phút, audit 0,7 phút, chẩn đoán
  0,6 phút. Dedup đọc lại các cột cần thiết của hai file cho từng phân vùng (28 lần với
  cấu hình mặc định), rồi ghi hai bản sao. Audit đọc từng endpoint và parse JSON của mọi
  dòng ChEMBL trong tập audit.

## 8. Tham số chính trong `config.json` (`stage2a`)

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `dedup.keep_source` | `chembl` | nguồn được giữ trong mỗi nhóm duplicate |
| `dedup.provenance_markers` | `bindingdb: ["chembl"]`, `chembl: ["bindingdb"]` | dấu hiệu nguồn nhập liệu |
| `dedup.document_keys` | `pmid`, `doi`, `patent` | loại tài liệu được dùng để ghép |
| `dedup.value_exact_relative_tolerance` | 1e-9 | ngưỡng "bằng nhau" của nồng độ |
| `dedup.value_close_p_tolerance` | 0,05 | ngưỡng "gần bằng" theo \|Δp\| |
| `dedup.letters_per_partition` | 1 | số chữ cái mỗi phân vùng (tăng lên thì ít lượt đọc hơn nhưng tốn bộ nhớ hơn) |
| `dedup.examples_per_level` | `1: 2`, `2: 2`, `3: 1` | số ví dụ in ra cho mỗi mức |
| `dedup.known_case` | 50126247–50126294 / 1423853–1423971 | ca kiểm chứng |
| `dedup.relative_stereo_pattern` | `[\|,]r[\|,]` | regex tìm cờ `r` trong `molecule_cx_extension`; chỉ dùng để đếm trong báo cáo |
| `audit.replicate_std_ddof` | 1 | ddof của STD replicate |
| `audit.replicate_std_thresholds` | 0,3; 0,5 | ngưỡng tỷ lệ STD trong mục 1 |
| `audit.offset_thresholds` | 1,0; 2,0; 3,0 | ngưỡng tỷ lệ \|Δ\| trong mục 3 |
| `audit.offset_histogram_bin_width` | 0,001 | độ rộng ô histogram của mục 3 |
| `audit.p_activity_low` / `p_activity_high` | 0 / 15 | khoảng pActivity hợp lý trong mục 4 |
| `audit.top_proteins` | 20 | số protein trong mục 5 |
| `audit.diagnostic_bin_edges` | 0,01; 0,1; 0,3; 0,5 | các mốc chia khoảng STD và \|Δ\| trong mục 7 |
| `audit.zero_tolerance` | 1e-9 | "bằng 0" trong mục 7 (bỏ qua sai số làm tròn của phép lấy trung bình) |
| `audit.near_zero_std` | 0,01 | "STD ≈ 0" trong mục 7.3 |
| `audit.potential_duplicate_key` / `_value` | `potential_duplicate` / 1 | cờ đọc từ `activity_context` của ChEMBL |
| `diagnose.copy_fraction_edges` | 0,1; 0,5; 0,9 | các mốc tỷ lệ \|Δ\| = 0 của cặp publication (mục 5.5 A) |
| `diagnose.copy_min_shared_pairs` | 3 | cặp publication chung ít hơn số cặp assay này thì để riêng, không xếp theo tỷ lệ |
| `diagnose.round_p_tolerance` | 1e-6 | "pActivity nguyên" (giá trị tròn) |
| `diagnose.top_publication_pairs` / `top_publications` / `top_keys` | 10 / 3 / 5 | số ví dụ in ra ở mục A và B |
