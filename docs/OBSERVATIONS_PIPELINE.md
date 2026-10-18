# Bước Raw → Observations

Tài liệu này mô tả đúng code đang có trong `run_observations.py` và gói
`src/molopt/observations/` (bốn module `chembl.py`, `bindingdb.py`, `normalize.py`,
`pipeline.py`), cùng kết quả lần chạy ngày 18–19/10/2026. Mọi tham số được nhắc tới
(tên file, tên cột, bảng đơn vị, kích thước lô) đều nằm trong `config.json`.

---

## 1. Mục tiêu của bước này

Bước này biến dữ liệu thô của ChEMBL 37 và BindingDB 202610 thành một bảng
*observation* thống nhất. Mỗi observation là đúng một phép đo: một phân tử, đo trên một
protein, trong một thí nghiệm (assay), theo một endpoint (Ki, Kd, IC50 hoặc EC50). Mỗi
dòng mang theo giá trị gốc đúng như trong nguồn, cộng thêm các cột suy ra như nồng độ
quy về mol/L, pActivity, canonical SMILES, InChIKey, và bốn cột trạng thái cho biết
từng phần của dòng đã được xác định chắc chắn hay chưa.

Ở bước này chưa có cặp phân tử A–B, chưa có hiệu Δp, chưa gộp các phép đo lặp lại và
chưa khử trùng lặp giữa hai nguồn. Code cũng không bỏ dòng nào và không áp thêm bộ lọc
khoa học nào. Dòng nào thiếu thông tin hoặc có giá trị sai dạng vẫn được ghi ra, với
trạng thái và lý do cụ thể, để bước sau tự quyết định có dùng hay không.

Có bốn chế độ chạy qua `python run_observations.py --mode ...`:
- `inspect` chỉ in danh sách file, schema và vị trí các cột cần dùng, không đọc dòng
  dữ liệu.
- `smoke` chạy thử 5.000 dòng mỗi nguồn vào một thư mục mới
  `runs/observations/smoke_<thời điểm>/`.
- `full` chạy toàn bộ vào `data/observations/`.
- `verify` đối chiếu output của `full` ngược về raw.

---

## 2. Hai nguồn đầu vào

**ChEMBL 37.** Thư mục `raw/chembl37/` chứa bản phát hành dạng file nén
`chembl_37_sqlite.tar.gz`. Code không bao giờ tự giải nén. Bản SQLite đã được giải nén
sẵn, bằng tay, vào `work/chembl_37/chembl_37_sqlite/chembl_37.db`, và code chỉ đọc file
này. Hàm `open_chembl_readonly` mở nó qua URI `file:...?mode=ro`, nên ở mức SQLite kết
nối không thể ghi vào file. ChEMBL phải đọc theo cách này vì dữ liệu cần thiết nằm rải
trên mười bảng quan hệ (activities, assays, variant_sequences, target_dictionary…), và
muốn ghép chúng thì phải chạy truy vấn SQL có join và có index. Việc đó đòi hỏi một file
cơ sở dữ liệu truy cập ngẫu nhiên được, không thể làm khi đọc tuần tự từ file `.tar.gz`.
Kết nối còn đặt `PRAGMA temp_store = MEMORY` để các cấu trúc tạm của SQLite nằm trong
RAM thay vì trong thư mục temp của hệ thống, tức là bên ngoài `E:\eval_training`.

**BindingDB 202610.** Thư mục `raw/bindingdb202610/` có ba file ZIP được dùng.
`BindingDB_All_202610_tsv.zip` chứa bảng dữ liệu chính `BindingDB_All.tsv`.
`BindingDB_rsid_eaids_202610_tsv.zip` là bảng nối từ Reactant_set_id sang assay.
`BindingDB_Assays_202610_tsv.zip` là bảng mô tả assay. Mỗi file được đọc bằng
`zipfile` của Python: mở thành viên `.tsv` duy nhất trong ZIP như một luồng giải nén và
đưa thẳng vào `csv.reader` với dấu tab, không ghi bản giải nén nào ra đĩa. Cách này dùng
được vì mỗi dòng của BindingDB_All tự chứa gần như mọi thông tin của nó, nên chỉ cần đọc
tuần tự từ đầu đến cuối một lần. Phần duy nhất cần tra cứu (assay) được nạp trước vào bộ
nhớ từ hai ZIP nhỏ. Đọc trực tiếp từ ZIP vừa giữ raw nguyên vẹn, vừa không phải tạo một
bản sao giải nén nhiều GB. Bộ đọc dùng `quoting=csv.QUOTE_NONE`, vì file TSV này không
dùng dấu ngoặc kép để bọc trường: dấu ngoặc kép xuất hiện trong SMILES hay tên chất phải
được giữ nguyên là ký tự. Giới hạn độ dài trường của module `csv` được nâng lên
2.147.483.647, vì một ô chuỗi protein hay danh sách PDB có thể rất dài.

**Hai file SDF không được đọc ở bước này**: `chembl_37.sdf.gz` và
`BindingDB_All_2D_202610_sdf.zip`. Cấu trúc phân tử ở bước này chỉ lấy từ cột SMILES
của mỗi nguồn. Tên hai file SDF được ghi vào `config.json` (`chembl.sdf_not_read`,
`bindingdb.sdf_zip_not_read`) để chế độ `inspect` in ra rõ ràng là chúng bị bỏ qua.

---

## 3. Nhánh ChEMBL (`chembl.py`)

### 3.1. Câu truy vấn

Bảng gốc là `activities`, lọc theo `standard_type IN ('Ki','Kd','IC50','EC50')`. Mỗi
dòng `activities` thỏa điều kiện cho ra đúng một observation. Câu `EXTRACTION_QUERY`
nối bảng này với các bảng khác, đều bằng `LEFT JOIN`, để một dòng activity không bị mất
khi bảng bên kia thiếu dữ liệu:

- `source` (theo `src_id`), lấy `src_short_name`, tức nguồn biên soạn như LITERATURE.
- `assays` (theo `assay_id`), lấy `chembl_id`, mô tả, `assay_type`, `variant_id`, `tid`
  và các trường ngữ cảnh: organism, tax_id, strain, tissue, cell_type, subcellular
  fraction, relationship_type, confidence_score, src_assay_id, bao_format.
- `variant_sequences` (theo `assays.variant_id`), lấy `sequence`, `accession`,
  `mutation`, `isoform`.
- `target_dictionary` (theo `assays.tid`), lấy `chembl_id`, `pref_name`,
  `target_type`, `organism`.
- Một truy vấn con trên `target_components`, gom theo `tid`, đếm số component của target
  (`component_count`) và lấy `MIN(component_id)`. Sau đó nối tiếp sang
  `component_sequences` để lấy `sequence` và `accession` của component đó. Component
  này chỉ thực sự được dùng khi target có đúng một component.
- `molecule_dictionary` và `compound_structures` (theo `molregno`), lấy ChEMBL ID của
  phân tử, `chirality`, `canonical_smiles`, `standard_inchi_key`.
- `docs` (theo `doc_id`), lấy ChEMBL ID của tài liệu, `pubmed_id`, `doi`, `patent_id`.

Từ `activities`, code lấy bộ giá trị đã chuẩn hóa của ChEMBL (`standard_value`,
`standard_units`, `standard_relation`) làm giá trị, đơn vị và qualifier của observation.
Bộ giá trị gốc như tác giả công bố (`value`, `units`, `relation`) cùng `record_id`,
`standard_flag`, `activity_comment`, `data_validity_comment`, `potential_duplicate`,
`pchembl_value` được giữ nguyên trong cột JSON `activity_context`. Trước khi đọc dữ
liệu, `check_schema` so danh sách cột cần dùng (`REQUIRED_COLUMNS`) với schema thật của
SQLite. Nếu thiếu cột nào, chương trình dừng và in tên cột thiếu.

### 3.2. Lấy protein theo từng phép đo

Hàm `resolve_protein` quyết định chuỗi protein của từng phép đo theo thứ tự sau:

1. Nếu assay có khai báo `variant_id`, chuỗi lấy từ `variant_sequences.sequence`, và
   `protein_sequence_origin = chembl_variant_sequences`. Nếu đã khai báo variant mà
   chuỗi variant rỗng, dòng thành `unresolved` với lý do
   `variant_declared_but_sequence_missing`. Code **không** lấy chuỗi wild type thay vào,
   vì phép đo đó được thực hiện trên đột biến, và gán chuỗi wild type sẽ cho ra một cặp
   phân tử–protein sai mà không có dấu hiệu gì.
2. Chỉ khi không khai báo variant, code mới dùng chuỗi component của target, với
   `protein_sequence_origin = chembl_component_sequences`:
   - target không có component nào → `unresolved`, `target_has_no_protein_component`;
   - target có nhiều hơn một component (phức hợp, họ protein…) →
     `unresolved`, `target_component_count_not_1:N`, vì không biết phân tử gắn vào
     chuỗi nào;
   - có đúng một component nhưng chuỗi rỗng → `unresolved`,
     `component_sequence_missing`;
   - còn lại → `resolved`.

### 3.3. Chia lô theo activity_id và lý do không dùng ORDER BY trên toàn bộ kết quả

Output của ChEMBL được ghi theo thứ tự `activity_id` tăng dần, để hai lần chạy trên cùng
dữ liệu cho ra file có cùng thứ tự dòng, thuận tiện cho việc đối chiếu và tái lập. Cách
đơn giản nhất là thêm `ORDER BY act.activity_id` vào cuối câu join lớn. Nhưng khi đó
SQLite phải dựng một B-tree tạm chứa toàn bộ khoảng 5,3 triệu dòng kết quả của phép join,
gồm cả chuỗi protein và mô tả assay dài, rồi mới sắp xếp. Kích thước lên tới nhiều GB,
và vì `temp_store = MEMORY` nên tất cả nằm trong RAM. EXPLAIN QUERY PLAN xác nhận
phương án này có bước "USE TEMP B-TREE FOR ORDER BY".

Vì vậy hàm `iter_records` làm hai bước:

1. `selected_activity_ids` chạy `SELECT activity_id FROM activities WHERE standard_type
   IN (...) ORDER BY activity_id` và lưu kết quả vào một `array('q')` (mảng số nguyên
   64-bit). Bước này chỉ sắp xếp số nguyên: 5,3 triệu × 8 byte, khoảng 43 MB.
2. Mảng id được cắt thành từng lô 10.000 phần tử (`chembl.id_batch_size`). Với mỗi lô,
   câu join lớn chạy với `WHERE act.activity_id IN (?, ?, ...) ORDER BY act.activity_id`.
   Trong một lô, SQLite đi theo khóa chính `activity_id` nên không cần B-tree tạm. Các
   lô lại nối tiếp nhau theo thứ tự tăng dần, nên thứ tự toàn cục vẫn đúng. Con số
   10.000 phải nhỏ hơn giới hạn số tham số của một câu SQLite (32.766 ở SQLite ≥ 3.32).

Với smoke, tham số `limit` được đưa vào bước 1 (`LIMIT ?`), nên smoke lấy đúng các
`activity_id` nhỏ nhất.

### 3.4. Bản ghi đầu ra

`build_record` tạo một dict cho mỗi dòng kết quả, chép nguyên giá trị nguồn vào các cột
`*_source`. Chuyển đổi đơn vị, qualifier và cấu trúc phân tử không làm ở đây mà ở
`normalize.py` (xem mục 5). `assay_status` là `resolved` khi tìm thấy dòng trong bảng
`assays`, ngược lại là `unresolved` với lý do `assay_row_missing`.

---

## 4. Nhánh BindingDB (`bindingdb.py`)

### 4.1. Cấu trúc file All và cột protein

`BindingDB_All.tsv` có 640 cột. 40 cột đầu là thông tin chung của một phép đo:
- Reactant_set_id, Ligand SMILES, InChI, InChI Key, MonomerID, tên ligand.
- Target Name, Target Source Organism.
- Bốn cột giá trị `Ki (nM)`, `IC50 (nM)`, `Kd (nM)`, `EC50 (nM)`; thêm `kon`, `koff`.
- pH, nhiệt độ, Curation/DataSource, Article DOI, Entry DOI, PMID, PubChem AID,
  Patent Number.
- Các liên kết và ID ngoài.
- Cột `Number of Protein Chains in Target (>1 implies a multichain complex)`.

600 cột còn lại là 50 nhóm, mỗi nhóm 12 cột, cho chain 1 đến chain 50. Mỗi nhóm có chuỗi
(`BindingDB Target Chain Sequence k`), PDB ID của chain, và các trường UniProt SwissProt
(Recommended Name, Entry Name, Primary ID, Secondary ID, Alternative ID) cùng TrEMBL
(Submitted Name, Entry Name, Primary ID, Secondary ID, Alternative ID).

Hàm `resolve_protein` chỉ dùng cột của **chính dòng đang đọc**, không tra chuỗi ở bất kỳ
đâu khác:

- Nếu số chain trống → `unresolved`, `chain_count_missing`. Nếu không phải số nguyên →
  `invalid`.
- Nếu số chain khác 1 → `unresolved`, `target_chain_count_not_1:N`. Với phức hợp nhiều
  chain, không biết ligand gắn vào chain nào.
- Nếu đúng một chain: chuỗi lấy từ `BindingDB Target Chain Sequence 1`. Accession lấy
  `UniProt (SwissProt) Primary ID of Target Chain 1` nếu có
  (`protein_accession_db = swissprot`), không thì `UniProt (TrEMBL) Primary ID of Target
  Chain 1` (`trembl`). Nếu cả hai đều trống, dòng vẫn `resolved` vì chuỗi đã có, nhưng
  ghi lý do `no_uniprot_accession`.

Chuỗi được lấy từ chính dòng đo vì cùng một accession UniProt có thể ứng với các
construct khác nhau ở các phép đo khác nhau. Ví dụ P03367 (HIV-1 protease) có phép đo
trên domain protease 99 aa và có phép đo trên polyprotein 1.447 aa. Tra chuỗi theo
accession sẽ gán nhầm construct cho một trong hai nhóm.

Ngoài ra, mỗi dòng được kiểm tra độ rộng. Dòng nào có nhiều trường hơn header nghĩa là
việc tách cột đã sai, nên chương trình dừng hẳn thay vì đoán cột nào là cột nào.

### 4.2. Lấy assay qua `rsid_eaids`

BindingDB_All không có cột assay. Đường đi của assay là:

```
BindingDB_All."BindingDB Reactant_set_id"
    → BindingDB_rsid_eaids.REACTANT_SET_ID
    → BindingDB_rsid_eaids.ENTRYID_ASSAYID   (ví dụ "1025_1")
    → tách tại "_" thành (ENTRYID = "1025", ASSAYID = "1")
    → BindingDB_Assays theo cặp (ENTRYID, ASSAYID)
```

`load_assay_links` đọc toàn bộ `rsid_eaids` vào một dict
`Reactant_set_id → danh sách các cặp (ENTRYID, ASSAYID) khác nhau`. Hàm
`parse_entry_assay_id` chỉ chấp nhận chuỗi tách được thành đúng hai số nguyên. Mọi dạng
khác đều làm chương trình dừng, kèm số dòng. `load_assays` đọc `BindingDB_Assays` vào
dict `(ENTRYID, ASSAYID) → (ASSAY_NAME, DESCRIPTION)`. Nếu một cặp khóa xuất hiện hai
lần, giá trị được đánh dấu `None` để không chọn bừa một trong hai. Khóa phải là cặp vì
ASSAYID được đánh số lại trong từng entry. Trong bảng Assays của bản 202610, ASSAYID 1
xuất hiện ở 54.412 trên 56.379 entry, nên ASSAYID một mình không xác định được assay.

**Vì sao không join thẳng Reactant_set_id vào ENTRYID.** Hai cột này là hai hệ đánh số
khác nhau. Reactant_set_id đánh số từng tập phép đo ligand–target, có hàng triệu giá trị.
ENTRYID đánh số từng entry, tức một bài báo, bằng sáng chế hay đợt nộp dữ liệu, chứa
nhiều phép đo. Cả hai đều là số nguyên, nên một phép join thẳng vẫn "chạy được" và cho ra
kết quả, nhưng ghép một phép đo với assay của một entry không liên quan. Lỗi này không
báo gì. Trong ví dụ ở mục 10, Reactant_set_id 13888 thuộc ENTRYID 1025, chứ không phải
ENTRYID 13888.

`resolve_assay` trả về một trong các trạng thái sau:
- `resolved`: có đúng một cặp liên kết, và cặp đó có đúng một dòng trong bảng Assays.
- `unresolved`, kèm một trong bốn lý do:
  - `no_rsid_assay_link`: Reactant_set_id không có trong `rsid_eaids`;
  - `multiple_assay_links:N`: có nhiều cặp, không chọn bừa;
  - `assay_pair_not_in_assay_table`;
  - `duplicate_assay_table_rows`.

Thiếu assay không làm mất protein hay giá trị đo: các trạng thái được tính độc lập.

### 4.3. Kiểm tra giao khóa lúc nạp

Ngay sau khi nạp hai bảng assay, `assert_domains_intersect` chạy hai phép kiểm tra:

1. Tập các cặp (ENTRYID, ASSAYID) lấy từ `rsid_eaids`, đối chiếu với khóa của bảng
   Assays.
2. Reactant_set_id của 20.000 dòng đầu BindingDB_All (`key_check_rows`), đối chiếu với
   khóa của `rsid_eaids`.

Mỗi phép kiểm tra đếm số khóa bên trái tìm thấy ở bên phải và ghi vào log. Nếu con số là
0, chương trình dừng với thông báo rằng hai miền khóa không giao nhau, tức đang join hai
hệ đánh số khác nhau hoặc tách khóa sai. Đây là chốt chặn chống lỗi cấu trúc, như tách
sai ký tự phân cách hay nhầm cột. Nó không phải phép đo độ phủ, vì chỉ cần khớp một khóa
là qua. Ở lần chạy 18/10, log ghi: 224.448/224.448 cặp khớp ở phép thứ nhất và
19.971/20.000 Reactant_set_id khớp ở phép thứ hai. Vì `parse_entry_assay_id` dừng ngay
khi gặp dạng lạ, việc lần chạy hoàn tất cũng cho thấy mọi dòng của `rsid_eaids` đều có
dạng `số_số`.

### 4.4. Một dòng raw thành nhiều observation

Một dòng BindingDB_All có thể có giá trị ở nhiều cột endpoint cùng lúc, ví dụ vừa Ki vừa
IC50. `build_records` tính phần chung của dòng (protein, assay, phân tử, tài liệu) một
lần, rồi duyệt bốn cột `Ki (nM)`, `Kd (nM)`, `IC50 (nM)`, `EC50 (nM)`. Mỗi ô không rỗng
tạo một bản ghi riêng: `endpoint` là tên endpoint, `activity_value_source` là nội dung ô
giữ nguyên, kể cả dấu cách đầu hay tiền tố như `>`, và `activity_unit_source = "nM"`
lấy từ tên cột qua `config.json`. Ô rỗng không tạo bản ghi. Các cột kon và koff không
thuộc phạm vi bước này. Vì vậy số observation của BindingDB bằng tổng số ô endpoint
không rỗng, không phải số dòng raw.

---

## 5. Phần chuẩn hóa dùng chung (`normalize.py`)

Mọi hàm trong `normalize.py` là hàm thuần: nhận giá trị và trả giá trị, không đọc file.
Chúng không bao giờ ghi đè cột nguồn: kết quả đi vào các cột suy ra riêng, do
`pipeline.build_observation` ghép vào cạnh cột gốc.

### 5.1. Quy đổi đơn vị về mol/L

Code chỉ quy đổi những đơn vị có trong bảng tường minh `unit_to_molar` của
`config.json`:

| Đơn vị | Hệ số nhân ra mol/L |
|---|---|
| `M` | 1 |
| `mM` | 1e-3 |
| `uM`, `µM` (micro sign U+00B5), `μM` (chữ Hy Lạp mu U+03BC) | 1e-6 |
| `nM` | 1e-9 |
| `pM` | 1e-12 |
| `fM` | 1e-15 |

Một đơn vị không có trong bảng sẽ không bị đoán: dòng thành `unresolved` với lý do
`unit_not_in_table:<đơn vị>`, nồng độ và pActivity để trống. Ví dụ `ug.mL-1` muốn đổi
ra mol/L thì phải có khối lượng phân tử, mà bước này không dùng. Đơn vị trống cho lý do
`missing_unit`. Với BindingDB, đơn vị luôn là `nM` vì nó nằm sẵn trong tên cột.

### 5.2. Qualifier

Qualifier được chia làm hai nhóm. Nhóm chính xác chỉ có `=`. Nhóm bị chặn (censored) gồm
`<`, `>`, `<=`, `>=`, `<<`, `>>`.

- ChEMBL lấy qualifier từ cột `standard_relation`.
- BindingDB không có cột qualifier riêng: qualifier dính ở đầu ô giá trị, như `>10000`.
  Hàm `split_embedded_qualifier` thử các qualifier đã biết, ưu tiên chuỗi dài trước để
  `>=` không bị nhận nhầm thành `>`, rồi tách ra. Ô không có tiền tố được coi là `=`
  (`implicit_qualifier`).

Hàm `parse_activity` kiểm tra theo thứ tự cố định. Điều kiện đầu tiên không đạt quyết
định status và lý do:

| Thứ tự | Điều kiện không đạt | Status | Lý do |
|---|---|---|---|
| 1 | Giá trị trống | `unresolved` | `missing_value` |
| 2 | Qualifier trống | `unresolved` | `missing_qualifier` |
| 3 | Qualifier ngoài bảng (ví dụ `~`) | `unresolved` | `qualifier_not_in_table:<q>` |
| 4 | Giá trị không phải số | `invalid` | `non_numeric_value:<v>` |
| 5 | Giá trị là nan hoặc inf | `invalid` | `non_finite_value:<v>` |
| 6 | Giá trị ≤ 0 | `invalid` | `non_positive_value:<v>` |
| 7 | Đơn vị trống | `unresolved` | `missing_unit` |
| 8 | Đơn vị ngoài bảng | `unresolved` | `unit_not_in_table:<u>` |

Qua hết tám bước thì dòng là `resolved` và có nồng độ mol/L, rồi tách làm hai trường
hợp:
- Qualifier bị chặn: `activity_censored = true`, pActivity để trống, lý do
  `censored_qualifier:<q>`. Giá trị như ">10 µM" chỉ là một cận, không phải một con số
  đo được, nên không quy thành một pActivity cụ thể.
- Qualifier `=`: `activity_censored = false` và có pActivity.

### 5.3. pActivity

pActivity = −log₁₀(nồng độ tính bằng mol/L). Ví dụ 300 nM = 3×10⁻⁷ M cho
pActivity ≈ 6,52.

### 5.4. Cấu trúc phân tử: canonical SMILES, InChIKey, CXSMILES

`describe_structure` nhận chuỗi SMILES nguồn: `compound_structures.canonical_smiles` ở
ChEMBL, `Ligand SMILES` ở BindingDB. Chuỗi nguồn luôn được giữ nguyên trong
`molecule_source_smiles`. Các bước:

1. Chuỗi trống → `unresolved`, `missing_source_smiles`.
2. `cx_extension` cắt phần mở rộng CXSMILES, tức mọi thứ sau `" |"` như `|r|` hay
   `|&1:1|`, và giữ nguyên văn vào `molecule_cx_extension`.
3. `Chem.MolFromSmiles` đọc chuỗi. Nếu RDKit ném lỗi → `unresolved`,
   `rdkit_parse_error:<lỗi>`. Nếu trả về `None` (SMILES không hợp lệ) → `invalid`,
   `rdkit_parse_failed`.
4. `Chem.MolToSmiles(..., isomericSmiles=True)` cho canonical SMILES có lập thể. Lỗi →
   `unresolved`, `canonical_smiles_failed:<lỗi>`. Rỗng → `invalid`.
5. `Chem.MolToInchiKey` cho InChIKey. Lỗi → `unresolved`, `inchikey_failed:<lỗi>`.
   Không đúng dạng `XXXXXXXXXXXXXX-XXXXXXXXXX-X` → `invalid`, `invalid_inchikey`.
6. Đếm tâm lập thể (`count_stereo_centers`, dùng `FindMolChiralCenters` với thuật toán
   CIP mới), gồm tổng số tâm và số tâm chưa xác định. Bước này nằm trong
   `try/except (RuntimeError, ValueError)` riêng. Nếu RDKit thất bại, như lỗi "Digraph
   generation failed: more than 100000 nodes found" ở phân tử rất lớn, hai cột đếm để
   trống, lý do ghi `stereo_annotation_failed:<lỗi>`, nhưng status vẫn `resolved`. Canonical
   SMILES và InChIKey đã tính xong ở bước 4–5 vẫn hợp lệ; chỉ phần mô tả phụ bị thiếu.
7. `describe_stereo_groups` ghi các nhóm lập thể nâng cao (enhanced stereo: ABSOLUTE,
   OR, AND) mà RDKit đọc được vào `molecule_stereo_groups`. Nếu có nhóm OR hoặc AND,
   tức lập thể tương đối hoặc racemic, lý do ghi thêm
   `relative_or_racemic_stereo_groups_not_encoded_in_canonical_ids`. Canonical SMILES và
   InChIKey không mã hóa các nhóm này, nên người dùng cần biết hai định danh đó đang
   biểu diễn quá chắc chắn về lập thể. Nhiều lý do được nối bằng dấu `;`.

Với RDKit 2026.09.1 đang dùng, cờ `|r|` (mọi tâm đều là lập thể tương đối) **không**
được chuyển thành stereo group. Các dòng có `|r|` vì thế giữ dấu này trong
`molecule_source_smiles` và `molecule_cx_extension`, nhưng `molecule_stereo_groups`
trống và `structure_reason` không ghi gì. Cờ chỉ được nhận diện khi viết dưới dạng nhóm
tường minh như `|&1:1|` hay `|o1:1|`.

Kết quả `describe_structure` được cache trong `pipeline.py` (`lru_cache`, 200.000 mục).
Cùng một SMILES xuất hiện ở nhiều phép đo chỉ phải tính một lần.

### 5.5. `observation_id`

`observation_id = "<source_database>:<native_measurement_id>:<endpoint>"`, ví dụ
`chembl:120007:EC50` hay `bindingdb:13888:Ki`. `native_measurement_id` là
`activity_id` ở ChEMBL và Reactant_set_id ở BindingDB. Endpoint phải có mặt vì một dòng
BindingDB tạo ra nhiều observation (mục 4.4).

Chuỗi protein **không** được hash vào `observation_id`. Định danh của một phép đo phải
chỉ phụ thuộc vào phép đo đó trong nguồn, không phụ thuộc vào kết quả suy luận của
pipeline. Chuỗi protein là kết quả của một bước giải quyết có thể thay đổi: sửa cách lấy
variant, hay một dòng chuyển từ unresolved sang resolved. Nếu hash chuỗi vào ID, mỗi lần
như vậy ID của phép đo sẽ đổi, mọi tham chiếu từ các bước sau bị gãy, và một phép đo có
thể trông như hai phép đo khác nhau. Dòng nào chưa có chuỗi cũng sẽ không có ID ổn
định. Chuỗi protein được hash riêng vào cột `protein_sequence_hash` (SHA-256 của chuỗi
đã bỏ khoảng trắng và viết hoa), dùng khi cần gom theo construct.

### 5.6. PMID

`document_pmid_source` giữ nguyên PMID của nguồn. `normalize_integer_id` tạo
`document_pmid` dạng số nguyên thuần (`"12345.0"` → `"12345"`). Giá trị không phải số
nguyên thì để trống thay vì đoán.

### 5.7. Bốn cột status

Mỗi observation có bốn cặp cột status/reason độc lập: `protein_status`, `assay_status`,
`structure_status`, `activity_status`. Mỗi cột nhận một trong ba giá trị:

- `resolved`: phần đó đã xác định được. Reason có thể vẫn chứa một ghi chú, như
  `censored_qualifier:>`, `no_uniprot_accession` hay `stereo_annotation_failed:...`.
- `unresolved`: nguồn thiếu thông tin hoặc thông tin không đủ để quyết định một cách
  chắc chắn (không có variant sequence, nhiều component, đơn vị lạ…). Giá trị không sai,
  chỉ là chưa biết.
- `invalid`: nguồn có giá trị nhưng giá trị sai dạng (không phải số, ≤ 0, SMILES không
  đọc được…).

Bốn status được tách riêng để một phần hỏng không kéo theo phần khác. Ví dụ dòng thiếu
assay vẫn có protein và giá trị đo dùng được, và bước sau có thể chọn tiêu chí lọc theo
từng mục đích.

---

## 6. Hai nguồn ra hai file riêng, không gộp

`run_extract` ghi mỗi nguồn ra một file Parquet riêng trong `data/observations/`:
`chembl37_observations.parquet` và `bindingdb202610_observations.parquet`. Hai file
không được gộp và không được khử trùng lặp với nhau, vì BindingDB chứa một lượng lớn dữ
liệu nhập lại từ ChEMBL. Ở lần chạy này, 1.657.429 trên 3.246.509 observation BindingDB
có `source_curation = "ChEMBL"`. Muốn biết một dòng ChEMBL và một dòng BindingDB có phải
cùng một phép đo hay không, phải so phân tử, protein, tài liệu, giá trị và assay. Việc đó
thuộc bước sau. Gộp sớm ở đây sẽ hoặc đếm trùng một phép đo hai lần, hoặc buộc phải ra
quyết định khử trùng lặp khi chưa có đủ căn cứ.

Hai file dùng **cùng một schema** (`OBSERVATION_SCHEMA` trong `pipeline.py`, 57 cột cùng
tên, cùng kiểu, cùng thứ tự). Cột nào một nguồn không có thì để trống (null). Nhờ vậy
bước sau có thể đọc cả hai bằng cùng một code, hoặc nối chúng lại khi cần.

Mỗi file được ghi theo từng khối 100.000 dòng (`chunk_rows`) vào
`<tên>.parquet.partial`, và chỉ được đổi tên thành `<tên>.parquet` sau khi ghi xong toàn
bộ. Nếu chương trình lỗi giữa chừng, file `.partial` bị xóa. Vì vậy một file `.parquet`
đã tồn tại luôn là một file hoàn chỉnh. `--mode full` bỏ qua nguồn đã có file hoàn chỉnh,
và chỉ chạy lại nguồn đó khi có thêm `--force`.

---

## 7. Bảng mô tả schema

Cột được xếp theo đúng thứ tự trong file. "—" nghĩa là nguồn đó không có thông tin
tương ứng, và cột luôn null với nguồn đó.

**Cột định danh và cột nguồn** (giá trị chép nguyên từ raw hoặc được giải quyết theo
từng phép đo):

| Cột | Kiểu | Ý nghĩa | ChEMBL | BindingDB |
|---|---|---|---|---|
| `observation_id` | string | ID ổn định của observation | `chembl:<activity_id>:<endpoint>` | `bindingdb:<Reactant_set_id>:<endpoint>` |
| `source_database` | string | Tên nguồn | `"chembl"` | `"bindingdb"` |
| `source_release` | string | Phiên bản | `"37"` | `"202610"` |
| `native_measurement_id` | string | ID phép đo trong nguồn | `activities.activity_id` | `BindingDB Reactant_set_id` |
| `source_curation` | string | Nơi biên soạn dữ liệu | `source.src_short_name` | `Curation/DataSource` |
| `document_id` | string | ID tài liệu/entry | `docs.chembl_id` | ENTRYID (từ `rsid_eaids`, chỉ có khi assay liên kết được đúng một cặp) |
| `document_pmid_source` | string | PMID nguyên văn | `docs.pubmed_id` (dạng text) | `PMID` |
| `document_doi` | string | DOI bài báo | `docs.doi` | `Article DOI` |
| `document_patent_number` | string | Số bằng sáng chế | `docs.patent_id` | `Patent Number` |
| `molecule_source_id` | string | ID phân tử trong nguồn | `molecule_dictionary.chembl_id` | `BindingDB MonomerID` |
| `molecule_source_smiles` | string | SMILES nguyên văn | `compound_structures.canonical_smiles` | `Ligand SMILES` (có thể là CXSMILES) |
| `molecule_source_inchikey` | string | InChIKey do nguồn cung cấp | `compound_structures.standard_inchi_key` | `Ligand InChI Key` |
| `molecule_source_chirality` | string | Cờ chirality của nguồn | `molecule_dictionary.chirality` | — |
| `protein_sequence` | string | Chuỗi protein của chính phép đo | `variant_sequences.sequence`, hoặc `component_sequences.sequence` khi không có variant | `BindingDB Target Chain Sequence 1` (chỉ khi có 1 chain) |
| `protein_sequence_origin` | string | Chuỗi lấy từ đâu | `chembl_variant_sequences` / `chembl_component_sequences` | `bindingdb_target_chain_1` |
| `protein_accession` | string | Accession UniProt | `variant_sequences.accession` hoặc `component_sequences.accession` | SwissProt Primary ID chain 1, nếu không có thì TrEMBL Primary ID |
| `protein_accession_db` | string | Accession thuộc SwissProt hay TrEMBL | — | `swissprot` / `trembl` |
| `protein_target_id` | string | ID target | `target_dictionary.chembl_id` | — |
| `protein_target_name` | string | Tên target | `target_dictionary.pref_name` | `Target Name` (văn bản tự do) |
| `protein_target_type` | string | Loại target | `target_dictionary.target_type` | — |
| `protein_organism` | string | Loài | `target_dictionary.organism` | `Target Source Organism According to Curator or DataSource` |
| `protein_variant_id` | string | ID variant | `assays.variant_id` | — |
| `protein_mutation` | string | Mô tả đột biến | `variant_sequences.mutation` | — |
| `protein_isoform` | string | Isoform | `variant_sequences.isoform` | — |
| `protein_chain_count` | string | Số component/chain của target | số dòng `target_components` của target | `Number of Protein Chains in Target` |
| `protein_pdb_ids` | string | PDB ID của chain | — | `PDB ID(s) of Target Chain 1` |
| `protein_status` | string | resolved / unresolved / invalid | mục 3.2 | mục 4.1 |
| `protein_reason` | string | Lý do của status | mục 3.2 | mục 4.1 |
| `assay_native_id` | string | ID assay | `assays.chembl_id` | ASSAYID |
| `assay_entry_id` | string | Entry chứa assay | — | ENTRYID |
| `assay_name` | string | Tên assay | — | `Assays.ASSAY_NAME` |
| `assay_description` | string | Mô tả assay | `assays.description` | `Assays.DESCRIPTION` |
| `assay_type` | string | Loại assay (B, F, A…) | `assays.assay_type` | — |
| `assay_context` | string (JSON) | Ngữ cảnh assay | JSON gồm assay_id, assay_test_type, assay_category, assay_organism, assay_tax_id, assay_strain, assay_tissue, assay_cell_type, assay_subcellular_fraction, relationship_type, confidence_score, src_assay_id, bao_format | JSON gồm ph, temperature_c, pubchem_aid, entry_doi |
| `assay_status` | string | resolved / unresolved | `assay_row_missing` nếu không thấy assay | mục 4.2 |
| `assay_reason` | string | Lý do của status | như trên | như trên |
| `endpoint` | string | Ki / Kd / IC50 / EC50 | `activities.standard_type` | Tên cột endpoint chứa giá trị |
| `activity_value_source` | string | Giá trị nguyên văn | `activities.standard_value` | Nội dung ô endpoint, giữ nguyên cả tiền tố qualifier |
| `activity_unit_source` | string | Đơn vị nguyên văn | `activities.standard_units` | `"nM"` (từ tên cột) |
| `activity_qualifier_source` | string | Qualifier nguyên văn | `activities.standard_relation` | — (qualifier nằm trong ô giá trị) |
| `activity_context` | string (JSON) | Ngữ cảnh giá trị | JSON gồm record_id, relation, value, units, standard_flag, activity_comment, data_validity_comment, potential_duplicate, pchembl_value | — |

**Cột suy ra** (tính trong `pipeline.build_observation` bằng các hàm của
`normalize.py`, giống nhau cho cả hai nguồn):

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `document_pmid` | string | PMID dạng số nguyên thuần, null nếu không phải số nguyên |
| `protein_sequence_hash` | string | SHA-256 của chuỗi protein (bỏ khoảng trắng, viết hoa) |
| `molecule_canonical_smiles` | string | Canonical isomeric SMILES của RDKit |
| `molecule_inchikey` | string | InChIKey do RDKit tính |
| `molecule_cx_extension` | string | Phần mở rộng CXSMILES nguyên văn (ví dụ `\|r\|`). Với ChEMBL luôn null, vì nguồn không có CXSMILES |
| `molecule_stereo_groups` | string | Nhóm lập thể nâng cao mà RDKit đọc được, dạng `TYPE:atom,atom;...` |
| `molecule_stereo_centers` | int32 | Tổng số tâm lập thể; null nếu RDKit không gán được |
| `molecule_unspecified_stereo_centers` | int32 | Số tâm lập thể chưa xác định cấu hình; null nếu RDKit không gán được |
| `structure_status` | string | resolved / unresolved / invalid (mục 5.4) |
| `structure_reason` | string | Lý do; nhiều lý do nối bằng `;` |
| `activity_qualifier` | string | Qualifier đã tách: `=`, `<`, `>`, `<=`, `>=`, `<<`, `>>` |
| `activity_concentration_molar` | float64 | Nồng độ quy về mol/L (có cả cho giá trị censored) |
| `activity_p_activity` | float64 | −log₁₀(mol/L); chỉ có khi qualifier là `=` |
| `activity_censored` | bool | true nếu qualifier là dạng bị chặn; null nếu qualifier không xác định |
| `activity_status` | string | resolved / unresolved / invalid (mục 5.2) |
| `activity_reason` | string | Lý do của status |

---

## 8. Kết quả lần chạy ngày 18–19/10/2026

Lần chạy `full` bắt đầu lúc 22:00 ngày 18/10/2026. Nhánh ChEMBL xong lúc 23:04, nhánh
BindingDB xong lúc 23:53. Sau đó `verify` chạy từ 23:57 đến 00:01 ngày 19/10/2026. Báo
cáo chi tiết nằm ở `runs/observations/verify_20261018_235759.json`. Cả năm phép kiểm
tra của verify đều đạt.

**Số dòng.** ChEMBL ra 5.338.213 dòng, BindingDB ra 3.246.509 dòng, tổng cộng
8.584.722 observation.

**Bảo toàn số dòng.** Ở cả hai nguồn, số dòng ghi ra bằng đúng số dòng được chọn từ raw:
- ChEMBL: so với `COUNT(*)` của `activities` theo bốn endpoint.
- BindingDB: so với tổng số ô endpoint không rỗng, đếm lại bằng một lượt đọc độc lập.

Không có `observation_id` nào trùng.

**Đối chiếu ngược về raw.** 5.000 dòng được chọn ngẫu nhiên (hạt giống cố định
20261018) và đối chiếu từng trường với raw bằng đường tra cứu độc lập: truy vấn riêng
từng bảng ChEMBL, và đọc lại ba file BindingDB. Kết quả có 0 sai lệch.

**Kiểm tra construct P03367.** Có 1.040 dòng mang accession P03367. Trong đó 854 dòng
giữ chuỗi 99 aa (domain protease HIV-1) và 186 dòng giữ chuỗi 1.447 aa (polyprotein).
So với raw của chính từng phép đo thì có 0 sai lệch. Đây là bằng chứng mỗi phép đo giữ
đúng construct của chính nó, không bị thay bằng một chuỗi chung theo accession.

**PMID.** Không có dòng nào có `document_pmid` ở dạng `.0` hay dạng không phải số
nguyên.

**ChEMBL, trạng thái protein.**
- 3.278.184 dòng `resolved`.
- Nhóm `unresolved` lớn nhất là `target_has_no_protein_component`, 1.823.185 dòng. Đây
  là các target không phải một protein đơn, và không có component nào trong
  `target_components`. Theo `target_type`, nhiều nhất là CELL-LINE (827.438),
  UNCHECKED (455.961), ORGANISM (399.560) và NON-MOLECULAR (95.542). Không dòng nào
  thuộc SINGLE PROTEIN.
- `variant_declared_but_sequence_missing`: 5.097 dòng.
- Phần còn lại chủ yếu là `target_component_count_not_1:N` (target nhiều component).

**BindingDB.**
- `protein_status = resolved`: 3.058.211 dòng (94,2%).
- `assay_status = unresolved` với `no_rsid_assay_link`: 58.159 dòng (1,8%).
- `structure_status = invalid` với `rdkit_parse_failed`: 5.816 dòng.

**Giá trị censored.** ChEMBL có 909.037 dòng `>` và 190.113 dòng `<`. BindingDB có
427.439 dòng `>` và 227.035 dòng `<`.

**Đơn vị không quy đổi được.** Lớn nhất là `ug.mL-1` với 92.929 dòng ở ChEMBL. Các đơn
vị này giữ `activity_status = unresolved` với lý do
`unit_not_in_table:<đơn vị>`.

---

## 9. Giới hạn đã biết

**31 dòng không có số tâm lập thể.** RDKit không gán được nhãn CIP cho 31 dòng (17 ở
ChEMBL, 14 ở BindingDB). Lỗi là "Digraph generation failed: more than 100000 nodes
found", gặp ở các phân tử rất lớn có SMILES dài 215–298 ký tự, phần lớn thuộc cùng một
loạt hợp chất. Các dòng này vẫn có canonical SMILES và InChIKey, `structure_status`
vẫn là `resolved`. Chỉ hai cột `molecule_stereo_centers` và
`molecule_unspecified_stereo_centers` để trống, và `structure_reason` ghi
`stereo_annotation_failed:...`.

**ChEMBL không có cột CXSMILES.** Bản SQLite của ChEMBL chỉ có `canonical_smiles` dạng
SMILES thường, nên thông tin lập thể nâng cao (racemic, tương đối) nếu có cũng không
truyền được qua cột này. `molecule_cx_extension` và `molecule_stereo_groups` luôn trống
với ChEMBL. Các file molfile/SDF có thể chứa thông tin này không được đọc ở bước này.

**RDKit bỏ qua marker `|r|` của BindingDB.** RDKit 2026.09.1 đọc được các nhóm lập
thể nâng cao viết tường minh như `|&1:1|` hay `|o1:1|`, nhưng bỏ qua cờ `|r|` (mọi
tâm lập thể đều là tương đối). Vì vậy có 421.184 dòng BindingDB mang `|r|` mà:
- canonical SMILES và InChIKey mã hóa lập thể như thể là tuyệt đối;
- `molecule_stereo_groups` trống và `structure_reason` trống;
- chỉ cột `molecule_cx_extension` (cùng chuỗi nguồn trong `molecule_source_smiles`)
  còn giữ dấu `|r|`.

Ở bước MMP, đây là chỗ một hợp chất racemic hoặc có lập thể tương đối có thể bị coi
là một đồng phân đối quang xác định. Nó có thể bị ghép cặp hay khử trùng lặp nhầm với
một đồng phân tuyệt đối có cùng canonical SMILES. Muốn nhận ra các dòng này, phải dựa
vào `molecule_cx_extension`, không dựa vào `structure_reason`.

**BindingDB không có cột variant hay mutation riêng.** Thông tin đột biến của BindingDB
chỉ có thể nằm trong `Target Name`, một trường văn bản tự do, hoặc ẩn trong chính chuỗi
`BindingDB Target Chain Sequence 1`. Code không phân tích `Target Name`, nên
`protein_variant_id`, `protein_mutation` và `protein_isoform` luôn trống với BindingDB.
Chuỗi protein vẫn đúng là chuỗi của chính phép đo, nhưng không có nhãn đột biến tách
riêng để lọc.

**1.051 dòng BindingDB có chuỗi protein nhưng không có mã UniProt.** Các dòng này có
`protein_status = resolved` và `protein_reason = no_uniprot_accession`, với
`protein_accession` trống. Nếu bước chia dữ liệu (train/validation/test split) khóa
theo accession, cần quyết định riêng cho nhóm này, ví dụ khóa theo
`protein_sequence_hash`. Nếu không, chúng có thể rơi vào một nhóm "accession rỗng" chung
hoặc bị bỏ sót.

---

## 10. Ví dụ lần theo một dòng

### 10.1. ChEMBL: `chembl:120007:EC50`

Ví dụ này đi qua nhánh variant, là nhánh quan trọng nhất của việc lấy protein theo từng
phép đo.

**Trong raw.** Bảng `activities`, dòng `activity_id = 120007`:
- `assay_id = 79621`, `doc_id = 14232`, `molregno = 91157`;
- `standard_type = 'EC50'`, `standard_relation = '='`, `standard_value = 300`,
  `standard_units = 'nM'`;
- giá trị gốc công bố là `value = 0.3`, `units = 'uM'`.

**Các bước:**

1. `selected_activity_ids` chọn id 120007 vì `standard_type` là EC50. Câu join chạy
   trong lô chứa id này.
2. Join sang `assays` (assay_id 79621) được:
   - `chembl_id = CHEMBL693550`, `tid = 228`, `variant_id = 2424`, `assay_type = 'B'`;
   - mô tả "Cross Resistance (antiviral activity) with L100I (Non nucleoside reverse
     transcriptase inhibitor) resistant HIV isolate from cytopathic cell killing assay".
3. Lấy protein: assay khai báo `variant_id = 2424`, nên `resolve_protein` lấy
   `variant_sequences` dòng 2424: `mutation = 'L100I'`, `accession = 'Q72547'`, chuỗi
   566 aa. Target 228 (CHEMBL247, "Human immunodeficiency virus type 1 reverse
   transcriptase", SINGLE PROTEIN) cũng có đúng một component (component_id 375,
   Q72547, 566 aa). Nhưng chuỗi component là wild type: so từng vị trí, hai chuỗi chỉ
   khác nhau ở vị trí 100 (L ở wild type, I ở variant). Code chọn chuỗi variant,
   `protein_sequence_origin = chembl_variant_sequences`. Nếu lấy chuỗi component, phép
   đo trên chủng kháng thuốc L100I sẽ bị gán nhầm vào enzyme wild type.
4. Phân tử: molregno 91157 → `CHEMBL293961`, SMILES
   `Cc1cccc2c1nc(-c1c(F)cccc1F)n2Cc1c(F)cccc1F`. RDKit cho canonical SMILES giống hệt
   và InChIKey `JIYRSPUMGRIYBL-UHFFFAOYSA-N`. Phân tử không có tâm lập thể.
5. Giá trị: qualifier `=`, 300 × 1e-9 = 3×10⁻⁷ M, pActivity = −log₁₀(3×10⁻⁷) ≈
   6,5229. Kết quả khớp với `pchembl_value = 6.52` mà ChEMBL tự tính, được giữ trong
   `activity_context`.
6. Tài liệu: doc_id 14232 → `CHEMBL1130403`, PMID 9435891, DOI `10.1021/jm970096g`.

**Dòng observation (các cột chính):**

| Cột | Giá trị |
|---|---|
| `observation_id` | `chembl:120007:EC50` |
| `native_measurement_id` | `120007` |
| `source_curation` | `LITERATURE` |
| `document_id` / `document_pmid` / `document_doi` | `CHEMBL1130403` / `9435891` / `10.1021/jm970096g` |
| `molecule_source_id` | `CHEMBL293961` |
| `molecule_source_smiles` | `Cc1cccc2c1nc(-c1c(F)cccc1F)n2Cc1c(F)cccc1F` |
| `molecule_source_chirality` | `-1` |
| `protein_sequence` | `PISPIETVPVKLKPGMDGPK...` (566 aa, có I ở vị trí 100) |
| `protein_sequence_origin` | `chembl_variant_sequences` |
| `protein_accession` | `Q72547` |
| `protein_target_id` / `protein_target_type` | `CHEMBL247` / `SINGLE PROTEIN` |
| `protein_variant_id` / `protein_mutation` / `protein_isoform` | `2424` / `L100I` / `1` |
| `protein_chain_count` | `1` |
| `protein_status` / `protein_reason` | `resolved` / null |
| `assay_native_id` / `assay_type` | `CHEMBL693550` / `B` |
| `assay_context` | `{"assay_id": 79621, ..., "relationship_type": "H", "confidence_score": 8, ..., "bao_format": "BAO_0000357"}` |
| `assay_status` | `resolved` |
| `endpoint` | `EC50` |
| `activity_value_source` / `activity_unit_source` / `activity_qualifier_source` | `300` / `nM` / `=` |
| `activity_context` | `{"record_id": 257090, "relation": "=", "value": 0.3, "units": "uM", "standard_flag": 1, ..., "pchembl_value": 6.52}` |
| `protein_sequence_hash` | `93b053b2...dfc102` |
| `molecule_canonical_smiles` | `Cc1cccc2c1nc(-c1c(F)cccc1F)n2Cc1c(F)cccc1F` |
| `molecule_inchikey` | `JIYRSPUMGRIYBL-UHFFFAOYSA-N` |
| `molecule_cx_extension` / `molecule_stereo_groups` | null / null |
| `molecule_stereo_centers` / `molecule_unspecified_stereo_centers` | `0` / `0` |
| `structure_status` | `resolved` |
| `activity_qualifier` | `=` |
| `activity_concentration_molar` | `3.0e-07` |
| `activity_p_activity` | `6.5229` |
| `activity_censored` | `false` |
| `activity_status` / `activity_reason` | `resolved` / null |

### 10.2. BindingDB: Reactant_set_id 13888 → hai observation

Ví dụ này cho thấy đường join assay qua `rsid_eaids`, và cách một dòng raw tách thành
nhiều observation.

**Trong raw.**
- `BindingDB_All.tsv`, dòng số 13.889 (dòng 1 là header), có đủ 640 trường:
  - `BindingDB Reactant_set_id = 13888`, `BindingDB MonomerID = 8129`;
  - `Ligand SMILES = N#Cc1ccc(Nc2nccc(n2)-c2cnn3ncccc23)cc1`,
    `Ligand InChI Key = YRSCDRTXWYUNBB-UHFFFAOYSA-N`;
  - `Target Name = Glycogen synthase kinase-3 beta`, organism `Homo sapiens`;
  - `Ki (nM) = " 2"` (có dấu cách đầu), `IC50 (nM) = "<10"`, `Kd (nM)` và `EC50 (nM)`
    rỗng;
  - `Curation/DataSource = Curated from the literature by BindingDB`, `PMID = 15341487`,
    `Article DOI = 10.1021/jm040063i`;
  - số chain = `1`, chain 1 có chuỗi 420 aa, `SwissProt Primary ID = P49841`, TrEMBL
    rỗng.
- `BindingDB_rsid_eaids.tsv`, dòng 13.860: `REACTANT_SET_ID = 13888`,
  `ENTRYID_ASSAYID = 1025_1`. Đây là dòng liên kết duy nhất của reactant set này.
- `BindingDB_Assays.tsv`: dòng có `ENTRYID = 1025`, `ASSAYID = 1`, với
  `ASSAY_NAME = "Kinase Inhibition SPA  Assay"`.

**Các bước:**

1. Lúc nạp, `load_assay_links` ghi `"13888" → [("1025", "1")]`, và `load_assays` có
   khóa `("1025", "1")`. Hai phép kiểm tra giao khóa đều qua.
2. Đọc dòng 13.889 của All. `resolve_assay("13888")` tìm được đúng một cặp
   `(1025, 1)`, cặp này có đúng một dòng trong bảng Assays, nên `assay_status =
   resolved`, `assay_entry_id = 1025`, `assay_native_id = 1`. Reactant_set_id 13888
   không bị đem so với ENTRYID: hai số này khác hẳn nhau (13888 và 1025).
3. `resolve_protein`: số chain là 1, chuỗi lấy từ `BindingDB Target Chain Sequence 1`
   (420 aa), accession `P49841` từ cột SwissProt, nên `protein_accession_db = swissprot`.
4. `build_records` duyệt bốn cột endpoint. Ô Ki và ô IC50 không rỗng nên tạo hai bản
   ghi. Ô Kd và EC50 rỗng nên bỏ qua.
5. Bản ghi Ki: `split_embedded_qualifier(" 2")` không thấy tiền tố, nên qualifier `=` và
   giá trị `2`. Ra 2 nM = 2×10⁻⁹ M, pActivity = −log₁₀(2×10⁻⁹) ≈ 8,6990.
6. Bản ghi IC50: `split_embedded_qualifier("<10")` cho qualifier `<` và giá trị `10`. Ra
   1×10⁻⁸ M, censored, không có pActivity, lý do `censored_qualifier:<`.
7. Cấu trúc: RDKit đọc SMILES và cho canonical SMILES
   `N#Cc1ccc(Nc2nccc(-c3cnn4ncccc34)n2)cc1`. Chuỗi này khác chuỗi nguồn về cách viết
   nhưng là cùng phân tử: InChIKey `YRSCDRTXWYUNBB-UHFFFAOYSA-N` trùng với InChIKey
   nguồn. Phân tử không có tâm lập thể và không có phần mở rộng CXSMILES. Hai bản ghi
   dùng chung kết quả từ cache.

**Hai dòng observation (các cột chính; cột không ghi giống nhau ở cả hai dòng):**

| Cột | `bindingdb:13888:Ki` | `bindingdb:13888:IC50` |
|---|---|---|
| `native_measurement_id` | `13888` | `13888` |
| `source_curation` | `Curated from the literature by BindingDB` | (giống) |
| `document_id` / `document_pmid` / `document_doi` | `1025` / `15341487` / `10.1021/jm040063i` | (giống) |
| `molecule_source_id` | `8129` | (giống) |
| `molecule_source_smiles` | `N#Cc1ccc(Nc2nccc(n2)-c2cnn3ncccc23)cc1` | (giống) |
| `protein_sequence` | `MSGRPRTTSFAESCKPVQQP...` (420 aa) | (giống) |
| `protein_sequence_origin` | `bindingdb_target_chain_1` | (giống) |
| `protein_accession` / `protein_accession_db` | `P49841` / `swissprot` | (giống) |
| `protein_target_name` / `protein_organism` | `Glycogen synthase kinase-3 beta` / `Homo sapiens` | (giống) |
| `protein_chain_count` | `1` | (giống) |
| `protein_status` | `resolved` | (giống) |
| `assay_entry_id` / `assay_native_id` | `1025` / `1` | (giống) |
| `assay_name` | `Kinase Inhibition SPA  Assay` | (giống) |
| `assay_context` | `{"ph": null, "temperature_c": null, "pubchem_aid": "aid1796148", "entry_doi": "10.7270/Q2VM49HJ"}` | (giống) |
| `assay_status` | `resolved` | (giống) |
| `endpoint` | `Ki` | `IC50` |
| `activity_value_source` | `" 2"` | `"<10"` |
| `activity_unit_source` | `nM` | `nM` |
| `activity_qualifier_source` | null | null |
| `protein_sequence_hash` | `2ca842f3...b06fa3` | (giống) |
| `molecule_canonical_smiles` | `N#Cc1ccc(Nc2nccc(-c3cnn4ncccc34)n2)cc1` | (giống) |
| `molecule_inchikey` | `YRSCDRTXWYUNBB-UHFFFAOYSA-N` | (giống) |
| `molecule_stereo_centers` | `0` | `0` |
| `structure_status` | `resolved` | `resolved` |
| `activity_qualifier` | `=` | `<` |
| `activity_concentration_molar` | `2.0e-09` | `1.0e-08` |
| `activity_p_activity` | `8.6990` | null |
| `activity_censored` | `false` | `true` |
| `activity_status` / `activity_reason` | `resolved` / null | `resolved` / `censored_qualifier:<` |
