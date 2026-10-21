"""ChEMBL reader: streams Ki/Kd/IC50/EC50 activities from the extracted SQLite release.

Protein per measurement: assays.variant_id -> variant_sequences first; the target's
component sequence is used only when no variant is declared. A declared variant without
a sequence stays unresolved (wild type is never substituted).

This module uses the standard library only, so `inspect` runs without RDKit/pyarrow.
Values are passed on verbatim; unit/qualifier/structure conversion happens in normalize.py.

Vị trí trong luồng raw -> observations:
    raw ChEMBL (bản SQLite đã giải nén sẵn trong work/, mở CHỈ ĐỌC)
    -> chembl.py (file này): chọn các activity Ki/Kd/IC50/EC50, join sang assay, variant,
       target, phân tử, tài liệu; quyết định chuỗi protein cho TỪNG phép đo; trả về mỗi
       activity một bản ghi với giá trị nguồn nguyên văn
    -> pipeline.py: thêm cột suy ra (normalize.py) và ghi Parquet.
"""

from __future__ import annotations

# Chỉ dùng thư viện chuẩn: SQLite, array.
import json
import logging
import sqlite3
from array import array
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

LOGGER = logging.getLogger(__name__)

# Tên nguồn, dùng trong observation_id.
SOURCE_DATABASE = "chembl"
# Giá trị của cột protein_sequence_origin: chuỗi protein lấy từ bảng nào.
SEQUENCE_ORIGIN_VARIANT = "chembl_variant_sequences"
SEQUENCE_ORIGIN_COMPONENT = "chembl_component_sequences"
SINGLE_COMPONENT = 1

RESOLVED = "resolved"
UNRESOLVED = "unresolved"

# Every column the extraction query reads; checked against the real schema before use.
# Các cột câu SQL cần; thiếu cột nào thì dừng trước khi đọc dữ liệu.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "activities": ("activity_id", "assay_id", "doc_id", "record_id", "molregno",
                   "standard_relation", "standard_value", "standard_units", "standard_flag",
                   "standard_type", "activity_comment", "data_validity_comment",
                   "potential_duplicate", "pchembl_value", "src_id", "relation", "value",
                   "units"),
    "assays": ("assay_id", "chembl_id", "description", "assay_type", "assay_test_type",
               "assay_category", "assay_organism", "assay_tax_id", "assay_strain",
               "assay_tissue", "assay_cell_type", "assay_subcellular_fraction", "tid",
               "relationship_type", "confidence_score", "src_assay_id", "bao_format",
               "variant_id"),
    "variant_sequences": ("variant_id", "mutation", "accession", "isoform", "sequence",
                          "organism"),
    "target_dictionary": ("tid", "chembl_id", "pref_name", "target_type", "organism"),
    "target_components": ("tid", "component_id"),
    "component_sequences": ("component_id", "accession", "sequence"),
    "molecule_dictionary": ("molregno", "chembl_id", "chirality"),
    "compound_structures": ("molregno", "canonical_smiles", "standard_inchi_key"),
    "docs": ("doc_id", "chembl_id", "pubmed_id", "doi", "patent_id"),
    "source": ("src_id", "src_short_name"),
}

# Điều kiện chọn activity theo endpoint (standard_type), dùng tham số "?" cho từng endpoint.
# Vd. 'act.standard_type IN (?, ?, ?, ?)'.
ENDPOINT_FILTER = "act.standard_type IN ({placeholders})"

# Câu truy vấn chính: mỗi dòng activities -> một dòng kết quả.
# - Dùng LEFT JOIN ở mọi chỗ để một activity không bao giờ bị mất chỉ vì bảng bên kia
#   thiếu dòng (thiếu sẽ thể hiện bằng NULL và status unresolved, không phải mất dòng).
# - variant_sequences được join qua assays.variant_id: đây là chuỗi protein của đột biến
#   thực sự được đo trong assay đó.
# - Truy vấn con "tc" đếm số component của target và lấy MIN(component_id); component này
#   chỉ được dùng khi target có đúng 1 component (xem resolve_protein).
# - Lấy bộ giá trị đã chuẩn hóa của ChEMBL (standard_*); bộ giá trị gốc (value, units,
#   relation) được giữ trong activity_context.
# - "{id_placeholders}" được thay bằng một lô activity_id (xem iter_records).
# Không nhân đôi dòng: mọi bảng bên phải join theo khóa chính; tc gom mỗi target còn một
# dòng.
EXTRACTION_QUERY = """
SELECT
    act.activity_id, act.record_id, act.standard_type, act.standard_relation,
    act.standard_value, act.standard_units, act.standard_flag, act.relation, act.value,
    act.units, act.activity_comment, act.data_validity_comment, act.potential_duplicate,
    act.pchembl_value,
    src.src_short_name,
    a.assay_id, a.chembl_id AS assay_chembl_id, a.description AS assay_description,
    a.assay_type, a.assay_test_type, a.assay_category, a.assay_organism, a.assay_tax_id,
    a.assay_strain, a.assay_tissue, a.assay_cell_type, a.assay_subcellular_fraction,
    a.relationship_type, a.confidence_score, a.src_assay_id, a.bao_format, a.variant_id,
    vs.sequence AS variant_sequence, vs.accession AS variant_accession,
    vs.mutation AS variant_mutation, vs.isoform AS variant_isoform,
    td.chembl_id AS target_chembl_id, td.pref_name AS target_name,
    td.target_type, td.organism AS target_organism,
    tc.component_count, comp.sequence AS component_sequence,
    comp.accession AS component_accession,
    md.chembl_id AS molecule_chembl_id, md.chirality,
    cs.canonical_smiles, cs.standard_inchi_key,
    d.chembl_id AS doc_chembl_id, d.pubmed_id, d.doi, d.patent_id
FROM activities AS act
LEFT JOIN source AS src ON src.src_id = act.src_id
LEFT JOIN assays AS a ON a.assay_id = act.assay_id
LEFT JOIN variant_sequences AS vs ON vs.variant_id = a.variant_id
LEFT JOIN target_dictionary AS td ON td.tid = a.tid
LEFT JOIN (
    SELECT tid, COUNT(*) AS component_count, MIN(component_id) AS component_id
    FROM target_components GROUP BY tid
) AS tc ON tc.tid = a.tid
LEFT JOIN component_sequences AS comp ON comp.component_id = tc.component_id
LEFT JOIN molecule_dictionary AS md ON md.molregno = act.molregno
LEFT JOIN compound_structures AS cs ON cs.molregno = act.molregno
LEFT JOIN docs AS d ON d.doc_id = act.doc_id
WHERE act.activity_id IN ({id_placeholders})
ORDER BY act.activity_id
"""

# Đếm số activity được chọn; verify dùng để kiểm tra số dòng vào = số dòng ra.
COUNT_QUERY = "SELECT COUNT(*) FROM activities AS act WHERE {endpoint_filter}"

# Output order is activity_id. Sorting the full joined result would build a multi-GB
# temporary B-tree, so only the selected IDs are sorted here (integers), and the join
# above runs per ID batch, where the activity_id index already gives the order.
SELECTED_IDS_QUERY = ("SELECT act.activity_id FROM activities AS act "
                      "WHERE {endpoint_filter} ORDER BY act.activity_id")
# Mã kiểu của array: số nguyên có dấu 64-bit (8 byte mỗi id, ~43 MB cho 5,3 triệu id).
SIGNED_64_BIT = "q"


class ProteinResolution(NamedTuple):
    """Kết quả chọn chuỗi protein cho một phép đo (xem resolve_protein)."""
    sequence: str | None
    accession: str | None
    origin: str | None
    status: str
    reason: str | None


# --------------------------------------------------------------------------- database access


def open_chembl_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the extracted ChEMBL SQLite file read-only; fail clearly if it is missing.

    Nhận: đường dẫn file chembl_37.db (đã giải nén sẵn trong work/).
    Trả về: kết nối SQLite chỉ đọc.
    Vì sao cần: dữ liệu ChEMBL nằm trên nhiều bảng quan hệ, phải dùng SQL join nên cần
    file SQLite truy cập ngẫu nhiên được. Mở ở chế độ "mode=ro" để pipeline không thể ghi
    vào cơ sở dữ liệu dù có lỗi lập trình.
    """
    # Pipeline không bao giờ tự giải nén file .tar.gz trong raw/; thiếu file thì dừng.
    # Không kiểm tra thì SQLite sẽ tự tạo một database rỗng.
    if not db_path.is_file():
        raise FileNotFoundError(
            f"ChEMBL SQLite not found at {db_path}. This pipeline never extracts archives; "
            "extract chembl_37_sqlite.tar.gz there yourself first."
        )
    # URI "file:...?mode=ro": SQLite từ chối mọi thao tác ghi trên kết nối này.
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    # Keep SQLite's temporary sort/group structures in RAM instead of the system temp dir.
    # (Thư mục temp của hệ thống nằm ngoài E:\eval_training, nơi pipeline không được ghi.)
    connection.execute("PRAGMA temp_store = MEMORY")
    return connection


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    """Column names of a table, read from the SQLite schema.

    Nhận: kết nối và tên bảng. Trả về: danh sách tên cột theo thứ tự trong bảng.
    """
    return [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]


def missing_columns(connection: sqlite3.Connection) -> list[str]:
    """Required 'table.column' names absent from the database.

    Nhận: kết nối. Trả về: danh sách "bảng.cột" có trong REQUIRED_COLUMNS nhưng không có
    trong cơ sở dữ liệu (rỗng nếu đủ).
    """
    missing = []
    for table, columns in REQUIRED_COLUMNS.items():
        present = set(table_columns(connection, table))
        missing += [f"{table}.{column}" for column in columns if column not in present]
    return missing


def check_schema(connection: sqlite3.Connection) -> None:
    """Fail loudly if any column the extraction query needs is missing.

    Nhận: kết nối. Trả về: không có; ném ValueError nêu tên mọi cột thiếu.
    Vì sao cần: nếu một bản ChEMBL khác đổi tên cột, lỗi phải xuất hiện ngay từ đầu với
    thông báo rõ ràng, thay vì một lỗi SQL khó hiểu giữa chừng.
    """
    missing = missing_columns(connection)
    if missing:
        raise ValueError(f"ChEMBL schema is missing required columns: {', '.join(missing)}")


def endpoint_filter(endpoints: Sequence[str]) -> str:
    """SQL filter on activities.standard_type with one placeholder per endpoint.

    Nhận: danh sách endpoint. Trả về: đoạn SQL "act.standard_type IN (?, ?, ...)".
    Dùng tham số "?" thay vì ghép chuỗi trực tiếp, để SQLite tự xử lý giá trị.
    """
    return ENDPOINT_FILTER.format(placeholders=", ".join("?" for _ in endpoints))


def count_selected_activities(connection: sqlite3.Connection, endpoints: Sequence[str]) -> int:
    """Number of activities rows the extraction selects (conservation check).

    Nhận: kết nối và danh sách endpoint. Trả về: số dòng activities thỏa điều kiện.
    Vì sao cần: verify so con số này với số dòng ChEMBL đã ghi ra để chứng minh không
    mất và không nhân đôi dòng nào.
    """
    query = COUNT_QUERY.format(endpoint_filter=endpoint_filter(endpoints))
    return connection.execute(query, list(endpoints)).fetchone()[0]


# --------------------------------------------------------------------------- resolution (pure)


def text_or_none(value: object) -> str | None:
    """SQLite value as text, verbatim; None for NULL or blank strings.

    Nhận: một giá trị SQLite (số, chuỗi, NULL). Trả về: chuỗi giữ nguyên nội dung, hoặc
    None nếu NULL/chỉ có khoảng trắng.
    Vì sao cần: mọi cột *_source trong output là kiểu chuỗi để giữ nguyên giá trị nguồn;
    số được chuyển thành chuỗi chứ không làm tròn hay định dạng lại.
    """
    # NULL hoặc chuỗi trắng -> None; số -> str(), không làm tròn.
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text if text.strip() else None


def resolve_protein(
    variant_id: object,
    variant_sequence: str | None,
    variant_accession: str | None,
    component_count: int | None,
    component_sequence: str | None,
    component_accession: str | None,
) -> ProteinResolution:
    """Protein sequence for one measurement: declared variant first, never wild type instead.

    Nhận: variant_id của assay và chuỗi/accession của variant đó; số component của target
    và chuỗi/accession của component (MIN component_id).
    Trả về: ProteinResolution (chuỗi, accession, nguồn của chuỗi, status, lý do).

    Vì sao lấy protein theo từng phép đo, và theo thứ tự này:
    - Cùng một target ChEMBL có thể được đo trên nhiều đột biến khác nhau. Assay khai báo
      variant_id cho biết chính xác đột biến nào được đo, nên chuỗi variant phải được ưu
      tiên.
    - Nếu assay đã khai báo variant mà không có chuỗi variant, KHÔNG được thay bằng chuỗi
      component của target: chuỗi đó là wild type, gán vào sẽ tạo một cặp phân tử-protein
      sai mà không để lại dấu vết. Dòng được để unresolved.
    - Chỉ khi không khai báo variant mới dùng chuỗi component, và chỉ khi target có đúng
      một component: với phức hợp/họ protein không biết phân tử gắn vào chuỗi nào.
    """
    # --- Nhánh 1: assay khai báo variant -> chỉ dùng chuỗi variant.
    # Rẽ nhánh theo việc assay CÓ KHAI BÁO variant, không theo việc có chuỗi.
    if variant_id is not None:
        # Không có chuỗi -> dừng ở đây, không lấy wild type.
        if not variant_sequence:
            return ProteinResolution(None, None, None, UNRESOLVED,
                                     "variant_declared_but_sequence_missing")
        return ProteinResolution(variant_sequence, variant_accession, SEQUENCE_ORIGIN_VARIANT,
                                 RESOLVED, None)
    # --- Nhánh 2: không có variant -> dùng component của target, nếu xác định được.
    # Target không có component nào (cell line, organism, ...): không có protein.
    # NULL (target không có component) hoặc 0.
    if not component_count:
        return ProteinResolution(None, None, None, UNRESOLVED, "target_has_no_protein_component")
    # Target nhiều component: không chọn bừa một chuỗi.
    if component_count != SINGLE_COMPONENT:
        return ProteinResolution(None, None, None, UNRESOLVED,
                                 f"target_component_count_not_1:{component_count}")
    if not component_sequence:
        return ProteinResolution(None, None, None, UNRESOLVED, "component_sequence_missing")
    return ProteinResolution(component_sequence, component_accession, SEQUENCE_ORIGIN_COMPONENT,
                             RESOLVED, None)


def build_record(row: sqlite3.Row, release: str) -> dict[str, Any]:
    """One observation record from one joined activities row (values verbatim).

    Nhận: một dòng kết quả của EXTRACTION_QUERY và phiên bản ChEMBL ("37").
    Trả về: dict đúng bằng tập cột nguồn (SOURCE_FIELDS trong pipeline.py), giá trị chép
    nguyên văn từ raw. Cột nào ChEMBL không có (ví dụ protein_pdb_ids) để None.
    Vì sao cần: đưa dữ liệu ChEMBL về cùng một dạng bản ghi với BindingDB, để pipeline.py
    xử lý hai nguồn bằng cùng một code.
    """
    # --- Chọn chuỗi protein cho chính phép đo này.
    protein = resolve_protein(row["variant_id"], row["variant_sequence"],
                              row["variant_accession"], row["component_count"],
                              row["component_sequence"], row["component_accession"])
    # --- Assay coi là tìm thấy khi LEFT JOIN sang bảng assays có dòng tương ứng.
    assay_found = row["assay_chembl_id"] is not None
    # --- Ngữ cảnh assay và ngữ cảnh giá trị: giữ nguyên dưới dạng JSON, không diễn giải,
    # để bước sau có thể lọc (ví dụ theo confidence_score, data_validity_comment).
    assay_context = {key: row[key] for key in (
        "assay_id", "assay_test_type", "assay_category", "assay_organism", "assay_tax_id",
        "assay_strain", "assay_tissue", "assay_cell_type", "assay_subcellular_fraction",
        "relationship_type", "confidence_score", "src_assay_id", "bao_format")}
    activity_context = {key: row[key] for key in (
        "record_id", "relation", "value", "units", "standard_flag", "activity_comment",
        "data_validity_comment", "potential_duplicate", "pchembl_value")}
    # --- Bản ghi: tên cột thống nhất với BindingDB.
    return {
        # Nguồn và ID.
        "source_database": SOURCE_DATABASE,
        "source_release": release,
        "native_measurement_id": str(row["activity_id"]),
        "source_curation": row["src_short_name"],
        # Tài liệu.
        "document_id": row["doc_chembl_id"],
        "document_pmid_source": text_or_none(row["pubmed_id"]),
        "document_doi": row["doi"],
        "document_patent_number": row["patent_id"],
        # Phân tử.
        "molecule_source_id": row["molecule_chembl_id"],
        "molecule_source_smiles": row["canonical_smiles"],
        "molecule_source_inchikey": row["standard_inchi_key"],
        "molecule_source_chirality": text_or_none(row["chirality"]),
        # Protein.
        "protein_sequence": protein.sequence,
        "protein_sequence_origin": protein.origin,
        "protein_accession": protein.accession,
        "protein_accession_db": None,
        "protein_target_id": row["target_chembl_id"],
        "protein_target_name": row["target_name"],
        "protein_target_type": row["target_type"],
        "protein_organism": row["target_organism"],
        "protein_variant_id": text_or_none(row["variant_id"]),
        "protein_mutation": row["variant_mutation"],
        "protein_isoform": text_or_none(row["variant_isoform"]),
        "protein_chain_count": text_or_none(row["component_count"]),
        "protein_pdb_ids": None,
        "protein_status": protein.status,
        "protein_reason": protein.reason,
        # Assay.
        "assay_native_id": row["assay_chembl_id"],
        "assay_entry_id": None,
        "assay_name": None,
        "assay_description": row["assay_description"],
        "assay_type": row["assay_type"],
        "assay_context": json.dumps(assay_context, ensure_ascii=False),
        "assay_status": RESOLVED if assay_found else UNRESOLVED,
        "assay_reason": None if assay_found else "assay_row_missing",
        # Giá trị đo (bộ standard_* của ChEMBL).
        "endpoint": row["standard_type"],
        "activity_value_source": text_or_none(row["standard_value"]),
        "activity_unit_source": text_or_none(row["standard_units"]),
        "activity_qualifier_source": text_or_none(row["standard_relation"]),
        "activity_context": json.dumps(activity_context, ensure_ascii=False),
    }


# --------------------------------------------------------------------------- records


def selected_activity_ids(connection: sqlite3.Connection, endpoints: Sequence[str],
                          limit: int | None, fetch_size: int) -> array:
    """activity_ids selected for extraction, ascending; `limit` keeps the first ones (smoke).

    Nhận: kết nối, danh sách endpoint, giới hạn số id (None = lấy hết; smoke dùng số nhỏ),
    và số dòng đọc mỗi lần từ con trỏ.
    Trả về: array số nguyên 64-bit các activity_id, tăng dần.
    Vì sao cần: đây là bước 1 của cách đọc theo lô. Chỉ sắp xếp danh sách id (số nguyên,
    vài chục MB) thay vì sắp xếp toàn bộ kết quả join (nhiều GB).
    """
    # Chỉ sắp xếp id (số nguyên), rất nhẹ.
    query = SELECTED_IDS_QUERY.format(endpoint_filter=endpoint_filter(endpoints))
    parameters: list[Any] = list(endpoints)
    # Smoke: chỉ lấy `limit` id nhỏ nhất.
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    # Đọc từng phần (fetchmany) vào array gọn bộ nhớ thay vì list các tuple Python.
    # 8 byte mỗi id; đọc 20.000 dòng mỗi lần.
    ids = array(SIGNED_64_BIT)
    cursor = connection.execute(query, parameters)
    while rows := cursor.fetchmany(fetch_size):
        ids.extend(row[0] for row in rows)
    return ids


def iter_records(config: Mapping[str, Any], root: Path, limit: int | None) -> Iterator[dict]:
    """Stream ChEMBL observation records in activity_id order; `limit` caps rows (smoke).

    Nhận: config, thư mục gốc dự án, giới hạn số dòng (None = toàn bộ).
    Trả về: iterator, mỗi phần tử là một bản ghi (dict) của một activity.
    Vì sao cần: là "nguồn bản ghi" ChEMBL mà pipeline.py tiêu thụ. Dùng iterator để không
    bao giờ giữ cả 5 triệu dòng trong bộ nhớ.

    Cách đọc theo lô và lý do không dùng ORDER BY trên toàn bộ kết quả: xem
    SELECTED_IDS_QUERY ở đầu file. Thứ tự output ổn định theo activity_id giúp đối chiếu
    và tái lập giữa các lần chạy.
    """
    settings = config["chembl"]
    release = config["releases"]["chembl"]
    connection = open_chembl_readonly(root / config["paths"]["chembl_db"])
    try:
        # --- Kiểm tra schema trước khi đọc dữ liệu.
        check_schema(connection)
        # --- Bước 1: danh sách activity_id đã sắp xếp.
        ids = selected_activity_ids(connection, config["endpoints"], limit,
                                    settings["fetch_size"])
        LOGGER.info("ChEMBL activities selected: %s", f"{len(ids):,}")
        # sqlite3.Row cho phép lấy cột theo tên (row["activity_id"]) trong build_record.
        connection.row_factory = sqlite3.Row
        # --- Bước 2: chạy câu join cho từng lô id. Kích thước lô (10.000) phải nhỏ hơn
        # giới hạn số tham số "?" của một câu SQLite (32.766).
        batch_size = settings["id_batch_size"]
        # Lô 10.000 id (5,34 triệu id -> 534 lô).
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size].tolist()
            # Mỗi id một dấu '?', truyền vào như tham số.
            query = EXTRACTION_QUERY.format(id_placeholders=", ".join("?" for _ in batch))
            for row in connection.execute(query, batch):
                yield build_record(row, release)
    finally:
        # Luôn đóng kết nối, kể cả khi bên tiêu thụ dừng giữa chừng hoặc có lỗi.
        connection.close()


# --------------------------------------------------------------------------- inspect


def describe(config: Mapping[str, Any], root: Path) -> list[str]:
    """Inspect report: raw files, SQLite schema and required-column check (no data rows).

    Nhận: config và thư mục gốc. Trả về: danh sách dòng văn bản để in.
    Vì sao cần: chế độ inspect dùng hàm này để cho thấy file nào có mặt, bảng/cột nào tồn
    tại, cột bắt buộc nào thiếu, và mỗi trường quan trọng lấy từ đâu. Không đọc dòng dữ
    liệu nào.
    """
    # --- Danh sách file trong raw/chembl37 (kể cả file SDF không được đọc).
    raw_dir = root / config["paths"]["raw_chembl_dir"]
    db_path = root / config["paths"]["chembl_db"]
    lines = [f"[chembl] raw dir: {raw_dir}"]
    lines += [f"  file: {path.name} ({path.stat().st_size:,} bytes)"
              for path in sorted(raw_dir.iterdir())]
    lines.append(f"  NOT READ in this stage (SDF): {config['chembl']['sdf_not_read']}")
    lines.append(f"  db (read-only): {db_path}")
    # --- Schema: liệt kê mọi bảng và cột, rồi kiểm tra cột bắt buộc.
    connection = open_chembl_readonly(db_path)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        lines += [f"  table {table}: {', '.join(table_columns(connection, table))}"
                  for table in tables]
        missing = missing_columns(connection)
    finally:
        connection.close()
    lines.append("  required columns: " + ("all present" if not missing
                                           else "MISSING " + ", ".join(missing)))
    # --- Mô tả đường lấy từng trường quan trọng.
    lines += [
        "  per-measurement protein sequence: assays.variant_id -> variant_sequences.sequence; "
        "else (no variant) assays.tid -> target_components -> component_sequences.sequence "
        "(single-component targets only)",
        "  native assay ID: activities.assay_id -> assays.chembl_id",
        "  variant info: assays.variant_id, variant_sequences.mutation/accession/isoform",
        "  CXSMILES: NO CXSMILES column in ChEMBL SQLite; compound_structures.canonical_smiles "
        "is stored verbatim as the source SMILES (molfile not parsed in this stage)",
    ]
    return lines
