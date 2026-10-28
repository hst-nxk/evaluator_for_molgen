"""Raw -> Observations pipeline: build rows, write Parquet in chunks, verify against raw.

Vị trí trong luồng raw -> observations: đây là phần điều phối.
    chembl.iter_records / bindingdb.iter_records   (bản ghi với giá trị nguồn nguyên văn)
    -> build_observation: thêm các cột suy ra bằng normalize.py
       (observation_id, PMID số nguyên, hash protein, canonical SMILES, InChIKey,
        qualifier, mol/L, pActivity, các cột status)
    -> write_observations: ghi Parquet theo khối, mỗi nguồn một file riêng
    -> run_verify: đối chiếu output ngược về raw bằng đường tra cứu độc lập.
"""

from __future__ import annotations

# Thư viện chuẩn, cộng pyarrow để dựng bảng và đọc/ghi Parquet.
import logging
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

# Hai reader và phần chuẩn hóa dùng chung.
from molopt.observations import bindingdb, chembl, normalize

LOGGER = logging.getLogger(__name__)

# Kiểu của hàm "nguồn bản ghi": (config, thư mục gốc, giới hạn) -> iterator các dict.
RecordSource = Callable[[Mapping[str, Any], Path, "int | None"], Iterator[dict]]

# Bốn cột trạng thái độc lập của mỗi observation (mỗi cột có cột *_reason đi kèm).
STATUS_COLUMNS = ("protein_status", "assay_status", "structure_status", "activity_status")
# Số SMILES khác nhau được nhớ kết quả RDKit (cùng phân tử xuất hiện ở nhiều phép đo).
STRUCTURE_CACHE_SIZE = 200_000
# Đuôi file tạm khi đang ghi; dạng mốc thời gian, vd. 20261018_205349.
PARTIAL_SUFFIX = ".partial"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
# Sai số khi so số thực; số ví dụ sai lệch tối đa; dạng PMID hợp lệ.
FLOAT_RELATIVE_TOLERANCE = 1e-12
MAX_REPORTED_MISMATCHES = 50
PMID_PATTERN = re.compile(r"^\d+$")

# Sources whose activity cell carries its own qualifier prefix (e.g. '>10000').
# Chỉ BindingDB có dấu so sánh dính trong ô giá trị, vd. '>10000'.
EMBEDDED_QUALIFIER_SOURCES = frozenset({bindingdb.SOURCE_DATABASE})

# Columns supplied by the readers (raw values, resolved per measurement).
# 42 cột mà mọi reader phải trả về: đúng tên, không thừa, không thiếu.
SOURCE_FIELDS: list[tuple[str, pa.DataType]] = [
    # Nguồn và ID phép đo.
    ("source_database", pa.string()),
    ("source_release", pa.string()),
    ("native_measurement_id", pa.string()),
    ("source_curation", pa.string()),
    # Tài liệu.
    ("document_id", pa.string()),
    ("document_pmid_source", pa.string()),
    ("document_doi", pa.string()),
    ("document_patent_number", pa.string()),
    # Phân tử, theo đúng nguồn ghi.
    ("molecule_source_id", pa.string()),
    ("molecule_source_smiles", pa.string()),
    ("molecule_source_inchikey", pa.string()),
    ("molecule_source_chirality", pa.string()),
    ("molecule_source_chembl_id", pa.string()),
    ("molecule_source_type", pa.string()),
    # Protein của đúng phép đo này.
    ("protein_sequence", pa.string()),
    ("protein_sequence_origin", pa.string()),
    ("protein_accession", pa.string()),
    ("protein_accession_db", pa.string()),
    ("protein_target_id", pa.string()),
    ("protein_target_name", pa.string()),
    ("protein_target_type", pa.string()),
    ("protein_organism", pa.string()),
    ("protein_variant_id", pa.string()),
    ("protein_mutation", pa.string()),
    ("protein_isoform", pa.string()),
    ("protein_chain_count", pa.string()),
    ("protein_pdb_ids", pa.string()),
    ("protein_status", pa.string()),
    ("protein_reason", pa.string()),
    # Assay.
    ("assay_native_id", pa.string()),
    ("assay_entry_id", pa.string()),
    ("assay_name", pa.string()),
    ("assay_description", pa.string()),
    ("assay_type", pa.string()),
    ("assay_context", pa.string()),
    ("assay_status", pa.string()),
    ("assay_reason", pa.string()),
    # Giá trị đo nguyên văn.
    ("endpoint", pa.string()),
    ("activity_value_source", pa.string()),
    ("activity_unit_source", pa.string()),
    ("activity_qualifier_source", pa.string()),
    ("activity_context", pa.string()),
]

# Columns computed here from the source columns (never written back over them).
# 24 cột pipeline tự tính bằng normalize.py.
DERIVED_FIELDS: list[tuple[str, pa.DataType]] = [
    # ID, PMID đã chuẩn hóa, hash chuỗi protein.
    ("observation_id", pa.string()),
    ("document_pmid", pa.string()),
    ("protein_sequence_hash", pa.string()),
    # Cấu trúc phân tử: định danh của PARENT (ChEMBL Structure Pipeline), dùng ở mọi
    # bước sau; định danh của cấu trúc nguồn đọc thẳng để đối chiếu.
    ("molecule_canonical_smiles", pa.string()),
    ("molecule_inchikey", pa.string()),
    ("molecule_input_canonical_smiles", pa.string()),
    ("molecule_input_inchikey", pa.string()),
    ("molecule_source_inchikey_check", pa.string()),
    ("molecule_parent_fragments", pa.int32()),
    ("molecule_parent_excluded", pa.bool_()),
    ("molecule_cx_extension", pa.string()),
    ("molecule_stereo_groups", pa.string()),
    ("molecule_stereo_centers", pa.int32()),
    ("molecule_unspecified_stereo_centers", pa.int32()),
    ("structure_status", pa.string()),
    ("structure_reason", pa.string()),
    # Mã ChEMBL đã kiểm dạng và loại phân tử do ChEMBL gán (BindingDB: tra theo mã).
    ("molecule_chembl_id", pa.string()),
    ("molecule_chembl_type", pa.string()),
    # Giá trị đo đã quy đổi.
    ("activity_qualifier", pa.string()),
    ("activity_concentration_molar", pa.float64()),
    ("activity_p_activity", pa.float64()),
    ("activity_censored", pa.bool_()),
    ("activity_status", pa.string()),
    ("activity_reason", pa.string()),
]

# Schema chung của cả hai file output: observation_id đứng đầu, rồi các cột nguồn, rồi
# các cột suy ra còn lại. Hai nguồn dùng đúng schema này (cột nguồn không có -> null).
OBSERVATION_SCHEMA = pa.schema(DERIVED_FIELDS[:1] + SOURCE_FIELDS + DERIVED_FIELDS[1:])
# Tên 42 cột nguồn, dùng để kiểm tra bản ghi. 42 + 24 = 66 cột.
SOURCE_COLUMN_NAMES = frozenset(name for name, _ in SOURCE_FIELDS)
# Hàm tra loại phân tử theo mã ChEMBL (chembl.MoleculeTypeLookup); None = không tra.
MoleculeTypes = Callable[[str], "str | None"]

# RDKit là bước chậm nhất; cache theo chuỗi SMILES để mỗi phân tử chỉ tính một lần.
# Cùng một chuỗi SMILES chỉ chạy RDKit một lần.
describe_structure_cached = lru_cache(maxsize=STRUCTURE_CACHE_SIZE)(normalize.describe_structure)


# --------------------------------------------------------------------------- row building


def derive_activity(
    source_database: str,
    value_text: str | None,
    unit: str | None,
    qualifier_source: str | None,
    config: Mapping[str, Any],
) -> normalize.ActivityResult:
    """Activity conversion shared by both sources (qualifier, mol/L, pActivity, status).

    Nhận: tên nguồn, giá trị/đơn vị/qualifier nguyên văn của nguồn, config.
    Trả về: ActivityResult của normalize.parse_activity.
    Vì sao cần: hai nguồn ghi qualifier theo hai cách khác nhau. Hàm này đưa chúng về
    cùng một dạng (qualifier riêng + chuỗi số) rồi dùng chung một hàm quy đổi.
    """
    # exact = ['=']; censored = ['<', '>', '<=', '>=', '<<', '>>'].
    exact = config["qualifiers"]["exact"]
    censored = config["qualifiers"]["censored"]
    # ChEMBL: qualifier có cột riêng (standard_relation), dùng nguyên.
    # Mặc định (ChEMBL): dùng nguyên qualifier và giá trị.
    qualifier, number_text = qualifier_source, value_text
    # BindingDB: qualifier dính trong ô giá trị (">10000") -> tách ra; không có tiền tố
    # thì coi là "=" (implicit_qualifier trong config).
    # BindingDB: ' 2' -> ('=', '2'); '<10' -> ('<', '10').
    if source_database in EMBEDDED_QUALIFIER_SOURCES and value_text is not None:
        qualifier, number_text = normalize.split_embedded_qualifier(
            value_text, [*exact, *censored], config["bindingdb"]["implicit_qualifier"])
    # Quy đổi đơn vị qua bảng tường minh, tính pActivity, quyết định status.
    # Từ đây hai nguồn dùng chung một quy tắc.
    return normalize.parse_activity(number_text, unit, qualifier, config["unit_to_molar"],
                                    exact, censored)


def derive_molecule_type(record: Mapping[str, Any], chembl_id: str | None,
                         molecule_types: MoleculeTypes | None) -> str | None:
    """Loại phân tử do ChEMBL gán cho phân tử của bản ghi.

    Nhận: bản ghi nguồn, mã ChEMBL đã kiểm dạng, hàm tra theo mã (None = không tra).
    Trả về: dòng ChEMBL -> molecule_type của chính dòng đó (cùng câu join lúc đọc raw);
    dòng BindingDB -> tra ChEMBL theo mã BindingDB ghi; không có mã, mã không có trong
    ChEMBL, hoặc không có hàm tra -> None. Không đoán theo cấu trúc ở bước này.
    """
    if record["source_database"] == chembl.SOURCE_DATABASE:
        return record["molecule_source_type"]
    if chembl_id is None or molecule_types is None:
        return None
    return molecule_types(chembl_id)


def build_observation(record: Mapping[str, Any], config: Mapping[str, Any],
                      molecule_types: MoleculeTypes | None = None) -> dict[str, Any]:
    """Full observation row: source columns unchanged plus derived columns.

    Nhận: một bản ghi từ chembl/bindingdb (đúng tập cột nguồn), config, và hàm tra loại
    phân tử theo mã ChEMBL (bắt buộc khi chạy BindingDB thật, xem run_extract).
    Trả về: dict một dòng observation đủ 66 cột của OBSERVATION_SCHEMA.
    Vì sao cần: đây là chỗ duy nhất tạo cột suy ra, áp dụng giống nhau cho hai nguồn. Cột
    nguồn được giữ nguyên (**record), cột suy ra nằm cạnh, không ghi đè lên cột nguồn.
    """
    # --- Bản ghi phải có đúng tập cột nguồn: thiếu hoặc thừa cột là lỗi lập trình ở
    # reader, phải dừng ngay thay vì ghi ra dòng lệch schema.
    # Thiếu hoặc thừa cột -> dừng (pyarrow sẽ không tự báo lỗi).
    keys = set(record)
    if keys != SOURCE_COLUMN_NAMES:
        raise ValueError(
            f"Record {record.get('source_database')}:{record.get('native_measurement_id')} "
            f"has unexpected keys {sorted(keys - SOURCE_COLUMN_NAMES)} / "
            f"missing keys {sorted(SOURCE_COLUMN_NAMES - keys)}")
    # --- Giá trị hoạt tính: qualifier, mol/L, pActivity, activity_status.
    # Qualifier, mol/L, pActivity, status.
    activity = derive_activity(record["source_database"], record["activity_value_source"],
                               record["activity_unit_source"],
                               record["activity_qualifier_source"], config)
    # --- Cấu trúc phân tử (RDKit, có cache): canonical SMILES, InChIKey, structure_status.
    # Canonical SMILES, InChIKey, lập thể, status.
    structure = describe_structure_cached(record["molecule_source_smiles"])
    # --- Mã ChEMBL đã kiểm dạng và loại phân tử ChEMBL.
    chembl_id = normalize.chembl_molecule_id(record["molecule_source_chembl_id"])
    # --- Ghép cột nguồn + cột suy ra.
    # Cột nguồn giữ nguyên; cột suy ra thêm bên cạnh.
    return {
        **record,
        # observation_id chỉ từ nguồn + ID phép đo + endpoint; không dùng chuỗi protein
        # (lý do: xem normalize.make_observation_id).
        "observation_id": normalize.make_observation_id(
            record["source_database"], record["native_measurement_id"], record["endpoint"]),
        # '12345.0' -> '12345'.
        "document_pmid": normalize.normalize_integer_id(record["document_pmid_source"]),
        # SHA-256 của chuỗi protein.
        "protein_sequence_hash": normalize.sequence_hash(record["protein_sequence"]),
        # Lấy từ kết quả chuẩn hóa cấu trúc (parent) và từ cấu trúc nguồn đọc thẳng.
        "molecule_canonical_smiles": structure.canonical_smiles,
        "molecule_inchikey": structure.inchikey,
        "molecule_input_canonical_smiles": structure.input_canonical_smiles,
        "molecule_input_inchikey": structure.input_inchikey,
        "molecule_source_inchikey_check": normalize.compare_source_inchikey(
            record["molecule_source_inchikey"], structure.input_inchikey),
        "molecule_parent_fragments": structure.parent_fragments,
        "molecule_parent_excluded": structure.parent_excluded,
        "molecule_cx_extension": structure.cx_extension,
        "molecule_stereo_groups": structure.stereo_groups,
        "molecule_stereo_centers": structure.stereo_centers,
        "molecule_unspecified_stereo_centers": structure.unspecified_stereo_centers,
        "structure_status": structure.status,
        "structure_reason": structure.reason,
        "molecule_chembl_id": chembl_id,
        "molecule_chembl_type": derive_molecule_type(record, chembl_id, molecule_types),
        # Lấy từ kết quả derive_activity.
        "activity_qualifier": activity.qualifier,
        "activity_concentration_molar": activity.concentration_molar,
        "activity_p_activity": activity.p_activity,
        "activity_censored": activity.censored,
        "activity_status": activity.status,
        "activity_reason": activity.reason,
    }


# --------------------------------------------------------------------------- writing


def write_observations(
    records: Iterable[dict], output_path: Path, config: Mapping[str, Any], label: str,
    molecule_types: MoleculeTypes | None = None,
) -> int:
    """Build and write observations in fixed-size Parquet chunks; return the row count.

    Rows go to '<name>.partial' first and are renamed only after a complete run.

    Nhận: iterator bản ghi của một nguồn, đường dẫn file output, config, nhãn để ghi log,
    hàm tra loại phân tử theo mã ChEMBL (xem build_observation).
    Trả về: số dòng đã ghi.
    Vì sao cần:
    - Ghi theo khối (chunk_rows = 100.000 dòng) để bộ nhớ chỉ giữ một khối tại một thời
      điểm, dù nguồn có hàng triệu dòng.
    - Ghi vào file ".partial" rồi mới đổi tên: một file tên chính thức luôn là file hoàn
      chỉnh. Nếu chạy hỏng giữa chừng, file .partial bị xóa và không để lại output dở.
    """
    # 100.000 dòng mỗi khối; ghi log mỗi 500.000 dòng.
    chunk_rows = config["run"]["chunk_rows"]
    progress_every = config["run"]["progress_every_rows"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi vào file .partial trước, xong hẳn mới đổi tên.
    partial_path = output_path.with_name(output_path.name + PARTIAL_SUFFIX)
    written = 0
    buffer: list[dict[str, Any]] = []
    # Schema cố định 57 cột cho mọi khối.
    writer = pq.ParquetWriter(partial_path, OBSERVATION_SCHEMA)
    try:
        # --- Dựng từng dòng observation, gom thành khối, ghi khối khi đủ kích thước.
        for record in records:
            # Bản ghi -> một dòng observation đủ 66 cột.
            buffer.append(build_observation(record, config, molecule_types))
            # Đủ một khối: ghi xuống file rồi làm rỗng buffer.
            if len(buffer) >= chunk_rows:
                writer.write_table(pa.Table.from_pylist(buffer, schema=OBSERVATION_SCHEMA))
                written += len(buffer)
                buffer.clear()
                # Ghi log tiến độ mỗi khi vượt qua một mốc progress_every (500.000 dòng).
                # Chỉ đúng tại 500.000, 1.000.000, ...
                if written % progress_every < chunk_rows:
                    LOGGER.info("[%s] %s rows written", label, f"{written:,}")
        # --- Ghi khối cuối (chưa đủ chunk_rows).
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=OBSERVATION_SCHEMA))
            written += len(buffer)
    except Exception:
        # --- Lỗi bất kỳ: đóng writer, xóa file dở, rồi ném lại lỗi gốc (không nuốt lỗi).
        # Ctrl+C không vào nhánh này; file .partial có thể còn lại nhưng vô hại.
        writer.close()
        partial_path.unlink(missing_ok=True)
        raise
    # --- Thành công: đóng file rồi đổi tên .partial -> tên chính thức.
    # close() ghi footer; phải close xong thì file mới đọc được.
    writer.close()
    # Đổi tên .partial -> tên chính thức.
    partial_path.replace(output_path)
    LOGGER.info("[%s] done: %s rows -> %s", label, f"{written:,}", output_path)
    return written


def source_outputs(config: Mapping[str, Any]) -> list[tuple[str, RecordSource, str]]:
    """(label, record iterator, output file name) for each source.

    Nhận: config. Trả về: danh sách (tên nguồn, hàm đọc nguồn, tên file output).
    Vì sao cần: mỗi nguồn ghi ra MỘT FILE RIÊNG, không gộp. BindingDB có chứa dữ liệu
    nhập lại từ ChEMBL; khử trùng lặp giữa hai nguồn để bước sau, khi đã đủ căn cứ xác
    định hai bản ghi có phải cùng một thí nghiệm hay không.
    """
    # ChEMBL chạy trước, BindingDB sau.
    return [
        (chembl.SOURCE_DATABASE, chembl.iter_records, config["output"]["chembl_file"]),
        (bindingdb.SOURCE_DATABASE, bindingdb.iter_records, config["output"]["bindingdb_file"]),
    ]


def run_extract(config: Mapping[str, Any], root: Path, output_dir: Path,
                limit: int | None, force: bool) -> dict[str, int]:
    """Extract both sources into `output_dir`; `limit` caps rows per source (smoke).

    A source whose final Parquet file already exists is skipped unless `force` is set.
    The final name only appears after a complete write (see write_observations), so an
    existing file is always a finished one.

    Nhận: config, thư mục gốc, thư mục output, giới hạn số dòng mỗi nguồn, cờ force.
    Trả về: {tên nguồn: số dòng} (với nguồn bị bỏ qua: số dòng đọc từ metadata file có sẵn).
    Vì sao có cơ chế bỏ qua: nếu nguồn thứ hai lỗi, lần chạy lại không phải làm lại nguồn
    thứ nhất (mất khoảng 1 giờ). Chỉ bỏ qua ở mức cả file, không nối tiếp giữa chừng,
    để không bao giờ trộn output của hai phiên bản code trong cùng một file.
    Phiên bản ChEMBL Structure Pipeline được kiểm trước khi đọc dòng nào.
    """
    normalize.check_structure_pipeline_version(
        config["structure"]["chembl_structure_pipeline_version"])
    # Tra loại phân tử ChEMBL cho dòng BindingDB; kết nối chỉ mở khi có mã cần tra.
    molecule_types = chembl.MoleculeTypeLookup(root / config["paths"]["chembl_db"],
                                               config["chembl"]["molecule_type_cache_size"])
    counts = {}
    try:
        # Mỗi nguồn: (tên, hàm đọc nguồn, tên file output).
        for label, iter_records, file_name in source_outputs(config):
            output_path = output_dir / file_name
            # --- File hoàn chỉnh đã có và không --force: bỏ qua, chỉ đọc số dòng.
            # Chỉ xét file tên chính thức; sửa code thì chạy lại với --force.
            if output_path.is_file() and not force:
                counts[label] = pq.ParquetFile(output_path).metadata.num_rows
                LOGGER.info("[%s] skipped: %s already exists (%s rows); use --force to "
                            "rebuild", label, output_path, f"{counts[label]:,}")
                continue
            # --- Chạy nguồn: đọc raw -> dựng observation -> ghi Parquet.
            LOGGER.info("[%s] stage: extract (limit=%s)", label, limit)
            # Generator: raw được đọc dần khi write_observations lặp qua.
            counts[label] = write_observations(iter_records(config, root, limit),
                                               output_path, config, label, molecule_types)
    finally:
        molecule_types.close()
    return counts


def run_smoke(config: Mapping[str, Any], root: Path) -> dict[str, int]:
    """Small real-data sample from both sources into a new runs/observations/smoke_<time>/.

    Nhận: config, thư mục gốc. Trả về: {tên nguồn: số dòng}.
    Vì sao cần: chạy thử nhanh trên dữ liệu thật (5.000 dòng mỗi nguồn) để phát hiện lỗi
    trước khi chạy full. Luôn ghi vào thư mục mới và luôn force, nên không đụng tới output
    của full.
    """
    # Thư mục smoke mới, 5.000 bản ghi mỗi nguồn.
    stamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    output_dir = root / config["paths"]["runs_dir"] / f"smoke_{stamp}"
    return run_extract(config, root, output_dir, config["run"]["smoke_rows_per_source"],
                       force=True)


def run_full(config: Mapping[str, Any], root: Path, force: bool) -> dict[str, int]:
    """All relevant observations into paths.observations_dir (Parquet only).

    Nhận: config, thư mục gốc, cờ force. Trả về: {tên nguồn: số dòng}.
    """
    # Toàn bộ dữ liệu, ghi vào paths.observations_dir (hiện là data/observations_v2/).
    return run_extract(config, root, root / config["paths"]["observations_dir"], None, force)


# --------------------------------------------------------------------------- verify: outputs


def iter_output_rows(path: Path, batch_rows: int) -> Iterator[dict[str, Any]]:
    """Stream rows of an output Parquet file as dicts, one batch in memory at a time.

    Nhận: đường dẫn file Parquet, số dòng mỗi lô. Trả về: iterator các dòng dạng dict.
    """
    # Đọc theo lô để không giữ cả file trong RAM.
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
        yield from batch.to_pylist()


def structure_flags(row: Mapping[str, Any]) -> list[str]:
    """Nhãn đếm cho một dòng: kết quả so InChIKey nguồn, parent có khác cấu trúc nguồn
    không, số mảnh của parent (1 / >1), cờ exclude, có mã ChEMBL không, loại phân tử."""
    resolved = row["structure_status"] == normalize.RESOLVED
    fragments = row["molecule_parent_fragments"]
    return [
        f"source_inchikey_check|{row['molecule_source_inchikey_check']}",
        "parent_changed|" + (str(row["molecule_canonical_smiles"]
                                 != row["molecule_input_canonical_smiles"])
                             if resolved else "no_structure"),
        "parent_fragments|" + ("none" if fragments is None
                               else "1" if fragments == normalize.SINGLE_FRAGMENT else ">1"),
        f"parent_excluded|{row['molecule_parent_excluded']}",
        f"chembl_id_present|{row['molecule_chembl_id'] is not None}",
        f"chembl_type|{row['molecule_chembl_type']}",
    ]


def scan_outputs(paths: list[Path], config: Mapping[str, Any]) -> dict[str, Any]:
    """One pass over the outputs: counts, ID uniqueness, PMID form, sample and regression rows.

    Nhận: danh sách file output, config.
    Trả về: dict gồm số dòng theo nguồn/endpoint, phân bố status+lý do, số ID trùng, các
    PMID sai dạng, mẫu ngẫu nhiên để đối chiếu, và các dòng dùng cho kiểm tra hồi quy
    (accession P03367, monomer 13590).
    Vì sao gom vào một lượt đọc: output có hơn 8 triệu dòng; đọc một lần cho mọi phép
    kiểm tra thay vì đọc lại cho từng phép.
    """
    # Các ca hồi quy trong config: P03367, monomer 13590, '|r|'.
    regression = config["regression"]
    # --- Chọn trước vị trí mẫu ngẫu nhiên (hạt giống cố định -> lần verify nào cũng ra
    # cùng một mẫu, kết quả tái lập được).
    # Seed cố định: lần verify nào cũng ra cùng một mẫu.
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    sample_size = min(config["run"]["verify_sample_rows"], total)
    sample_positions = set(random.Random(config["run"]["verify_seed"])
                           .sample(range(total), sample_size))
    # ID đã gặp, để phát hiện ID trùng.
    seen_ids: set[str] = set()
    scan: dict[str, Any] = {
        "rows_by_source": Counter(), "rows_by_source_endpoint": Counter(),
        "status_reasons": Counter(), "duplicate_ids": 0, "bad_pmids": [],
        "sample": [], "accession_rows": [], "monomer_rows": [], "structure_flags": Counter(),
    }
    # Số thứ tự dòng, tính liền qua cả hai file.
    position = 0
    for path in paths:
        LOGGER.info("[verify] scanning %s", path)
        for row in iter_output_rows(path, config["run"]["chunk_rows"]):
            # --- Đếm dòng theo nguồn và endpoint; đếm từng cặp status + lý do.
            source = row["source_database"]
            scan["rows_by_source"][source] += 1
            scan["rows_by_source_endpoint"][f"{source}|{row['endpoint']}"] += 1
            # Đếm theo nguồn | cột status | status | lý do.
            for column in STATUS_COLUMNS:
                reason_column = column.replace("_status", "_reason")
                scan["status_reasons"][
                    f"{source}|{column}|{row[column]}|{row[reason_column]}"] += 1
            # --- Đếm các cờ cấu trúc và loại phân tử (để báo cáo, không có ngưỡng).
            for flag in structure_flags(row):
                scan["structure_flags"][f"{source}|{flag}"] += 1
            # --- observation_id phải duy nhất trên cả hai file.
            if row["observation_id"] in seen_ids:
                scan["duplicate_ids"] += 1
            seen_ids.add(row["observation_id"])
            # --- document_pmid phải là số nguyên thuần (không còn dạng "12345.0").
            pmid = row["document_pmid"]
            if pmid is not None and not PMID_PATTERN.match(pmid):
                scan["bad_pmids"].append(f"{row['observation_id']}={pmid}")
            # --- Gom mẫu ngẫu nhiên và các dòng cho kiểm tra hồi quy.
            # Giữ dòng cho mẫu ngẫu nhiên và cho các ca hồi quy.
            if position in sample_positions:
                scan["sample"].append(row)
            if row["protein_accession"] == regression["accession_construct_check"]:
                scan["accession_rows"].append(row)
            if (source == bindingdb.SOURCE_DATABASE
                    and row["molecule_source_id"]
                    == regression["bindingdb_monomer_enhanced_stereo"]):
                scan["monomer_rows"].append(row)
            position += 1
    return scan


# --------------------------------------------------------------------------- verify: raw side


def read_bindingdb_raw(config: Mapping[str, Any], root: Path,
                       wanted_ids: set[str]) -> dict[str, Any]:
    """Independent pass over the raw BindingDB files for the rows under verification.

    Nhận: config, thư mục gốc, tập Reactant_set_id cần đối chiếu.
    Trả về: dict gồm tổng số ô endpoint không rỗng trong raw, dòng raw của các id cần
    đối chiếu, các liên kết rsid_eaids của chúng, và tên assay tương ứng.
    Vì sao cần: verify phải đọc lại raw bằng một đường riêng (không dùng iter_records /
    resolve_* của lúc extract), để lỗi trong code extract không tự "xác nhận" chính nó.
    """
    settings = config["bindingdb"]
    raw_dir = root / config["paths"]["raw_bindingdb_dir"]
    all_zip = raw_dir / settings["all_zip"]
    # Đọc lại raw bằng đường riêng, không dùng code extract.
    rows = bindingdb.iter_tsv_rows(all_zip, settings["encoding"])
    _, header = next(rows)
    roles = {**settings["all_columns"],
             **{endpoint: spec["column"] for endpoint, spec in settings["endpoint_columns"].items()
                if endpoint in config["endpoints"]}}
    columns = bindingdb.column_indexes(header, roles, all_zip.name)
    endpoint_indexes = [columns[endpoint] for endpoint in config["endpoints"]
                        if endpoint in columns]
    # --- Lượt 1: đọc toàn bộ BindingDB_All; đếm ô endpoint không rỗng (để so với số dòng
    # đã ghi) và giữ lại dòng raw của các id cần đối chiếu.
    selected_cells = 0
    raw_rows: dict[str, dict[str, str | None]] = {}
    for line_number, fields in rows:
        # Đếm ô endpoint không rỗng; phải bằng số dòng đã ghi.
        selected_cells += sum(bindingdb.field(fields, index) is not None
                              for index in endpoint_indexes)
        reactant_set_id = (bindingdb.field(fields, columns["reactant_set_id"]) or "").strip()
        # Giữ dòng raw của các id cần đối chiếu.
        if reactant_set_id in wanted_ids:
            raw_rows[reactant_set_id] = {role: bindingdb.field(fields, index)
                                         for role, index in columns.items()}
        if line_number % config["run"]["progress_every_rows"] == 0:
            LOGGER.info("[verify] BindingDB_All raw lines read: %s", f"{line_number:,}")

    # --- Lượt 2: liên kết rsid_eaids của các id cần đối chiếu.
    links: dict[str, list[tuple[str, str]]] = defaultdict(list)
    link_rows = bindingdb.iter_tsv_rows(raw_dir / settings["rsid_eaids_zip"], settings["encoding"])
    _, link_header = next(link_rows)
    link_columns = bindingdb.column_indexes(link_header, settings["rsid_eaids_columns"],
                                            settings["rsid_eaids_zip"])
    # Reactant_set_id -> cặp (ENTRYID, ASSAYID), chỉ cho id cần đối chiếu.
    for line_number, fields in link_rows:
        reactant_set_id = (bindingdb.field(fields, link_columns["reactant_set_id"]) or "").strip()
        if reactant_set_id in wanted_ids:
            pair = bindingdb.parse_entry_assay_id(
                fields[link_columns["entry_assay_id"]], settings["entry_assay_separator"],
                line_number)
            if pair not in links[reactant_set_id]:
                links[reactant_set_id].append(pair)

    # --- Lượt 3: tên assay của các cặp (ENTRYID, ASSAYID) liên quan; giữ dạng danh sách
    # để phát hiện cặp trùng.
    wanted_pairs = {pair for pairs in links.values() for pair in pairs}
    assay_names: dict[tuple[str, str], list[str | None]] = defaultdict(list)
    assay_rows = bindingdb.iter_tsv_rows(raw_dir / settings["assays_zip"], settings["encoding"])
    _, assay_header = next(assay_rows)
    assay_columns = bindingdb.column_indexes(assay_header, settings["assays_columns"],
                                             settings["assays_zip"])
    # Cặp -> danh sách tên assay (hai tên = không xác định).
    for _, fields in assay_rows:
        pair = ((bindingdb.field(fields, assay_columns["entry_id"]) or "").strip(),
                (bindingdb.field(fields, assay_columns["assay_id"]) or "").strip())
        if pair in wanted_pairs:
            assay_names[pair].append(bindingdb.field(fields, assay_columns["assay_name"]))
    return {"selected_cells": selected_cells, "rows": raw_rows, "links": links,
            "assay_names": assay_names}


def expected_bindingdb(observation: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Source columns as they should be, read directly from the raw BindingDB files.

    Nhận: một dòng observation BindingDB, kết quả read_bindingdb_raw.
    Trả về: dict {cột: giá trị đáng lẽ phải có}, tính lại từ raw bằng quy tắc viết riêng.
    """
    native_id = observation["native_measurement_id"]
    raw_row = raw["rows"].get(native_id)
    # Không tìm thấy dòng raw: trả về giá trị chắc chắn không khớp để báo sai lệch.
    if raw_row is None:
        return {"native_measurement_id": f"<absent from raw: {native_id}>"}
    # --- Protein: chỉ khi target một chain; accession SwissProt trước, rồi TrEMBL.
    # Viết lại quy tắc protein độc lập, không gọi resolve_protein.
    single_chain = (raw_row["chain_count"] or "").strip() == str(bindingdb.SINGLE_CHAIN)
    sequence = raw_row["chain_1_sequence"] if single_chain else None
    accession = (raw_row["chain_1_swissprot_id"] or raw_row["chain_1_trembl_id"]
                 if sequence else None)
    # --- Assay: chỉ khi có đúng một cặp liên kết và đúng một tên assay.
    # Assay chỉ xác định khi có đúng một cặp.
    pairs = raw["links"].get(native_id, [])
    entry_id, assay_id = pairs[0] if len(pairs) == 1 else (None, None)
    names = raw["assay_names"].get((entry_id, assay_id), [])
    # Giá trị nguồn đáng lẽ phải có, đọc thẳng từ raw.
    return {
        "native_measurement_id": native_id,
        "molecule_source_id": raw_row["monomer_id"],
        "molecule_source_smiles": raw_row["ligand_smiles"],
        "molecule_source_inchikey": raw_row["ligand_inchikey"],
        "molecule_source_chembl_id": raw_row["chembl_id"],
        "molecule_source_type": None,
        "protein_sequence": sequence,
        "protein_accession": accession.strip() if accession else None,
        "activity_value_source": raw_row[observation["endpoint"]],
        "document_pmid_source": raw_row["pmid"],
        "document_doi": raw_row["article_doi"],
        "document_id": entry_id,
        "assay_entry_id": entry_id,
        "assay_native_id": assay_id,
        "assay_name": names[0] if len(names) == 1 else None,
    }


# Truy vấn riêng từng bảng ChEMBL cho verify. Cố ý KHÔNG dùng lại EXTRACTION_QUERY, để
# lỗi trong câu join lúc extract không được "xác nhận" bởi chính câu join đó.
# Mỗi truy vấn đọc một bảng theo khóa chính.
CHEMBL_LOOKUPS = {
    "activity": "SELECT assay_id, doc_id, molregno, standard_type, standard_relation, "
                "standard_value, standard_units FROM activities WHERE activity_id = ?",
    "assay": "SELECT chembl_id, tid, variant_id FROM assays WHERE assay_id = ?",
    "variant": "SELECT sequence, accession FROM variant_sequences WHERE variant_id = ?",
    "components": "SELECT component_id FROM target_components WHERE tid = ?",
    "component": "SELECT sequence, accession FROM component_sequences WHERE component_id = ?",
    "molecule": "SELECT chembl_id, molecule_type FROM molecule_dictionary WHERE molregno = ?",
    "structure": "SELECT canonical_smiles, standard_inchi_key FROM compound_structures "
                 "WHERE molregno = ?",
    "document": "SELECT chembl_id, pubmed_id, doi FROM docs WHERE doc_id = ?",
    # Loại phân tử theo mã ChEMBL (dòng BindingDB), viết riêng khỏi chembl.MoleculeTypeLookup.
    "molecule_type_by_id": "SELECT molecule_type FROM molecule_dictionary WHERE chembl_id = ?",
    # Parent do ChEMBL tính (molecule_hierarchy), để đối chiếu parent của pipeline.
    "parent_molregno": "SELECT parent_molregno FROM molecule_hierarchy WHERE molregno = ?",
    "activity_molregno": "SELECT molregno FROM activities WHERE activity_id = ?",
}


def lookup_one(connection: Any, query_name: str, key: object) -> tuple | None:
    """Single-row lookup used by verify (independent of the extraction query).

    Nhận: kết nối, tên truy vấn trong CHEMBL_LOOKUPS, khóa. Trả về: một dòng hoặc None
    (kể cả khi khóa là None).
    """
    if key is None:
        return None
    return connection.execute(CHEMBL_LOOKUPS[query_name], (key,)).fetchone()


def expected_chembl(observation: Mapping[str, Any], connection: Any) -> dict[str, Any]:
    """Source columns as they should be, from separate per-table ChEMBL lookups.

    Nhận: một dòng observation ChEMBL, kết nối SQLite chỉ đọc.
    Trả về: dict {cột: giá trị đáng lẽ phải có}, tính lại bằng các truy vấn từng bảng.
    """
    # --- Dòng activities gốc.
    activity_id = int(observation["native_measurement_id"])
    activity = lookup_one(connection, "activity", activity_id)
    if activity is None:
        return {"native_measurement_id": f"<absent from raw: {activity_id}>"}
    # Tra assay của activity.
    assay_id, doc_id, molregno, standard_type, relation, value, units = activity
    assay = lookup_one(connection, "assay", assay_id) or (None, None, None)
    assay_chembl_id, tid, variant_id = assay
    # --- Protein, viết lại độc lập cùng quy tắc: có variant -> chuỗi variant (không thay
    # bằng wild type); không variant -> chuỗi component khi target có đúng một component.
    sequence = accession = None
    # Có variant: chỉ dùng chuỗi variant.
    if variant_id is not None:
        sequence, accession = lookup_one(connection, "variant", variant_id) or (None, None)
        sequence = sequence or None
        accession = accession if sequence else None
    else:
        # Không variant: chỉ khi target có đúng 1 component.
        components = connection.execute(CHEMBL_LOOKUPS["components"], (tid,)).fetchall()
        if len(components) == 1:
            sequence, accession = (lookup_one(connection, "component", components[0][0])
                                   or (None, None))
            sequence = sequence or None
            accession = accession if sequence else None
    # --- Phân tử và tài liệu.
    molecule = lookup_one(connection, "molecule", molregno)
    structure = lookup_one(connection, "structure", molregno)
    document = lookup_one(connection, "document", doc_id) or (None, None, None)
    return {
        "native_measurement_id": str(activity_id),
        "endpoint": standard_type,
        "activity_value_source": chembl.text_or_none(value),
        "activity_unit_source": chembl.text_or_none(units),
        "activity_qualifier_source": chembl.text_or_none(relation),
        "assay_native_id": assay_chembl_id,
        "protein_sequence": sequence,
        "protein_accession": accession,
        "molecule_source_id": molecule[0] if molecule else None,
        "molecule_source_chembl_id": molecule[0] if molecule else None,
        "molecule_source_type": molecule[1] if molecule else None,
        "molecule_source_smiles": structure[0] if structure else None,
        "molecule_source_inchikey": structure[1] if structure else None,
        "document_id": document[0],
        "document_pmid_source": chembl.text_or_none(document[1]),
        "document_doi": document[2],
    }


def expected_molecule_type(observation: Mapping[str, Any], expected: Mapping[str, Any],
                           chembl_id: str | None, connection: Any) -> str | None:
    """Loại phân tử kỳ vọng: ChEMBL lấy từ dòng molecule_dictionary đã tra lại; BindingDB
    tra theo mã bằng truy vấn riêng của verify (không dùng chembl.MoleculeTypeLookup)."""
    if observation["source_database"] == chembl.SOURCE_DATABASE:
        return expected.get("molecule_source_type")
    row = lookup_one(connection, "molecule_type_by_id", chembl_id)
    return chembl.text_or_none(row[0]) if row else None


def expected_derived(observation: Mapping[str, Any], expected: Mapping[str, Any],
                     config: Mapping[str, Any], connection: Any) -> dict[str, Any]:
    """Derived columns recomputed from the raw-side values.

    Nhận: dòng observation, các giá trị nguồn đọc lại từ raw (expected_*), config, kết nối
    ChEMBL chỉ đọc (tra loại phân tử của dòng BindingDB).
    Trả về: dict các cột suy ra tính lại từ giá trị raw.
    Vì sao cần: kiểm tra rằng cột suy ra trong output đúng là kết quả của giá trị raw thật
    (không chỉ của giá trị đã lưu trong output).
    """
    # Tính lại cột suy ra từ giá trị raw, bằng hàm của normalize.py.
    source = observation["source_database"]

    def raw_value(column: str) -> Any:
        # Ưu tiên giá trị đọc lại từ raw; cột nào verify không đọc lại thì dùng giá trị lưu.
        return expected.get(column, observation[column])

    activity = derive_activity(source, raw_value("activity_value_source"),
                               raw_value("activity_unit_source"),
                               raw_value("activity_qualifier_source"), config)
    structure = normalize.describe_structure(raw_value("molecule_source_smiles"))
    chembl_id = normalize.chembl_molecule_id(raw_value("molecule_source_chembl_id"))
    return {
        "observation_id": normalize.make_observation_id(
            source, raw_value("native_measurement_id"), raw_value("endpoint")),
        "document_pmid": normalize.normalize_integer_id(raw_value("document_pmid_source")),
        "protein_sequence_hash": normalize.sequence_hash(raw_value("protein_sequence")),
        "molecule_canonical_smiles": structure.canonical_smiles,
        "molecule_inchikey": structure.inchikey,
        "molecule_input_canonical_smiles": structure.input_canonical_smiles,
        "molecule_input_inchikey": structure.input_inchikey,
        "molecule_source_inchikey_check": normalize.compare_source_inchikey(
            raw_value("molecule_source_inchikey"), structure.input_inchikey),
        "molecule_parent_fragments": structure.parent_fragments,
        "molecule_parent_excluded": structure.parent_excluded,
        "structure_status": structure.status,
        "structure_reason": structure.reason,
        "molecule_chembl_id": chembl_id,
        "molecule_chembl_type": expected_molecule_type(observation, expected, chembl_id,
                                                       connection),
        "activity_qualifier": activity.qualifier,
        "activity_concentration_molar": activity.concentration_molar,
        "activity_p_activity": activity.p_activity,
        "activity_censored": activity.censored,
        "activity_status": activity.status,
        "activity_reason": activity.reason,
    }


def values_match(stored: Any, expected: Any) -> bool:
    """Exact match, except floats which are compared with a tight relative tolerance.

    Nhận: giá trị đã lưu và giá trị kỳ vọng. Trả về: True nếu khớp.
    Số thực so với sai số tương đối 1e-12 để bỏ qua sai khác làm tròn ở bit cuối.
    """
    # Số thực so gần đúng; kiểu khác so bằng tuyệt đối.
    if isinstance(stored, float) and isinstance(expected, float):
        return math.isclose(stored, expected, rel_tol=FLOAT_RELATIVE_TOLERANCE)
    return stored == expected


def compare_row(observation: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """Mismatch descriptions for one observation.

    Nhận: dòng observation và dict giá trị kỳ vọng. Trả về: danh sách mô tả các cột lệch
    (rỗng nếu khớp hết).
    """
    # Mỗi cột lệch -> một dòng mô tả; danh sách rỗng = khớp.
    return [f"{observation['observation_id']}.{column}: stored={observation.get(column)!r} "
            f"raw={value!r}"
            for column, value in expected.items()
            if not values_match(observation.get(column), value)]


# --------------------------------------------------------------------------- verify: checks


def reconcile_sample(scan: Mapping[str, Any], bindingdb_raw: Mapping[str, Any],
                     connection: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare every sampled row back to raw, field by field.

    Nhận: kết quả scan_outputs, dữ liệu raw BindingDB, kết nối ChEMBL, config.
    Trả về: dict báo cáo (số dòng đã kiểm, số sai lệch, ví dụ sai lệch, passed).
    """
    mismatches: list[str] = []
    for observation in scan["sample"]:
        # Giá trị nguồn kỳ vọng theo đúng nguồn của dòng, rồi thêm cột suy ra kỳ vọng.
        if observation["source_database"] == chembl.SOURCE_DATABASE:
            expected = expected_chembl(observation, connection)
        else:
            expected = expected_bindingdb(observation, bindingdb_raw)
        # Gộp giá trị nguồn và cột suy ra kỳ vọng.
        expected = {**expected, **expected_derived(observation, expected, config, connection)}
        mismatches += compare_row(observation, expected)
    # Mẫu rỗng không được tính là đạt.
    return {"rows_checked": len(scan["sample"]), "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:MAX_REPORTED_MISMATCHES],
            "passed": bool(scan["sample"]) and not mismatches}


def chembl_parent_inchikey(observation: Mapping[str, Any], connection: Any) -> str | None:
    """InChIKey parent do ChEMBL tự tính cho phân tử của một dòng ChEMBL, hoặc None.

    activity -> molregno -> molecule_hierarchy.parent_molregno -> compound_structures.
    Không có dòng hierarchy thì phân tử là parent của chính nó (lấy cấu trúc của nó).
    """
    activity = lookup_one(connection, "activity_molregno",
                          int(observation["native_measurement_id"]))
    if activity is None:
        return None
    hierarchy = lookup_one(connection, "parent_molregno", activity[0])
    parent_molregno = hierarchy[0] if hierarchy and hierarchy[0] is not None else activity[0]
    structure = lookup_one(connection, "structure", parent_molregno)
    return structure[1] if structure else None


def check_chembl_parent_agreement(scan: Mapping[str, Any], connection: Any) -> dict[str, Any]:
    """Mức khớp giữa parent của pipeline và parent ChEMBL tự tính, trên mẫu ngẫu nhiên.

    Nhận: kết quả scan_outputs, kết nối ChEMBL.
    Trả về: số dòng mẫu ChEMBL có cấu trúc, đếm theo normalize.compare_source_inchikey
    (khớp hoàn toàn / cùng khung khác lớp / khác khung / ChEMBL không có parent), ví dụ.
    Vì sao cần: kiểm tra ĐỘC LẬP bước chuẩn hóa. ChEMBL tạo parent bằng chính ChEMBL
    Structure Pipeline (Bento et al. 2020) nhưng tính InChI bằng công cụ của họ, nên có
    thể lệch nhỏ ở lớp lập thể; đây là số đo để báo cáo, không phải điều kiện đạt/không đạt.
    """
    counts: Counter = Counter()
    examples = []
    for observation in scan["sample"]:
        if (observation["source_database"] != chembl.SOURCE_DATABASE
                or observation["molecule_inchikey"] is None):
            continue
        reference = chembl_parent_inchikey(observation, connection)
        outcome = (normalize.compare_source_inchikey(reference, observation["molecule_inchikey"])
                   or "chembl_parent_missing")
        counts[outcome] += 1
        if outcome != normalize.INCHIKEY_MATCH and len(examples) < MAX_REPORTED_MISMATCHES:
            examples.append(f"{observation['observation_id']}: pipeline "
                            f"{observation['molecule_inchikey']} chembl {reference} "
                            f"source {observation['molecule_source_smiles']}")
    return {"rows_checked": sum(counts.values()), "outcomes": dict(counts),
            "examples": examples}


def check_accession_constructs(scan: Mapping[str, Any], bindingdb_raw: Mapping[str, Any],
                               connection: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Regression: each stored sequence for the accession equals its own measurement's raw one.

    Nhận: kết quả scan_outputs, raw BindingDB, kết nối ChEMBL, config.
    Trả về: dict báo cáo, gồm phân bố độ dài chuỗi đã lưu và số sai lệch.
    Vì sao cần: P03367 có nhiều construct (99 aa và 1.447 aa). Kiểm tra này chứng minh mỗi
    phép đo giữ chuỗi của chính nó, không bị thay bằng một chuỗi chung theo accession.
    """
    mismatches = []
    stored_lengths: Counter = Counter()
    # So chuỗi đã lưu với chuỗi raw của chính phép đo đó.
    for observation in scan["accession_rows"]:
        if observation["source_database"] == chembl.SOURCE_DATABASE:
            expected = expected_chembl(observation, connection)
        else:
            expected = expected_bindingdb(observation, bindingdb_raw)
        stored = observation["protein_sequence"]
        stored_lengths[len(stored) if stored else 0] += 1
        if stored != expected.get("protein_sequence"):
            raw_sequence = expected.get("protein_sequence")
            mismatches.append(f"{observation['observation_id']}: stored "
                              f"{len(stored) if stored else 0} aa, raw "
                              f"{len(raw_sequence) if raw_sequence else 0} aa")
    return {"accession": config["regression"]["accession_construct_check"],
            "rows_checked": len(scan["accession_rows"]),
            "stored_lengths": {str(length): count for length, count in stored_lengths.items()},
            "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:MAX_REPORTED_MISMATCHES],
            "passed": bool(scan["accession_rows"]) and not mismatches}


def check_enhanced_stereo(scan: Mapping[str, Any],
                          bindingdb_raw: Mapping[str, Any],
                          config: Mapping[str, Any]) -> dict[str, Any]:
    """Regression: the '|r|' marker of the monomer survives verbatim in the source-SMILES column.

    Nhận: kết quả scan_outputs, raw BindingDB, config.
    Trả về: dict báo cáo cho monomer 13590.
    Vì sao cần: dấu lập thể tương đối "|r|" bị mất trong canonical SMILES; kiểm tra rằng
    nó vẫn còn nguyên trong cột SMILES nguồn và cột đó giống hệt raw.
    """
    marker = config["regression"]["enhanced_stereo_marker"]
    problems = []
    # Phải còn '|r|' và giống hệt raw.
    for observation in scan["monomer_rows"]:
        stored = observation["molecule_source_smiles"] or ""
        raw_row = bindingdb_raw["rows"].get(observation["native_measurement_id"]) or {}
        if marker not in stored:
            problems.append(f"{observation['observation_id']}: marker missing in {stored!r}")
        if stored != raw_row.get("ligand_smiles"):
            problems.append(f"{observation['observation_id']}: differs from raw "
                            f"{raw_row.get('ligand_smiles')!r}")
    return {"monomer_id": config["regression"]["bindingdb_monomer_enhanced_stereo"],
            "rows_checked": len(scan["monomer_rows"]),
            "problems": problems[:MAX_REPORTED_MISMATCHES],
            "passed": bool(scan["monomer_rows"]) and not problems}


def run_verify(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Integrity checks of paths.observations_dir against raw; returns the report.

    Nhận: config, thư mục gốc.
    Trả về: báo cáo (dict) gồm năm phép kiểm tra và các bảng đếm; "passed" = True khi cả
    năm phép đều đạt.
    Năm phép kiểm tra: bảo toàn số dòng + không trùng ID; đối chiếu mẫu ngẫu nhiên về raw;
    construct P03367; dấu "|r|" của monomer 13590; PMID dạng số nguyên.
    """
    # --- Cùng phiên bản gói chuẩn hóa với lúc extract (cột suy ra được tính lại).
    normalize.check_structure_pipeline_version(
        config["structure"]["chembl_structure_pipeline_version"])
    # --- Cần đủ hai file output của full.
    output_dir = root / config["paths"]["observations_dir"]
    paths = [output_dir / file_name for _, _, file_name in source_outputs(config)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run --mode full first; missing outputs: {missing}")

    # --- Một lượt đọc output.
    LOGGER.info("[verify] stage: scan outputs")
    scan = scan_outputs(paths, config)
    # --- Một lượt đọc raw BindingDB cho những id cần đối chiếu.
    # Chỉ giữ dòng raw của các id cần đối chiếu.
    wanted_bindingdb_ids = {row["native_measurement_id"]
                            for row in scan["sample"] + scan["accession_rows"]
                            + scan["monomer_rows"]
                            if row["source_database"] == bindingdb.SOURCE_DATABASE}
    LOGGER.info("[verify] stage: read raw BindingDB")
    bindingdb_raw = read_bindingdb_raw(config, root, wanted_bindingdb_ids)

    # --- Tra cứu ChEMBL và các phép đối chiếu.
    LOGGER.info("[verify] stage: ChEMBL lookups and comparisons")
    # COUNT(*) số activity Ki/Kd/IC50/EC50 trong raw.
    connection = chembl.open_chembl_readonly(root / config["paths"]["chembl_db"])
    try:
        chembl_selected = chembl.count_selected_activities(connection, config["endpoints"])
        sample = reconcile_sample(scan, bindingdb_raw, connection, config)
        accession = check_accession_constructs(scan, bindingdb_raw, connection, config)
        parent_agreement = check_chembl_parent_agreement(scan, connection)
    finally:
        connection.close()

    # --- Bảo toàn: số dòng chọn từ raw phải bằng số dòng đã ghi, ở cả hai nguồn, và
    # không có observation_id trùng.
    # Số dòng chọn từ raw phải bằng số dòng đã ghi.
    rows_by_source = scan["rows_by_source"]
    conservation = {
        chembl.SOURCE_DATABASE: {"selected_from_raw": chembl_selected,
                                 "written": rows_by_source[chembl.SOURCE_DATABASE]},
        bindingdb.SOURCE_DATABASE: {"selected_from_raw": bindingdb_raw["selected_cells"],
                                    "written": rows_by_source[bindingdb.SOURCE_DATABASE]},
        "duplicate_observation_ids": scan["duplicate_ids"],
    }
    conservation["passed"] = scan["duplicate_ids"] == 0 and all(
        conservation[source]["selected_from_raw"] == conservation[source]["written"]
        for source in (chembl.SOURCE_DATABASE, bindingdb.SOURCE_DATABASE))
    pmid = {"non_integer_pmids": len(scan["bad_pmids"]),
            "examples": scan["bad_pmids"][:MAX_REPORTED_MISMATCHES],
            "passed": not scan["bad_pmids"]}
    # --- Gộp báo cáo.
    report = {
        "conservation": conservation,
        "sample_reconciliation": sample,
        "regression_accession_construct": accession,
        "regression_enhanced_stereo": check_enhanced_stereo(scan, bindingdb_raw, config),
        "regression_pmid_integer_form": pmid,
        # Thông tin, không tính vào "passed" (xem check_chembl_parent_agreement).
        "chembl_parent_agreement": parent_agreement,
        "counts_by_source": dict(rows_by_source),
        "counts_by_source_endpoint": dict(scan["rows_by_source_endpoint"]),
        "counts_by_source_status_reason": dict(scan["status_reasons"]),
        "counts_by_source_structure_flag": dict(scan["structure_flags"]),
    }
    # Chỉ đạt khi cả năm phép đều đạt.
    report["passed"] = all(report[key]["passed"] for key in (
        "conservation", "sample_reconciliation", "regression_accession_construct",
        "regression_enhanced_stereo", "regression_pmid_integer_form"))
    return report
