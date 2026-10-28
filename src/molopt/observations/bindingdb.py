"""BindingDB reader: streams BindingDB_All from its ZIP and resolves protein and assay.

Assay mapping goes through two different key namespaces:
    BindingDB_All.Reactant_set_id
        -> BindingDB_rsid_eaids (REACTANT_SET_ID -> ENTRYID_ASSAYID)
        -> BindingDB_Assays on (ENTRYID, ASSAYID).
Reactant_set_id is never joined to ENTRYID directly.

This module uses the standard library only, so `inspect` runs without RDKit/pyarrow.
Values are passed on verbatim; unit/qualifier/structure conversion happens in normalize.py.

Vị trí trong luồng raw -> observations:
    raw BindingDB (3 file ZIP, đọc thẳng từ ZIP, KHÔNG giải nén ra đĩa)
    -> bindingdb.py (file này): nạp bảng nối assay, đọc tuần tự BindingDB_All, lấy
       protein từ cột chain của chính dòng, lấy assay qua rsid_eaids, và tách mỗi ô
       endpoint không rỗng thành một bản ghi riêng (giá trị nguồn nguyên văn)
    -> pipeline.py: thêm cột suy ra (normalize.py) và ghi Parquet.
"""

from __future__ import annotations

# Chỉ dùng thư viện chuẩn: đọc ZIP, đọc TSV.
import csv
import io
import json
import logging
import zipfile
from collections.abc import Container, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

LOGGER = logging.getLogger(__name__)

# Tên nguồn, dùng trong observation_id.
SOURCE_DATABASE = "bindingdb"
TSV_SUFFIX = ".tsv"
# Giá trị của cột protein_accession_db: accession lấy từ SwissProt hay TrEMBL.
SWISSPROT = "swissprot"
TREMBL = "trembl"
# Chuỗi lấy từ chain 1; chỉ nhận target có 1 chain.
SEQUENCE_ORIGIN_CHAIN_1 = "bindingdb_target_chain_1"
SINGLE_CHAIN = 1

# Ba giá trị của các cột *_status.
RESOLVED = "resolved"
UNRESOLVED = "unresolved"
INVALID = "invalid"

# Khóa assay, vd. ('1025', '1').
AssayPair = tuple[str, str]  # (ENTRYID, ASSAYID)


class ProteinResolution(NamedTuple):
    """Kết quả chọn chuỗi protein cho một dòng BindingDB (xem resolve_protein)."""
    sequence: str | None
    accession: str | None
    accession_db: str | None
    status: str
    reason: str | None


class AssayResolution(NamedTuple):
    """Kết quả tìm assay cho một Reactant_set_id (xem resolve_assay)."""
    entry_id: str | None
    assay_id: str | None
    name: str | None
    description: str | None
    status: str
    reason: str | None


# --------------------------------------------------------------------------- ZIP / TSV access


def find_tsv_member(zip_path: Path) -> str:
    """Return the single .tsv member of a ZIP; fail if there is not exactly one.

    Nhận: đường dẫn file ZIP. Trả về: tên file .tsv bên trong ZIP.
    Vì sao cần: không viết cứng tên file bên trong ZIP; nếu ZIP chứa 0 hoặc nhiều file
    .tsv thì dừng thay vì đoán đọc file nào.
    """
    # Tìm file .tsv bên trong ZIP.
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(TSV_SUFFIX)]
    # Phải có đúng một file .tsv.
    if len(members) != 1:
        raise ValueError(f"{zip_path}: expected exactly one {TSV_SUFFIX} member, found {members}")
    return members[0]


def iter_tsv_rows(zip_path: Path, encoding: str) -> Iterator[tuple[int, list[str]]]:
    """Stream (line_number, fields) from the ZIP's TSV member without extracting it.

    Line 1 is the header. Decoding errors are re-raised with the line number.

    Nhận: đường dẫn ZIP và bảng mã (utf-8).
    Trả về: iterator (số dòng, danh sách trường) cho từng dòng, kể cả header.
    Vì sao cần: đọc trực tiếp từ ZIP giữ raw nguyên vẹn và không tạo bản giải nén nhiều
    GB; đọc tuần tự từng dòng nên bộ nhớ không phụ thuộc kích thước file.
    """
    member = find_tsv_member(zip_path)
    # archive.open trả về luồng giải nén; TextIOWrapper giải mã byte -> chuỗi.
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as raw_stream:
        text_stream = io.TextIOWrapper(raw_stream, encoding=encoding, newline="")
        # QUOTE_NONE: file TSV này không bọc trường bằng dấu ngoặc kép; dấu " trong SMILES
        # hay tên chất phải được giữ là ký tự thường, không được hiểu là ký hiệu bọc trường.
        reader = csv.reader(text_stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        line_number = 0
        try:
            # Trả từng dòng một, không giữ cả file trong RAM.
            for line_number, fields in enumerate(reader, start=1):
                yield line_number, fields
        except UnicodeDecodeError as error:
            # Báo lỗi kèm số dòng để có thể tìm lại đúng chỗ trong raw.
            raise ValueError(f"{zip_path}::{member} line {line_number + 1}: {error}") from error


def read_header(zip_path: Path, encoding: str) -> list[str]:
    """Read only the header line of the ZIP's TSV member.

    Nhận: đường dẫn ZIP và bảng mã. Trả về: danh sách tên cột (dòng 1).
    """
    # Chỉ đọc dòng header.
    for _, fields in iter_tsv_rows(zip_path, encoding):
        return fields
    raise ValueError(f"{zip_path}: TSV member is empty")


def column_indexes(header: list[str], wanted: Mapping[str, str], source: str) -> dict[str, int]:
    """Map role -> column index; fail naming every missing or duplicated column.

    Nhận: header, dict {vai trò: tên cột thật} (lấy từ config.json), tên nguồn để báo lỗi.
    Trả về: dict {vai trò: chỉ số cột}.
    Vì sao cần: tìm cột theo TÊN thay vì theo vị trí cố định, để một thay đổi thứ tự cột
    ở bản BindingDB khác không làm đọc nhầm cột. Tên cột phải xuất hiện đúng một lần.
    """
    problems = []
    indexes = {}
    for role, name in wanted.items():
        # Tìm cột theo tên, khớp chính xác.
        positions = [index for index, column in enumerate(header) if column == name]
        # Tên cột phải xuất hiện đúng một lần.
        if len(positions) != 1:
            problems.append(f"{role}={name!r} found {len(positions)} times")
        else:
            indexes[role] = positions[0]
    # Gom mọi vấn đề rồi báo một lần, để người dùng thấy hết các cột lỗi.
    if problems:
        raise ValueError(f"{source}: required columns not usable: {'; '.join(problems)}")
    return indexes


def field(fields: list[str], index: int) -> str | None:
    """Value at a column index, or None when empty or absent from a short row.

    Nhận: danh sách trường của một dòng và chỉ số cột.
    Trả về: chuỗi nguyên văn (không cắt khoảng trắng), hoặc None nếu ô rỗng/không tồn tại.
    """
    # Dòng ngắn hơn header -> ô không tồn tại.
    if index >= len(fields):
        return None
    # Giữ nguyên văn (' 2' vẫn là ' 2'); ô chỉ có khoảng trắng -> None.
    value = fields[index]
    return value if value.strip() else None


def check_row_width(fields: list[str], header_width: int, line_number: int, source: str) -> None:
    """A row wider than its header means the TSV was split wrongly: stop loudly.

    Nhận: trường của một dòng, số cột của header, số dòng, tên file.
    Trả về: không có; ném ValueError nếu dòng có nhiều trường hơn header.
    Vì sao cần: nếu một ô chứa ký tự tab thừa, mọi cột phía sau bị lệch; khi đó giá trị
    sẽ bị gán nhầm cột mà không ai biết. Dừng hẳn an toàn hơn là đoán.
    """
    # Chỉ bắt dòng dài hơn header; dòng ngắn hơn không bị bắt.
    if len(fields) > header_width:
        raise ValueError(f"{source} line {line_number}: {len(fields)} fields but header has "
                         f"{header_width}; refusing to guess column alignment")


# --------------------------------------------------------------------------- assay tables


def parse_entry_assay_id(text: str, separator: str, line_number: int) -> AssayPair:
    """Split ENTRYID_ASSAYID into (ENTRYID, ASSAYID); fail loudly on any other shape.

    Nhận: chuỗi ENTRYID_ASSAYID (ví dụ "1025_1"), ký tự phân tách ("_"), số dòng.
    Trả về: cặp (ENTRYID, ASSAYID) dạng chuỗi, ví dụ ("1025", "1").
    Vì sao chặt chẽ như vậy: ký tự phân tách được cấu hình chứ không được tài liệu hóa
    trong file; nếu giả định sai, mọi khóa assay đều sai. Chỉ chấp nhận đúng hai phần
    đều là số nguyên, mọi dạng khác làm chương trình dừng kèm số dòng.
    """
    # '1025_1' -> ['1025', '1'].
    parts = text.strip().split(separator)
    # Phải đúng 2 phần, đều là số; khác thì dừng.
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"BindingDB_rsid_eaids line {line_number}: ENTRYID_ASSAYID {text!r} "
                         f"does not split into two integers on {separator!r}")
    # Giữ dạng chuỗi để khớp với khóa của bảng Assays.
    return parts[0], parts[1]


def load_assay_links(zip_path: Path, settings: Mapping[str, Any]) -> dict[str, list[AssayPair]]:
    """Reactant_set_id -> distinct (ENTRYID, ASSAYID) pairs from BindingDB_rsid_eaids.

    Nhận: đường dẫn ZIP rsid_eaids và phần cấu hình bindingdb.
    Trả về: dict {Reactant_set_id: [các cặp (ENTRYID, ASSAYID) khác nhau]}.
    Vì sao cần: BindingDB_All không có cột assay. Bảng rsid_eaids là cầu nối DUY NHẤT từ
    Reactant_set_id (hệ đánh số phép đo) sang ENTRYID/ASSAYID (hệ đánh số entry/assay).
    Giữ danh sách cặp (không chỉ một cặp) để phát hiện Reactant_set_id liên kết với
    nhiều assay thay vì lặng lẽ ghi đè.
    """
    rows = iter_tsv_rows(zip_path, settings["encoding"])
    _, header = next(rows)
    columns = column_indexes(header, settings["rsid_eaids_columns"], zip_path.name)
    # Vd. '13888' -> [('1025', '1')].
    links: dict[str, list[AssayPair]] = {}
    for line_number, fields in rows:
        check_row_width(fields, len(header), line_number, zip_path.name)
        reactant_set_id = field(fields, columns["reactant_set_id"])
        entry_assay_text = field(fields, columns["entry_assay_id"])
        # Bảng nối mà thiếu khóa thì dữ liệu hỏng: dừng thay vì bỏ qua.
        if reactant_set_id is None or entry_assay_text is None:
            raise ValueError(f"{zip_path.name} line {line_number}: empty key field {fields!r}")
        pair = parse_entry_assay_id(entry_assay_text, settings["entry_assay_separator"],
                                    line_number)
        # Chỉ thêm cặp chưa có, để các dòng trùng lặp hoàn toàn không bị đếm thành "nhiều assay".
        pairs = links.setdefault(reactant_set_id.strip(), [])
        if pair not in pairs:
            pairs.append(pair)
    LOGGER.info("BindingDB assay links loaded: %s reactant sets", f"{len(links):,}")
    return links


def load_assays(
    zip_path: Path, settings: Mapping[str, Any]
) -> dict[AssayPair, tuple[str | None, str | None] | None]:
    """(ENTRYID, ASSAYID) -> (assay name, description); None marks duplicated keys.

    Nhận: đường dẫn ZIP Assays và phần cấu hình bindingdb.
    Trả về: dict {(ENTRYID, ASSAYID): (tên assay, mô tả)}; giá trị None nghĩa là khóa đó
    xuất hiện nhiều lần.
    Vì sao khóa là CẶP: ASSAYID được đánh số lại trong từng entry (ASSAYID 1 có ở hầu hết
    các entry), nên ASSAYID một mình không xác định được assay.
    """
    rows = iter_tsv_rows(zip_path, settings["encoding"])
    _, header = next(rows)
    columns = column_indexes(header, settings["assays_columns"], zip_path.name)
    # None = khóa bị trùng.
    assays: dict[AssayPair, tuple[str | None, str | None] | None] = {}
    for line_number, fields in rows:
        check_row_width(fields, len(header), line_number, zip_path.name)
        entry_id = field(fields, columns["entry_id"])
        assay_id = field(fields, columns["assay_id"])
        if entry_id is None or assay_id is None:
            raise ValueError(f"{zip_path.name} line {line_number}: empty key field {fields!r}")
        # Khóa là cặp, vì ASSAYID được đánh số lại trong mỗi entry.
        key = (entry_id.strip(), assay_id.strip())
        value = (field(fields, columns["assay_name"]), field(fields, columns["description"]))
        # Khóa trùng -> đánh dấu None để resolve_assay báo duplicate_assay_table_rows,
        # thay vì lấy bừa dòng đầu hoặc dòng cuối.
        assays[key] = None if key in assays else value
    LOGGER.info("BindingDB assay table loaded: %s assays", f"{len(assays):,}")
    return assays


def assert_domains_intersect(label: str, left: Iterable[Any], right: Container[Any]) -> int:
    """Count left keys found in right; fail loudly if the key domains do not intersect.

    Nhận: nhãn để ghi log, tập khóa bên trái, tập khóa bên phải.
    Trả về: số khóa bên trái tìm thấy ở bên phải (ghi vào log).
    Vì sao cần: đây là chốt chặn chống join sai hệ khóa. Nếu tách ENTRYID_ASSAYID sai
    ký tự, hoặc vô tình join hai cột thuộc hai hệ đánh số khác nhau, thì gần như không
    khóa nào khớp; khi đó mọi dòng sẽ thành "unresolved" mà chương trình vẫn chạy hết.
    Kiểm tra này biến lỗi âm thầm đó thành lỗi dừng ngay. Lưu ý: chỉ dừng khi KHÔNG có
    khóa nào khớp; đây là kiểm tra cấu trúc, không phải đo độ phủ.
    """
    # LƯU Ý: kiểm tra này không bắt được lỗi join Reactant_set_id vào ENTRYID (hai dãy số
    # chồng nhau, 12.665/20.000 vẫn khớp). Lỗi đó do resolve_assay, test và verify chặn.
    checked = matched = 0
    for key in left:
        checked += 1
        # True/False được cộng như 1/0.
        matched += key in right
    # Không khóa nào khớp -> dừng ngay.
    if matched == 0:
        raise ValueError(f"Key domains do not intersect for {label}: 0 of {checked:,} keys "
                         "matched. The join keys are in different namespaces or misparsed.")
    LOGGER.info("Key check %s: %s of %s keys matched", label, f"{matched:,}", f"{checked:,}")
    return matched


def first_reactant_set_ids(zip_path: Path, settings: Mapping[str, Any]) -> list[str]:
    """Reactant_set_ids of the first rows of BindingDB_All, for the key-domain check.

    Nhận: đường dẫn ZIP BindingDB_All và cấu hình. Trả về: Reactant_set_id của tối đa
    key_check_rows (20.000) dòng đầu.
    Vì sao chỉ lấy mẫu đầu file: đủ để phát hiện join sai hệ khóa mà không phải đọc cả
    file All thêm một lượt trước khi xử lý.
    """
    rows = iter_tsv_rows(zip_path, settings["encoding"])
    _, header = next(rows)
    index = column_indexes(header, settings["all_columns"], zip_path.name)["reactant_set_id"]
    # Chỉ lấy 20.000 dòng đầu.
    ids = []
    for _, fields in rows:
        value = field(fields, index)
        if value is not None:
            ids.append(value.strip())
        if len(ids) >= settings["key_check_rows"]:
            break
    return ids


# --------------------------------------------------------------------------- resolution (pure)


def resolve_protein(
    chain_count_text: str | None,
    chain_sequence: str | None,
    swissprot_id: str | None,
    trembl_id: str | None,
) -> ProteinResolution:
    """Protein of one measurement from its own target-chain columns; never looked up elsewhere.

    Nhận: số chain của target, chuỗi chain 1, SwissProt Primary ID và TrEMBL Primary ID
    của chain 1 — tất cả lấy từ CHÍNH dòng BindingDB_All đang đọc.
    Trả về: ProteinResolution (chuỗi, accession, SwissProt/TrEMBL, status, lý do).

    Vì sao lấy protein theo từng phép đo và không tra theo accession: cùng một accession
    UniProt có thể ứng với nhiều construct khác nhau (ví dụ P03367: domain protease 99 aa
    ở một số phép đo, polyprotein 1.447 aa ở phép đo khác). Tra chuỗi theo accession sẽ
    gán nhầm construct. Chuỗi trên dòng là chuỗi thực sự dùng trong phép đo đó.
    """
    # --- Số chain phải có và phải là số nguyên.
    # Thiếu số chain -> unresolved.
    if chain_count_text is None:
        return ProteinResolution(None, None, None, UNRESOLVED, "chain_count_missing")
    # Số chain không phải số -> invalid.
    try:
        chain_count = int(chain_count_text)
    except ValueError:
        return ProteinResolution(None, None, None, INVALID,
                                 f"chain_count_not_integer:{chain_count_text}")
    # --- Chỉ xử lý target một chain: với phức hợp nhiều chain, không biết ligand gắn vào
    # chain nào, nên không chọn bừa chain 1.
    # Nhiều chain -> unresolved, không chọn bừa chain 1.
    if chain_count != SINGLE_CHAIN:
        return ProteinResolution(None, None, None, UNRESOLVED,
                                 f"target_chain_count_not_1:{chain_count}")
    # Có 1 chain nhưng thiếu chuỗi.
    if chain_sequence is None:
        return ProteinResolution(None, None, None, UNRESOLVED, "chain_sequence_missing")
    # --- Accession: ưu tiên SwissProt (đã được kiểm duyệt), sau đó TrEMBL.
    if swissprot_id is not None:
        return ProteinResolution(chain_sequence, swissprot_id.strip(), SWISSPROT, RESOLVED, None)
    if trembl_id is not None:
        return ProteinResolution(chain_sequence, trembl_id.strip(), TREMBL, RESOLVED, None)
    # --- Có chuỗi nhưng không có accession: protein vẫn xác định được (resolved) vì chuỗi
    # mới là thông tin chính; lý do ghi lại để bước chia dữ liệu theo accession lưu ý.
    return ProteinResolution(chain_sequence, None, None, RESOLVED, "no_uniprot_accession")


def resolve_assay(
    reactant_set_id: str,
    links: Mapping[str, list[AssayPair]],
    assays: Mapping[AssayPair, tuple[str | None, str | None] | None],
) -> AssayResolution:
    """Assay of one measurement via Reactant_set_id -> (ENTRYID, ASSAYID) -> assay row.

    Nhận: Reactant_set_id của dòng, bảng nối (từ load_assay_links), bảng assay (từ
    load_assays).
    Trả về: AssayResolution (ENTRYID, ASSAYID, tên, mô tả, status, lý do).

    Vì sao đi qua rsid_eaids và KHÔNG join thẳng Reactant_set_id vào ENTRYID: đó là hai hệ
    đánh số khác nhau (Reactant_set_id đánh số phép đo, hàng triệu giá trị; ENTRYID đánh
    số entry = bài báo/bằng sáng chế/đợt nộp dữ liệu). Cả hai đều là số nguyên nên join
    thẳng vẫn "chạy" và ra kết quả, nhưng ghép phép đo với assay của entry không liên
    quan, không có báo lỗi nào. Ví dụ: Reactant_set_id 13888 thuộc ENTRYID 1025.

    Cách quyết định status: chỉ resolved khi có ĐÚNG MỘT cặp liên kết VÀ cặp đó có ĐÚNG
    MỘT dòng trong bảng assay. Mọi trường hợp mơ hồ đều unresolved với lý do riêng,
    không chọn bừa.
    """
    # --- Reactant_set_id không có trong bảng nối.
    pairs = links.get(reactant_set_id)
    if not pairs:
        return AssayResolution(None, None, None, None, UNRESOLVED, "no_rsid_assay_link")
    # --- Liên kết với nhiều assay: không biết phép đo thuộc assay nào.
    if len(pairs) > 1:
        return AssayResolution(None, None, None, None, UNRESOLVED,
                               f"multiple_assay_links:{len(pairs)}")
    # --- Đúng một cặp: tra bảng assay theo cả cặp (ENTRYID, ASSAYID).
    entry_id, assay_id = pairs[0]
    # Xét 'not in' trước, vì None trong assays nghĩa là khóa trùng.
    if (entry_id, assay_id) not in assays:
        return AssayResolution(entry_id, assay_id, None, None, UNRESOLVED,
                               "assay_pair_not_in_assay_table")
    assay_row = assays[(entry_id, assay_id)]
    # None = khóa xuất hiện nhiều lần trong bảng assay (xem load_assays).
    if assay_row is None:
        return AssayResolution(entry_id, assay_id, None, None, UNRESOLVED,
                               "duplicate_assay_table_rows")
    name, description = assay_row
    return AssayResolution(entry_id, assay_id, name, description, RESOLVED, None)


# --------------------------------------------------------------------------- records


def build_records(
    fields: list[str],
    columns: Mapping[str, int],
    endpoint_columns: Mapping[str, int],
    endpoint_units: Mapping[str, str],
    assay: AssayResolution,
    release: str,
) -> list[dict[str, Any]]:
    """One record per non-empty endpoint cell of a BindingDB_All row (values verbatim).

    Nhận: trường của một dòng All, chỉ số các cột cần dùng, chỉ số bốn cột endpoint, đơn
    vị của từng endpoint ("nM", lấy từ tên cột), kết quả assay của dòng, phiên bản.
    Trả về: danh sách bản ghi — 0 đến 4 bản ghi, mỗi ô endpoint không rỗng một bản ghi.
    Vì sao cần: một dòng BindingDB có thể chứa đồng thời Ki, IC50, Kd, EC50. Mỗi giá trị
    là một phép đo riêng, nên phải tách thành observation riêng (observation_id khác nhau
    nhờ phần endpoint).
    """

    # Đọc ô theo tên vai trò trong config.
    def get(role: str) -> str | None:
        # Lấy giá trị theo vai trò (tên trong config.json), None nếu ô rỗng.
        return field(fields, columns[role])

    # --- Phần chung của dòng: tính một lần, dùng cho mọi endpoint của dòng.
    protein = resolve_protein(get("chain_count"), get("chain_1_sequence"),
                              get("chain_1_swissprot_id"), get("chain_1_trembl_id"))
    # Ngữ cảnh assay: giữ nguyên dưới dạng JSON, không diễn giải.
    assay_context = {"ph": get("ph"), "temperature_c": get("temperature_c"),
                     "pubchem_aid": get("pubchem_aid"), "entry_doi": get("entry_doi")}
    # Tên cột thống nhất với ChEMBL; cột BindingDB không có (target_type, variant...) để None.
    # BindingDB không có cột qualifier riêng: qualifier nằm trong ô giá trị và được tách ở
    # pipeline.derive_activity.
    shared = {
        "source_database": SOURCE_DATABASE,
        "source_release": release,
        # Reactant_set_id = ID phép đo.
        "native_measurement_id": (get("reactant_set_id") or "").strip(),
        "source_curation": get("curation_source"),
        # ENTRYID; None nếu dòng không có liên kết assay.
        "document_id": assay.entry_id,
        "document_pmid_source": get("pmid"),
        "document_doi": get("article_doi"),
        "document_patent_number": get("patent_number"),
        "molecule_source_id": get("monomer_id"),
        "molecule_source_smiles": get("ligand_smiles"),
        "molecule_source_inchikey": get("ligand_inchikey"),
        "molecule_source_chirality": None,
        # Mã ChEMBL mà BindingDB tự ghi cho phân tử (nguyên văn); loại phân tử BindingDB
        # không có -> None (pipeline tra ChEMBL theo mã này).
        "molecule_source_chembl_id": get("chembl_id"),
        "molecule_source_type": None,
        "protein_sequence": protein.sequence,
        "protein_sequence_origin": SEQUENCE_ORIGIN_CHAIN_1 if protein.sequence else None,
        "protein_accession": protein.accession,
        "protein_accession_db": protein.accession_db,
        # Cột BindingDB không có -> None.
        "protein_target_id": None,
        "protein_target_name": get("target_name"),
        "protein_target_type": None,
        "protein_organism": get("target_organism"),
        "protein_variant_id": None,
        "protein_mutation": None,
        "protein_isoform": None,
        "protein_chain_count": get("chain_count"),
        "protein_pdb_ids": get("chain_1_pdb_ids"),
        "protein_status": protein.status,
        "protein_reason": protein.reason,
        "assay_native_id": assay.assay_id,
        "assay_entry_id": assay.entry_id,
        "assay_name": assay.name,
        "assay_description": assay.description,
        "assay_type": None,
        "assay_context": json.dumps(assay_context, ensure_ascii=False),
        "assay_status": assay.status,
        "assay_reason": assay.reason,
        "activity_qualifier_source": None,
        "activity_context": None,
    }
    # --- Tách theo endpoint: mỗi ô không rỗng -> một bản ghi; ô rỗng bỏ qua.
    # Vd. Ki ' 2' và IC50 '<10' trên cùng dòng -> 2 bản ghi.
    records = []
    for endpoint, index in endpoint_columns.items():
        value_text = field(fields, index)
        if value_text is None:
            continue
        # Phần chung + endpoint, giá trị nguyên văn, đơn vị nM.
        records.append({**shared, "endpoint": endpoint, "activity_value_source": value_text,
                        "activity_unit_source": endpoint_units[endpoint]})
    return records


def iter_records(config: Mapping[str, Any], root: Path, limit: int | None) -> Iterator[dict]:
    """Stream BindingDB observation records; `limit` caps the number of records (smoke).

    Nhận: config, thư mục gốc dự án, giới hạn số bản ghi (None = toàn bộ).
    Trả về: iterator các bản ghi observation của BindingDB.
    Vì sao cần: là "nguồn bản ghi" BindingDB mà pipeline.py tiêu thụ, cùng giao diện với
    chembl.iter_records.
    """
    settings = config["bindingdb"]
    raw_dir = root / config["paths"]["raw_bindingdb_dir"]
    all_zip = raw_dir / settings["all_zip"]
    # Nâng giới hạn độ dài một trường của module csv: ô chuỗi protein hay danh sách PDB
    # có thể dài hơn giới hạn mặc định 131.072 ký tự.
    csv.field_size_limit(settings["csv_field_size_limit"])

    # --- Nạp hai bảng nhỏ dùng để tra assay vào bộ nhớ.
    links = load_assay_links(raw_dir / settings["rsid_eaids_zip"], settings)
    assays = load_assays(raw_dir / settings["assays_zip"], settings)
    # --- Kiểm tra giao khóa: (1) cặp trong rsid_eaids phải khớp khóa bảng Assays;
    # (2) Reactant_set_id của All phải khớp khóa rsid_eaids. Không khớp khóa nào -> dừng.
    assert_domains_intersect("rsid_eaids(ENTRYID,ASSAYID) -> Assays(ENTRYID,ASSAYID)",
                             {pair for pairs in links.values() for pair in pairs}, assays)
    assert_domains_intersect("All.Reactant_set_id -> rsid_eaids.REACTANT_SET_ID",
                             first_reactant_set_ids(all_zip, settings), links)

    # --- Mở BindingDB_All, tìm cột theo tên; chỉ giữ các endpoint có trong config.
    rows = iter_tsv_rows(all_zip, settings["encoding"])
    _, header = next(rows)
    columns = column_indexes(header, settings["all_columns"], all_zip.name)
    # Chỉ các endpoint có trong config.
    endpoint_names = {endpoint: spec["column"]
                      for endpoint, spec in settings["endpoint_columns"].items()
                      if endpoint in config["endpoints"]}
    endpoint_columns = column_indexes(header, endpoint_names, all_zip.name)
    endpoint_units = {endpoint: spec["unit"]
                      for endpoint, spec in settings["endpoint_columns"].items()}
    release = config["releases"]["bindingdb"]

    # --- Đọc tuần tự từng dòng: kiểm tra độ rộng, tìm assay, tách thành bản ghi.
    # Đếm bản ghi (không phải dòng) cho limit.
    yielded = 0
    for line_number, fields in rows:
        check_row_width(fields, len(header), line_number, all_zip.name)
        reactant_set_id = field(fields, columns["reactant_set_id"])
        # Reactant_set_id là ID phép đo (dùng trong observation_id): thiếu thì dừng.
        if reactant_set_id is None:
            raise ValueError(f"{all_zip.name} line {line_number}: empty Reactant_set_id")
        # Assay tính một lần cho cả dòng.
        assay = resolve_assay(reactant_set_id.strip(), links, assays)
        for record in build_records(fields, columns, endpoint_columns, endpoint_units,
                                    assay, release):
            yield record
            yielded += 1
            # Smoke: dừng khi đủ số bản ghi yêu cầu.
            if limit is not None and yielded >= limit:
                return


# --------------------------------------------------------------------------- inspect


def describe(config: Mapping[str, Any], root: Path) -> list[str]:
    """Inspect report: files, ZIP members, TSV headers and required-column check (no data rows).

    Nhận: config và thư mục gốc. Trả về: danh sách dòng văn bản để in.
    Vì sao cần: chế độ inspect dùng hàm này để xác nhận file, thành viên trong ZIP, header
    và vị trí từng cột bắt buộc trước khi chạy thật. Chỉ đọc dòng header.
    """
    # --- Danh sách file trong raw/bindingdb202610 (kể cả file SDF không được đọc).
    settings = config["bindingdb"]
    raw_dir = root / config["paths"]["raw_bindingdb_dir"]
    lines = [f"[bindingdb] raw dir: {raw_dir}"]
    lines += [f"  file: {path.name} ({path.stat().st_size:,} bytes)"
              for path in sorted(raw_dir.iterdir())]
    lines.append(f"  NOT READ in this stage (SDF): {settings['sdf_zip_not_read']}")

    # --- Với từng ZIP: in thành viên, header, và trạng thái từng cột bắt buộc
    # (tìm thấy / thiếu / xuất hiện nhiều lần).
    required = {
        "all_zip": {**settings["all_columns"],
                    **{endpoint: spec["column"]
                       for endpoint, spec in settings["endpoint_columns"].items()}},
        "rsid_eaids_zip": settings["rsid_eaids_columns"],
        "assays_zip": settings["assays_columns"],
    }
    for zip_key, wanted in required.items():
        zip_path = raw_dir / settings[zip_key]
        with zipfile.ZipFile(zip_path) as archive:
            lines += [f"  {zip_path.name} :: {info.filename} "
                      f"(uncompressed {info.file_size:,} bytes)" for info in archive.infolist()]
        header = read_header(zip_path, settings["encoding"])
        lines.append(f"  header ({len(header)} columns):")
        lines += [f"    [{index}] {name}" for index, name in enumerate(header)]
        for role, name in wanted.items():
            count = header.count(name)
            state = f"found at [{header.index(name)}]" if count == 1 else (
                "MISSING" if count == 0 else f"AMBIGUOUS ({count} columns)")
            lines.append(f"  required {role} = {name!r}: {state}")

    # --- Mô tả đường lấy từng trường quan trọng.
    lines += [
        "  per-measurement protein sequence: 'BindingDB Target Chain Sequence 1' "
        "(+ chains 2..N for multichain targets; only single-chain rows are resolved)",
        "  native assay ID: All.'BindingDB Reactant_set_id' -> rsid_eaids.REACTANT_SET_ID / "
        "ENTRYID_ASSAYID -> Assays.(ENTRYID, ASSAYID)",
        f"  TODO(confirm): {settings['entry_assay_separator_TODO']} "
        f"Configured separator: {settings['entry_assay_separator']!r}",
        "  variant info: NO dedicated variant/mutation column in BindingDB_All; only free-text "
        "'Target Name' and the chain sequence itself. Not parsed, stored verbatim.",
        "  CXSMILES: 'Ligand SMILES' (stored verbatim; CX extensions such as |r| cannot be "
        "confirmed from the header alone)",
    ]
    return lines
