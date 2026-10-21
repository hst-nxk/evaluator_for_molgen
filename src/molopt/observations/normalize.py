"""Shared, pure normalization functions used by both sources.

Every function takes plain values and returns plain values, so it can be tested
without raw data. Nothing here overwrites a source value: callers store the
results in separate derived columns.

Vị trí trong luồng raw -> observations:
    chembl.py / bindingdb.py  đọc raw, trả về bản ghi với giá trị nguồn nguyên văn
    -> normalize.py (file này) tính các giá trị suy ra: qualifier, nồng độ mol/L,
       pActivity, canonical SMILES, InChIKey, thông tin lập thể, observation_id,
       PMID dạng số nguyên, hash chuỗi protein
    -> pipeline.py ghép giá trị nguồn + giá trị suy ra thành một dòng observation.

Mọi hàm ở đây là hàm thuần (không đọc file, không có trạng thái), dùng chung cho cả
hai nguồn, nên ChEMBL và BindingDB được chuẩn hóa theo đúng một quy tắc và có thể
kiểm thử bằng unit test mà không cần dữ liệu raw.
"""

from __future__ import annotations

# Hash, toán, regex; RDKit để xử lý phân tử.
import hashlib
import math
import re
from collections.abc import Collection, Mapping
from typing import NamedTuple

from rdkit import Chem, RDLogger

# RDKit failures are recorded per row in structure_status/structure_reason,
# so its console warnings would only add noise to multi-million-row runs.
RDLogger.DisableLog("rdApp.*")

# Ba giá trị dùng cho mọi cột *_status:
#   resolved   = đã xác định được giá trị;
#   unresolved = nguồn thiếu/không đủ thông tin để quyết định (không sai, chỉ chưa biết);
#   invalid    = nguồn có giá trị nhưng sai dạng (không phải số, <= 0, SMILES hỏng...).
RESOLVED = "resolved"
UNRESOLVED = "unresolved"
INVALID = "invalid"

# Phần mở rộng CXSMILES bắt đầu sau " |" (ví dụ "C[C@H](N)C(=O)O |r|").
CX_BLOCK_START = "|"
CXSMILES_SEPARATOR = " " + CX_BLOCK_START
# Dạng chuẩn của InChIKey: 14 chữ - 10 chữ - 1 chữ.
INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
# Dùng để chuẩn hóa PMID: "12345" hoặc "12345.0" (PMID bị lưu như số thực).
INTEGER_PATTERN = re.compile(r"^\d+$")
INTEGER_WITH_ZERO_FRACTION_PATTERN = re.compile(r"^(\d+)\.0+$")
# RDKit trả nhãn "?" cho tâm lập thể chưa xác định cấu hình.
UNASSIGNED_CENTER_LABEL = "?"
# Nhóm lập thể tuyệt đối; khác loại này là tương đối/racemic.
ABSOLUTE_STEREO_GROUP = Chem.StereoGroupType.STEREO_ABSOLUTE
# RDKit raises these for molecules it cannot process (e.g. CIP digraph > 100000 nodes).
RDKIT_ERRORS = (RuntimeError, ValueError)
# Lý do lỗi tối đa 80 ký tự; nhiều lý do nối bằng ';'.
MAX_ERROR_DETAIL_CHARS = 80
REASON_SEPARATOR = ";"


class ActivityResult(NamedTuple):
    """Kết quả chuẩn hóa một giá trị hoạt tính (xem parse_activity)."""
    qualifier: str | None
    concentration_molar: float | None
    p_activity: float | None
    censored: bool | None
    status: str
    reason: str | None


class StructureResult(NamedTuple):
    """Kết quả xử lý cấu trúc một phân tử bằng RDKit (xem describe_structure)."""
    canonical_smiles: str | None
    inchikey: str | None
    cx_extension: str | None
    stereo_groups: str | None
    stereo_centers: int | None
    unspecified_stereo_centers: int | None
    status: str
    reason: str | None


def make_observation_id(source_database: str, native_measurement_id: str, endpoint: str) -> str:
    """Stable ID from source + native record ID + endpoint (never from protein data).

    Nhận: tên nguồn ("chembl"/"bindingdb"), ID phép đo trong nguồn (activity_id của
    ChEMBL, Reactant_set_id của BindingDB) và endpoint (Ki/Kd/IC50/EC50).
    Trả về: chuỗi "<nguồn>:<id>:<endpoint>", ví dụ "bindingdb:13888:Ki".
    Vì sao cần: mỗi observation cần một ID ổn định để các bước sau tham chiếu.
    - Endpoint phải có mặt vì một dòng BindingDB có thể có nhiều cột endpoint và sinh
      ra nhiều observation cùng Reactant_set_id.
    - KHÔNG hash chuỗi protein vào ID: chuỗi protein là kết quả suy luận của pipeline
      (có thể thay đổi khi sửa cách lấy variant, hoặc một dòng chuyển từ unresolved sang
      resolved). Nếu ID phụ thuộc vào nó, ID của cùng một phép đo sẽ đổi giữa các lần
      chạy và mọi tham chiếu phía sau bị gãy. Chuỗi protein được hash riêng vào cột
      protein_sequence_hash.
    """
    # Vd. 'bindingdb:13888:Ki'.
    return f"{source_database}:{native_measurement_id}:{endpoint}"


def normalize_integer_id(value: object) -> str | None:
    """Integer-like identifier as a string ('12345.0' -> '12345'); None if not integer-like.

    Nhận: một giá trị ID bất kỳ (chuỗi, int, float, None).
    Trả về: chuỗi số nguyên thuần, hoặc None nếu giá trị không phải số nguyên.
    Vì sao cần: PMID trong nguồn đôi khi bị lưu như số thực ("12345.0"); cột
    document_pmid cần dạng thống nhất để so khớp tài liệu giữa hai nguồn. Giá trị gốc
    vẫn được giữ trong document_pmid_source.
    """
    # bool là lớp con của int trong Python: loại ra trước để True không thành "1".
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    # Số thực chỉ chấp nhận khi hữu hạn và không có phần lẻ (12345.0, không phải 12345.5).
    if isinstance(value, float):
        return str(int(value)) if math.isfinite(value) and value.is_integer() else None
    # Chuỗi: chấp nhận "12345" hoặc "12345.000"; mọi dạng khác trả None, không đoán.
    text = str(value).strip()
    # Nhận '12345' hoặc '12345.000'; dạng khác -> None.
    if INTEGER_PATTERN.match(text):
        return text
    zero_fraction = INTEGER_WITH_ZERO_FRACTION_PATTERN.match(text)
    return zero_fraction.group(1) if zero_fraction else None


def sequence_hash(sequence: str | None) -> str | None:
    """SHA-256 of the sequence with whitespace removed and letters upper-cased.

    Nhận: chuỗi protein (có thể None).
    Trả về: mã hex SHA-256, hoặc None nếu không có chuỗi.
    Vì sao cần: so sánh/gom các phép đo theo đúng construct protein (cùng accession
    nhưng khác chuỗi là khác construct) mà không phải so chuỗi dài hàng nghìn ký tự.
    Bỏ khoảng trắng và viết hoa để cùng một chuỗi viết khác kiểu vẫn cho cùng hash.
    """
    if not sequence:
        return None
    # Bỏ khoảng trắng, viết hoa trước khi hash.
    cleaned = "".join(sequence.split()).upper()
    # Lưu ý: ký tự ngoài ASCII sẽ gây lỗi.
    return hashlib.sha256(cleaned.encode("ascii")).hexdigest() if cleaned else None


def split_embedded_qualifier(
    value_text: str, known_qualifiers: Collection[str], implicit_qualifier: str
) -> tuple[str, str]:
    """Split a value like '>10000' into ('>', '10000'); no prefix means the implicit qualifier.

    Nhận: nội dung ô giá trị (ví dụ ">10000", " 2"), danh sách qualifier đã biết, và
    qualifier mặc định khi ô không có tiền tố ("=").
    Trả về: (qualifier, phần số dạng chuỗi).
    Vì sao cần: BindingDB không có cột qualifier riêng; qualifier dính ở đầu ô giá trị.
    """
    stripped = value_text.strip()
    # Thử qualifier dài trước, để ">=" không bị nhận nhầm thành ">" + "=5".
    for qualifier in sorted(known_qualifiers, key=len, reverse=True):
        # Vd. '<10' -> ('<', '10').
        if stripped.startswith(qualifier):
            return qualifier, stripped[len(qualifier):].strip()
    # Không có tiền tố -> '='.
    return implicit_qualifier, stripped


def to_molar(value: float, unit: str, unit_to_molar: Mapping[str, float]) -> float | None:
    """Convert via the explicit unit table; None for units outside the table.

    Nhận: giá trị số, đơn vị, và bảng hệ số đơn vị -> mol/L (unit_to_molar trong
    config.json: M, mM, uM/µM/μM, nM, pM, fM).
    Trả về: nồng độ mol/L, hoặc None nếu đơn vị không có trong bảng.
    Vì sao chỉ dùng bảng tường minh: các đơn vị khác (ug.mL-1, %, ppm...) không thể đổi
    ra mol/L nếu không có thêm thông tin (ví dụ khối lượng phân tử) hoặc vốn không phải
    nồng độ. Đoán sẽ tạo ra pActivity sai mà không ai biết; trả None để dòng được đánh
    dấu unresolved với lý do unit_not_in_table.
    """
    # Đơn vị ngoài bảng -> None, không đoán.
    factor = unit_to_molar.get(unit)
    return None if factor is None else value * factor


def p_activity(concentration_molar: float) -> float:
    """pActivity = -log10(concentration in mol/L).

    Nhận: nồng độ mol/L (> 0, đã được parse_activity kiểm tra).
    Trả về: pActivity, ví dụ 1 nM = 1e-9 M -> 9.0.
    """
    # Vd. 300 nM = 3e-7 M -> 6,52.
    return -math.log10(concentration_molar)


def parse_activity(
    value_text: str | None,
    unit: str | None,
    qualifier: str | None,
    unit_to_molar: Mapping[str, float],
    exact_qualifiers: Collection[str],
    censored_qualifiers: Collection[str],
) -> ActivityResult:
    """Resolve one activity value into mol/L and pActivity, with an exact status reason.

    Nhận: chuỗi giá trị (đã tách qualifier), đơn vị, qualifier, bảng đơn vị, danh sách
    qualifier chính xác ("=") và qualifier bị chặn ("<", ">", "<=", ">=", "<<", ">>").
    Trả về: ActivityResult gồm qualifier, nồng độ mol/L, pActivity, cờ censored, status
    và lý do.
    Vì sao cần: một chỗ duy nhất quyết định một giá trị đo có dùng được không và vì sao,
    áp dụng giống hệt cho cả hai nguồn.

    Cách quyết định status: kiểm tra theo thứ tự cố định, điều kiện đầu tiên không đạt
    quyết định status/lý do. "unresolved" khi thiếu thông tin hoặc thông tin nằm ngoài
    bảng đã biết; "invalid" khi giá trị có mặt nhưng sai dạng. Không dòng nào bị bỏ:
    dòng hỏng vẫn được ghi ra cùng lý do để bước sau tự quyết định.
    """

    def failed(status: str, reason: str) -> ActivityResult:
        # Kết quả thất bại: giữ qualifier và cờ censored nếu đã biết, không có số liệu.
        return ActivityResult(qualifier, None, None, censored, status, reason)

    # Phân loại qualifier: True = bị chặn, False = chính xác, None = không có trong bảng.
    censored = (True if qualifier in censored_qualifiers
                else False if qualifier in exact_qualifiers else None)
    # --- Bước 1-3: thiếu giá trị / thiếu qualifier / qualifier lạ -> unresolved.
    if value_text is None or not value_text.strip():
        return failed(UNRESOLVED, "missing_value")
    if not qualifier:
        return failed(UNRESOLVED, "missing_qualifier")
    if censored is None:
        return failed(UNRESOLVED, f"qualifier_not_in_table:{qualifier}")
    # --- Bước 4-6: giá trị có mặt nhưng không dùng được -> invalid.
    # Lưu ý: float('nan') không lỗi, nên cần isfinite ở dưới.
    try:
        value = float(value_text)
    except ValueError:
        return failed(INVALID, f"non_numeric_value:{value_text}")
    if not math.isfinite(value):
        return failed(INVALID, f"non_finite_value:{value_text}")
    # Nồng độ <= 0 không có log, nên không thể có pActivity.
    if value <= 0:
        return failed(INVALID, f"non_positive_value:{value_text}")
    # --- Bước 7-8: quy đổi đơn vị qua bảng tường minh; đơn vị thiếu/lạ -> unresolved.
    if not unit:
        return failed(UNRESOLVED, "missing_unit")
    concentration = to_molar(value, unit, unit_to_molar)
    if concentration is None:
        return failed(UNRESOLVED, f"unit_not_in_table:{unit}")
    # --- Giá trị bị chặn (">10 uM"): vẫn resolved và có nồng độ, nhưng KHÔNG có
    # pActivity, vì đó chỉ là một cận chứ không phải một con số đo được.
    if censored:
        return ActivityResult(qualifier, concentration, None, True, RESOLVED,
                              f"censored_qualifier:{qualifier}")
    # --- Giá trị chính xác: có đủ nồng độ và pActivity.
    return ActivityResult(qualifier, concentration, p_activity(concentration), False,
                          RESOLVED, None)


def cx_extension(source_smiles: str) -> str | None:
    """The CXSMILES extension block (e.g. '|r|') kept verbatim, or None if absent.

    Nhận: chuỗi SMILES nguồn.
    Trả về: phần sau " |" (kể cả dấu "|" đầu), giữ nguyên văn; None nếu không có.
    Vì sao cần: canonical SMILES và InChIKey làm mất thông tin lập thể nâng cao trong
    phần CX (ví dụ "|r|" = lập thể tương đối); giữ riêng phần này để không mất dấu.
    """
    # Cắt tại ' |': '... |r|' -> '|r|'.
    _, separator, extension = source_smiles.partition(CXSMILES_SEPARATOR)
    return (CX_BLOCK_START + extension).strip() if separator else None


def describe_stereo_groups(molecule: Chem.Mol) -> str | None:
    """Enhanced-stereo groups as 'TYPE:atom,atom;...' (atom indices in source order).

    Nhận: phân tử RDKit.
    Trả về: chuỗi mô tả các nhóm lập thể nâng cao mà RDKit đọc được (ABSOLUTE/OR/AND),
    hoặc None nếu không có nhóm nào.
    """
    # Vd. 'STEREO_OR:1'. RDKit 2026.09.1 không tạo nhóm cho '|r|'.
    groups = [
        f"{group.GetGroupType().name}:{','.join(str(atom.GetIdx()) for atom in group.GetAtoms())}"
        for group in molecule.GetStereoGroups()
    ]
    return ";".join(groups) if groups else None


def has_relative_stereo(molecule: Chem.Mol) -> bool:
    """True if any enhanced-stereo group is not absolute (OR/AND, e.g. from '|r|').

    Nhận: phân tử RDKit.
    Trả về: True nếu có nhóm lập thể tương đối/racemic (OR hoặc AND).
    Vì sao cần: canonical SMILES/InChIKey không mã hóa các nhóm này, nên cần gắn cờ để
    người dùng biết hai định danh đó đang biểu diễn lập thể "chắc chắn" hơn thực tế.
    """
    # Vì vậy '|r|' không bật cờ này.
    return any(group.GetGroupType() != ABSOLUTE_STEREO_GROUP
               for group in molecule.GetStereoGroups())


def short_error(error: Exception) -> str:
    """First line of an exception message, trimmed for use in a reason string.

    Nhận: một exception.
    Trả về: dòng đầu của thông báo lỗi, tối đa 80 ký tự (hoặc tên lớp lỗi nếu thông báo
    rỗng), để ghi vào cột *_reason mà không làm phình dữ liệu.
    """
    lines = str(error).strip().splitlines()
    message = lines[0] if lines else type(error).__name__
    return message[:MAX_ERROR_DETAIL_CHARS]


def count_stereo_centers(molecule: Chem.Mol) -> tuple[int, int]:
    """(all stereocenters, unspecified ones). May raise when RDKit's CIP labelling fails.

    Nhận: phân tử RDKit.
    Trả về: (tổng số tâm lập thể, số tâm chưa xác định cấu hình).
    Tách thành hàm riêng để describe_structure bọc riêng bước này trong try/except, và
    để unit test có thể giả lập lỗi của nó.
    """
    # Thuật toán CIP mới (useLegacyImplementation=False) chính xác hơn nhưng với phân tử
    # rất lớn có thể ném RuntimeError "Digraph generation failed: more than 100000 nodes".
    centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True,
                                        useLegacyImplementation=False)
    # Nhãn '?' = tâm chưa xác định cấu hình.
    unspecified = sum(1 for _, label in centers if label == UNASSIGNED_CENTER_LABEL)
    return len(centers), unspecified


def describe_structure(source_smiles: str | None) -> StructureResult:
    """RDKit canonical isomeric SMILES, InChIKey and stereo metadata for a source (CX)SMILES.

    The source string itself is stored separately by the caller; canonical SMILES and
    InChIKey drop enhanced-stereo markers, so rows with relative/racemic groups are flagged.

    Nhận: chuỗi SMILES/CXSMILES nguồn.
    Trả về: StructureResult gồm canonical SMILES, InChIKey, phần CX, nhóm lập thể, số tâm
    lập thể, status và lý do.
    Vì sao cần: hai nguồn viết SMILES theo kiểu khác nhau; canonical SMILES và InChIKey
    do cùng một RDKit tính cho phép nhận ra cùng một phân tử ở hai nguồn.

    Cách quyết định status:
    - Các bước BẮT BUỘC (đọc SMILES, canonical SMILES, InChIKey): RDKit ném lỗi ->
      unresolved (không biết phân tử có hỏng không, chỉ biết công cụ không xử lý được);
      RDKit trả kết quả rỗng/sai dạng -> invalid (chuỗi nguồn hỏng).
    - Bước PHỤ (đếm tâm lập thể): lỗi ở đây KHÔNG làm hỏng định danh; status vẫn
      resolved, chỉ ghi lý do stereo_annotation_failed.
    """

    def failed(status: str, reason: str, extension: str | None = None) -> StructureResult:
        # Kết quả thất bại: không có định danh phân tử, nhưng vẫn giữ phần CX nếu đã tách.
        return StructureResult(None, None, extension, None, None, None, status, reason)

    # --- Không có SMILES nguồn.
    if source_smiles is None or not source_smiles.strip():
        return failed(UNRESOLVED, "missing_source_smiles")
    # --- Giữ nguyên văn phần mở rộng CXSMILES trước khi RDKit xử lý.
    extension = cx_extension(source_smiles)
    # --- Bước bắt buộc 1: đọc SMILES. Ném lỗi -> unresolved; trả None -> SMILES hỏng.
    try:
        molecule = Chem.MolFromSmiles(source_smiles)
    except RDKIT_ERRORS as error:
        return failed(UNRESOLVED, f"rdkit_parse_error:{short_error(error)}", extension)
    if molecule is None:
        return failed(INVALID, "rdkit_parse_failed", extension)
    # --- Bước bắt buộc 2: canonical SMILES có lập thể (isomeric).
    try:
        # Giữ ký hiệu lập thể trong SMILES.
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    except RDKIT_ERRORS as error:
        return failed(UNRESOLVED, f"canonical_smiles_failed:{short_error(error)}", extension)
    if not canonical_smiles:
        return failed(INVALID, "empty_canonical_smiles", extension)
    # --- Bước bắt buộc 3: InChIKey, kiểm tra đúng dạng chuẩn.
    try:
        inchikey = Chem.MolToInchiKey(molecule)
    except RDKIT_ERRORS as error:
        return failed(UNRESOLVED, f"inchikey_failed:{short_error(error)}", extension)
    if not inchikey or not INCHIKEY_PATTERN.match(inchikey):
        return failed(INVALID, "invalid_inchikey", extension)

    # Stereo metadata is optional: if RDKit cannot annotate it (e.g. CIP digraph too
    # large), the identifiers above stay valid and the failure is recorded as the reason.
    # --- Bước phụ: đếm tâm lập thể, bọc riêng. Chỉ bắt RuntimeError/ValueError (không
    # bắt mọi exception), để lỗi lập trình khác vẫn làm chương trình dừng. Ở lần chạy
    # 18/10/2026 có 31 dòng rơi vào nhánh này (phân tử SMILES 215-298 ký tự).
    # Ghi chú phụ; dòng vẫn resolved.
    reasons = []
    try:
        stereo_centers, unspecified = count_stereo_centers(molecule)
    except RDKIT_ERRORS as error:
        stereo_centers = unspecified = None
        reasons.append(f"stereo_annotation_failed:{short_error(error)}")
    # --- Gắn cờ nếu có nhóm lập thể tương đối/racemic mà canonical ID không mã hóa được.
    if has_relative_stereo(molecule):
        reasons.append("relative_or_racemic_stereo_groups_not_encoded_in_canonical_ids")
    # --- Kết quả: luôn resolved ở đây; các ghi chú (nếu có) nối bằng ";".
    return StructureResult(canonical_smiles, inchikey, extension,
                           describe_stereo_groups(molecule), stereo_centers, unspecified,
                           RESOLVED, REASON_SEPARATOR.join(reasons) or None)
