"""Shared, pure normalization functions used by both sources.

Every function takes plain values and returns plain values, so it can be tested
without raw data. Nothing here overwrites a source value: callers store the
results in separate derived columns.

Vị trí trong luồng raw -> observations:
    chembl.py / bindingdb.py  đọc raw, trả về bản ghi với giá trị nguồn nguyên văn
    -> normalize.py (file này) tính các giá trị suy ra: qualifier, nồng độ mol/L,
       pActivity, cấu trúc parent (ChEMBL Structure Pipeline), canonical SMILES,
       InChIKey, thông tin lập thể, observation_id, PMID dạng số nguyên, hash chuỗi protein
    -> pipeline.py ghép giá trị nguồn + giá trị suy ra thành một dòng observation.

Mọi hàm ở đây là hàm thuần (không đọc file dữ liệu, không có trạng thái), dùng chung
cho cả hai nguồn, nên ChEMBL và BindingDB được chuẩn hóa theo đúng một quy tắc và có
thể kiểm thử bằng unit test mà không cần dữ liệu raw.

Cấu trúc phân tử: định danh (canonical SMILES, InChIKey) tính trên PARENT của ChEMBL
Structure Pipeline (Bento et al. 2020, J. Cheminform. 12:51,
DOI 10.1186/s13321-020-00456-1): standardize_mol rồi get_parent_mol, bỏ muối và dung
môi, trung hòa điện tích, bỏ đồng vị. Lý do: ion đối không tham gia phép đo; cùng một
chất ở dạng muối, dạng base tự do hay dạng proton hóa phải có cùng một định danh, và
phân tử còn ion đối không ghép được cặp MMP (cắt một liên kết cho ba mảnh). Cấu trúc
nguồn đọc thẳng (chưa chuẩn hóa) vẫn được giữ ở cột molecule_input_* để đối chiếu.
"""

from __future__ import annotations

# Hash, toán, regex; RDKit để xử lý phân tử; ChEMBL Structure Pipeline để lấy parent.
import hashlib
import math
import re
from collections.abc import Collection, Mapping
from typing import NamedTuple

import chembl_structure_pipeline
from chembl_structure_pipeline import standardizer as chembl_standardizer
from chembl_structure_pipeline.exclude_flag import exclude_flag
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
# Một phân tử liền (không có dấu '.').
SINGLE_FRAGMENT = 1
# InChIKey: 14 ký tự đầu mã hóa khung nối nguyên tử (connectivity), phần sau là các lớp
# lập thể, đồng vị, proton.
INCHIKEY_SKELETON_CHARS = 14
# Giá trị của cột molecule_source_inchikey_check (InChIKey nguồn so với InChIKey RDKit
# tính trên CHÍNH cấu trúc nguồn, trước chuẩn hóa).
INCHIKEY_MATCH = "match"
INCHIKEY_OTHER_LAYERS_DIFFER = "skeleton_match_other_layers_differ"
INCHIKEY_SKELETON_DIFFERS = "skeleton_differs"
# Mã phân tử ChEMBL, vd. 'CHEMBL25'.
CHEMBL_MOLECULE_ID_PATTERN = re.compile(r"^CHEMBL\d+$")
# Ghi chú trong structure_reason khi chuẩn hóa thay đổi cấu trúc theo cách cần biết.
REASON_FRAGMENTS_REMOVED = "salt_or_solvent_fragments_removed"
REASON_ISOTOPES_REMOVED = "isotopes_removed"
REASON_EXCLUDED = "chembl_exclude_flag_parent_not_stripped"
REASON_MULTIPLE_FRAGMENTS = "parent_has_multiple_fragments"


class ActivityResult(NamedTuple):
    """Kết quả chuẩn hóa một giá trị hoạt tính (xem parse_activity)."""
    qualifier: str | None
    concentration_molar: float | None
    p_activity: float | None
    censored: bool | None
    status: str
    reason: str | None


class StructureResult(NamedTuple):
    """Kết quả xử lý cấu trúc một phân tử (xem describe_structure).

    canonical_smiles, inchikey, stereo_centers, unspecified_stereo_centers: của PARENT.
    input_canonical_smiles, input_inchikey: của cấu trúc nguồn đọc thẳng, chưa chuẩn hóa.
    cx_extension, stereo_groups: của chuỗi/cấu trúc nguồn (chuẩn hóa không giữ chúng).
    parent_fragments: số mảnh rời của parent (> 1: hỗn hợp, hoặc nhóm bị loại trừ).
    parent_excluded: cờ exclude của ChEMBL (kim loại, > 7 nguyên tử B): không bỏ muối.
    """
    canonical_smiles: str | None
    inchikey: str | None
    input_canonical_smiles: str | None
    input_inchikey: str | None
    cx_extension: str | None
    stereo_groups: str | None
    stereo_centers: int | None
    unspecified_stereo_centers: int | None
    parent_fragments: int | None
    parent_excluded: bool | None
    status: str
    reason: str | None


def check_structure_pipeline_version(expected: str) -> None:
    """Dừng nếu bản ChEMBL Structure Pipeline đang cài khác bản ghi trong config.

    Nhận: phiên bản kỳ vọng (config structure.chembl_structure_pipeline_version).
    Vì sao cần: danh sách muối/dung môi và quy tắc chuẩn hóa nằm trong gói; khác phiên
    bản thì parent có thể khác, nên dữ liệu không tái lập được nếu không khóa phiên bản.
    """
    installed = chembl_structure_pipeline.__version__
    if installed != expected:
        raise RuntimeError(f"chembl_structure_pipeline {installed} is installed but config "
                           f"requires {expected}; install the pinned version first")


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


def chembl_molecule_id(text: str | None) -> str | None:
    """Mã phân tử ChEMBL đã kiểm dạng ('CHEMBL25'), hoặc None.

    Nhận: chuỗi nguồn (molecule_dictionary.chembl_id của ChEMBL, hoặc cột 'ChEMBL ID of
    Ligand' của BindingDB). Trả về: chuỗi đã bỏ khoảng trắng hai đầu nếu đúng dạng
    CHEMBL + số; mọi dạng khác (rỗng, nhiều mã, ký tự lạ) -> None, không đoán. Chuỗi
    nguồn nguyên văn vẫn nằm ở cột molecule_source_chembl_id.
    """
    if text is None:
        return None
    stripped = text.strip()
    return stripped if CHEMBL_MOLECULE_ID_PATTERN.match(stripped) else None


def compare_source_inchikey(source_inchikey: str | None,
                            input_inchikey: str | None) -> str | None:
    """So InChIKey nguồn ghi sẵn với InChIKey RDKit tính trên chính cấu trúc nguồn.

    Nhận: InChIKey của nguồn; InChIKey của cấu trúc nguồn đọc thẳng (chưa chuẩn hóa).
    Trả về: 'match'; 'skeleton_match_other_layers_differ' (14 ký tự đầu giống, khác ở lớp
    lập thể/đồng vị/proton, vd. BindingDB ghi InChIKey không có lập thể trong khi SMILES
    có); 'skeleton_differs' (SMILES và InChIKey của nguồn mô tả hai khung khác nhau);
    None khi thiếu một trong hai. Chỉ gắn cờ, không loại dòng nào.
    """
    if not source_inchikey or not input_inchikey:
        return None
    source = source_inchikey.strip()
    if source == input_inchikey:
        return INCHIKEY_MATCH
    if source[:INCHIKEY_SKELETON_CHARS] == input_inchikey[:INCHIKEY_SKELETON_CHARS]:
        return INCHIKEY_OTHER_LAYERS_DIFFER
    return INCHIKEY_SKELETON_DIFFERS


def chembl_parent(molecule: Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Parent theo ChEMBL Structure Pipeline: standardize_mol -> get_parent_mol.

    Nhận: phân tử RDKit đọc từ SMILES nguồn (không bị sửa).
    Trả về: (parent chưa sanitize, cờ exclude của ChEMBL).
    Tương đương get_parent_mol(standardize_mol(m), neutralize=True, check_exclusion=True)
    của gói. Riêng phân tử chỉ có MỘT mảnh, gói đọc lại file danh sách muối và dịch lại
    163 mẫu SMARTS ở mỗi lần gọi dù kết quả luôn là "giữ nguyên mảnh, trung hòa điện
    tích"; nhánh đó được viết lại đúng theo code của gói (standardizer.py,
    get_fragment_parent_mol) để không tốn hàng giờ trên 3 triệu phân tử. Test
    test_single_fragment_parent_equals_package so kết quả với gói trên nhiều phân tử.
    """
    standardized = chembl_standardizer.standardize_mol(Chem.Mol(molecule))
    isotope_parent = chembl_standardizer.get_isotope_parent_mol(standardized)
    pieces = Chem.GetMolFrags(isotope_parent, asMols=True, sanitizeFrags=False)
    if len(pieces) == SINGLE_FRAGMENT:
        # Gói: một mảnh thì kết quả là chính mảnh đó (khớp muối/dung môi thì "giữ tất
        # cả"), cờ exclude tính trên mảnh đó, rồi trung hòa điện tích.
        return (chembl_standardizer.uncharge_mol(pieces[0]),
                exclude_flag(pieces[0], includeRDKitSanitization=False))
    return chembl_standardizer.get_fragment_parent_mol(isotope_parent, check_exclusion=True,
                                                       neutralize=True)


def sanitized_copy(molecule: Chem.Mol) -> Chem.Mol:
    """Bản sao đã sanitize (vòng, thơm, hóa trị) để tính SMILES/InChI.

    Gói ChEMBL trả về mảnh chưa sanitize; sanitize không đổi cấu trúc, chỉ tính lại
    thông tin RDKit cần. Lỗi (RDKIT_ERRORS) để hàm gọi ghi lý do.
    """
    copy = Chem.Mol(molecule)
    Chem.SanitizeMol(copy)
    return copy


def identifiers(molecule: Chem.Mol) -> tuple[str, str]:
    """(canonical SMILES có lập thể, InChIKey) của một phân tử đã sanitize.

    Ném ValueError khi kết quả rỗng hoặc InChIKey sai dạng (hàm gọi chuyển thành status).
    """
    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    if not smiles:
        raise ValueError("empty canonical SMILES")
    inchikey = Chem.MolToInchiKey(molecule)
    if not inchikey or not INCHIKEY_PATTERN.match(inchikey):
        raise ValueError(f"invalid InChIKey {inchikey!r}")
    return smiles, inchikey


def describe_structure(source_smiles: str | None) -> StructureResult:
    """Parent ChEMBL, canonical SMILES, InChIKey và thông tin lập thể của một (CX)SMILES nguồn.

    Nhận: chuỗi SMILES/CXSMILES nguồn (cột nguồn được lưu nguyên văn ở chỗ khác).
    Trả về: StructureResult; định danh chính là của parent, định danh của cấu trúc nguồn
    đọc thẳng nằm ở input_*.
    Vì sao cần: hai nguồn viết SMILES theo kiểu khác nhau, có dạng muối, dạng proton hóa;
    parent do cùng một quy trình tính cho phép nhận ra cùng một chất ở mọi dạng.

    Cách quyết định status:
    - Các bước BẮT BUỘC (đọc SMILES; định danh của cấu trúc nguồn; parent; định danh của
      parent): RDKit/gói ném lỗi -> unresolved (công cụ không xử lý được, không kết luận
      phân tử hỏng); RDKit trả rỗng/sai dạng khi đọc -> invalid (chuỗi nguồn hỏng).
      Không bao giờ dùng cấu trúc chưa chuẩn hóa thay cho parent.
    - Bước PHỤ (đếm tâm lập thể): lỗi KHÔNG làm hỏng định danh; status vẫn resolved, chỉ
      ghi lý do stereo_annotation_failed.
    - Ghi chú (không đổi status): đã bỏ mảnh muối/dung môi, đã bỏ đồng vị, cờ exclude,
      parent còn nhiều mảnh, nhóm lập thể tương đối/racemic.
    """

    def failed(status: str, reason: str, extension: str | None = None,
               input_ids: tuple[str | None, str | None] = (None, None)) -> StructureResult:
        # Không có định danh parent; vẫn giữ phần CX và định danh nguồn nếu đã tính được.
        return StructureResult(None, None, *input_ids, extension, None, None, None, None,
                               None, status, reason)

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
    # --- Bước bắt buộc 2: định danh của cấu trúc nguồn đọc thẳng (để đối chiếu với
    # InChIKey nguồn và để biết chuẩn hóa đã đổi gì).
    try:
        input_ids = identifiers(molecule)
    except RDKIT_ERRORS as error:
        return failed(UNRESOLVED, f"input_identifiers_failed:{short_error(error)}", extension)
    # --- Bước bắt buộc 3: parent theo ChEMBL Structure Pipeline, rồi định danh của parent.
    try:
        parent, excluded = chembl_parent(molecule)
        parent = sanitized_copy(parent)
        canonical_smiles, inchikey = identifiers(parent)
    except RDKIT_ERRORS as error:
        return failed(UNRESOLVED, f"chembl_parent_failed:{short_error(error)}", extension,
                      input_ids)

    reasons = []
    # --- Ghi chú về những gì chuẩn hóa đã làm (để audit đếm được, không đổi status).
    input_fragments = len(Chem.GetMolFrags(molecule))
    parent_fragments = len(Chem.GetMolFrags(parent))
    if parent_fragments < input_fragments:
        reasons.append(REASON_FRAGMENTS_REMOVED)
    if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
        reasons.append(REASON_ISOTOPES_REMOVED)
    if excluded:
        reasons.append(REASON_EXCLUDED)
    if parent_fragments > SINGLE_FRAGMENT:
        reasons.append(f"{REASON_MULTIPLE_FRAGMENTS}:{parent_fragments}")
    # --- Bước phụ: đếm tâm lập thể của parent, bọc riêng. Chỉ bắt RuntimeError/ValueError
    # (không bắt mọi exception), để lỗi lập trình khác vẫn làm chương trình dừng. Ở lần
    # chạy 18/10/2026 có 31 dòng rơi vào nhánh này (phân tử SMILES 215-298 ký tự).
    try:
        stereo_centers, unspecified = count_stereo_centers(parent)
    except RDKIT_ERRORS as error:
        stereo_centers = unspecified = None
        reasons.append(f"stereo_annotation_failed:{short_error(error)}")
    # --- Nhóm lập thể nâng cao đọc từ cấu trúc nguồn (chuẩn hóa không mang chúng theo).
    if has_relative_stereo(molecule):
        reasons.append("relative_or_racemic_stereo_groups_not_encoded_in_canonical_ids")
    return StructureResult(canonical_smiles, inchikey, *input_ids, extension,
                           describe_stereo_groups(molecule), stereo_centers, unspecified,
                           parent_fragments, excluded, RESOLVED,
                           REASON_SEPARATOR.join(reasons) or None)
