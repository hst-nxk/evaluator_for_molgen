"""Stage 2b, bước 2: khóa định danh phân tử và tạo cặp MMP trong cùng một assay.

Khóa định danh:
    InChIKey của Stage 1, cộng hậu tố (config stage2b.identity.relative_stereo_suffix) khi
    dòng mang cờ lập thể tương đối 'r' trong molecule_cx_extension. Lý do: cờ 'r' nghĩa
    là nguồn chỉ biết quan hệ giữa các tâm lập thể, không biết cấu hình tuyệt đối; RDKit
    bỏ qua cờ này nên InChIKey lại khẳng định một cấu hình cụ thể. Hậu tố làm hai dạng
    mang khóa khác nhau, và cặp giữa một phân tử có cờ với một phân tử không cờ bị loại
    (đếm riêng để báo cáo).

Cặp MMP (single cut, như mmpdb):
    - RDKit rdMMPA.FragmentMol với minCuts = maxCuts = 1 và mẫu liên kết cắt ghi tường
      minh trong config (cut_smarts, đúng mẫu mặc định của RDKit): mỗi lần cắt một liên kết
      khớp mẫu cho hai mảnh; mỗi mảnh lần lượt đóng vai lõi chung (constant) và phần biến
      đổi (variable);
    - maxCutBonds của RDKit KHÔNG phải giới hạn số lần cắt: nếu phân tử có nhiều liên kết
      khớp mẫu hơn maxCutBonds thì FragmentMol trả về rỗng cho CẢ phân tử (đo 24/10/2026,
      docs/AUDIT_20261024.md mục 3.3). Vì vậy max_cut_bonds được đặt lớn hơn mọi phân tử
      trong dữ liệu, và fragment_molecule báo lỗi nếu gặp phân tử vượt giới hạn thay vì
      lặng lẽ bỏ nó. Kích thước của phép sửa do hai điều kiện dưới đây khống chế, không do
      maxCutBonds;
    - A và B là một cặp khi có cùng constant (SMILES chính tắc, gồm vị trí nối) và khác
      variable;
    - giới hạn kích thước theo Hu et al. 2012 (MMP-Cliffs, J. Chem. Inf. Model. 52, 1138;
      DOI 10.1021/ci3001138): variable ≤ max_variable_heavy_atoms (13) nguyên tử nặng;
      constant ≥ min_core_to_variable_ratio (2) lần variable, kiểm tra trên từng phân tử vì
      constant giống hệt ở A và B; hai variable của một cặp chênh nhau ≤
      max_variable_size_difference (8) nguyên tử nặng, kiểm tra khi ghép cặp;
    - mỗi mảnh đúng một điểm nối, và nguyên tử nối (dummy) có bậc 1.
    Khi một cặp (A, B) có nhiều cách cắt hợp lệ (vd. cắt ở hai liên kết liền nhau trên
    cùng nhánh thay đổi), giữ cách cắt có phần biến đổi nhỏ nhất (quy ước "biến đổi tối
    thiểu"); hòa thì so tổng kích thước rồi so chuỗi, để kết quả không đổi giữa các lần chạy.

Cặp chỉ được tìm trong một nhóm (assay, protein, endpoint); hàm ở đây không biết gì về
assay, chỉ nhận danh sách phân tử của một nhóm.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA

# Cảnh báo của RDKit khi đọc lại SMILES được in rất nhiều lần với dữ liệu lớn; tắt đi để
# log tiến độ đọc được. Lỗi thật vẫn được đếm (FragmentResult.failed).
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class FragmentSettings:
    """Tham số cắt và lọc mảnh (config stage2b.mmp)."""
    min_cuts: int
    max_cuts: int
    max_cut_bonds: int
    cut_smarts: str
    min_core_to_variable_ratio: float
    max_variable_heavy_atoms: int
    max_variable_size_difference: int
    attachment_points: int
    attachment_degree: int


@dataclass(frozen=True)
class Cut:
    """Một cách chia phân tử thành constant + variable đã qua các điều kiện lọc."""
    constant: str
    variable: str
    variable_heavy: int


@dataclass(frozen=True)
class FragmentResult:
    """Kết quả cắt một phân tử: các cách chia hợp lệ và số liệu để báo cáo."""
    cuts: tuple[Cut, ...]
    attachment_violations: int
    failed: bool


@dataclass(frozen=True)
class GroupMolecule:
    """Một phân tử trong một nhóm assay: chỉ số dòng ở bảng bước 1, SMILES, cờ |r|."""
    row: int
    smiles: str
    relative: bool


@dataclass(frozen=True)
class Pair:
    """Một cặp MMP trong nhóm: row_x < row_y theo chỉ số dòng; transform của cặp."""
    row_x: int
    row_y: int
    constant: str
    variable_x: str
    variable_y: str


# --------------------------------------------------------------------------- khóa định danh


def relative_stereo_mask(extensions: pa.ChunkedArray | pa.Array,
                         pattern: str) -> pa.ChunkedArray | pa.Array:
    """True ở dòng có cờ lập thể tương đối 'r' trong khối CX; ô rỗng -> False."""
    return pc.fill_null(pc.match_substring_regex(extensions, pattern=pattern), False)


def identity_keys(inchikeys: pa.ChunkedArray | pa.Array,
                  relative: pa.ChunkedArray | pa.Array,
                  suffix: str) -> pa.ChunkedArray | pa.Array:
    """Khóa định danh: InChIKey, thêm `suffix` ở dòng có cờ |r|.

    Vd. 'AAAA-BBBB-N' + '-REL' -> 'AAAA-BBBB-N-REL'. InChIKey null -> null.
    """
    suffixed = pc.binary_join_element_wise(inchikeys, pa.scalar(suffix), "")
    return pc.if_else(relative, suffixed, inchikeys)


def is_relative_key(key: str, suffix: str) -> bool:
    """Khóa có mang hậu tố lập thể tương đối không."""
    return key.endswith(suffix)


# --------------------------------------------------------------------------- cắt phân tử


def fragment_pieces(chains: Chem.Mol, settings: FragmentSettings
                    ) -> tuple[list[tuple[str, int]], bool]:
    """Hai mảnh của một lần cắt: [(SMILES chính tắc, số nguyên tử nặng)], mảnh có hợp lệ.

    Hợp lệ khi mỗi mảnh có đúng `attachment_points` nguyên tử dummy và mọi dummy có bậc
    `attachment_degree`. Với single cut trên liên kết đơn điều này luôn đúng; kiểm tra để
    lỗi đầu vào bất thường hiện ra thành số đếm thay vì lặng lẽ sinh cặp sai.
    """
    pieces, valid = [], True
    for fragment in Chem.GetMolFrags(chains, asMols=True, sanitizeFrags=False):
        dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
        if (len(dummies) != settings.attachment_points
                or any(atom.GetDegree() != settings.attachment_degree for atom in dummies)):
            valid = False
        # GetNumHeavyAtoms đếm nguyên tử số hiệu > 1, nên không tính dummy (số hiệu 0).
        pieces.append((Chem.MolToSmiles(fragment), fragment.GetNumHeavyAtoms()))
    return pieces, valid and len(pieces) == 2


@functools.lru_cache(maxsize=None)
def cut_pattern(smarts: str) -> Chem.Mol:
    """Mẫu liên kết cắt đã biên dịch (một lần cho mỗi chuỗi SMARTS)."""
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"cut_smarts không đọc được: {smarts!r}")
    return pattern


def check_cut_bonds(molecule: Chem.Mol, smiles: str, settings: FragmentSettings) -> None:
    """Báo lỗi khi FragmentMol trả rỗng VÌ phân tử vượt max_cut_bonds.

    Vì sao: RDKit bỏ cả phân tử trong trường hợp này mà không báo gì; dataset v1 mất
    166.585 bản ghi vì vậy. Phân tử không có liên kết nào khớp mẫu thì trả rỗng là đúng.
    """
    matches = len(molecule.GetSubstructMatches(cut_pattern(settings.cut_smarts)))
    if matches > settings.max_cut_bonds:
        raise ValueError(f"phân tử có {matches} liên kết cắt được, vượt max_cut_bonds = "
                         f"{settings.max_cut_bonds}; tăng giới hạn trong config: {smiles}")


def fragment_molecule(smiles: str, settings: FragmentSettings) -> FragmentResult:
    """Mọi cách chia hợp lệ (constant, variable) của một phân tử.

    Nhận: SMILES chính tắc (Stage 1); tham số.
    Trả về: FragmentResult. SMILES không đọc được -> failed = True, không có cách chia nào.
    Mỗi lần cắt cho hai cách chia (mảnh nào là constant). Giữ cách chia khi variable
    ≤ max_variable_heavy_atoms và constant ≥ min_core_to_variable_ratio × variable.
    Cách chia trùng nhau (do đối xứng, hai liên kết tương đương) chỉ giữ một.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return FragmentResult((), 0, True)
    results = rdMMPA.FragmentMol(molecule, minCuts=settings.min_cuts,
                                 maxCuts=settings.max_cuts,
                                 maxCutBonds=settings.max_cut_bonds,
                                 pattern=settings.cut_smarts, resultsAsMols=True)
    if not results:
        check_cut_bonds(molecule, smiles, settings)
    cuts: set[Cut] = set()
    violations = 0
    for _, chains in results:
        pieces, valid = fragment_pieces(chains, settings)
        if not valid:
            violations += 1
            continue
        for (constant, constant_heavy), (variable, variable_heavy) in (
                (pieces[0], pieces[1]), (pieces[1], pieces[0])):
            if (variable_heavy <= settings.max_variable_heavy_atoms
                    and constant_heavy
                    >= settings.min_core_to_variable_ratio * variable_heavy):
                cuts.add(Cut(constant, variable, variable_heavy))
    return FragmentResult(tuple(sorted(cuts, key=lambda cut: (cut.constant, cut.variable))),
                          violations, False)


# --------------------------------------------------------------------------- tạo cặp


def cut_score(first: Cut, second: Cut) -> tuple[int, int, str, str, str]:
    """Thứ tự ưu tiên giữa các cách cắt của cùng một cặp: nhỏ hơn là tốt hơn."""
    return (max(first.variable_heavy, second.variable_heavy),
            first.variable_heavy + second.variable_heavy,
            first.constant, first.variable, second.variable)


def find_pairs(molecules: Sequence[GroupMolecule],
               fragments: Callable[[str], FragmentResult],
               max_size_difference: int) -> tuple[list[Pair], set[tuple[int, int]]]:
    """Mọi cặp MMP trong một nhóm (assay, protein, endpoint).

    Nhận: các phân tử của nhóm (mỗi khóa định danh một dòng); hàm cắt (có cache); chênh
    lệch tối đa số nguyên tử nặng giữa hai variable của một cặp (Hu et al. 2012). Cách cắt
    vượt mức chênh lệch bị bỏ trước khi chọn cách cắt tốt nhất, nên cặp vẫn được giữ nếu
    còn cách cắt khác hợp lệ.
    Trả về: (danh sách cặp, tập (row_x, row_y) bị loại vì một bên có cờ |r| còn bên kia
    không). Mỗi cặp không thứ tự xuất hiện một lần, với row_x < row_y.
    Hai phân tử khác khóa nhưng cùng cấu trúc (dạng |r| và dạng tuyệt đối) cho cùng mọi
    variable nên không bao giờ thành cặp.
    """
    buckets: dict[str, dict[tuple[int, str], Cut]] = defaultdict(dict)
    relative = {molecule.row: molecule.relative for molecule in molecules}
    for molecule in molecules:
        for cut in fragments(molecule.smiles).cuts:
            buckets[cut.constant][(molecule.row, cut.variable)] = cut
    best: dict[tuple[int, int], tuple[tuple[int, int, str, str, str], Pair]] = {}
    lost: set[tuple[int, int]] = set()
    for entries in buckets.values():
        if len(entries) < 2:
            continue
        items = sorted(entries.items())
        for index, ((row_i, variable_i), cut_i) in enumerate(items):
            for (row_j, variable_j), cut_j in items[index + 1:]:
                if row_i == row_j or variable_i == variable_j:
                    continue
                if abs(cut_i.variable_heavy - cut_j.variable_heavy) > max_size_difference:
                    continue
                key = (min(row_i, row_j), max(row_i, row_j))
                if relative[row_i] != relative[row_j]:
                    lost.add(key)
                    continue
                first, second = (cut_i, cut_j) if row_i < row_j else (cut_j, cut_i)
                score = cut_score(first, second)
                if key not in best or score < best[key][0]:
                    best[key] = (score, Pair(key[0], key[1], first.constant, first.variable,
                                             second.variable))
    return [best[key][1] for key in sorted(best)], lost


# --------------------------------------------------------------------------- tiến trình con

# Hàm cắt có cache và mức chênh lệch kích thước của tiến trình hiện tại; đặt bởi
# init_worker (kể cả khi chạy một tiến trình), vì cả hai đọc từ config lúc chạy.
_FRAGMENTS: Callable[[str], FragmentResult] | None = None
_MAX_SIZE_DIFFERENCE: int | None = None


@dataclass
class TaskResult:
    """Kết quả một lô nhóm assay, gửi từ tiến trình con về tiến trình chính."""
    pairs: list[tuple[int, Pair]]
    lost: list[tuple[int, int, int]]
    molecules: int
    attachment_violations: int
    failed_smiles: int


def init_worker(settings: FragmentSettings, cache_size: int) -> None:
    """Đặt hàm cắt có cache LRU và mức chênh lệch kích thước cho tiến trình hiện tại."""
    global _FRAGMENTS, _MAX_SIZE_DIFFERENCE
    _FRAGMENTS = functools.lru_cache(maxsize=cache_size)(
        functools.partial(fragment_molecule, settings=settings))
    _MAX_SIZE_DIFFERENCE = settings.max_variable_size_difference


def process_task(task: Iterable[tuple[int, Sequence[GroupMolecule]]]) -> TaskResult:
    """Tìm cặp cho một lô nhóm assay: [(số hiệu nhóm, phân tử của nhóm)].

    Trả về: TaskResult; mỗi cặp và mỗi cặp bị loại kèm số hiệu nhóm của nó.
    Đếm vi phạm điểm nối và SMILES lỗi theo số lần cắt thực sự (lần gặp đầu của mỗi SMILES
    trong tiến trình), nên con số là cận dưới khi cache bị đẩy ra rồi cắt lại.
    """
    if _FRAGMENTS is None or _MAX_SIZE_DIFFERENCE is None:
        raise RuntimeError("init_worker chưa được gọi trong tiến trình này")
    pairs: list[tuple[int, Pair]] = []
    lost: list[tuple[int, int, int]] = []
    molecules = violations = failed = 0
    seen: set[str] = set()
    for group, members in task:
        for molecule in members:
            if molecule.smiles not in seen:
                seen.add(molecule.smiles)
                result = _FRAGMENTS(molecule.smiles)
                violations += result.attachment_violations
                failed += int(result.failed)
        found, dropped = find_pairs(members, _FRAGMENTS, _MAX_SIZE_DIFFERENCE)
        pairs.extend((group, pair) for pair in found)
        lost.extend((group, row_x, row_y) for row_x, row_y in sorted(dropped))
        molecules += len(members)
    return TaskResult(pairs, lost, molecules, violations, failed)
