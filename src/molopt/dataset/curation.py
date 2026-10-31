"""Làm sạch dòng đầu vào Stage 2b: các quy tắc loại dòng, mỗi dòng bị loại ghi rõ lý do.

Chạy TRƯỚC lượt gom của pipeline (pipeline.build_exclusions điều phối việc đọc file). Module
này chỉ chứa logic thuần trên bảng, không đọc hay ghi file.

Hai loại quy tắc:
    quy tắc theo dòng (row_reasons): chỉ cần chính dòng đó, ví dụ cờ data_validity_comment
        của ChEMBL, confidence_score, pActivity ngoài khoảng.
    quy tắc theo nhóm (group_reasons): cần nhiều dòng cùng lúc, ví dụ hai assay lệch đúng
        3 bậc (lỗi đơn vị), assay chỉ có vài giá trị lặp lại (dữ liệu chia khoảng).

Mọi quy tắc tính trên các dòng đã qua bộ lọc cơ bản (pipeline.filter_steps). Quy tắc theo
dòng tính độc lập với nhau; hai quy tắc theo nhóm về cách assay ghi giá trị chỉ nhìn các
dòng mà quy tắc theo dòng không bắt (thứ tự ở group_reasons). Một dòng có thể mang nhiều lý
do, để báo cáo đếm được từng quy tắc. Tham số, câu trích nguồn và lý do chọn nằm ở
config.json mục stage2b.curation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

# Tên lý do, theo đúng thứ tự in trong báo cáo. Phải khớp các khóa trong config
# stage2b.curation.rules (kiểm ở check_rule_names).
CHEMBL_DATA_VALIDITY = "chembl_data_validity"
CHEMBL_NO_PCHEMBL = "chembl_no_pchembl"
CHEMBL_INACTIVE_COMMENT = "chembl_inactive_comment"
CHEMBL_POTENTIAL_DUPLICATE = "chembl_potential_duplicate"
CHEMBL_CONFIDENCE = "chembl_confidence_not_direct"
CHEMBL_RELATIONSHIP = "chembl_relationship_not_direct"
CHEMBL_MUTANT = "chembl_mutant_without_variant"
P_OUTSIDE = "p_outside_plausible"
BINDINGDB_TYPICAL_RANGE = "bindingdb_outside_typical_range"
BINDINGDB_CONFLICT = "bindingdb_conflicts_chembl"
NONSTANDARD_SEQUENCE = "nonstandard_sequence"
UNIT_ERROR = "unit_error_between_assays"
TOP_CONCENTRATION = "top_concentration_ties"
BINNED_ASSAY = "binned_assay"
ROW_REASONS = (CHEMBL_DATA_VALIDITY, CHEMBL_NO_PCHEMBL, CHEMBL_INACTIVE_COMMENT,
               CHEMBL_POTENTIAL_DUPLICATE, CHEMBL_CONFIDENCE, CHEMBL_RELATIONSHIP,
               CHEMBL_MUTANT, P_OUTSIDE, BINDINGDB_TYPICAL_RANGE, BINDINGDB_CONFLICT,
               NONSTANDARD_SEQUENCE)
GROUP_REASONS = (UNIT_ERROR, TOP_CONCENTRATION, BINNED_ASSAY)
REASONS = ROW_REASONS + GROUP_REASONS

# Trường ChEMBL đọc từ JSON Stage 1 lưu (json.dumps mặc định: '"khóa": giá trị').
# (tên trường, cột chứa JSON).
CHEMBL_FIELDS = (("data_validity_comment", "activity_context"),
                 ("pchembl_value", "activity_context"),
                 ("activity_comment", "activity_context"),
                 ("potential_duplicate", "activity_context"),
                 ("confidence_score", "assay_context"),
                 ("relationship_type", "assay_context"))
# Một giá trị JSON vô hướng: null, true/false, số, hoặc chuỗi (có thể chứa \" bên trong).
JSON_SCALAR = r'(?P<v>null|true|false|-?[0-9][0-9.eE+-]*|"(?:[^"\\]|\\.)*")'

# Cột của bảng nhóm (đầu vào group_reasons).
GROUP_COLUMNS = ("observation_id", "assay_key", "protein_hash", "endpoint", "molecule_key",
                 "p_activity")
# Khóa một assay (một protein, một endpoint) và khóa một phân tử trong assay đó.
ASSAY_COLUMNS = ["assay_key", "protein_hash", "endpoint"]
MOLECULE_COLUMNS = ASSAY_COLUMNS + ["molecule_key"]


def check_rule_names(rules: Mapping[str, Any]) -> None:
    """Config phải có đúng một mục cho mỗi lý do, không thừa không thiếu."""
    missing = sorted(set(REASONS) - set(rules))
    extra = sorted(set(rules) - set(REASONS))
    if missing or extra:
        raise KeyError(f"stage2b.curation.rules không khớp danh sách lý do: thiếu {missing}, "
                       f"thừa {extra}")


# --------------------------------------------------------------------------- trường ChEMBL


def json_field(texts: pa.ChunkedArray | pa.Array, key: str) -> list[Any]:
    """Giá trị của `key` trong từng chuỗi JSON (None nếu null, vắng khóa hoặc dòng null).

    Dùng regex thay vì json.loads từng dòng vì có hàng triệu dòng; đúng vì Stage 1 ghi JSON
    bằng json.dumps với định dạng cố định. check_json_fields so lại với json.loads trên một
    mẫu để bảo đảm điều đó.
    """
    pattern = f'"{key}": {JSON_SCALAR}'
    found = pc.struct_field(pc.extract_regex(pc.fill_null(texts, ""), pattern=pattern), "v")
    return [None if token is None else json.loads(token) for token in found.to_pylist()]


def chembl_fields(table: pa.Table) -> dict[str, list[Any]]:
    """Sáu trường ChEMBL của mọi dòng: {tên trường: danh sách giá trị}."""
    return {key: json_field(table.column(column), key) for key, column in CHEMBL_FIELDS}


def check_json_fields(table: pa.Table, fields: Mapping[str, Sequence[Any]],
                      rows: int) -> int:
    """So kết quả regex với json.loads trên `rows` dòng đầu; lệch thì dừng.

    Trả về: số dòng đã so. Dòng có JSON null thì mọi trường phải là None.
    """
    checked = 0
    texts = {column: table.column(column).slice(0, rows).to_pylist()
             for column in {column for _, column in CHEMBL_FIELDS}}
    for index in range(min(rows, table.num_rows)):
        for key, column in CHEMBL_FIELDS:
            text = texts[column][index]
            expected = None if text is None else json.loads(text).get(key)
            if expected != fields[key][index]:
                raise ValueError(f"regex đọc {key} ở dòng {index} ra {fields[key][index]!r}, "
                                 f"json.loads ra {expected!r}")
        checked += 1
    return checked


# --------------------------------------------------------------------------- theo dòng


def as_bool(values: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Danh sách bool -> mảng numpy bool."""
    return np.asarray(values, dtype=bool)


def chembl_row_flags(fields: Mapping[str, Sequence[Any]], descriptions: Sequence[str | None],
                     variants: Sequence[str | None],
                     rules: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Bảy quy tắc chỉ áp cho dòng ChEMBL (trường của nguồn này)."""
    inactive = {value.strip().lower() for value in rules[CHEMBL_INACTIVE_COMMENT]["values"]}
    duplicate = str(rules[CHEMBL_POTENTIAL_DUPLICATE]["value"])
    direct = rules[CHEMBL_CONFIDENCE]["value"]
    relation = rules[CHEMBL_RELATIONSHIP]["value"]
    mutant = pc.match_substring_regex(pa.array(list(descriptions), pa.string()),
                                      pattern=rules[CHEMBL_MUTANT]["pattern"], ignore_case=True)
    mutant = as_bool(pc.fill_null(mutant, False).to_pylist())
    return {
        CHEMBL_DATA_VALIDITY: as_bool([value is not None and str(value).strip() != ""
                                       for value in fields["data_validity_comment"]]),
        CHEMBL_NO_PCHEMBL: as_bool([value is None for value in fields["pchembl_value"]]),
        CHEMBL_INACTIVE_COMMENT: as_bool([value is not None
                                          and str(value).strip().lower() in inactive
                                          for value in fields["activity_comment"]]),
        CHEMBL_POTENTIAL_DUPLICATE: as_bool([value is not None and str(value) == duplicate
                                             for value in fields["potential_duplicate"]]),
        # Thiếu confidence_score tức ChEMBL chưa gán mức tin cậy: không phải mức trực tiếp.
        CHEMBL_CONFIDENCE: as_bool([value is None or int(value) != direct
                                    for value in fields["confidence_score"]]),
        CHEMBL_RELATIONSHIP: as_bool([value != relation
                                      for value in fields["relationship_type"]]),
        CHEMBL_MUTANT: mutant & as_bool([not value for value in variants]),
    }


def outside(values: np.ndarray, low: float, high: float, slack: float) -> np.ndarray:
    """True khi giá trị nằm ngoài [low, high] (có bù sai số dấu phẩy động)."""
    return (values < low - slack) | (values > high + slack)


def conflict_flags(reasons: Sequence[str | None], markers: Sequence[str]) -> np.ndarray:
    """dedup_reason (các lý do nối bằng ';') chứa một trong `markers` (so cả token)."""
    wanted = set(markers)
    return as_bool([bool(wanted & set((value or "").split(";"))) for value in reasons])


def nonstandard_flags(hashes: Sequence[str], sequences: Sequence[str | None], alphabet: str,
                      cache: dict[str, bool]) -> np.ndarray:
    """Chuỗi protein (bỏ khoảng trắng, viết hoa như hash Stage 1) có ký tự ngoài `alphabet`.

    cache: protein_hash -> kết quả, dùng chung mọi lô vì một protein lặp lại hàng nghìn dòng.
    """
    allowed = set(alphabet)
    out = np.zeros(len(hashes), dtype=bool)
    for index, (key, sequence) in enumerate(zip(hashes, sequences)):
        value = cache.get(key)
        if value is None:
            cleaned = "".join((sequence or "").split()).upper()
            value = bool(set(cleaned) - allowed)
            cache[key] = value
        out[index] = value
    return out


def row_reasons(table: pa.Table, is_chembl: bool, settings: Mapping[str, Any],
                cache: dict[str, bool], json_check_rows: int) -> dict[str, np.ndarray]:
    """Cờ của mọi quy tắc theo dòng cho một lô dòng đã qua bộ lọc cơ bản.

    Nhận: bảng có các cột vai trò (xem pipeline.CURATION_ROLES); dòng của ChEMBL hay không;
    config stage2b.curation; cache chuỗi protein; số dòng đầu để so regex với json.loads
    (0 = không so).
    Trả về: {lý do: mảng bool}, đủ mọi lý do trong ROW_REASONS (lý do không áp cho nguồn
    này thì toàn False).
    """
    rules = settings["rules"]
    slack = settings["float_slack"]
    rows = table.num_rows
    flags = {reason: np.zeros(rows, dtype=bool) for reason in ROW_REASONS}
    if is_chembl:
        fields = chembl_fields(table)
        if json_check_rows:
            check_json_fields(table, fields, json_check_rows)
        flags.update(chembl_row_flags(fields, table.column("assay_description").to_pylist(),
                                      table.column("protein_variant_id").to_pylist(), rules))
    p = np.asarray(pc.cast(table.column("p_activity"), pa.float64()).to_numpy(
        zero_copy_only=False), dtype=np.float64)
    low, high = rules[P_OUTSIDE]["range"]
    flags[P_OUTSIDE] = outside(p, low, high, slack)
    if not is_chembl:
        low, high = rules[BINDINGDB_TYPICAL_RANGE]["range"]
        flags[BINDINGDB_TYPICAL_RANGE] = outside(p, low, high, slack)
        flags[BINDINGDB_CONFLICT] = conflict_flags(table.column("dedup_reason").to_pylist(),
                                                   rules[BINDINGDB_CONFLICT]["markers"])
    flags[NONSTANDARD_SEQUENCE] = nonstandard_flags(
        table.column("protein_hash").to_pylist(), table.column("protein_sequence").to_pylist(),
        rules[NONSTANDARD_SEQUENCE]["alphabet"], cache)
    return flags


# --------------------------------------------------------------------------- theo nhóm


def molecule_means(frame: pd.DataFrame) -> pd.DataFrame:
    """p trung bình của mỗi (assay, protein, endpoint, phân tử) — một phân tử một giá trị."""
    return frame.groupby(MOLECULE_COLUMNS, sort=False, observed=True)["p_activity"].mean().rename(
        "p").reset_index()


def unit_error_keys(means: pd.DataFrame, shifts: Sequence[float], tolerance: float,
                    decimals: int) -> pd.DataFrame:
    """(assay, protein, endpoint, phân tử) mà ở một assay khác giá trị lệch đúng `shifts` bậc.

    So trên đơn vị 10^-decimals log để ghép bằng khóa chính xác (một phân tử chuẩn có thể nằm
    trong hàng trăm assay của cùng protein, ghép mọi cặp assay sẽ quá lớn). Hai phía của mỗi
    cặp đều được trả về (nguồn: "Remove both entries").
    """
    keys = ["protein_hash", "endpoint", "molecule_key"]
    scale = 10 ** decimals
    shared = means[means.groupby(keys, observed=True)["assay_key"].transform("size") >= 2]
    shared = shared.assign(code=np.round(shared["p"].to_numpy() * scale).astype(np.int64))
    step = int(round(tolerance * scale))
    found = []
    for shift in shifts:
        for offset in range(-step, step + 1):
            target = shared.assign(code=shared["code"] - int(round(shift * scale)) - offset)
            merged = shared.merge(target, on=keys + ["code"], suffixes=("_x", "_y"))
            merged = merged[merged["assay_key_x"] != merged["assay_key_y"]]
            found.append(merged[keys + ["assay_key_x"]].rename(
                columns={"assay_key_x": "assay_key"}))
            found.append(merged[keys + ["assay_key_y"]].rename(
                columns={"assay_key_y": "assay_key"}))
    if not found:
        return pd.DataFrame(columns=["assay_key", *keys])
    return pd.concat(found, ignore_index=True).drop_duplicates()


def tie_keys(means: pd.DataFrame, min_ties: int, decimals: int) -> pd.DataFrame:
    """Khóa (assay, protein, endpoint, phân tử) của phân tử nằm ở giá trị yếu nhất của assay
    khi >= min_ties phân tử cùng chung giá trị đó (nồng độ cao nhất đã thử ghi thành '=').

    Giá trị làm tròn `decimals` chữ số trước khi so (cùng độ phân giải 0,01 của pChEMBL).
    Chỉ một lượt: bỏ mức thấp nhất rồi tìm tiếp sẽ ăn dần vào assay đo thật ở độ phân giải
    0,01 (vd. 4,01 / 4,02 / 4,03 mỗi mức ba phân tử).
    """
    rounded = means.assign(p=means["p"].round(decimals))
    group = rounded.groupby(ASSAY_COLUMNS, sort=False, observed=True)["p"]
    at_low = rounded["p"] == group.transform("min")
    ties = at_low.groupby([rounded[name] for name in ASSAY_COLUMNS], sort=False,
                          observed=True).transform("sum")
    return rounded.loc[at_low & (ties >= min_ties), MOLECULE_COLUMNS]


def binned_keys(means: pd.DataFrame, min_molecules: int, top_values: int, min_share: float,
                decimals: int, slack: float) -> pd.DataFrame:
    """Khóa (assay, protein, endpoint) của assay có >= min_molecules phân tử mà `top_values`
    giá trị phổ biến nhất chiếm >= min_share số phân tử (dữ liệu chia khoảng).

    min_share = 1 nghĩa là assay chỉ có <= top_values giá trị khác nhau. `slack` bù sai số
    dấu phẩy động khi so tỉ phần.
    """
    rounded = means.assign(p=means["p"].round(decimals))
    sizes = rounded.groupby(ASSAY_COLUMNS, sort=False, observed=True).size()
    counts = rounded.groupby(ASSAY_COLUMNS + ["p"], sort=False, observed=True).size()
    counts = counts.rename("n").reset_index().sort_values(
        ASSAY_COLUMNS + ["n"], ascending=[True] * len(ASSAY_COLUMNS) + [False])
    top = counts.groupby(ASSAY_COLUMNS, sort=False, observed=True).head(top_values)
    covered = top.groupby(ASSAY_COLUMNS, observed=True)["n"].sum().reindex(sizes.index)
    binned = sizes[(sizes >= min_molecules) & (covered / sizes >= min_share - slack)]
    return binned.reset_index()[ASSAY_COLUMNS]


def without_keys(means: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Các dòng của `means` có (assay, protein, endpoint, phân tử) KHÔNG nằm trong `keys`."""
    return means[~member_flags(means, keys, MOLECULE_COLUMNS)]


def group_reasons(frame: pd.DataFrame, settings: Mapping[str, Any],
                  row_excluded: np.ndarray) -> dict[str, np.ndarray]:
    """Cờ của ba quy tắc theo nhóm cho MỌI dòng của `frame` (đúng thứ tự dòng).

    Nhận: bảng các dòng đã qua bộ lọc cơ bản của cả hai nguồn, cột GROUP_COLUMNS;
    config stage2b.curation; mảng bool, True ở dòng đã bị một quy tắc theo dòng bắt.

    Thứ tự tính, mỗi bước một lượt:
      1. unit_error_between_assays trên MỌI dòng: hai phía của một cặp lệch đúng 3/6/9 bậc
         đều bị loại ("Remove both entries"), kể cả khi một phía đã bị quy tắc khác bắt.
      2. top_concentration_ties trên dòng KHÔNG bị quy tắc theo dòng bắt: giá trị yếu nhất
         phải là giá trị yếu nhất của các phép đo còn lại, nếu không một dòng sai đơn vị nằm
         thấp hơn sẽ che mất cả trăm phân tử cùng ghi nồng độ trần (đo 31/10/2026:
         bindingdb:11549:3 có 607 phân tử đúng p = 4,00 bị che bởi một dòng p < 4).
      3. binned_assay trên dòng còn lại sau bước 2: giá trị trần không phải số đo, nên không
         tính vào câu hỏi "số đo của assay có bị chia khoảng không".
    Cờ của bước 2 và 3 gắn cho mọi dòng có khóa tương ứng, kể cả dòng đã bị quy tắc khác bắt.
    """
    rules = settings["rules"]
    decimals = settings["value_decimals"]
    unit = rules[UNIT_ERROR]
    shifted = unit_error_keys(molecule_means(frame), unit["shifts"], unit["tolerance"],
                              decimals)
    measured = molecule_means(frame[~np.asarray(row_excluded, dtype=bool)])
    ties = tie_keys(measured, rules[TOP_CONCENTRATION]["min_molecules_at_minimum"], decimals)
    binned_rule = rules[BINNED_ASSAY]
    binned = binned_keys(without_keys(measured, ties), binned_rule["min_molecules"],
                         binned_rule["top_values"], binned_rule["min_share_of_top_values"],
                         decimals, settings["float_slack"])
    return {UNIT_ERROR: member_flags(frame, shifted, MOLECULE_COLUMNS),
            TOP_CONCENTRATION: member_flags(frame, ties, MOLECULE_COLUMNS),
            BINNED_ASSAY: member_flags(frame, binned, ASSAY_COLUMNS)}


def member_flags(frame: pd.DataFrame, keys: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """True ở dòng của `frame` có (columns) nằm trong `keys`."""
    if keys.empty:
        return np.zeros(len(frame), dtype=bool)
    marked = keys[list(columns)].drop_duplicates().assign(_hit=True)
    joined = frame[list(columns)].merge(marked, on=list(columns), how="left")
    return joined["_hit"].notna().to_numpy()


# --------------------------------------------------------------------------- tổng hợp


def reason_lists(flags: Mapping[str, np.ndarray]) -> list[list[str]]:
    """Danh sách lý do của từng dòng (rỗng = không bị loại), theo thứ tự REASONS."""
    names = [reason for reason in REASONS if reason in flags]
    matrix = np.column_stack([flags[name] for name in names])
    return [[names[column] for column in np.flatnonzero(row)] for row in matrix]


def reason_counts(flags: Mapping[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Mỗi lý do: số dòng mang nó, và số dòng CHỈ bị loại vì nó."""
    names = [reason for reason in REASONS if reason in flags]
    matrix = np.column_stack([flags[name] for name in names])
    alone = matrix.sum(axis=1) == 1
    return {name: {"rows": int(matrix[:, index].sum()),
                   "only_this_rule": int((matrix[:, index] & alone).sum())}
            for index, name in enumerate(names)}
