"""Stage 3: kiến trúc evaluator v1 — encoder dùng chung, edit encoder, hai kiểu đầu ra.

Luồng dữ liệu (kiểu `direct`):

    fingerprint(smiles_a) -> ligand encoder ─┐
    fingerprint(smiles_b) -> ligand encoder ─┤  CÙNG một bộ trọng số
    fingerprint(variable_a) -> edit encoder ─┼─> concat -> MLP -> Δp
    fingerprint(variable_b) -> edit encoder ─┤
    embedding protein -> MLP -> protein_out ─┤
    endpoint -> embedding ───────────────────┘

Kiểu `twin` chấm điểm từng phân tử rồi lấy hiệu: Δp = s(B) − s(A), với
s(x) = MLP([ligand(x), protein, endpoint]).

Ba đầu ra tồn tại để ablation tách được hai yếu tố khác nhau:

    đầu ra          | thấy A và B cùng lúc | có nhánh edit
    ----------------|----------------------|---------------
    direct          | có                   | có
    direct_no_edit  | có                   | không
    twin            | không (tách rời)     | không

    direct vs direct_no_edit -> tác dụng riêng của NHÁNH EDIT
    direct_no_edit vs twin   -> tác dụng riêng của việc tính Δ THẲNG thay vì lấy hiệu

Ô thứ tư (twin có nhánh edit) không tồn tại được: một phép sửa là thuộc tính của CẶP,
không phải của một phân tử, nên hàm chấm điểm một phân tử không có chỗ đặt nó.

Module này KHÔNG import transformers: vector protein được truyền vào từ bên ngoài, dù nó
đến từ cache (cấu hình frozen) hay từ ESM2 đang được train cùng (cấu hình finetune).
Nhờ vậy kiến trúc test được mà không cần tải trọng số ESM2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

HEAD_DIRECT = "direct"
HEAD_DIRECT_NO_EDIT = "direct_no_edit"
HEAD_TWIN = "twin"
HEADS = (HEAD_DIRECT, HEAD_DIRECT_NO_EDIT, HEAD_TWIN)

# Hai đầu ra "joint" nhìn thấy cả A và B cùng lúc trong một MLP; `twin` thì không.
JOINT_HEADS = (HEAD_DIRECT, HEAD_DIRECT_NO_EDIT)

PROTEIN_FROZEN = "frozen"
PROTEIN_FINETUNE = "finetune"
PROTEIN_NONE = "none"
PROTEIN_MODES = (PROTEIN_FROZEN, PROTEIN_FINETUNE, PROTEIN_NONE)

# Cấu hình chạy -> chế độ nhánh protein. `no_protein` là đối chứng: bỏ hẳn nhánh protein
# để biết protein có đóng góp gì không.
CONFIGURATION_MODES = {"frozen": PROTEIN_FROZEN, "finetune": PROTEIN_FINETUNE,
                       "no_protein": PROTEIN_NONE}


def build_mlp(in_features: int, hidden: Sequence[int], out_features: int,
              dropout: float) -> nn.Sequential:
    """MLP: các lớp ẩn có ReLU và dropout, lớp cuối tuyến tính thuần.

    Lớp cuối không có kích hoạt để đầu ra không bị chặn dấu — Δp có cả âm và dương.
    """
    layers: list[nn.Module] = []
    width = in_features
    for size in hidden:
        layers.append(nn.Linear(width, size))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        width = size
    layers.append(nn.Linear(width, out_features))
    return nn.Sequential(*layers)


class EvaluatorModel(nn.Module):
    """Evaluator v1: dự đoán Δp cho một cặp (A, B) trên một protein và một endpoint.

    Tham số:
        settings       : khối config stage3.model
        ligand_bits    : số bit fingerprint phân tử
        fragment_bits  : số bit fingerprint nhóm thế
        protein_dim    : chiều embedding protein ĐO ĐƯỢC (None khi protein_mode='none')
        endpoint_count : số endpoint
        head           : 'direct', 'direct_no_edit' hoặc 'twin' (xem ghi chú module)
        protein_mode   : 'frozen', 'finetune' hoặc 'none'

    Một encoder dùng chung cho A và B (không phải hai encoder riêng): hai phía cùng là
    SMILES nên không có lý do để hai bộ trọng số khác nhau, và dùng chung thì cùng một
    phân tử luôn được biểu diễn như nhau dù nó nằm ở phía nào của cặp.
    """

    def __init__(self, settings: Mapping[str, Any], ligand_bits: int, fragment_bits: int,
                 protein_dim: int | None, endpoint_count: int, head: str,
                 protein_mode: str) -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head lạ: {head!r}; chỉ có {HEADS}")
        if protein_mode not in PROTEIN_MODES:
            raise ValueError(f"protein_mode lạ: {protein_mode!r}; chỉ có {PROTEIN_MODES}")
        if protein_mode != PROTEIN_NONE and not protein_dim:
            raise ValueError("cần protein_dim khi nhánh protein được bật")
        self.head = head
        self.protein_mode = protein_mode
        dropout = float(settings["dropout"])
        ligand_out = int(settings["ligand_out"])
        edit_out = int(settings["edit_out"])
        protein_out = int(settings["protein_out"])
        endpoint_dim = int(settings["endpoint_dim"])

        self.ligand_encoder = build_mlp(ligand_bits, settings["ligand_hidden"], ligand_out,
                                        dropout)
        self.endpoint_embedding = nn.Embedding(endpoint_count, endpoint_dim)
        self.protein_projection = (
            build_mlp(int(protein_dim), [], protein_out, 0.0)
            if protein_mode != PROTEIN_NONE else None)

        context = endpoint_dim + (protein_out if protein_mode != PROTEIN_NONE else 0)
        # Chỉ `direct` có nhánh edit. `direct_no_edit` tồn tại để tách tác dụng của nhánh
        # đó khỏi tác dụng của việc tính Δ thẳng (xem bảng trong ghi chú module).
        if head == HEAD_DIRECT:
            self.edit_encoder = build_mlp(fragment_bits, settings["edit_hidden"], edit_out,
                                          dropout)
        else:
            self.edit_encoder = None
        if head in JOINT_HEADS:
            width = 2 * ligand_out + context + (2 * edit_out if head == HEAD_DIRECT else 0)
        else:
            width = ligand_out + context
        self.predictor = build_mlp(width, settings["head_hidden"], 1, dropout)

    @property
    def uses_edit_branch(self) -> bool:
        """True khi kiến trúc thực sự dùng variable_a/variable_b."""
        return self.edit_encoder is not None

    def context_vector(self, protein: Tensor | None, endpoint: Tensor) -> Tensor:
        """Ghép phần không phụ thuộc phân tử: protein đã giảm chiều và endpoint."""
        pieces = [self.endpoint_embedding(endpoint)]
        if self.protein_projection is not None:
            if protein is None:
                raise ValueError("nhánh protein đang bật nhưng không nhận được vector protein")
            pieces.insert(0, torch.relu(self.protein_projection(protein)))
        elif protein is not None:
            raise ValueError("protein_mode='none' nhưng vẫn nhận được vector protein")
        return torch.cat(pieces, dim=1)

    def forward(self, ligand_a: Tensor, ligand_b: Tensor, fragment_a: Tensor | None,
                fragment_b: Tensor | None, protein: Tensor | None,
                endpoint: Tensor) -> Tensor:
        """Δp dự đoán, hình dạng (batch,).

        ligand_a/ligand_b   : fingerprint float 0/1 của hai phân tử
        fragment_a/fragment_b: fingerprint nhóm thế (bỏ qua ở twin và direct_no_edit)
        protein             : vector protein, hoặc None khi protein_mode='none'
        endpoint            : mã endpoint, kiểu long
        """
        context = self.context_vector(protein, endpoint)
        hidden_a = self.ligand_encoder(ligand_a)
        hidden_b = self.ligand_encoder(ligand_b)
        if self.head == HEAD_TWIN:
            score_a = self.predictor(torch.cat([hidden_a, context], dim=1))
            score_b = self.predictor(torch.cat([hidden_b, context], dim=1))
            return (score_b - score_a).squeeze(1)
        pieces = [hidden_a, hidden_b]
        if self.edit_encoder is not None:
            if fragment_a is None or fragment_b is None:
                raise ValueError("kiểu direct cần fingerprint của variable_a và variable_b")
            pieces.append(self.edit_encoder(fragment_a))
            pieces.append(self.edit_encoder(fragment_b))
        pieces.append(context)
        return self.predictor(torch.cat(pieces, dim=1)).squeeze(1)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Số tham số tổng và số tham số được train, để ghi vào manifest."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def protein_mode_for(configuration: str) -> str:
    """Chế độ nhánh protein của một cấu hình chạy ('frozen', 'finetune', 'no_protein')."""
    if configuration not in CONFIGURATION_MODES:
        raise ValueError(f"cấu hình lạ: {configuration!r}; "
                         f"chỉ có {tuple(CONFIGURATION_MODES)}")
    return CONFIGURATION_MODES[configuration]
