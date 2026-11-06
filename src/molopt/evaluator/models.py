"""Stage 3: kiến trúc evaluator — encoder dùng chung, edit encoder, ba kiểu đầu ra, ba kiểu ghép.

Luồng dữ liệu (kiểu `direct`, cách ghép `concat`):

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

Cách ghép các nhánh (khóa `fusion` trong stage3.model, việc giao ở họp 02/11/2026):

    concat     : ghép thẳng các vector, mỗi nhánh một chiều riêng (bản v1, mặc định).
    projector  : mỗi nhánh qua một lớp chiếu Linear + LayerNorm về CÙNG chiều
                 `fusion_dim` rồi mới ghép, để không nhánh nào áp đảo chỉ vì to hơn.
    attention  : như projector, rồi coi mỗi nhánh là một token và cho các token nhìn nhau
                 qua `attention_layers` lớp Transformer encoder (self-attention). Mỗi vị trí
                 token có một embedding vai trò học được, vì A và B dùng chung encoder:
                 thiếu nó thì attention không biết token nào là A, token nào là B, trong khi
                 dấu của Δp phụ thuộc đúng vào điều đó. Sau attention các token được nối
                 lại theo đúng thứ tự, nên đầu dự đoán vẫn biết vai trò từng phần.

Giới hạn của `attention` ở đây: mỗi nhánh chỉ là MỘT vector (fingerprint, embedding protein
đã mean-pool), nên attention chỉ trộn giữa các NHÁNH với nhau, không nhìn được từng
nguyên tử hay từng acid amin. Muốn attention ở mức đó (kiểu DrugBAN, MolTrans) thì đầu
vào phải đổi sang token: đồ thị phân tử (GNN) và embedding ESM2 theo từng residue.

Khối đầu dự đoán (khóa `head_block`): `plain` là MLP thường như v1; `residual` là ResNet
cho dữ liệu dạng vector của Gorishniy và cs. 2021 (NeurIPS, arXiv:2106.11959, phương trình
2), ý tưởng pre-activation từ He và cs. 2016 (ECCV, DOI 10.1007/978-3-319-46493-0_38) —
phần normalization + skip connection anh Tiến đề xuất. Một chỗ KHÁC bài gốc: bài dùng
BatchNorm, ở đây dùng LayerNorm (lựa chọn riêng, chưa có nguồn): cấu hình finetune gom
batch theo protein (98 % batch chỉ một protein, xem finetune_overrides_note), nên thống kê
của BatchNorm sẽ lệch theo protein.

Nguồn của từng phần (kiểm 03/11/2026):
    embedding vai trò cộng vào token  : ViLT, Kim và cs. 2021, ICML, arXiv:2102.03334, pt. 3
    pre-LN + LayerNorm cuối            : Xiong và cs. 2020, ICML, arXiv:2002.04745
    feed-forward 4 x width, dropout 0,1: Vaswani và cs. 2017, arXiv:1706.03762
    LayerNorm sau lớp chiếu, nối token thay vì token CLS, GELU, mọi con số kích thước
    (fusion_dim, số head, số lớp, residual_width, residual_blocks): CHƯA CÓ NGUỒN.

Mọi giá trị mặc định giữ nguyên hành vi v1: config hay checkpoint cũ không có `fusion` và
`head_block` thì được hiểu là `concat` và `plain`, tên tham số cũng không đổi, nên các
checkpoint đã train vẫn nạp được.

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

FUSION_CONCAT = "concat"
FUSION_PROJECTOR = "projector"
FUSION_ATTENTION = "attention"
FUSIONS = (FUSION_CONCAT, FUSION_PROJECTOR, FUSION_ATTENTION)

HEAD_BLOCK_PLAIN = "plain"
HEAD_BLOCK_RESIDUAL = "residual"
HEAD_BLOCKS = (HEAD_BLOCK_PLAIN, HEAD_BLOCK_RESIDUAL)

# Chuẩn hóa trong head plain (khóa `head_norm`). `layernorm` đặt LayerNorm giữa Linear và
# ReLU của mỗi lớp ẩn (Ba và cs. 2016, arXiv:1607.06450). Thêm sau khi --mode diagnose đo
# được head plain của fusion=projector chết 100 % ở lớp đầu (06/11/2026).
HEAD_NORM_NONE = "none"
HEAD_NORM_LAYER = "layernorm"
HEAD_NORMS = (HEAD_NORM_NONE, HEAD_NORM_LAYER)

# Loại của từng token, theo đúng thứ tự ghép. A và B cùng loại `ligand` nên dùng chung
# lớp chiếu, giống như chúng dùng chung encoder.
KIND_LIGAND = "ligand"
KIND_EDIT = "edit"
KIND_PROTEIN = "protein"
KIND_ENDPOINT = "endpoint"

# Cấu hình chạy -> chế độ nhánh protein. `no_protein` là đối chứng: bỏ hẳn nhánh protein
# để biết protein có đóng góp gì không.
CONFIGURATION_MODES = {"frozen": PROTEIN_FROZEN, "finetune": PROTEIN_FINETUNE,
                       "no_protein": PROTEIN_NONE}


def build_mlp(in_features: int, hidden: Sequence[int], out_features: int,
              dropout: float, norm: bool = False) -> nn.Sequential:
    """MLP: các lớp ẩn có ReLU và dropout, lớp cuối tuyến tính thuần.

    Lớp cuối không có kích hoạt để đầu ra không bị chặn dấu — Δp có cả âm và dương.
    ``norm`` = True thêm LayerNorm giữa Linear và ReLU của mỗi lớp ẩn (xem HEAD_NORMS).
    """
    layers: list[nn.Module] = []
    width = in_features
    for size in hidden:
        layers.append(nn.Linear(width, size))
        if norm:
            layers.append(nn.LayerNorm(size))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        width = size
    layers.append(nn.Linear(width, out_features))
    return nn.Sequential(*layers)


def build_projector(in_features: int, width: int) -> nn.Sequential:
    """Lớp chiếu một nhánh về không gian chung: Linear rồi LayerNorm.

    LayerNorm đưa mọi nhánh về cùng thang đo, nên khi ghép, nhánh có giá trị lớn hơn (ví dụ
    embedding protein so với embedding endpoint) không lấn các nhánh còn lại.
    """
    return nn.Sequential(nn.Linear(in_features, width), nn.LayerNorm(width))


class ResidualBlock(nn.Module):
    """Khối dư: x + Dropout(Linear(Dropout(ReLU(Linear(LN(x)))))).

    Đúng phương trình ResNetBlock của Gorishniy và cs. 2021 (arXiv:2106.11959), chỉ thay
    BatchNorm bằng LayerNorm (xem ghi chú module).
    """

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.body = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, inputs: Tensor) -> Tensor:
        """Cùng hình dạng với đầu vào; nhánh tắt giữ nguyên tín hiệu gốc."""
        return inputs + self.body(self.norm(inputs))


class ResidualHead(nn.Module):
    """Đầu dự đoán có skip connection: chiếu về `width`, qua `blocks` khối dư, ra 1 số.

    Lớp ra là Linear(ReLU(LN(x))), đúng phương trình Prediction của Gorishniy và cs. 2021
    (bài gốc dùng BatchNorm). Lớp ra tuyến tính thuần để Δp mang được cả hai dấu.
    """

    def __init__(self, in_features: int, width: int, blocks: int, dropout: float) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError(f"residual_blocks phải >= 1, nhận {blocks}")
        self.input = nn.Linear(in_features, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Hình dạng (batch, 1), giống MLP thường để forward của model không phải đổi."""
        return self.output(torch.relu(self.norm(self.blocks(self.input(inputs)))))


class AttentionFusion(nn.Module):
    """Self-attention giữa các token nhánh, rồi nối các token lại theo thứ tự.

    Tham số:
        tokens  : số token (cố định theo đầu ra và chế độ protein)
        width   : chiều mỗi token (fusion_dim)
        heads   : số head attention, phải chia hết width
        layers  : số lớp Transformer encoder
        dropout : dropout bên trong attention và feed-forward

    Dùng pre-LN (norm_first) và một LayerNorm cuối, cách đặt norm ổn định hơn post-LN khi
    train (Xiong và cs. 2020, ICML, arXiv:2002.04745). Feed-forward rộng 4 lần width như bản
    Transformer gốc (Vaswani và cs. 2017, arXiv:1706.03762). Embedding vai trò cộng vào token
    theo cách ViLT cộng modal-type embedding (Kim và cs. 2021, arXiv:2102.03334, pt. 3).
    Nối các token lại thay vì dùng token CLS là lựa chọn riêng, chưa có nguồn.
    """

    def __init__(self, tokens: int, width: int, heads: int, layers: int,
                 dropout: float) -> None:
        super().__init__()
        if heads < 1 or width % heads != 0:
            raise ValueError(f"fusion_dim ({width}) phải chia hết cho attention_heads "
                             f"({heads})")
        if layers < 1:
            raise ValueError(f"attention_layers phải >= 1, nhận {layers}")
        # Embedding vai trò: mỗi vị trí (A, B, F_A, F_B, protein, endpoint) một vector.
        self.roles = nn.Parameter(torch.empty(tokens, width))
        nn.init.normal_(self.roles, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        # enable_nested_tensor=False: tối ưu nested tensor không dùng được với norm_first
        # và chỉ có ích khi có mask đệm, ở đây mọi mẫu luôn đủ token.
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """(batch, tokens, width) -> (batch, tokens * width)."""
        return self.encoder(tokens + self.roles).flatten(1)


class EvaluatorModel(nn.Module):
    """Evaluator: dự đoán Δp cho một cặp (A, B) trên một protein và một endpoint.

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
        fusion = str(settings.get("fusion", FUSION_CONCAT))
        if fusion not in FUSIONS:
            raise ValueError(f"fusion lạ: {fusion!r}; chỉ có {FUSIONS}")
        head_block = str(settings.get("head_block", HEAD_BLOCK_PLAIN))
        if head_block not in HEAD_BLOCKS:
            raise ValueError(f"head_block lạ: {head_block!r}; chỉ có {HEAD_BLOCKS}")
        head_norm = str(settings.get("head_norm", HEAD_NORM_NONE))
        if head_norm not in HEAD_NORMS:
            raise ValueError(f"head_norm lạ: {head_norm!r}; chỉ có {HEAD_NORMS}")
        if head_norm != HEAD_NORM_NONE and head_block != HEAD_BLOCK_PLAIN:
            # Head residual đã có LayerNorm trước mọi ReLU; nhận im lặng thì một lần quét
            # tham số tưởng đang thử head_norm mà thật ra không đổi gì.
            raise ValueError("head_norm chỉ dùng với head_block='plain'")
        self.head_norm = head_norm
        self.head = head
        self.protein_mode = protein_mode
        self.fusion = fusion
        self.head_block = head_block
        dropout = float(settings["dropout"])
        ligand_out = int(settings["ligand_out"])
        edit_out = int(settings["edit_out"])

        self.ligand_encoder = build_mlp(ligand_bits, settings["ligand_hidden"], ligand_out,
                                        dropout)
        # Chỉ `direct` có nhánh edit. `direct_no_edit` tồn tại để tách tác dụng của nhánh
        # đó khỏi tác dụng của việc tính Δ thẳng (xem bảng trong ghi chú module).
        if head == HEAD_DIRECT:
            self.edit_encoder = build_mlp(fragment_bits, settings["edit_hidden"], edit_out,
                                          dropout)
        else:
            self.edit_encoder = None

        # Thứ tự token cũng là thứ tự ghép của bản v1, nên `concat` cho đúng kết quả cũ.
        kinds = [KIND_LIGAND] if head == HEAD_TWIN else [KIND_LIGAND, KIND_LIGAND]
        if self.edit_encoder is not None:
            kinds += [KIND_EDIT, KIND_EDIT]
        if protein_mode != PROTEIN_NONE:
            kinds.append(KIND_PROTEIN)
        kinds.append(KIND_ENDPOINT)
        self.token_kinds = tuple(kinds)

        self.protein_projection: nn.Module | None = None
        self.projectors: nn.ModuleDict | None = None
        self.attention: AttentionFusion | None = None
        if fusion == FUSION_CONCAT:
            protein_out = int(settings["protein_out"])
            endpoint_dim = int(settings["endpoint_dim"])
            self.endpoint_embedding = nn.Embedding(endpoint_count, endpoint_dim)
            if protein_mode != PROTEIN_NONE:
                self.protein_projection = build_mlp(int(protein_dim), [], protein_out, 0.0)
            sizes = {KIND_LIGAND: ligand_out, KIND_EDIT: edit_out,
                     KIND_PROTEIN: protein_out, KIND_ENDPOINT: endpoint_dim}
            width = sum(sizes[kind] for kind in self.token_kinds)
        else:
            shared = int(settings["fusion_dim"])
            # Endpoint là nhãn rời rạc nên embedding thẳng ra chiều chung, không cần chiếu.
            self.endpoint_embedding = nn.Embedding(endpoint_count, shared)
            sources = {KIND_LIGAND: ligand_out, KIND_EDIT: edit_out,
                       KIND_PROTEIN: protein_dim}
            self.projectors = nn.ModuleDict({
                kind: build_projector(int(sources[kind]), shared)
                for kind in dict.fromkeys(self.token_kinds) if kind != KIND_ENDPOINT})
            width = len(self.token_kinds) * shared
            if fusion == FUSION_ATTENTION:
                self.attention = AttentionFusion(
                    len(self.token_kinds), shared, int(settings["attention_heads"]),
                    int(settings["attention_layers"]),
                    float(settings["attention_dropout"]))

        if head_block == HEAD_BLOCK_RESIDUAL:
            self.predictor: nn.Module = ResidualHead(
                width, int(settings["residual_width"]), int(settings["residual_blocks"]),
                dropout)
        else:
            self.predictor = build_mlp(width, settings["head_hidden"], 1, dropout,
                                       norm=head_norm == HEAD_NORM_LAYER)

    @property
    def uses_edit_branch(self) -> bool:
        """True khi kiến trúc thực sự dùng variable_a/variable_b."""
        return self.edit_encoder is not None

    def project(self, kind: str, vector: Tensor) -> Tensor:
        """Đưa vector của một nhánh về không gian chung; ở `concat` thì giữ nguyên."""
        if self.projectors is None:
            return vector
        return self.projectors[kind](vector)

    def context_pieces(self, protein: Tensor | None, endpoint: Tensor) -> list[Tensor]:
        """Phần không phụ thuộc phân tử, theo thứ tự [protein], endpoint."""
        pieces: list[Tensor] = []
        if self.protein_mode != PROTEIN_NONE:
            if protein is None:
                raise ValueError("nhánh protein đang bật nhưng không nhận được vector protein")
            if self.protein_projection is not None:
                pieces.append(torch.relu(self.protein_projection(protein)))
            else:
                pieces.append(self.project(KIND_PROTEIN, protein))
        elif protein is not None:
            raise ValueError("protein_mode='none' nhưng vẫn nhận được vector protein")
        pieces.append(self.endpoint_embedding(endpoint))
        return pieces

    def combine(self, pieces: Sequence[Tensor]) -> Tensor:
        """Ghép các nhánh thành một vector cho đầu dự đoán."""
        if len(pieces) != len(self.token_kinds):
            raise ValueError(f"cần {len(self.token_kinds)} nhánh, nhận {len(pieces)}")
        if self.attention is not None:
            return self.attention(torch.stack(list(pieces), dim=1))
        return torch.cat(list(pieces), dim=1)

    def representations(self, ligand_a: Tensor, ligand_b: Tensor, fragment_a: Tensor | None,
                        fragment_b: Tensor | None, protein: Tensor | None,
                        endpoint: Tensor) -> list[Tensor]:
        """Vector đi vào đầu dự đoán: một vector ở kiểu joint, hai (A rồi B) ở twin.

        Đây là "đặc trưng sau encoder" dùng cho t-SNE (việc 4, họp 02/11/2026): mọi nhánh
        đã mã hóa và đã ghép, ngay trước MLP cuối. Tham số như forward.
        """
        context = self.context_pieces(protein, endpoint)
        hidden_a = self.project(KIND_LIGAND, self.ligand_encoder(ligand_a))
        hidden_b = self.project(KIND_LIGAND, self.ligand_encoder(ligand_b))
        if self.head == HEAD_TWIN:
            return [self.combine([hidden_a, *context]), self.combine([hidden_b, *context])]
        pieces = [hidden_a, hidden_b]
        if self.edit_encoder is not None:
            if fragment_a is None or fragment_b is None:
                raise ValueError("kiểu direct cần fingerprint của variable_a và variable_b")
            pieces.append(self.project(KIND_EDIT, self.edit_encoder(fragment_a)))
            pieces.append(self.project(KIND_EDIT, self.edit_encoder(fragment_b)))
        pieces.extend(context)
        return [self.combine(pieces)]

    def forward(self, ligand_a: Tensor, ligand_b: Tensor, fragment_a: Tensor | None,
                fragment_b: Tensor | None, protein: Tensor | None,
                endpoint: Tensor) -> Tensor:
        """Δp dự đoán, hình dạng (batch,).

        ligand_a/ligand_b   : fingerprint float 0/1 của hai phân tử
        fragment_a/fragment_b: fingerprint nhóm thế (bỏ qua ở twin và direct_no_edit)
        protein             : vector protein, hoặc None khi protein_mode='none'
        endpoint            : mã endpoint, kiểu long
        """
        vectors = self.representations(ligand_a, ligand_b, fragment_a, fragment_b, protein,
                                       endpoint)
        if self.head == HEAD_TWIN:
            score_a, score_b = (self.predictor(vector) for vector in vectors)
            return (score_b - score_a).squeeze(1)
        return self.predictor(vectors[0]).squeeze(1)


def architecture_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Khối stage3.model bỏ các khóa ghi chú (*note), để lưu vào checkpoint và manifest.

    Checkpoint phải mang theo kiến trúc của chính nó: eval dựng lại model từ đây chứ không
    từ config hiện tại, nếu không đổi config (ví dụ sang fusion attention) là checkpoint
    cũ không nạp được hoặc, tệ hơn, nạp vào nhầm kiến trúc.
    """
    return {key: value for key, value in settings.items() if not key.endswith("note")}


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
