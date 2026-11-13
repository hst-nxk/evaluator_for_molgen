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

PHẢN ĐỐI XỨNG (họp 09/11/2026). Yêu cầu là f(A, B, P) = −f(B, A, P) phải đúng do CẤU TẠO,
với mọi bộ trọng số, chưa cần train. Hai đầu ra đạt được điều đó:

    twin          : Δp = s(B) − s(A). Đổi chỗ A và B thì hai số hạng đổi chỗ -> đổi dấu.
    antisymmetric : Δp = <g(P, e), h(d)> với d = [enc(A) − enc(B), enc(F_A) − enc(F_B)]
                    và h là hàm LẺ (h(−d) = −h(d)). Đổi chỗ A và B thì d -> −d, nên
                    h(d) -> −h(d) còn g(P, e) không đổi -> đầu ra đổi dấu.

Vì sao `antisymmetric` cần tồn tại khi đã có `twin`: ở twin, hàm chấm điểm nhận MỘT phân
tử, nên nó không thể biểu diễn tương tác giữa A và B (ví dụ "nhóm thế này tốt chỉ khi
khung kia có mặt"). Ở `antisymmetric`, h là hàm phi tuyến của HIỆU hai biểu diễn, nên
tương tác trong cặp vẫn học được mà tính chất toán học không mất. Đây là đường anh Long
đề xuất (bỏ bias, dùng hàm kích hoạt lẻ), cộng thêm một điểm: ngữ cảnh P vào bằng phép
NHÂN (cổng g), không bằng phép cộng. Cộng P vào trong h sẽ phá tính lẻ; nhân thì không,
vì g không phụ thuộc d. Nhờ vậy P vẫn được phép ảnh hưởng đến cả ĐỘ LỚN và DẤU của dự
đoán — đúng chỗ mà ablation ngày 07/11/2026 thấy mô hình đang phớt lờ P.

MỘT GIỚI HẠN PHẢI GHI VÀO PAPER. Tính chất trên đúng TUYỆT ĐỐI ở chế độ eval, tức ở mọi
con số được báo cáo. Trong lúc TRAIN với dropout > 0 thì nó chỉ đúng theo kỳ vọng: encoder
dùng chung được gọi hai lần trong một lượt forward (một lần cho A, một lần cho B) và mỗi
lần lấy một mặt nạ dropout độc lập, nên enc_{m1}(A) − enc_{m2}(B) không đúng bằng
−(enc_{m1}(B) − enc_{m2}(A)). Đặt dropout = 0 thì đúng tuyệt đối ở cả hai chế độ. Xem
test_dropout_breaks_antisymmetry_during_training_but_not_at_eval.

Ô thứ tư của bảng trên (twin CÓ nhánh edit) bật bằng khóa `twin_edit`. Trước đây module
này ghi rằng ô đó "không tồn tại được" vì một phép sửa là thuộc tính của cặp — lập luận
đó SAI, và anh Tiến đã chỉ ra ở họp 09/11/2026: F_A là phần bị cắt khỏi A nên nó thuộc về
A, F_B là phần lắp vào tạo B nên nó thuộc về B. Vậy s(A) nhận [enc(A), enc(F_A)] và s(B)
nhận [enc(B), enc(F_B)], hiệu hai bên vẫn đổi dấu khi đổi chỗ cặp. Đúng công thức
g(v_A, v_{F_A}, v_P) − g(v_B, v_{F_B}, v_P) của Task 4.

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
HEAD_ANTISYMMETRIC = "antisymmetric"
HEADS = (HEAD_DIRECT, HEAD_DIRECT_NO_EDIT, HEAD_TWIN, HEAD_ANTISYMMETRIC)

# Hai đầu ra "joint" nhìn thấy cả A và B cùng lúc trong một MLP; `twin` thì không.
JOINT_HEADS = (HEAD_DIRECT, HEAD_DIRECT_NO_EDIT)

# Hai đầu ra phản đối xứng ĐÚNG THEO CẤU TẠO: f(A, B, P) = −f(B, A, P) với mọi bộ trọng
# số, không cần train và không cần augment. Xem ghi chú module, mục "Phản đối xứng".
ANTISYMMETRIC_HEADS = (HEAD_TWIN, HEAD_ANTISYMMETRIC)

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
# `linear` là baseline anh Tiến giao ở họp 09/11/2026 (Task 3): bỏ HẾT phần phi tuyến —
# không encoder, không MLP ghép — chỉ còn MỘT lớp Linear đặt ngay sau các vector đặc trưng
# thô (fingerprint phân tử, fingerprint nhóm thế, embedding protein, embedding endpoint).
# Ridge = chính head này với weight_decay > 0: AdamW dùng weight decay tách rời
# (Loshchilov & Hutter 2019, ICLR, arXiv:1711.05101), nên với một lớp Linear duy nhất nó
# đúng là số hạng phạt L2 của hồi quy Ridge.
HEAD_BLOCK_LINEAR = "linear"
HEAD_BLOCKS = (HEAD_BLOCK_PLAIN, HEAD_BLOCK_RESIDUAL, HEAD_BLOCK_LINEAR)

# Chuẩn hóa trong head plain (khóa `head_norm`). `layernorm` đặt LayerNorm giữa Linear và
# ReLU của mỗi lớp ẩn (Ba và cs. 2016, arXiv:1607.06450). Thêm sau khi --mode diagnose đo
# được head plain của fusion=projector chết 100 % ở lớp đầu (06/11/2026).
# Nhánh phân tử lấy gì làm đầu vào. fingerprint = Morgan bit như v1; embedding =
# vector đã cache từ stage3.molecule_encoder. Mặc định fingerprint để kết quả cũ
# không đổi; xem ligand_features_note trong config.json.
LIGAND_FINGERPRINT = "fingerprint"
LIGAND_EMBEDDING = "embedding"
LIGAND_FEATURES = (LIGAND_FINGERPRINT, LIGAND_EMBEDDING)

HEAD_NORM_NONE = "none"
HEAD_NORM_LAYER = "layernorm"
HEAD_NORMS = (HEAD_NORM_NONE, HEAD_NORM_LAYER)

# Đặc trưng CẶP ghép thêm vào đầu vào của head joint — việc 3 họp 09/11/2026: "thêm đặc
# trưng hiệu A − B vào input, theo PBCNet 2.0: lấy hai vector trừ nhau rồi nhân tương tác".
#   none               : bản v1, chỉ [enc(A), enc(B)]. Mặc định, giữ nguyên mọi số cũ.
#   difference         : ghép thêm enc(A) − enc(B).
#   difference_product : ghép thêm enc(A) − enc(B) VÀ enc(A) * enc(B) (nhân theo từng chiều).
# MLP vốn xấp xỉ được hiệu của hai vector, nên cái thêm vào đây KHÔNG phải thông tin mới —
# nó là một ƯU TIÊN QUY NẠP: đặt sẵn phép toán mà bài toán cần, để mạng không phải học.
# Vì vậy kết quả có thể bằng 0, và nếu bằng 0 thì đó cũng là một câu trả lời có nghĩa.
# CHƯA ĐỐI CHIẾU VỚI BÀI GỐC: việc 6 (đọc lại PBCNet 2.0) chưa làm, nên ở đây chỉ cài đúng
# mô tả trong biên bản, không khẳng định trùng với thiết kế của bài.
# Vì sao chỉ cho head joint: `twin` chấm điểm TỪNG phân tử nên không có chỗ nào nhìn thấy
# cả hai vector cùng lúc; `antisymmetric` thì ĐÃ dựng trên vector hiệu, và ghép thêm tích
# enc(A) * enc(B) — một hàm CHẴN khi đổi chỗ A và B — sẽ phá tính phản đối xứng do cấu tạo.
PAIR_FEATURES_NONE = "none"
PAIR_FEATURES_DIFFERENCE = "difference"
PAIR_FEATURES_DIFFERENCE_PRODUCT = "difference_product"
PAIR_FEATURES = (PAIR_FEATURES_NONE, PAIR_FEATURES_DIFFERENCE,
                 PAIR_FEATURES_DIFFERENCE_PRODUCT)

# Loại của từng token, theo đúng thứ tự ghép. A và B cùng loại `ligand` nên dùng chung
# lớp chiếu, giống như chúng dùng chung encoder.
KIND_LIGAND = "ligand"
KIND_EDIT = "edit"
KIND_PROTEIN = "protein"
KIND_ENDPOINT = "endpoint"
# Hai loại token của đặc trưng cặp. Chúng sống trong cùng không gian với token ligand
# (hiệu và tích của hai vector ligand) nên chiều của chúng bằng `ligand_out`.
KIND_DIFFERENCE = "difference"
KIND_PRODUCT = "product"

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


def build_odd_mlp(in_features: int, hidden: Sequence[int], out_features: int
                  ) -> nn.Sequential:
    """MLP LẺ: h(−x) = −h(x) đúng từng bit, với mọi bộ trọng số.

    Hai điều kiện, cả hai đều bắt buộc, bỏ một cái là mất tính chất:
        1. Mọi lớp Linear KHÔNG CÓ BIAS. Có bias thì W(−x) + b ≠ −(Wx + b).
        2. Hàm kích hoạt phải LẺ. tanh lẻ; ReLU thì không (ReLU(−x) = 0 với x > 0, đúng
           chỗ anh Tiến phản biện ở dòng 123 biên bản 09/11/2026).
    Hợp của các hàm lẻ là hàm lẻ, nên xếp mấy lớp cũng giữ tính chất.

    KHÔNG có dropout và KHÔNG có LayerNorm ở đây, cũng là điều kiện:
        - dropout ở chế độ train nhân một mặt nạ ngẫu nhiên, mà hai lượt forward khác nhau
          lấy mặt nạ khác nhau, nên f(A,B) + f(B,A) = 0 chỉ còn đúng theo kỳ vọng. Ở chế
          độ eval dropout là phép đồng nhất nên không sao, nhưng để lại thì tính chất chỉ
          đúng lúc eval — đúng kiểu bẫy mà anh Tiến cảnh báo.
        - LayerNorm chia cho độ lệch chuẩn (lẻ thì được) nhưng còn TRỪ trung bình và cộng
          tham số dịch, hai phép đó phá tính lẻ.
    Chống quá khớp cho nhánh này đặt ở encoder phía trước (encoder vẫn được phép có bias,
    ReLU và dropout: tính chất chỉ cần đúng SAU phép lấy hiệu — xem ghi chú module).
    """
    layers: list[nn.Module] = []
    width = in_features
    for size in hidden:
        layers.append(nn.Linear(width, size, bias=False))
        layers.append(nn.Tanh())
        width = size
    layers.append(nn.Linear(width, out_features, bias=False))
    return nn.Sequential(*layers)


def build_projector(in_features: int, width: int) -> nn.Sequential:
    """Lớp chiếu một nhánh về không gian chung: Linear rồi LayerNorm.

    LayerNorm đưa mọi nhánh về cùng thang đo, nên khi ghép, nhánh có giá trị lớn hơn (ví dụ
    embedding protein so với embedding endpoint) không lấn các nhánh còn lại.
    """
    return nn.Sequential(nn.Linear(in_features, width), nn.LayerNorm(width))


def hidden_width(width: int, factor: float) -> int:
    """Chiều lớp trong của khối dư = width * factor, làm tròn, tối thiểu 1.

    `hidden factor` là tham số của Bảng 14 (Gorishniy và cs. 2021): tỉ lệ giữa lớp trong
    của khối và chiều khối, quét trong Uniform[1, 4].
    """
    return max(1, int(round(width * float(factor))))


class ResidualBlock(nn.Module):
    """Khối dư: x + Drop_r(Linear(Drop_h(ReLU(Linear(LN(x)))))).

    Đúng phương trình ResNetBlock của Gorishniy và cs. 2021 (arXiv:2106.11959), chỉ thay
    BatchNorm bằng LayerNorm (xem ghi chú module). Hai dropout TÁCH RIÊNG như Bảng 14 của
    bài: `hidden_dropout` nằm trong khối, `residual_dropout` nằm ở cuối nhánh dư. Mục F.1
    của bài ghi rằng ở cấu hình tốt nhất, dropout trong khối thường CAO HƠN dropout nhánh
    dư, và dropout nhánh dư nhiều khi bằng 0 — nên để chung một giá trị là tự bỏ mất một
    chiều quét.
    """

    def __init__(self, width: int, hidden_factor: float, hidden_dropout: float,
                 residual_dropout: float) -> None:
        super().__init__()
        inner = hidden_width(width, hidden_factor)
        self.norm = nn.LayerNorm(width)
        self.body = nn.Sequential(nn.Linear(width, inner), nn.ReLU(),
                                  nn.Dropout(hidden_dropout),
                                  nn.Linear(inner, width), nn.Dropout(residual_dropout))

    def forward(self, inputs: Tensor) -> Tensor:
        """Cùng hình dạng với đầu vào; nhánh tắt giữ nguyên tín hiệu gốc."""
        return inputs + self.body(self.norm(inputs))


class ResidualHead(nn.Module):
    """Đầu dự đoán có skip connection: chiếu về `width`, qua `blocks` khối dư, ra 1 số.

    Lớp ra là Linear(ReLU(LN(x))), đúng phương trình Prediction của Gorishniy và cs. 2021
    (bài gốc dùng BatchNorm). Lớp ra tuyến tính thuần để Δp mang được cả hai dấu.
    """

    def __init__(self, in_features: int, width: int, blocks: int, hidden_factor: float,
                 hidden_dropout: float, residual_dropout: float) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError(f"residual_blocks phải >= 1, nhận {blocks}")
        self.input = nn.Linear(in_features, width)
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, hidden_factor, hidden_dropout, residual_dropout)
            for _ in range(blocks)])
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
        twin_edit = bool(settings.get("twin_edit", False))
        if head_block == HEAD_BLOCK_LINEAR:
            # Baseline tuyến tính phải là tuyến tính TỪ ĐẦU ĐẾN CUỐI mới nói được điều gì:
            # giữ lại lớp chiếu của projector/attention thì nó không còn là baseline nữa.
            if fusion != FUSION_CONCAT:
                raise ValueError("head_block='linear' chỉ dùng với fusion='concat'; "
                                 "projector và attention đều thêm lớp chiếu phi tuyến")
            if head == HEAD_ANTISYMMETRIC:
                raise ValueError("head='antisymmetric' đã có đường tuyến tính riêng: đặt "
                                 "antisymmetric_hidden=[] (một lớp Linear không bias trên "
                                 "vector hiệu); head_block='linear' ở đây là trùng nút")
        if head == HEAD_ANTISYMMETRIC and fusion != FUSION_CONCAT:
            raise ValueError("head='antisymmetric' tự dựng vector hiệu nên không đi qua "
                             "tầng ghép; chỉ dùng với fusion='concat'")
        if twin_edit and head != HEAD_TWIN:
            raise ValueError(f"twin_edit chỉ có nghĩa với head='twin', nhận {head!r}")
        pair_features = str(settings.get("pair_features", PAIR_FEATURES_NONE))
        if pair_features not in PAIR_FEATURES:
            raise ValueError(f"pair_features lạ: {pair_features!r}; chỉ có {PAIR_FEATURES}")
        if pair_features != PAIR_FEATURES_NONE:
            # Ba điều kiện này đều là "không có nghĩa", không phải "chưa kịp làm": xem khối
            # chú thích ở PAIR_FEATURES. Báo lỗi thay vì nhận im lặng, vì nhận im lặng thì
            # cả một đợt chạy tưởng đang thử việc 3 mà thật ra không đổi gì.
            if head not in JOINT_HEADS:
                raise ValueError(
                    f"pair_features chỉ dùng được với head {JOINT_HEADS} (hai head nhìn "
                    f"thấy A và B cùng lúc), nhận {head!r}. `twin` chấm từng phân tử nên "
                    f"không có chỗ ghép; `antisymmetric` đã dựng sẵn trên vector hiệu, và "
                    f"tích enc(A)*enc(B) là hàm chẵn nên sẽ phá tính phản đối xứng.")
            if fusion != FUSION_CONCAT:
                raise ValueError(
                    f"pair_features mới chỉ cài cho fusion='concat', nhận {fusion!r}. "
                    f"projector và attention đã có đường cho các nhánh tác động lẫn nhau, "
                    f"nên hai thứ này chồng lên nhau và hiệu đo được sẽ không đọc được.")
            if head_block == HEAD_BLOCK_LINEAR:
                raise ValueError(
                    "head_block='linear' + pair_features là trùng nút ở phần hiệu: với "
                    "head linear, enc(x) = x nên enc(A) − enc(B) là tổ hợp TUYẾN TÍNH của "
                    "chính đầu vào, lớp Linear duy nhất đã biểu diễn được. Muốn thử tích "
                    "A*B trên baseline tuyến tính thì phải dựng nút riêng và nói rõ rằng "
                    "mô hình không còn tuyến tính theo đầu vào nữa.")
        self.pair_features = pair_features
        self.head_norm = head_norm
        self.head = head
        self.protein_mode = protein_mode
        self.fusion = fusion
        self.head_block = head_block
        self.twin_edit = twin_edit
        dropout = float(settings["dropout"])
        # Head `linear` bỏ hẳn encoder: vector đặc trưng THÔ đi thẳng vào lớp Linear duy
        # nhất, nên "chiều ra của nhánh" đúng bằng chiều đầu vào của nhánh.
        raw = head_block == HEAD_BLOCK_LINEAR
        ligand_out = ligand_bits if raw else int(settings["ligand_out"])
        edit_out = fragment_bits if raw else int(settings["edit_out"])

        self.ligand_encoder: nn.Module = (
            nn.Identity() if raw
            else build_mlp(ligand_bits, settings["ligand_hidden"], ligand_out, dropout))
        # Nhánh edit có ở: `direct`, `antisymmetric`, và `twin` khi bật twin_edit (ô thứ tư
        # của bảng trong ghi chú module). `direct_no_edit` tồn tại để tách tác dụng của
        # nhánh đó khỏi tác dụng của việc tính Δ thẳng.
        self.edit_encoder: nn.Module | None = None
        if head == HEAD_DIRECT or head == HEAD_ANTISYMMETRIC or twin_edit:
            self.edit_encoder = (
                nn.Identity() if raw
                else build_mlp(fragment_bits, settings["edit_hidden"], edit_out, dropout))

        # Thứ tự token cũng là thứ tự ghép của bản v1, nên `concat` cho đúng kết quả cũ.
        # `twin` chấm từng phân tử nên mỗi lượt chỉ có MỘT ligand và MỘT edit.
        single = head in (HEAD_TWIN, HEAD_ANTISYMMETRIC)
        kinds = [KIND_LIGAND] if single else [KIND_LIGAND, KIND_LIGAND]
        # Đặc trưng cặp đứng NGAY SAU hai token ligand và trước nhánh edit: thứ tự này chỉ
        # là quy ước, nhưng phải cố định vì head `linear` và mọi checkpoint đọc vector ghép
        # theo vị trí. Khi pair_features='none' danh sách không đổi, nên checkpoint cũ nạp
        # lại vẫn đúng từng chiều.
        if pair_features != PAIR_FEATURES_NONE:
            kinds.append(KIND_DIFFERENCE)
            if pair_features == PAIR_FEATURES_DIFFERENCE_PRODUCT:
                kinds.append(KIND_PRODUCT)
        if self.edit_encoder is not None:
            kinds += [KIND_EDIT] if single else [KIND_EDIT, KIND_EDIT]
        if protein_mode != PROTEIN_NONE:
            kinds.append(KIND_PROTEIN)
        kinds.append(KIND_ENDPOINT)
        self.token_kinds = tuple(kinds)

        self.protein_projection: nn.Module | None = None
        self.projectors: nn.ModuleDict | None = None
        self.attention: AttentionFusion | None = None
        if fusion == FUSION_CONCAT:
            endpoint_dim = int(settings["endpoint_dim"])
            self.endpoint_embedding = nn.Embedding(endpoint_count, endpoint_dim)
            # Head `linear` và head `antisymmetric` KHÔNG chiếu nhánh protein: lớp chiếu
            # của bản v1 có ReLU (xem context_pieces), mà ReLU vừa phá tính tuyến tính của
            # baseline vừa triệt tiêu nửa âm của vector protein. Hai head đó nhận thẳng
            # vector protein, nên chiều của nhánh là protein_dim.
            plain_protein = raw or head == HEAD_ANTISYMMETRIC
            protein_out = int(protein_dim or 0) if plain_protein else int(
                settings["protein_out"])
            if protein_mode != PROTEIN_NONE and not plain_protein:
                self.protein_projection = build_mlp(int(protein_dim), [], protein_out, 0.0)
            sizes = {KIND_LIGAND: ligand_out, KIND_EDIT: edit_out,
                     KIND_PROTEIN: protein_out, KIND_ENDPOINT: endpoint_dim,
                     KIND_DIFFERENCE: ligand_out, KIND_PRODUCT: ligand_out}
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

        # Head `antisymmetric` không dùng `predictor`: nó có hai nhánh riêng, một nhánh LẺ
        # trên vector hiệu và một cổng phụ thuộc ngữ cảnh (xem ghi chú module).
        self.odd_branch: nn.Module | None = None
        self.gate: nn.Module | None = None
        self.predictor: nn.Module | None = None
        if head == HEAD_ANTISYMMETRIC:
            gate_dim = int(settings["antisymmetric_gate_dim"])
            if gate_dim < 1:
                raise ValueError(f"antisymmetric_gate_dim phải >= 1, nhận {gate_dim}")
            difference_dim = ligand_out + (edit_out if self.edit_encoder is not None else 0)
            self.odd_branch = build_odd_mlp(difference_dim,
                                            settings["antisymmetric_hidden"], gate_dim)
            context_dim = endpoint_dim + (0 if protein_mode == PROTEIN_NONE
                                          else int(protein_dim or 0))
            # Cổng được phép phi tuyến tùy ý (bias, ReLU, dropout): nó KHÔNG phụ thuộc
            # vector hiệu, nên nó không ảnh hưởng tới tính lẻ của nhánh kia.
            self.gate = build_mlp(context_dim, settings["antisymmetric_gate_hidden"],
                                  gate_dim, dropout)
        elif head_block == HEAD_BLOCK_RESIDUAL:
            self.predictor = ResidualHead(
                width, int(settings["residual_width"]), int(settings["residual_blocks"]),
                float(settings.get("residual_hidden_factor", 1.0)),
                dropout_or_default(settings.get("residual_hidden_dropout"), dropout),
                dropout_or_default(settings.get("residual_dropout"), dropout))
        elif head_block == HEAD_BLOCK_LINEAR:
            # MỘT lớp Linear, không kích hoạt, không dropout. bias mặc định TẮT để head này
            # cũng là mô hình mà anh Tiến nói ở dòng 135 biên bản ("Linear Model không
            # bias"); bật lại bằng model.linear_bias=true nếu muốn hồi quy có chặn.
            self.predictor = nn.Linear(width, 1, bias=bool(settings.get("linear_bias",
                                                                        False)))
        else:
            self.predictor = build_mlp(width, settings["head_hidden"], 1, dropout,
                                       norm=head_norm == HEAD_NORM_LAYER)

    @property
    def uses_edit_branch(self) -> bool:
        """True khi kiến trúc thực sự dùng variable_a/variable_b."""
        return self.edit_encoder is not None

    @property
    def antisymmetric_by_construction(self) -> bool:
        """True khi f(A, B, P) = −f(B, A, P) đúng do cấu tạo, với mọi bộ trọng số.

        Ghi vào manifest để phân biệt "sai số hai chiều nhỏ vì kiến trúc đảm bảo" với "sai
        số hai chiều nhỏ vì train trên dữ liệu đã augment" — hai chuyện khác nhau, và chỉ
        chuyện thứ nhất là điều anh Tiến yêu cầu ở họp 09/11/2026.
        """
        return self.head in ANTISYMMETRIC_HEADS

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

    def pair_pieces(self, hidden_a: Tensor, hidden_b: Tensor) -> list[Tensor]:
        """Đặc trưng cặp ghép thêm: [] | [A − B] | [A − B, A * B] (việc 3 họp 09/11/2026).

        Lấy trên vector ĐÃ MÃ HÓA chứ không trên fingerprint thô: hiệu của hai fingerprint
        Morgan chỉ là tập bit có ở A mà không có ở B (và ngược lại, mang dấu âm), còn hiệu
        của hai vector sau encoder là hiệu trong không gian mà mạng đang dùng để dự đoán.
        """
        if self.pair_features == PAIR_FEATURES_NONE:
            return []
        pieces = [hidden_a - hidden_b]
        if self.pair_features == PAIR_FEATURES_DIFFERENCE_PRODUCT:
            pieces.append(hidden_a * hidden_b)
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
        edit_a, edit_b = self.edit_vectors(fragment_a, fragment_b)
        if self.head == HEAD_ANTISYMMETRIC:
            odd, gate = self.antisymmetric_parts(hidden_a, hidden_b, edit_a, edit_b, context)
            return [torch.cat([odd, gate], dim=1)]
        if self.head == HEAD_TWIN:
            side_a = [hidden_a] + ([] if edit_a is None else [edit_a]) + list(context)
            side_b = [hidden_b] + ([] if edit_b is None else [edit_b]) + list(context)
            return [self.combine(side_a), self.combine(side_b)]
        pieces = [hidden_a, hidden_b]
        pieces.extend(self.pair_pieces(hidden_a, hidden_b))
        if edit_a is not None and edit_b is not None:
            pieces.append(edit_a)
            pieces.append(edit_b)
        pieces.extend(context)
        return [self.combine(pieces)]

    def edit_vectors(self, fragment_a: Tensor | None, fragment_b: Tensor | None
                     ) -> tuple[Tensor | None, Tensor | None]:
        """Hai vector nhánh edit đã mã hóa, hoặc (None, None) khi kiến trúc không có nhánh.

        Cùng MỘT bộ trọng số cho F_A và F_B, như ligand encoder dùng chung cho A và B: nhờ
        vậy ở twin và antisymmetric, đổi chỗ cặp chỉ đổi chỗ hai vector, không đổi giá trị.
        """
        if self.edit_encoder is None:
            return None, None
        if fragment_a is None or fragment_b is None:
            raise ValueError(f"kiến trúc {self.head!r} cần fingerprint của variable_a và "
                             f"variable_b")
        return (self.project(KIND_EDIT, self.edit_encoder(fragment_a)),
                self.project(KIND_EDIT, self.edit_encoder(fragment_b)))

    def antisymmetric_parts(self, hidden_a: Tensor, hidden_b: Tensor, edit_a: Tensor | None,
                            edit_b: Tensor | None, context: Sequence[Tensor]
                            ) -> tuple[Tensor, Tensor]:
        """(nhánh lẻ trên vector hiệu, cổng theo ngữ cảnh) của head `antisymmetric`.

        Vector hiệu lấy TRƯỚC khi đi vào nhánh lẻ: d = [enc(A) − enc(B), enc(F_A) − enc(F_B)].
        Đổi chỗ (A, B) kèm (F_A, F_B) làm d đổi dấu, nhánh lẻ đổi dấu theo, cổng không đổi.
        """
        if self.odd_branch is None or self.gate is None:
            raise ValueError("head này không có nhánh phản đối xứng")
        pieces = [hidden_a - hidden_b]
        if edit_a is not None and edit_b is not None:
            pieces.append(edit_a - edit_b)
        return self.odd_branch(torch.cat(pieces, dim=1)), self.gate(
            torch.cat(list(context), dim=1))

    def forward(self, ligand_a: Tensor, ligand_b: Tensor, fragment_a: Tensor | None,
                fragment_b: Tensor | None, protein: Tensor | None,
                endpoint: Tensor) -> Tensor:
        """Δp dự đoán, hình dạng (batch,).

        ligand_a/ligand_b   : fingerprint float 0/1 của hai phân tử
        fragment_a/fragment_b: fingerprint nhóm thế (bỏ qua ở twin và direct_no_edit)
        protein             : vector protein, hoặc None khi protein_mode='none'
        endpoint            : mã endpoint, kiểu long
        """
        if self.head == HEAD_ANTISYMMETRIC:
            context = self.context_pieces(protein, endpoint)
            edit_a, edit_b = self.edit_vectors(fragment_a, fragment_b)
            odd, gate = self.antisymmetric_parts(
                self.project(KIND_LIGAND, self.ligand_encoder(ligand_a)),
                self.project(KIND_LIGAND, self.ligand_encoder(ligand_b)),
                edit_a, edit_b, context)
            # Tích vô hướng: cổng nhân theo từng chiều rồi cộng lại. Phép nhân giữ tính lẻ
            # của `odd` vì `gate` không phụ thuộc vector hiệu.
            return (odd * gate).sum(dim=1)
        vectors = self.representations(ligand_a, ligand_b, fragment_a, fragment_b, protein,
                                       endpoint)
        if self.predictor is None:
            raise ValueError(f"head {self.head!r} không có đầu dự đoán")
        if self.head == HEAD_TWIN:
            score_a, score_b = (self.predictor(vector) for vector in vectors)
            return (score_b - score_a).squeeze(1)
        return self.predictor(vectors[0]).squeeze(1)


def dropout_or_default(value: Any, fallback: float) -> float:
    """Giá trị dropout riêng của khối dư; số âm nghĩa là "dùng model.dropout".

    Config không nhận None ở --set (xem run_evaluator.apply_overrides chỉ cho số và
    chuỗi), nên quy ước số âm làm dấu hiệu "chưa đặt riêng". Mặc định -1 giữ nguyên hành
    vi cũ: cả hai dropout của khối dư bằng model.dropout.
    """
    if value is None:
        return float(fallback)
    number = float(value)
    return float(fallback) if number < 0.0 else number


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
