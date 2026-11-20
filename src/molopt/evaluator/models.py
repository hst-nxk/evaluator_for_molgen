"""Stage 3: the evaluator architecture - shared encoder, edit encoder, four heads, three
ways of fusing the branches.

The data path (head `direct`, fusion `concat`):

    fingerprint(smiles_a) -> ligand encoder -+
    fingerprint(smiles_b) -> ligand encoder -+  THE SAME weights
    fingerprint(variable_a) -> edit encoder -+-> concat -> MLP -> Δp
    fingerprint(variable_b) -> edit encoder -+
    protein embedding -> MLP -> protein_out -+
    endpoint -> embedding -------------------+

The `twin` head scores each molecule and subtracts: Δp = s(B) - s(A), where
s(x) = MLP([ligand(x), protein, endpoint]).

The heads exist so that an ablation can separate two different factors:

    head            | sees A and B together | has an edit branch
    ----------------|-----------------------|-------------------
    direct          | yes                   | yes
    direct_no_edit  | yes                   | no
    twin            | no (scored apart)     | no

    direct vs direct_no_edit -> the effect of the EDIT BRANCH alone
    direct_no_edit vs twin   -> the effect of predicting Δ DIRECTLY instead of subtracting

ANTISYMMETRY (2026-11-09 meeting). The requirement is that f(A, B, P) = -f(B, A, P) hold BY
CONSTRUCTION, for every set of weights, before any training. Two heads achieve that:

    twin          : Δp = s(B) - s(A). Swapping A and B swaps the two terms -> the sign
                    flips.
    antisymmetric : Δp = <g(P, e), h(d)> with d = [enc(A) - enc(B), enc(F_A) - enc(F_B)]
                    and h an ODD function (h(-d) = -h(d)). Swapping A and B sends d -> -d,
                    so h(d) -> -h(d) while g(P, e) does not move -> the output flips sign.

WHY `antisymmetric` IS NEEDED WHEN `twin` ALREADY EXISTS: in twin the scoring function sees
ONE molecule, so it cannot represent an interaction between A and B ("this substituent only
helps when that scaffold is present"). In `antisymmetric`, h is a non-linear function of
the DIFFERENCE of the two representations, so interaction within the pair can still be
learnt without losing the mathematical property. This is the route proposed at the meeting
(drop the biases, use an odd activation), plus one addition: the context P enters by
MULTIPLICATION (the gate g), not by addition. Adding P inside h would destroy oddness;
multiplying does not, because g does not depend on d. P is therefore still allowed to
affect both the MAGNITUDE and the SIGN of a prediction - exactly where the 2026-11-07
ablation found the model ignoring P.

ONE LIMITATION THAT BELONGS IN THE PAPER. The property holds EXACTLY in eval mode, which is
to say in every reported number. During TRAINING with dropout > 0 it holds only in
expectation: the shared encoder is called twice in one forward pass (once for A, once for
B) and each call draws its own dropout mask, so enc_{m1}(A) - enc_{m2}(B) is not exactly
-(enc_{m1}(B) - enc_{m2}(A)). With dropout = 0 it is exact in both modes. See
test_dropout_breaks_antisymmetry_during_training_but_not_at_eval.

The fourth cell of the table above (twin WITH an edit branch) is switched on with the
`twin_edit` key. This module used to state that the cell "cannot exist" because an edit is a
property of the pair - that argument is WRONG, as was pointed out at the 2026-11-09 meeting:
F_A is the part cut out of A, so it belongs to A; F_B is the part put in to make B, so it
belongs to B. Then s(A) takes [enc(A), enc(F_A)] and s(B) takes [enc(B), enc(F_B)], and the
difference still flips sign when the pair is swapped. That is exactly task 4's formula
g(v_A, v_{F_A}, v_P) - g(v_B, v_{F_B}, v_P).

How the branches are fused (the `fusion` key in stage3.model, assigned at the 2026-11-02
meeting):

    concat     : concatenate the vectors as they are, each branch keeping its own width
                 (the v1 behaviour, and the default).
    projector  : send every branch through a Linear + LayerNorm projection to ONE shared
                 width `fusion_dim` before concatenating, so no branch dominates merely by
                 being larger.
    attention  : as projector, then treat every branch as a token and let the tokens see
                 each other through `attention_layers` Transformer encoder layers
                 (self-attention). Every token position carries a learnt role embedding,
                 because A and B share one encoder: without it the attention could not tell
                 which token is A and which is B, and the sign of Δp depends on exactly
                 that. After the attention the tokens are concatenated back in order, so the
                 prediction head still knows which part played which role.

The limit of `attention` here: every branch is just ONE vector (a fingerprint, a mean-pooled
protein embedding), so the attention only mixes BRANCHES with each other and cannot look at
individual atoms or residues. Attention at that level (as in DrugBAN or MolTrans) would
require token inputs: a molecular graph (GNN) and per-residue ESM2 embeddings.

The prediction block (the `head_block` key): `plain` is an ordinary MLP as in v1;
`residual` is the ResNet for tabular data of Gorishniy et al. 2021 (NeurIPS,
arXiv:2106.11959, equation 2), whose pre-activation idea comes from He et al. 2016 (ECCV,
DOI 10.1007/978-3-319-46493-0_38) - the normalization + skip connection proposed at a
meeting. One DEPARTURE from the paper: it uses BatchNorm, this uses LayerNorm (our own
choice, no source): the finetune configuration groups batches by protein (98 % of batches
hold a single protein, see finetune_overrides_note), so BatchNorm statistics would be
skewed per protein.

The source of every part (checked 2026-11-03):
    role embedding added to a token     : ViLT, Kim et al. 2021, ICML, arXiv:2102.03334,
                                          section 3
    pre-LN + a final LayerNorm          : Xiong et al. 2020, ICML, arXiv:2002.04745
    feed-forward 4 x width, dropout 0.1 : Vaswani et al. 2017, arXiv:1706.03762
    LayerNorm after the projection, concatenating tokens instead of a CLS token, GELU, and
    every size (fusion_dim, the head count, the layer count, residual_width,
    residual_blocks): NO SOURCE.

Every default keeps the v1 behaviour: a config or checkpoint without `fusion` and
`head_block` is read as `concat` and `plain`, and no parameter was renamed, so checkpoints
already trained still load.

This module imports NO transformers: the protein vector is passed in from outside, whether
it came from the cache (the frozen configuration) or from an ESM2 that is being trained
alongside (the finetune configuration). That is what makes the architecture testable
without downloading ESM2's weights.
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

# The two "joint" heads see A and B together inside one MLP; `twin` does not.
JOINT_HEADS = (HEAD_DIRECT, HEAD_DIRECT_NO_EDIT)

# The two heads that are antisymmetric BY CONSTRUCTION: f(A, B, P) = -f(B, A, P) for every
# set of weights, with no training and no augmentation. See the module note, "ANTISYMMETRY".
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
# `linear` is the baseline assigned at the 2026-11-09 meeting (task 3): drop EVERY
# non-linearity - no encoder, no fusion MLP - leaving ONE Linear layer directly on the raw
# feature vectors (molecule fingerprint, substituent fingerprint, protein embedding,
# endpoint embedding). Ridge is this very head with weight_decay > 0: AdamW uses decoupled
# weight decay (Loshchilov & Hutter 2019, ICLR, arXiv:1711.05101), so with a single Linear
# layer that is exactly the L2 penalty of ridge regression.
HEAD_BLOCK_LINEAR = "linear"
HEAD_BLOCKS = (HEAD_BLOCK_PLAIN, HEAD_BLOCK_RESIDUAL, HEAD_BLOCK_LINEAR)

# Normalization inside the plain head (the `head_norm` key). `layernorm` puts a LayerNorm
# between the Linear and the ReLU of every hidden layer (Ba et al. 2016, arXiv:1607.06450).
# Added after --mode diagnose measured the plain head of fusion=projector as 100 % dead in
# its first layer (2026-11-06).
# What the molecule branch takes as input. fingerprint = Morgan bits, as in v1; embedding =
# the cached vectors from stage3.molecule_encoder. The default is fingerprint so that
# earlier results do not move; see ligand_features_note in config.json.
LIGAND_FINGERPRINT = "fingerprint"
LIGAND_EMBEDDING = "embedding"
LIGAND_FEATURES = (LIGAND_FINGERPRINT, LIGAND_EMBEDDING)

HEAD_NORM_NONE = "none"
HEAD_NORM_LAYER = "layernorm"
HEAD_NORMS = (HEAD_NORM_NONE, HEAD_NORM_LAYER)

# PAIR features appended to a joint head's input - task 3 of the 2026-11-09 meeting: "add
# the A - B difference feature to the input, following PBCNet 2.0: subtract the two vectors
# and multiply them for the interaction".
#   none               : the v1 input, [enc(A), enc(B)] only. The default, so earlier numbers
#                        do not move.
#   difference         : also append enc(A) - enc(B).
#   difference_product : also append enc(A) - enc(B) AND enc(A) * enc(B) (element-wise).
# An MLP can already approximate the difference of two vectors, so what is added here is NOT
# new information - it is an INDUCTIVE PRIOR: the operation the problem needs, supplied
# rather than learnt. The measured effect may therefore be zero, and a zero is itself a
# meaningful answer.
# NOT YET CHECKED AGAINST THE PAPER: task 6 (re-reading PBCNet 2.0) has not been done, so
# this implements exactly what the minutes describe and claims no agreement with that
# paper's design.
# WHY JOINT HEADS ONLY: `twin` scores EACH molecule, so nothing in it ever sees both vectors
# at once; `antisymmetric` is ALREADY built on the difference vector, and appending the
# product enc(A) * enc(B) - an EVEN function under swapping A and B - would destroy the
# antisymmetry it has by construction.
PAIR_FEATURES_NONE = "none"
PAIR_FEATURES_DIFFERENCE = "difference"
PAIR_FEATURES_DIFFERENCE_PRODUCT = "difference_product"
PAIR_FEATURES = (PAIR_FEATURES_NONE, PAIR_FEATURES_DIFFERENCE,
                 PAIR_FEATURES_DIFFERENCE_PRODUCT)

# The kind of every token, in concatenation order. A and B are both of kind `ligand` and
# therefore share one projection, just as they share one encoder.
KIND_LIGAND = "ligand"
KIND_EDIT = "edit"
KIND_PROTEIN = "protein"
KIND_ENDPOINT = "endpoint"
# The two token kinds of the pair features. They live in the same space as a ligand token
# (the difference and the product of two ligand vectors), so their width is `ligand_out`.
KIND_DIFFERENCE = "difference"
KIND_PRODUCT = "product"

# Run configuration -> protein branch mode. `no_protein` is a control: the protein branch
# is removed entirely, to see whether the protein contributes anything.
CONFIGURATION_MODES = {"frozen": PROTEIN_FROZEN, "finetune": PROTEIN_FINETUNE,
                       "no_protein": PROTEIN_NONE}


def build_mlp(in_features: int, hidden: Sequence[int], out_features: int,
              dropout: float, norm: bool = False) -> nn.Sequential:
    """An MLP: hidden layers with ReLU and dropout, a purely linear last layer.

    The last layer has no activation so the output is not restricted in sign - Δp takes both
    signs. ``norm`` = True adds a LayerNorm between the Linear and the ReLU of every hidden
    layer (see HEAD_NORMS).
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
    """An ODD MLP: h(-x) = -h(x) exactly, bit for bit, for every set of weights.

    Two conditions, both required; dropping either loses the property:
        1. NO BIAS on any Linear layer. With a bias, W(-x) + b != -(Wx + b).
        2. The activation has to be ODD. tanh is; ReLU is not (ReLU(-x) = 0 for x > 0, which
           is exactly the objection raised on line 123 of the 2026-11-09 minutes).
    A composition of odd functions is odd, so any number of such layers keeps the property.

    NO dropout and NO LayerNorm here, which are conditions too:
        - in train mode dropout multiplies by a random mask, and two forward passes draw
          different masks, so f(A,B) + f(B,A) = 0 would hold only in expectation. In eval
          mode dropout is the identity and would do no harm, but leaving it in would mean
          the property held at eval time only - the very trap the meeting warned about.
        - LayerNorm divides by a standard deviation (which is fine for oddness) but it also
          SUBTRACTS the mean and adds a shift parameter, and those two break it.
    Overfitting is held back in the encoder ahead of this branch instead (the encoder may
    still have biases, ReLU and dropout: the property only has to hold AFTER the difference
    is taken - see the module note).
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
    """The projection of one branch into the shared space: a Linear, then a LayerNorm.

    The LayerNorm puts every branch on one scale, so that at concatenation time a branch
    with larger values (a protein embedding against an endpoint embedding, say) does not
    swamp the others.
    """
    return nn.Sequential(nn.Linear(in_features, width), nn.LayerNorm(width))


def hidden_width(width: int, factor: float) -> int:
    """The inner width of a residual block = width * factor, rounded, at least 1.

    `hidden factor` is the parameter from Table 14 of Gorishniy et al. 2021: the ratio of a
    block's inner layer to the block's width, swept over Uniform[1, 4].
    """
    return max(1, int(round(width * float(factor))))


class ResidualBlock(nn.Module):
    """A residual block: x + Drop_r(Linear(Drop_h(ReLU(Linear(LN(x)))))).

    Exactly the ResNetBlock equation of Gorishniy et al. 2021 (arXiv:2106.11959), with
    BatchNorm replaced by LayerNorm (see the module note). The two dropouts are SEPARATE, as
    in that paper's Table 14: `hidden_dropout` sits inside the block, `residual_dropout` at
    the end of the residual branch. Section F.1 reports that in the best configurations the
    dropout inside the block is usually HIGHER than the one on the residual branch, and the
    residual one is often 0 - so tying them to one value would throw away a sweep dimension.
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
        """The same shape as the input; the skip path carries the original signal."""
        return inputs + self.body(self.norm(inputs))


class ResidualHead(nn.Module):
    """A prediction head with skip connections: project to `width`, pass through `blocks`
    residual blocks, output one number.

    The output layer is Linear(ReLU(LN(x))), exactly the Prediction equation of Gorishniy
    et al. 2021 (which uses BatchNorm). It is purely linear so that Δp can take both signs.
    """

    def __init__(self, in_features: int, width: int, blocks: int, hidden_factor: float,
                 hidden_dropout: float, residual_dropout: float) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError(f"residual_blocks has to be >= 1, got {blocks}")
        self.input = nn.Linear(in_features, width)
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, hidden_factor, hidden_dropout, residual_dropout)
            for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Shape (batch, 1), like an ordinary MLP, so the model's forward need not change."""
        return self.output(torch.relu(self.norm(self.blocks(self.input(inputs)))))


class AttentionFusion(nn.Module):
    """Self-attention among the branch tokens, then the tokens concatenated in order.

    Parameters:
        tokens  : the token count (fixed by the head and the protein mode)
        width   : the width of each token (fusion_dim)
        heads   : the number of attention heads; has to divide width
        layers  : the number of Transformer encoder layers
        dropout : the dropout inside the attention and the feed-forward

    Uses pre-LN (norm_first) and one final LayerNorm, a placement that trains more stably
    than post-LN (Xiong et al. 2020, ICML, arXiv:2002.04745). The feed-forward is 4 x width,
    as in the original Transformer (Vaswani et al. 2017, arXiv:1706.03762). The role
    embedding is added to the token the way ViLT adds its modal-type embedding (Kim et al.
    2021, arXiv:2102.03334, section 3). Concatenating the tokens instead of using a CLS
    token is our own choice, with no source.
    """

    def __init__(self, tokens: int, width: int, heads: int, layers: int,
                 dropout: float) -> None:
        super().__init__()
        if heads < 1 or width % heads != 0:
            raise ValueError(f"fusion_dim ({width}) has to be divisible by "
                             f"attention_heads ({heads})")
        if layers < 1:
            raise ValueError(f"attention_layers has to be >= 1, got {layers}")
        # The role embedding: one vector per position (A, B, F_A, F_B, protein, endpoint).
        self.roles = nn.Parameter(torch.empty(tokens, width))
        nn.init.normal_(self.roles, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        # enable_nested_tensor=False: the nested-tensor optimisation does not apply with
        # norm_first and only helps when there is a padding mask; here every sample always
        # carries every token.
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """(batch, tokens, width) -> (batch, tokens * width)."""
        return self.encoder(tokens + self.roles).flatten(1)


class EvaluatorModel(nn.Module):
    """The evaluator: predicts Δp for a pair (A, B) on one protein and one endpoint.

    Parameters:
        settings       : the stage3.model config block
        ligand_bits    : the molecule fingerprint's bit count
        fragment_bits  : the substituent fingerprint's bit count
        protein_dim    : the MEASURED protein embedding dimension (None when
                         protein_mode='none')
        endpoint_count : the number of endpoints
        head           : 'direct', 'direct_no_edit', 'twin' or 'antisymmetric' (see the
                         module note)
        protein_mode   : 'frozen', 'finetune' or 'none'

    ONE encoder shared by A and B, not two separate ones: both sides are SMILES, so there is
    no reason for two sets of weights, and sharing them means one molecule is always
    represented the same way whichever side of the pair it is on.
    """

    def __init__(self, settings: Mapping[str, Any], ligand_bits: int, fragment_bits: int,
                 protein_dim: int | None, endpoint_count: int, head: str,
                 protein_mode: str) -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head: {head!r}; only {HEADS}")
        if protein_mode not in PROTEIN_MODES:
            raise ValueError(f"unknown protein_mode: {protein_mode!r}; only "
                             f"{PROTEIN_MODES}")
        if protein_mode != PROTEIN_NONE and not protein_dim:
            raise ValueError("protein_dim is needed when the protein branch is on")
        fusion = str(settings.get("fusion", FUSION_CONCAT))
        if fusion not in FUSIONS:
            raise ValueError(f"unknown fusion: {fusion!r}; only {FUSIONS}")
        head_block = str(settings.get("head_block", HEAD_BLOCK_PLAIN))
        if head_block not in HEAD_BLOCKS:
            raise ValueError(f"unknown head_block: {head_block!r}; only {HEAD_BLOCKS}")
        head_norm = str(settings.get("head_norm", HEAD_NORM_NONE))
        if head_norm not in HEAD_NORMS:
            raise ValueError(f"unknown head_norm: {head_norm!r}; only {HEAD_NORMS}")
        if head_norm != HEAD_NORM_NONE and head_block != HEAD_BLOCK_PLAIN:
            # The residual head already has a LayerNorm before every ReLU; accepting this
            # quietly would let a whole sweep believe it was testing head_norm while nothing
            # changed at all.
            raise ValueError("head_norm only applies with head_block='plain'")
        twin_edit = bool(settings.get("twin_edit", False))
        if head_block == HEAD_BLOCK_LINEAR:
            # A linear baseline only says something if it is linear END TO END: keeping the
            # projection of projector/attention would stop it being a baseline at all.
            if fusion != FUSION_CONCAT:
                raise ValueError("head_block='linear' only applies with fusion='concat'; "
                                 "projector and attention both add a non-linear projection")
            if head == HEAD_ANTISYMMETRIC:
                raise ValueError("head='antisymmetric' has its own linear route: set "
                                 "antisymmetric_hidden=[] (one bias-free Linear on the "
                                 "difference vector); head_block='linear' here would be a "
                                 "second knob for the same thing")
        if head == HEAD_ANTISYMMETRIC and fusion != FUSION_CONCAT:
            raise ValueError("head='antisymmetric' builds its own difference vector and "
                             "does not pass through the fusion layer; it only applies with "
                             "fusion='concat'")
        if twin_edit and head != HEAD_TWIN:
            raise ValueError(f"twin_edit only means something with head='twin', got "
                             f"{head!r}")
        pair_features = str(settings.get("pair_features", PAIR_FEATURES_NONE))
        if pair_features not in PAIR_FEATURES:
            raise ValueError(f"unknown pair_features: {pair_features!r}; only "
                             f"{PAIR_FEATURES}")
        if pair_features != PAIR_FEATURES_NONE:
            # All three conditions below are "this would mean nothing", not "this has not
            # been built yet": see the comment block at PAIR_FEATURES. They raise instead of
            # being accepted quietly, because being accepted quietly would let a whole run
            # believe it was testing task 3 while nothing changed.
            if head not in JOINT_HEADS:
                raise ValueError(
                    f"pair_features only applies to the heads {JOINT_HEADS} (the two that "
                    f"see A and B together), got {head!r}. `twin` scores each molecule, so "
                    f"there is nowhere to append them; `antisymmetric` is already built on "
                    f"the difference vector, and the product enc(A)*enc(B) is an even "
                    f"function, so it would destroy the antisymmetry.")
            if fusion != FUSION_CONCAT:
                raise ValueError(
                    f"pair_features is implemented for fusion='concat' only, got "
                    f"{fusion!r}. projector and attention already give the branches a way "
                    f"to act on each other, so the two would overlap and the measured "
                    f"effect could not be read.")
            if head_block == HEAD_BLOCK_LINEAR:
                raise ValueError(
                    "head_block='linear' + pair_features is a second knob for the same "
                    "thing on the difference part: with the linear head enc(x) = x, so "
                    "enc(A) - enc(B) is a LINEAR combination of the input itself, which the "
                    "single Linear layer already represents. Testing the product A*B on a "
                    "linear baseline needs its own knob and a plain statement that the "
                    "model is no longer linear in its input.")
        self.pair_features = pair_features
        self.head_norm = head_norm
        self.head = head
        self.protein_mode = protein_mode
        self.fusion = fusion
        self.head_block = head_block
        self.twin_edit = twin_edit
        dropout = float(settings["dropout"])
        # The `linear` head drops the encoders entirely: the RAW feature vectors go straight
        # into the one Linear layer, so a branch's "output width" is its input width.
        raw = head_block == HEAD_BLOCK_LINEAR
        ligand_out = ligand_bits if raw else int(settings["ligand_out"])
        edit_out = fragment_bits if raw else int(settings["edit_out"])

        self.ligand_encoder: nn.Module = (
            nn.Identity() if raw
            else build_mlp(ligand_bits, settings["ligand_hidden"], ligand_out, dropout))
        # The edit branch exists in `direct`, in `antisymmetric`, and in `twin` when
        # twin_edit is on (the fourth cell of the table in the module note).
        # `direct_no_edit` exists to separate that branch's effect from the effect of
        # predicting Δ directly.
        self.edit_encoder: nn.Module | None = None
        if head == HEAD_DIRECT or head == HEAD_ANTISYMMETRIC or twin_edit:
            self.edit_encoder = (
                nn.Identity() if raw
                else build_mlp(fragment_bits, settings["edit_hidden"], edit_out, dropout))

        # The token order is also v1's concatenation order, so `concat` reproduces the
        # earlier results exactly. `twin` scores one molecule at a time, so each pass holds
        # ONE ligand and ONE edit.
        single = head in (HEAD_TWIN, HEAD_ANTISYMMETRIC)
        kinds = [KIND_LIGAND] if single else [KIND_LIGAND, KIND_LIGAND]
        # The pair features sit RIGHT AFTER the two ligand tokens and before the edit
        # branch: the order is a convention, but it has to be fixed, because the `linear`
        # head and every checkpoint read the concatenated vector by position. With
        # pair_features='none' the list is unchanged, so an older checkpoint still loads
        # with every width in the right place.
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
            # The `linear` and `antisymmetric` heads do NOT project the protein branch:
            # v1's projection carries a ReLU (see context_pieces), and that ReLU would both
            # break the baseline's linearity and wipe out the negative half of the protein
            # vector. Those two heads take the protein vector as it is, so the branch's
            # width is protein_dim.
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
            # The endpoint is a discrete label, so its embedding goes straight to the
            # shared width and needs no projection.
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

        # The `antisymmetric` head uses no `predictor`: it has two branches of its own, an
        # ODD one on the difference vector and a context-dependent gate (see the module
        # note).
        self.odd_branch: nn.Module | None = None
        self.gate: nn.Module | None = None
        self.predictor: nn.Module | None = None
        if head == HEAD_ANTISYMMETRIC:
            gate_dim = int(settings["antisymmetric_gate_dim"])
            if gate_dim < 1:
                raise ValueError(f"antisymmetric_gate_dim has to be >= 1, got "
                                 f"{gate_dim}")
            difference_dim = ligand_out + (edit_out if self.edit_encoder is not None else 0)
            self.odd_branch = build_odd_mlp(difference_dim,
                                            settings["antisymmetric_hidden"], gate_dim)
            context_dim = endpoint_dim + (0 if protein_mode == PROTEIN_NONE
                                          else int(protein_dim or 0))
            # The gate may be as non-linear as it likes (biases, ReLU, dropout): it does
            # NOT depend on the difference vector, so it cannot affect the other branch's
            # oddness.
            self.gate = build_mlp(context_dim, settings["antisymmetric_gate_hidden"],
                                  gate_dim, dropout)
        elif head_block == HEAD_BLOCK_RESIDUAL:
            self.predictor = ResidualHead(
                width, int(settings["residual_width"]), int(settings["residual_blocks"]),
                float(settings.get("residual_hidden_factor", 1.0)),
                dropout_or_default(settings.get("residual_hidden_dropout"), dropout),
                dropout_or_default(settings.get("residual_dropout"), dropout))
        elif head_block == HEAD_BLOCK_LINEAR:
            # ONE Linear layer, no activation, no dropout. The bias is OFF by default so
            # that this head is the model named on line 135 of the minutes ("a Linear Model
            # with no bias"); switch it back on with model.linear_bias=true for a regression
            # with an intercept.
            self.predictor = nn.Linear(width, 1, bias=bool(settings.get("linear_bias",
                                                                        False)))
        else:
            self.predictor = build_mlp(width, settings["head_hidden"], 1, dropout,
                                       norm=head_norm == HEAD_NORM_LAYER)

    @property
    def uses_edit_branch(self) -> bool:
        """True when the architecture really uses variable_a/variable_b."""
        return self.edit_encoder is not None

    @property
    def antisymmetric_by_construction(self) -> bool:
        """True when f(A, B, P) = -f(B, A, P) holds by construction, for every set of
        weights.

        Written into the manifest to separate "the two-direction error is small because the
        architecture guarantees it" from "the two-direction error is small because the model
        was trained on augmented data" - two different things, and only the first is what
        the 2026-11-09 meeting asked for.
        """
        return self.head in ANTISYMMETRIC_HEADS

    def project(self, kind: str, vector: Tensor) -> Tensor:
        """Put a branch's vector into the shared space; under `concat` it is unchanged."""
        if self.projectors is None:
            return vector
        return self.projectors[kind](vector)

    def context_pieces(self, protein: Tensor | None, endpoint: Tensor) -> list[Tensor]:
        """The parts that do not depend on a molecule, in the order [protein], endpoint."""
        pieces: list[Tensor] = []
        if self.protein_mode != PROTEIN_NONE:
            if protein is None:
                raise ValueError("the protein branch is on but no protein vector arrived")
            if self.protein_projection is not None:
                pieces.append(torch.relu(self.protein_projection(protein)))
            else:
                pieces.append(self.project(KIND_PROTEIN, protein))
        elif protein is not None:
            raise ValueError("protein_mode='none' but a protein vector arrived anyway")
        pieces.append(self.endpoint_embedding(endpoint))
        return pieces

    def pair_pieces(self, hidden_a: Tensor, hidden_b: Tensor) -> list[Tensor]:
        """The appended pair features: [] | [A - B] | [A - B, A * B] (task 3 of the
        2026-11-09 meeting).

        Taken on the ENCODED vectors and not on the raw fingerprints: the difference of two
        Morgan fingerprints is only the set of bits present in A and absent in B (and the
        reverse, with a minus sign), whereas the difference of two post-encoder vectors is a
        difference in the space the network actually predicts from.
        """
        if self.pair_features == PAIR_FEATURES_NONE:
            return []
        pieces = [hidden_a - hidden_b]
        if self.pair_features == PAIR_FEATURES_DIFFERENCE_PRODUCT:
            pieces.append(hidden_a * hidden_b)
        return pieces

    def combine(self, pieces: Sequence[Tensor]) -> Tensor:
        """Fuse the branches into one vector for the prediction head."""
        if len(pieces) != len(self.token_kinds):
            raise ValueError(f"{len(self.token_kinds)} branches are needed, got "
                             f"{len(pieces)}")
        if self.attention is not None:
            return self.attention(torch.stack(list(pieces), dim=1))
        return torch.cat(list(pieces), dim=1)

    def representations(self, ligand_a: Tensor, ligand_b: Tensor, fragment_a: Tensor | None,
                        fragment_b: Tensor | None, protein: Tensor | None,
                        endpoint: Tensor) -> list[Tensor]:
        """The vectors that enter the prediction head: one for a joint head, two (A then B)
        for twin.

        These are the "post-encoder features" used for the t-SNE (task 4, 2026-11-02
        meeting): every branch encoded and fused, immediately before the final MLP. The
        parameters are forward's.
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
        """The two encoded edit-branch vectors, or (None, None) when the architecture has
        no such branch.

        ONE set of weights for F_A and F_B, just as the ligand encoder is shared by A and B:
        that is what makes swapping the pair, under twin and antisymmetric, swap the two
        vectors without changing either value.
        """
        if self.edit_encoder is None:
            return None, None
        if fragment_a is None or fragment_b is None:
            raise ValueError(f"the {self.head!r} architecture needs the fingerprints of "
                             f"variable_a and variable_b")
        return (self.project(KIND_EDIT, self.edit_encoder(fragment_a)),
                self.project(KIND_EDIT, self.edit_encoder(fragment_b)))

    def antisymmetric_parts(self, hidden_a: Tensor, hidden_b: Tensor, edit_a: Tensor | None,
                            edit_b: Tensor | None, context: Sequence[Tensor]
                            ) -> tuple[Tensor, Tensor]:
        """(the odd branch on the difference vector, the context gate) of the
        `antisymmetric` head.

        The difference is taken BEFORE the odd branch: d = [enc(A) - enc(B),
        enc(F_A) - enc(F_B)]. Swapping (A, B) together with (F_A, F_B) flips the sign of d,
        the odd branch flips with it, and the gate does not move.
        """
        if self.odd_branch is None or self.gate is None:
            raise ValueError("this head has no antisymmetric branch")
        pieces = [hidden_a - hidden_b]
        if edit_a is not None and edit_b is not None:
            pieces.append(edit_a - edit_b)
        return self.odd_branch(torch.cat(pieces, dim=1)), self.gate(
            torch.cat(list(context), dim=1))

    def forward(self, ligand_a: Tensor, ligand_b: Tensor, fragment_a: Tensor | None,
                fragment_b: Tensor | None, protein: Tensor | None,
                endpoint: Tensor) -> Tensor:
        """The predicted Δp, shape (batch,).

        ligand_a/ligand_b    : the 0/1 float fingerprints of the two molecules
        fragment_a/fragment_b: the substituent fingerprints (ignored by twin and
                               direct_no_edit)
        protein              : the protein vector, or None when protein_mode='none'
        endpoint             : the endpoint code, as long
        """
        if self.head == HEAD_ANTISYMMETRIC:
            context = self.context_pieces(protein, endpoint)
            edit_a, edit_b = self.edit_vectors(fragment_a, fragment_b)
            odd, gate = self.antisymmetric_parts(
                self.project(KIND_LIGAND, self.ligand_encoder(ligand_a)),
                self.project(KIND_LIGAND, self.ligand_encoder(ligand_b)),
                edit_a, edit_b, context)
            # An inner product: the gate multiplies element-wise and the result is summed.
            # The multiplication preserves `odd`'s oddness because `gate` does not depend on
            # the difference vector.
            return (odd * gate).sum(dim=1)
        vectors = self.representations(ligand_a, ligand_b, fragment_a, fragment_b, protein,
                                       endpoint)
        if self.predictor is None:
            raise ValueError(f"the {self.head!r} head has no prediction head")
        if self.head == HEAD_TWIN:
            score_a, score_b = (self.predictor(vector) for vector in vectors)
            return (score_b - score_a).squeeze(1)
        return self.predictor(vectors[0]).squeeze(1)


def dropout_or_default(value: Any, fallback: float) -> float:
    """A residual block's own dropout value; a negative number means "use model.dropout".

    The config cannot carry None through --set (run_evaluator.apply_overrides accepts
    numbers and strings only), so a negative number is the convention for "not set
    separately". The default of -1 keeps the earlier behaviour: both residual dropouts equal
    model.dropout.
    """
    if value is None:
        return float(fallback)
    number = float(value)
    return float(fallback) if number < 0.0 else number


def architecture_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The stage3.model block without its note keys (*note), for the checkpoint and the
    manifest.

    A checkpoint has to carry its own architecture: eval rebuilds the model from this and
    not from the current config, because otherwise changing the config (to fusion attention,
    say) would make an older checkpoint fail to load or, worse, load into the wrong
    architecture.
    """
    return {key: value for key, value in settings.items() if not key.endswith("note")}


def count_parameters(model: nn.Module) -> dict[str, int]:
    """The total and trainable parameter counts, for the manifest."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def protein_mode_for(configuration: str) -> str:
    """The protein branch mode of a run configuration ('frozen', 'finetune',
    'no_protein')."""
    if configuration not in CONFIGURATION_MODES:
        raise ValueError(f"unknown configuration: {configuration!r}; only "
                         f"{tuple(CONFIGURATION_MODES)}")
    return CONFIGURATION_MODES[configuration]
