"""Per-block gradient norms (`--mode diagnose`) and the split lock inside a checkpoint.

Both answer review points raised on 2026-11-19 against the diagnostics of the 2026-11-18
meeting:

1. The ReLU count and the input-sensitivity measurement both run under `torch.no_grad()`,
   so neither could answer "do all 7 residual blocks receive gradient". `gradient_norms`
   does, and these tests pin the two properties that make it readable: a parameter the
   model cannot reach has gradient exactly zero, and the per-group numbers add up to the
   total.
2. A checkpoint recorded the protein-shuffle seed but NOT the split, so re-scoring a
   PAIR75 run without `--set inputs.partition_override_pairs=...` quietly scored the
   default validation partition instead.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="gradient norms need torch")

from molopt.evaluator import data, models, train  # noqa: E402

from test_evaluator_train_loop import (FRAGMENT_BITS, LIGAND_BITS,  # noqa: E402
                                       FakeCaches, PROTEIN_DIM, ROWS, fake_arrays)

SETTINGS = {"ligand_hidden": [16], "ligand_out": 8, "edit_hidden": [8], "edit_out": 4,
            "protein_out": 4, "endpoint_dim": 3, "head_hidden": [8], "dropout": 0.0}


def built(**changes: Any) -> Any:
    torch.manual_seed(0)
    return models.EvaluatorModel({**SETTINGS, **changes}, LIGAND_BITS, FRAGMENT_BITS,
                                 PROTEIN_DIM, 4, models.HEAD_DIRECT, models.PROTEIN_FROZEN)


@pytest.fixture()
def sample() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    arrays, _ = fake_arrays(rng, [{"name": "all"}])
    return {"arrays": arrays, "caches": FakeCaches(rng),
            "protein": np.random.default_rng(1).normal(size=(3, PROTEIN_DIM)
                                                       ).astype(np.float32)}


def measure(model: Any, sample: dict[str, Any], batches: int = 2) -> dict[str, Any]:
    return train.gradient_norms(model, sample["arrays"], sample["caches"],
                                sample["protein"], "cpu", 16,
                                torch.nn.MSELoss(), seed=0, batches=batches,
                                loss_name="mse")


# ------------------------------------------------------------------ grouping


@pytest.mark.parametrize("name,group", (
    ("head.blocks.3.linear.weight", "head.blocks.3"),
    ("head.blocks.0.norm.bias", "head.blocks.0"),
    ("ligand_encoder.0.weight", "ligand_encoder.0"),
    ("protein_projection", "protein_projection"),
    ("endpoint_embedding.weight", "endpoint_embedding"),
))
def test_a_parameter_is_grouped_by_its_first_numbered_component(name: str,
                                                                group: str) -> None:
    # The index of an nn.Sequential IS the block, which is what the meeting asked about.
    assert train.gradient_group(name) == group


# ------------------------------------------------------------------ the measurement


def test_every_parameter_group_of_the_model_is_reported(sample: dict[str, Any]) -> None:
    model = built()
    report = measure(model, sample)
    expected = {train.gradient_group(name)
                for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert {group["group"] for group in report["groups"]} == expected
    assert report["batches"] == 2
    assert sum(group["parameters"] for group in report["groups"]) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_the_groups_add_up_to_the_total(sample: dict[str, Any]) -> None:
    """share_of_total has to be a share: the squares add up, the norms do not."""
    report = measure(model=built(), sample=sample)
    total = sum(group["grad_norm"] ** 2 for group in report["groups"])
    assert total == pytest.approx(report["total_grad_norm"] ** 2, rel=1e-6)
    assert sum(group["share_of_total"] for group in report["groups"]) == pytest.approx(1.0)


def test_a_block_the_model_cannot_reach_has_gradient_exactly_zero(
        sample: dict[str, Any]) -> None:
    """The failure this measurement exists to catch, forced on purpose.

    An extra Linear is attached to the model but never used in `forward`, so it is exactly
    the case a dead branch produces: parameters that are there, that have ordinary
    activations nowhere, and that receive no gradient at all. The ReLU count and the
    sensitivity measurement cannot tell this apart from a healthy block.
    """
    model = built()
    model.detached_branch = torch.nn.Linear(4, 4)
    report = measure(model, sample)
    dead = [group for group in report["groups"] if group["group"] == "detached_branch"]
    assert len(dead) == 1
    assert dead[0]["grad_norm"] == 0.0
    assert dead[0]["zero_fraction"] == 1.0
    assert dead[0]["share_of_total"] == 0.0
    # Every other group still learns, so a zero is a finding and not the normal state.
    assert all(group["grad_norm"] > 0 for group in report["groups"]
               if group["group"] != "detached_branch")


def test_each_residual_block_is_reported_separately(sample: dict[str, Any]) -> None:
    """The meeting's question is about the 7 residual blocks, one by one."""
    model = built(head_block=models.HEAD_BLOCK_RESIDUAL, residual_blocks=7,
                  residual_width=12, residual_hidden_factor=1.5,
                  residual_hidden_dropout=0.0, residual_dropout=0.0)
    groups = {group["group"]: group for group in measure(model, sample)["groups"]}
    blocks = [name for name in groups if ".blocks." in name]
    assert len(blocks) == 7, sorted(groups)
    assert all(groups[name]["grad_norm"] > 0 for name in blocks)


def test_the_measurement_leaves_no_gradient_behind(sample: dict[str, Any]) -> None:
    """A diagnostic must not change the thing it measures: no .grad left on the model."""
    model = built()
    measure(model, sample)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_the_same_seed_gives_the_same_numbers(sample: dict[str, Any]) -> None:
    model = built()
    first = measure(model, sample)
    second = measure(model, sample)
    assert first["total_grad_norm"] == pytest.approx(second["total_grad_norm"])


def test_dropout_is_off_while_measuring(sample: dict[str, Any]) -> None:
    """Measured in eval mode, so repeating the command cannot give a different answer."""
    model = built(dropout=0.9)
    model.train()
    measure(model, sample)
    assert not model.training


@pytest.mark.parametrize("batches", (0, -1))
def test_at_least_one_batch_is_required(sample: dict[str, Any], batches: int) -> None:
    with pytest.raises(ValueError, match="batches must be >= 1"):
        measure(built(), sample, batches=batches)


def test_an_empty_sample_is_refused(sample: dict[str, Any]) -> None:
    from molopt.evaluator import data

    empty = data.take_rows(sample["arrays"], np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="no rows"):
        train.gradient_norms(built(), empty, sample["caches"], sample["protein"], "cpu",
                             16, torch.nn.MSELoss(), seed=0)


def test_it_reads_as_many_rows_as_it_was_given(sample: dict[str, Any]) -> None:
    report = train.gradient_norms(built(), sample["arrays"], sample["caches"],
                                  sample["protein"], "cpu", ROWS, torch.nn.MSELoss(),
                                  seed=0, batches=99)
    # Only one batch of ROWS rows exists, so the budget cannot be filled: the report says
    # how many batches it really averaged over instead of claiming 99.
    assert report["batches"] == 1
    assert report["batch_size"] == ROWS


# ------------------------------------------------------------------ the split lock


def with_inputs(**changes: str) -> dict[str, Any]:
    inputs = {"train_subset": "", "partition_override": "",
              "partition_override_pairs": "", **changes}
    return {"stage3": {"inputs": inputs}}


def test_the_fingerprint_holds_the_three_split_keys_and_nothing_else() -> None:
    fingerprint = train.input_fingerprint(with_inputs(train_subset="a.parquet"))
    assert fingerprint == {"train_subset": "a.parquet", "partition_override": "",
                           "partition_override_pairs": ""}


def test_the_same_split_passes() -> None:
    config = with_inputs(partition_override_pairs="work/splits/pair_split_1411.parquet")
    state = {"inputs": train.input_fingerprint(config)}
    assert train.check_input_split(state, config) == state["inputs"]


def test_a_forgotten_split_flag_is_refused() -> None:
    """THE TRAP: trained on PAIR75 (24,946 validation rows), re-scored on the default
    validation partition (493,852 rows) because one flag was left out."""
    trained = with_inputs(partition_override_pairs="work/splits/pair_split_1411.parquet")
    state = {"inputs": train.input_fingerprint(trained)}
    with pytest.raises(ValueError, match="different split"):
        train.check_input_split(state, with_inputs())
    # The message has to carry the flag that fixes it, not just the complaint.
    with pytest.raises(ValueError,
                       match="--set inputs.partition_override_pairs=work/splits/"):
        train.check_input_split(state, with_inputs())


def test_a_different_subset_is_refused() -> None:
    state = {"inputs": train.input_fingerprint(with_inputs(train_subset="a.parquet"))}
    with pytest.raises(ValueError, match="different split"):
        train.check_input_split(state, with_inputs(train_subset="b.parquet"))


def test_scoring_the_default_partition_without_a_flag_is_still_allowed() -> None:
    state = {"inputs": train.input_fingerprint(with_inputs())}
    assert train.check_input_split(state, with_inputs()) == state["inputs"]


def test_an_old_checkpoint_without_the_key_only_warns(
        caplog: pytest.LogCaptureFixture) -> None:
    """Checkpoints from before 2026-11-19 carry no split; refusing them all would make
    every earlier checkpoint unreadable, so this one warns and says what to check."""
    with caplog.at_level("WARNING"):
        assert train.check_input_split({}, with_inputs(train_subset="a.parquet")) is None
    assert "before 2026-11-19" in caplog.text
    # Review point of 2026-11-19: it must warn when the command sets NO flag either. That
    # is the dangerous case - an old PAIR75 checkpoint scored by a bare command reads the
    # default validation partition - so silence there would clear exactly what nobody can
    # catch by eye.
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert train.check_input_split({}, with_inputs()) is None
    assert "CANNOT be checked" in caplog.text
    assert "no inputs.* flag" in caplog.text


def test_a_checkpoint_carries_the_split_it_was_trained_with(tmp_path, monkeypatch,
                                                            config) -> None:
    """End to end: run_train writes the split into the .pt, and load_checkpoint reads it."""
    import json

    rng = np.random.default_rng(0)
    run_config = json.loads(json.dumps(config))
    run_config["stage3"]["train"].update(
        {"device": "cpu", "epochs": 1, "batch_size": 16, "gpu_memory_limit_mib": None,
         "eval_every_steps": 0, "eval_batch_size": 32})
    run_config["stage3"]["model"].update(SETTINGS)
    run_config["stage3"]["inputs"]["train_subset"] = "work/splits/fake_subset.parquet"
    arrays, keys = fake_arrays(rng, run_config["stage3"]["metrics"]["slices"])
    caches = FakeCaches(rng)
    monkeypatch.setattr(train, "Caches", lambda *args, **kwargs: caches)
    monkeypatch.setattr(train, "prepare_arrays",
                        lambda *args, **kwargs: (arrays, dict(keys), None))
    monkeypatch.setattr(train, "layout", lambda config, root: (tmp_path, tmp_path))
    manifest = train.run_train(run_config, tmp_path, "frozen", "direct", smoke=False,
                               stamp="s")
    state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False)
    assert state["inputs"]["train_subset"] == "work/splits/fake_subset.parquet"
    # And the same config passes its own check, while a bare one does not.
    assert train.check_input_split(state, run_config) == state["inputs"]
    bare = json.loads(json.dumps(run_config))
    bare["stage3"]["inputs"]["train_subset"] = ""
    with pytest.raises(ValueError, match="different split"):
        train.check_input_split(state, bare)



def test_the_report_names_the_loss_it_used(sample: dict[str, Any]) -> None:
    """A gradient norm without the loss behind it cannot be compared with anything."""
    assert measure(built(), sample)["loss"] == "mse"


def test_esm2_is_kept_out_of_the_backward_pass(sample: dict[str, Any]) -> None:
    """Review point of 2026-11-19: under `finetune` the protein vectors come from ESM2, and
    a backward pass through them would hold its activations for every batch while the
    report counts only the evaluator's parameters.

    The stand-in here records whether the tensor it hands back is asked for a gradient: the
    batch is built under torch.no_grad(), so the vectors arrive detached and the graph stops
    at the evaluator.
    """
    class Recording:
        """The interface train.batch_tensors needs of a non-array protein source."""

        def __init__(self, vectors: np.ndarray) -> None:
            self.vectors = vectors
            self.grad_enabled: list[bool] = []

        def train(self, mode: bool) -> None:
            pass

        def encode(self, rows: np.ndarray) -> Any:
            self.grad_enabled.append(torch.is_grad_enabled())
            tensor = torch.from_numpy(self.vectors[rows])
            tensor.requires_grad_(torch.is_grad_enabled())
            return tensor

    source = Recording(sample["protein"])
    report = train.gradient_norms(built(), sample["arrays"], sample["caches"], source,
                                  "cpu", 16, torch.nn.MSELoss(), seed=0, batches=2,
                                  loss_name="mse")
    assert source.grad_enabled and not any(source.grad_enabled)
    # And the evaluator's own parameters still got their gradients.
    assert report["total_grad_norm"] > 0


def test_the_sample_is_measured_in_the_space_it_is_given(sample: dict[str, Any]) -> None:
    """The scale of the labels changes the gradient, which is why run_diagnose has to put
    them back into the space the checkpoint was trained in (`LabelScaler.forward`)."""
    arrays = sample["arrays"]
    scaled = data.PairArrays(
        molecule_a=arrays.molecule_a, molecule_b=arrays.molecule_b,
        fragment_a=arrays.fragment_a, fragment_b=arrays.fragment_b,
        protein=arrays.protein, endpoint=arrays.endpoint,
        target=(arrays.target * 10.0).astype(arrays.target.dtype), slices=arrays.slices)
    model = built()
    raw = measure(model, sample)["total_grad_norm"]
    other = train.gradient_norms(model, scaled, sample["caches"], sample["protein"], "cpu",
                                 16, torch.nn.MSELoss(), seed=0, batches=2,
                                 loss_name="mse")["total_grad_norm"]
    assert raw != pytest.approx(other)


# ----------------------------------------------------------- cửa sổ chuỗi của finetune
# Review 2026-11-19: `--mode train` narrows max_residues through finetune_overrides (512
# instead of the shared 1022), while the re-scoring path rebuilt the encoder from the shared
# config. For a protein longer than 512 residues the same weights then see different input,
# so the numbers reported differ from the ones that selected the checkpoint.


def encoder_config(max_residues: int = 1022) -> dict:
    return {"stage3": {"protein_encoder": {"model": "facebook/esm2_t12_35M_UR50D",
                                           "embedding_dim": 480,
                                           "max_residues": max_residues,
                                           "long_sequence": "windows"}}}


def test_the_window_comes_from_the_checkpoint_not_from_the_config() -> None:
    state = {"protein_encoder_settings": {"model": "facebook/esm2_t12_35M_UR50D",
                                          "max_residues": 512,
                                          "long_sequence": "truncate"}}
    settings = {"finetune_overrides": {"max_residues": 512}}
    chosen = train.finetune_encoder_settings(state, encoder_config(), settings)
    assert chosen["max_residues"] == 512
    # Keys the checkpoint does not carry still come from the config.
    assert chosen["embedding_dim"] == 480


def test_an_old_checkpoint_always_warns_that_the_window_cannot_be_checked(
        caplog: pytest.LogCaptureFixture) -> None:
    settings = {"finetune_overrides": {"max_residues": 512}}
    with caplog.at_level(logging.WARNING):
        chosen = train.finetune_encoder_settings({}, encoder_config(), settings)
    assert chosen["max_residues"] == 1022
    assert chosen["long_sequence"] == "truncate"   # finetune never uses windows
    message = caplog.text
    assert "KHÔNG kiểm được" in message
    # Both numbers are named, so the reader can see how far apart they are.
    assert "1022" in message and "512" in message


def test_the_warning_says_so_even_when_no_override_is_set(
        caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        train.finetune_encoder_settings({}, encoder_config(), {})
    assert "không đặt" in caplog.text


