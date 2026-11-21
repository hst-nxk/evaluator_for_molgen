"""`molopt.evaluator.progress` and `scripts/plot_progress.py`: the shape of a run while
it is still running.

Two properties matter more than the rest and are tested hardest. One: the stream must be
readable the instant a measurement happens, because that is the whole reason it exists.
Two: nothing in it may ever take a training run down - a lost line is a lost figure, a
raised exception is a lost GPU batch.

Weights & Biases is OFFLINE by default and must never become online by itself; a fake
module stands in for the package so the tests neither need it installed nor touch a
network.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molopt.evaluator import progress  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "plot_progress", ROOT / "scripts" / "plot_progress.py")
assert SPEC is not None and SPEC.loader is not None
plot_progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_progress)


def record(epoch: int, step: int | None, mae: float, pairwise: float,
           loss: float | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch, "step": step,
        "validation": {"regression": {"n": 24946, "mae": mae},
                       "ranking": {"pairwise_accuracy_macro": pairwise}},
        "validation_selection": {"regression": {"n": 24946, "mae": mae},
                                 "ranking": {"pairwise_accuracy_macro": pairwise + 0.01}},
        "seconds": 9.6}
    if loss is not None:
        payload["train_loss"] = loss
    return payload


# ------------------------------------------------------------------ the stream itself
def test_a_line_is_readable_the_moment_it_is_written(tmp_path: Path) -> None:
    """The reason the stream exists. A reader must see measurement 1 before measurement 2
    is taken - otherwise a run in flight still publishes nothing.
    """
    stream = progress.ProgressStream.open(tmp_path, "20261120_121104")
    stream.record(record(0, 200, 0.60, 0.70))
    assert stream.path is not None
    lines = stream.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["step"] == 200
    stream.record(record(0, None, 0.58, 0.72, loss=0.30))
    assert len(stream.path.read_text(encoding="utf-8").splitlines()) == 2


def test_the_stream_is_named_after_the_run(tmp_path: Path) -> None:
    stream = progress.ProgressStream.open(tmp_path, "20261120_121104")
    assert stream.path is not None
    assert stream.path.name == "progress_20261120_121104.jsonl"


def test_an_old_file_is_not_appended_to(tmp_path: Path) -> None:
    # Two runs could share a stamp only by accident, but a stream that silently continued
    # someone else's file would plot two runs as one curve.
    (tmp_path / "progress_s.jsonl").write_text("{\"epoch\": 99}\n", encoding="utf-8")
    stream = progress.ProgressStream.open(tmp_path, "s")
    assert stream.path is not None
    assert stream.path.read_text(encoding="utf-8") == ""


def test_an_unwritable_directory_does_not_raise(tmp_path: Path, monkeypatch,
                                                caplog) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", refuse)
    with caplog.at_level(logging.WARNING):
        stream = progress.ProgressStream.open(tmp_path / "nope", "s")
    assert stream.path is None
    # And it stays usable: every later call is a no-op rather than an exception.
    stream.record(record(0, None, 0.6, 0.7))
    stream.close()
    assert any("tiếp tục" in message for message in caplog.messages)


def test_a_write_that_fails_warns_once_and_keeps_going(tmp_path: Path, monkeypatch,
                                                       caplog) -> None:
    stream = progress.ProgressStream.open(tmp_path, "s")

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", refuse)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            stream.record(record(0, None, 0.6, 0.7))
    # Five failures, one warning: a warning per measurement would bury the training log.
    assert len([message for message in caplog.messages if "disk full" in message]) == 1
    assert stream.count == 5


# ------------------------------------------------------------------ flattening for charts
def test_nested_scores_become_chart_keys() -> None:
    flat = progress.flatten(record(1, 400, 0.55, 0.75, loss=0.2))
    assert flat["validation/regression/mae"] == pytest.approx(0.55)
    assert flat["validation/ranking/pairwise_accuracy_macro"] == pytest.approx(0.75)
    assert flat["train_loss"] == pytest.approx(0.2)
    assert flat["epoch"] == 1.0 and flat["step"] == 400.0


def test_an_undefined_metric_is_absent_rather_than_zero() -> None:
    # A chart point at 0 for a metric that was UNDEFINED is a false reading of the run.
    flat = progress.flatten({"validation": {"ranking": {"pairwise_accuracy_macro": None}},
                             "epoch": 0})
    assert "validation/ranking/pairwise_accuracy_macro" not in flat
    assert flat == {"epoch": 0.0}


def test_strings_and_flags_are_dropped() -> None:
    flat = progress.flatten({"note": "epoch 3", "stopped": True, "mae": 0.5})
    assert flat == {"mae": 0.5}


# ------------------------------------------------------------------ W&B
class FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, values: dict[str, float], step: int | None = None) -> None:
        self.logged.append((values, step or 0))

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self, run: FakeRun | None = None, fail: bool = False) -> None:
        self.run = run or FakeRun()
        self.fail = fail
        self.kwargs: dict[str, Any] = {}

    def init(self, **kwargs: Any) -> FakeRun:
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("no api key")
        return self.run


@pytest.fixture()
def fake(monkeypatch) -> FakeWandb:
    module = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return module


def test_the_mode_asked_for_is_the_mode_requested_and_exported(
        fake: FakeWandb, tmp_path: Path) -> None:
    """The mode is passed to wandb.init AND written into the environment, because wandb
    reads WANDB_MODE on its own in places; a run that silently changed mode would either
    ship data it should not or fail to ship the charts that were asked for.
    """
    import os
    stream = progress.ProgressStream.open(
        tmp_path, "s", {"enabled": True, "mode": "online", "project": "molopt-stage3"},
        {"seed": 20261023})
    assert fake.kwargs["mode"] == "online"
    assert fake.kwargs["project"] == "molopt-stage3"
    assert os.environ["WANDB_MODE"] == "online"
    assert fake.kwargs["config"]["seed"] == 20261023
    stream.record(record(0, None, 0.6, 0.7))
    assert fake.run.logged[0][0]["validation/regression/mae"] == pytest.approx(0.6)
    stream.close()
    assert fake.run.finished

    progress.ProgressStream.open(tmp_path, "s2", {"mode": "offline"})
    assert fake.kwargs["mode"] == "offline"
    assert os.environ["WANDB_MODE"] == "offline"


def test_the_privacy_switches_are_forced_in_every_mode(fake: FakeWandb, tmp_path: Path,
                                                       monkeypatch) -> None:
    """Console capture is the one that matters: left on, W&B captures stdout, and the
    training log carries the absolute paths of the data and the checkpoints. Code and git
    capture go the same way. Forced in every mode, so switching to offline and back
    cannot leave one on, and an inherited 'true'/'on' from the shell cannot survive.
    """
    import os
    monkeypatch.setenv("WANDB_CONSOLE", "on")
    monkeypatch.setenv("WANDB_DISABLE_CODE", "false")
    for mode in ("online", "offline"):
        progress.ProgressStream.open(tmp_path, f"s-{mode}", {"mode": mode})
        assert os.environ["WANDB_CONSOLE"] == "off"
        assert os.environ["WANDB_DISABLE_CODE"] == "true"
        assert os.environ["WANDB_DISABLE_GIT"] == "true"


def test_extra_environment_from_the_config_is_applied(fake: FakeWandb,
                                                      tmp_path: Path) -> None:
    # So a further WANDB_* switch needs a config edit, not a code change.
    import os
    progress.ProgressStream.open(
        tmp_path, "s", {"mode": "online",
                        "extra_environment": {"WANDB_SILENT": "true"}})
    assert os.environ["WANDB_SILENT"] == "true"


def test_the_wandb_client_version_is_recorded(fake: FakeWandb, tmp_path: Path) -> None:
    # A chart opened in a year should say which client produced it.
    fake_module = sys.modules["wandb"]
    fake_module.__version__ = "0.0.0-test"   # type: ignore[attr-defined]
    progress.ProgressStream.open(tmp_path, "s", {"mode": "online"}, {"seed": 1})
    assert fake.kwargs["config"]["wandb_version"] == "0.0.0-test"


def test_a_missing_login_is_a_warning_that_names_the_fix(tmp_path: Path, monkeypatch,
                                                         caplog) -> None:
    """Online fails for ordinary reasons - nobody ran `wandb login` on this machine, or
    there is no route out. The run must continue, and the warning has to say what to do.
    """
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(fail=True))
    with caplog.at_level(logging.WARNING):
        stream = progress.ProgressStream.open(tmp_path, "s", {"mode": "online"})
    assert stream.run is None
    assert any("wandb login" in message for message in caplog.messages)
    # And every number is still on disk.
    stream.record(record(0, None, 0.6, 0.7))
    assert stream.path is not None
    assert len(stream.path.read_text(encoding="utf-8").splitlines()) == 1


def test_an_inherited_mode_does_not_override_the_config(fake: FakeWandb, tmp_path: Path,
                                                        monkeypatch) -> None:
    # The config decides, not whatever WANDB_MODE the shell happened to carry.
    import os
    monkeypatch.setenv("WANDB_MODE", "online")
    progress.ProgressStream.open(tmp_path, "s", {"mode": "offline"})
    assert os.environ["WANDB_MODE"] == "offline"
    assert fake.kwargs["mode"] == "offline"


def test_disabled_and_false_both_mean_no_wandb(fake: FakeWandb, tmp_path: Path) -> None:
    assert progress.ProgressStream.open(tmp_path, "s", {"mode": "disabled"}).run is None
    assert progress.ProgressStream.open(
        tmp_path, "s", {"enabled": False, "mode": "offline"}).run is None
    # And the file is still written: every figure is drawn from it, not from W&B.
    stream = progress.ProgressStream.open(tmp_path, "s2", {"mode": "disabled"})
    stream.record(record(0, None, 0.6, 0.7))
    assert stream.path is not None and stream.path.read_text(encoding="utf-8")


def test_no_settings_at_all_means_no_wandb(fake: FakeWandb, tmp_path: Path) -> None:
    assert progress.ProgressStream.open(tmp_path, "s").run is None
    assert progress.ProgressStream.open(tmp_path, "s", {}).run is None


def test_an_unknown_mode_is_refused_rather_than_guessed(fake: FakeWandb, tmp_path: Path,
                                                        caplog) -> None:
    with caplog.at_level(logging.WARNING):
        stream = progress.ProgressStream.open(tmp_path, "s", {"mode": "ofline"})
    assert stream.run is None
    assert any("không hợp lệ" in message for message in caplog.messages)


def test_a_missing_package_is_a_warning_not_a_failed_run(tmp_path: Path, monkeypatch,
                                                         caplog) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)   # import raises
    with caplog.at_level(logging.WARNING):
        stream = progress.ProgressStream.open(tmp_path, "s", {"mode": "offline"})
    assert stream.run is None
    assert stream.path is not None
    stream.record(record(0, None, 0.6, 0.7))
    assert stream.path.read_text(encoding="utf-8")


def test_a_failing_init_is_a_warning_not_a_failed_run(tmp_path: Path, monkeypatch,
                                                      caplog) -> None:
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(fail=True))
    with caplog.at_level(logging.WARNING):
        stream = progress.ProgressStream.open(tmp_path, "s", {"mode": "offline"})
    assert stream.run is None


def test_a_failing_log_drops_wandb_and_keeps_the_file(tmp_path: Path, monkeypatch,
                                                      caplog) -> None:
    class Breaking(FakeRun):
        def log(self, values: dict[str, float], step: int | None = None) -> None:
            raise RuntimeError("socket closed")

    module = FakeWandb(run=Breaking())
    monkeypatch.setitem(sys.modules, "wandb", module)
    stream = progress.ProgressStream.open(tmp_path, "s", {"mode": "offline"})
    with caplog.at_level(logging.WARNING):
        stream.record(record(0, None, 0.6, 0.7))
        stream.record(record(1, None, 0.5, 0.8))
    assert stream.run is None
    assert stream.path is not None
    assert len(stream.path.read_text(encoding="utf-8").splitlines()) == 2


# ------------------------------------------------------------------ the config default
def test_the_project_config_ships_online_and_says_what_that_means() -> None:
    """Online was chosen by the project owner on 2026-11-20. The note has to state what
    leaves the machine, and - the part that is easy to overstate - what is NOT blocked.
    """
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    block = config["stage3"]["train"]["wandb"]
    assert block["mode"] == "online"
    assert block["enabled"] is True
    assert block["extra_environment"] == {}
    note = config["stage3"]["train"]["wandb_note"]
    assert "uploads the metrics" in note
    assert "No molecules, no labels, no SMILES" in note
    assert "are NOT blocked" in note          # system metrics; no false comfort
    assert "wandb login" in note
    assert "NO VIEWER" in note                # why offline was not the answer
    assert "never a failed run" in note


# ------------------------------------------------------------------ the renderer
def stream_file(tmp_path: Path, count: int = 6) -> Path:
    path = tmp_path / "progress_20261120_121104.jsonl"
    lines = []
    for index in range(count):
        step = None if index % 3 == 2 else 200 * (index % 3 + 1)
        lines.append(json.dumps(record(index // 3, step, 0.60 - 0.01 * index,
                                       0.70 + 0.01 * index,
                                       loss=0.3 if step is None else None)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_figure_is_drawn_from_the_stream(tmp_path: Path) -> None:
    out = tmp_path / "progress.png"
    assert plot_progress.draw(stream_file(tmp_path), out, "strict") == 0
    assert out.is_file() and out.stat().st_size > 0


def test_a_half_written_last_line_is_ignored(tmp_path: Path) -> None:
    """The run writes while this reads, so a truncated final line is normal. It must be
    skipped, not treated as corruption - it will be complete at the next redraw.
    """
    path = stream_file(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"epoch": 2, "step": 200, "valid')
    assert len(plot_progress.records(path)) == 6
    assert plot_progress.draw(path, tmp_path / "x.png", "strict") == 0


def test_an_empty_stream_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "progress_empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert plot_progress.draw(path, tmp_path / "x.png", "strict") == 1
    assert not (tmp_path / "x.png").exists()


def test_nothing_in_an_unfinished_run_is_called_a_selection(tmp_path: Path) -> None:
    source = (ROOT / "scripts" / "plot_progress.py").read_text(encoding="utf-8")
    # The sentence is wrapped across source lines, so the distinctive part is matched.
    assert "point here is a selection" in source
    assert "Quote the figure drawn" in source and "not this one" in source
    # The best measurement so far is marked, and it is not the run's selection rule.
    best = plot_progress.best_so_far(plot_progress.records(stream_file(tmp_path)),
                                     "strict")
    assert best is not None and best[0] == 5


def test_the_latest_stream_of_a_directory_is_found(tmp_path: Path) -> None:
    import os
    first = tmp_path / "progress_a.jsonl"
    first.write_text("{}\n", encoding="utf-8")
    second = tmp_path / "progress_b.jsonl"
    second.write_text("{}\n", encoding="utf-8")
    os.utime(first, (1, 1))
    assert plot_progress.latest(tmp_path) == second
    assert plot_progress.latest(tmp_path / "empty" if False else tmp_path) is not None


def test_a_directory_without_a_stream_exits_two(tmp_path: Path) -> None:
    empty = tmp_path / "runs"
    empty.mkdir()
    assert plot_progress.main(["--latest", str(empty)]) == 2


def test_no_argument_at_all_exits_two() -> None:
    assert plot_progress.main([]) == 2


# ------------------------------------------------------------------ grouping a batch
def test_the_runs_of_one_batch_share_a_group() -> None:
    """A stamp is `<date>_<time>_<task>-<pid>` because the batch passes --run-id, so the
    pid is what every task of one batch shares. As the W&B group it puts the eleven runs
    of a diagnostics batch together instead of eleven unrelated rows.
    """
    assert progress.batch_group("20261120_092300_t3a-4174688") == "batch-4174688"
    assert progress.batch_group("20261120_092637_t3b-4174688") == "batch-4174688"
    # Two different batches do not merge.
    assert progress.batch_group("20261120_121104_t5r1-3597805") == "batch-3597805"


def test_a_hand_started_run_simply_has_no_group() -> None:
    assert progress.batch_group("20261120_092300") is None


def test_the_group_is_passed_to_wandb_and_the_config_can_override_it(
        fake: FakeWandb, tmp_path: Path) -> None:
    progress.ProgressStream.open(tmp_path, "20261120_092300_t3a-4174688",
                                 {"mode": "online"})
    assert fake.kwargs["group"] == "batch-4174688"
    progress.ProgressStream.open(tmp_path, "20261120_092300_t3a-4174688",
                                 {"mode": "online", "group": "ablation-heads"})
    assert fake.kwargs["group"] == "ablation-heads"

# ------------------------------------------- đường dẫn và thứ tự cờ (review 20/11)
def test_only_the_file_name_of_a_path_setting_leaves_the_machine() -> None:
    """Review 20/11: `train_subset` và `partition_override_pairs` được truyền nguyên chuỗi
    vào wandb.init(config=...). Script của dự án truyền đường dẫn tương đối, nhưng ai chạy
    bằng đường dẫn tuyệt đối thì cả cây thư mục - tên người dùng, tên dự án - sẽ lên mạng.
    Tên file là tất cả những gì một biểu đồ cần để biết nó là split nào.
    """
    assert progress.file_name("/home/khainx/work/splits/pair_split_1411.parquet") == \
        "pair_split_1411.parquet"
    assert progress.file_name("E:\\\\eval_training\\\\work\\\\splits\\\\s.parquet") == "s.parquet"
    assert progress.file_name("work/splits/x.parquet") == "x.parquet"
    assert progress.file_name("") == "" and progress.file_name(None) == ""


def test_the_privacy_switches_cannot_be_turned_off_from_the_config(
        fake: FakeWandb, tmp_path: Path, caplog) -> None:
    """Review 20/11: extra_environment được áp dụng SAU ba cờ riêng tư, nên một dòng
    config - hay một lỗi gõ trong đó - bật lại được console hoặc code capture. Console
    capture sẽ đưa cả log train kèm đường dẫn tuyệt đối lên mạng.
    """
    import os
    with caplog.at_level(logging.WARNING):
        progress.ProgressStream.open(
            tmp_path, "s",
            {"mode": "online",
             "extra_environment": {"WANDB_CONSOLE": "on", "WANDB_DISABLE_CODE": "false",
                                   "WANDB_SILENT": "true"}})
    assert os.environ["WANDB_CONSOLE"] == "off"
    assert os.environ["WANDB_DISABLE_CODE"] == "true"
    # Biến không nằm trong ba cờ đó thì vẫn được áp dụng.
    assert os.environ["WANDB_SILENT"] == "true"
    # Và việc bỏ qua phải được nói ra, không im lặng.
    assert any("WANDB_CONSOLE" in message for message in caplog.messages)

# ------------------------------------------- host thật sự (20/11, phát hiện trên server)
def test_the_host_is_recorded_in_the_run_configuration(fake: FakeWandb, tmp_path: Path,
                                                       monkeypatch) -> None:
    """"online" không nói gì nếu không kèm host.

    Trên server của dự án, wandb KHÔNG trỏ về wandb.ai: nó đọc base URL từ
    `WANDB_BASE_URL` hoặc `~/.netrc`, và ở đó nó ra instance của nhà cung cấp compute
    (`api.forge.coreweave.com`). Một biểu đồ mở ra sau một năm phải biết nó nằm ở đâu.
    """
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.forge.coreweave.com")
    progress.ProgressStream.open(tmp_path, "s", {"mode": "online"}, {"seed": 1})
    assert fake.kwargs["config"]["wandb_base_url"] == "https://api.forge.coreweave.com"


def test_an_unset_host_is_said_to_be_unset_not_guessed(fake: FakeWandb, tmp_path: Path,
                                                       monkeypatch) -> None:
    # Không được điền "wandb.ai" vào chỗ mình không biết.
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    progress.ProgressStream.open(tmp_path, "s", {"mode": "online"})
    recorded = fake.kwargs["config"]["wandb_base_url"]
    assert "default" in recorded
    assert "wandb.ai" not in recorded


def test_the_config_can_name_a_host_but_never_overrides_the_machine(
        fake: FakeWandb, tmp_path: Path, monkeypatch) -> None:
    """Máy tự đặt host thì máy thắng: credential nằm ở đó, không nằm trong config.json."""
    import os
    monkeypatch.delenv("WANDB_BASE_URL", raising=False)
    progress.ProgressStream.open(tmp_path, "s",
                                 {"mode": "online", "base_url": "https://wb.example"})
    assert os.environ["WANDB_BASE_URL"] == "https://wb.example"

    monkeypatch.setenv("WANDB_BASE_URL", "https://api.forge.coreweave.com")
    progress.ProgressStream.open(tmp_path, "s",
                                 {"mode": "online", "base_url": "https://wb.example"})
    assert os.environ["WANDB_BASE_URL"] == "https://api.forge.coreweave.com"
    assert fake.kwargs["config"]["wandb_base_url"] == "https://api.forge.coreweave.com"


def test_the_note_names_the_real_host_and_not_only_wandb_ai() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    note = config["stage3"]["train"]["wandb_note"]
    assert "api.forge.coreweave.com" in note
    assert "not necessarily wandb.ai" in note
    # Và vẫn phải giữ những gì đã hứa trước đó.
    assert "No molecules, no labels, no SMILES" in note
    assert "are NOT blocked" in note


def test_a_new_client_is_asked_to_finish_the_previous_run_by_name(
        fake: FakeWandb, tmp_path: Path) -> None:
    """wandb 0.31.1 cảnh báo `reinit=True` đã deprecated, trên cả 9 lượt đêm 20/11."""
    fake_module = sys.modules["wandb"]
    fake_module.__version__ = "0.31.1"   # type: ignore[attr-defined]
    progress.ProgressStream.open(tmp_path, "s", {"mode": "online"})
    assert fake.kwargs["reinit"] == "finish_previous"


def test_an_old_client_still_gets_the_boolean(fake: FakeWandb, tmp_path: Path) -> None:
    # Dạng chuỗi chỉ có ở client mới; gửi nó cho client cũ sẽ hỏng lúc init.
    fake_module = sys.modules["wandb"]
    fake_module.__version__ = "0.16.2"   # type: ignore[attr-defined]
    progress.ProgressStream.open(tmp_path, "s", {"mode": "online"})
    assert fake.kwargs["reinit"] is True


def test_an_unparsable_version_falls_back_to_the_boolean(fake: FakeWandb,
                                                         tmp_path: Path) -> None:
    fake_module = sys.modules["wandb"]
    fake_module.__version__ = "weird-build"   # type: ignore[attr-defined]
    progress.ProgressStream.open(tmp_path, "s", {"mode": "online"})
    assert fake.kwargs["reinit"] is True


def test_the_boundary_of_the_string_form_is_the_patch_number(fake: FakeWandb,
                                                             tmp_path: Path) -> None:
    """0.19.8 chi nhan boolean, 0.19.9 moi nhan chuoi - review 21/11.

    Doc chi so minor la sai: `reinit="finish_previous"` nem loi tren 0.19.8. Bo test nay
    di qua dung ranh gioi do, va them mot ban build co hau to.
    """
    fake_module = sys.modules["wandb"]
    cases = {"0.19.8": True, "0.19.9": "finish_previous", "0.19.10": "finish_previous",
             "0.20.0": "finish_previous", "0.18.9": True, "0.19.9rc1": "finish_previous",
             "0.19": True}
    for version, expected in cases.items():
        fake_module.__version__ = version   # type: ignore[attr-defined]
        progress.ProgressStream.open(tmp_path, "s", {"mode": "online"})
        assert fake.kwargs["reinit"] == expected, version


def test_the_docstring_does_not_claim_a_check_it_does_not_do() -> None:
    # Ban dau docstring viet "hoi client, khong so sanh phien ban" trong khi no so sanh
    # phien ban. Mot cau mo ta sai la mot cai bay cho nguoi doc sau.
    assert "not by comparing versions" not in (progress._reinit.__doc__ or "")
