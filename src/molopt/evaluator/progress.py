"""One line per measurement, written the moment it is measured.

WHY THIS EXISTS. A training run used to publish nothing until it finished: the manifest,
with its whole `history`, is written at the end, and everything in between lived only in
the log as prose. So a run in flight could not be plotted, and the only way to read its
shape was to scroll a log. This writes each measurement as a JSON line and flushes it, so
the file is complete up to the last measurement at every instant - `tail` sees it, a
figure can be drawn from it while the run continues, and five lines of it carry the
numbers that a page of log text carries.

The stream is the single source the renderers sit on: `scripts/plot_progress.py` draws
from it, and Weights & Biases - when it is switched on - is handed the same records. That
ordering is deliberate. The file is written first and never depends on W&B, so a run's
progress survives a missing package, a bad key, or a decision later not to use W&B at all.

WEIGHTS & BIASES, and what leaves the machine. `stage3.train.wandb.mode` is the only place
this changes, and the project ships `"online"` by its owner's decision of 2026-11-20:
metrics and the run configuration are uploaded to wandb.ai while the run proceeds, which
is what makes the charts live. `"offline"` writes into `wandb/` and sends nothing, but it
has no viewer - offline data becomes a chart only after `wandb sync`, which is an upload
and needs an account, so offline is a staging step and not a way to see anything.
`"disabled"` keeps the JSON stream alone, which every figure in this project is drawn from
anyway.

WHAT IS UPLOADED in online mode: the metrics of each measurement, the run configuration
this module is handed (seed, learning rate, batch size, split FILE NAMES, row counts,
configuration and head), and the run name. No molecules, no labels, no SMILES, no data
rows. On top of that, W&B collects host and hardware metadata of its own. The documented
switches for the parts that can be refused are set here - code and git capture, and
console capture, which would otherwise ship the training log with its absolute paths -
and `extra_environment` in the config takes any further variable without a code change.
System metrics (host name, GPU model, OS) are NOT claimed to be blocked: W&B gathers them
and this module does not pretend otherwise.

NOTHING HERE MAY KILL A RUN. A logging stream that raises takes a GPU batch with it, so
every failure - the directory, the file, the import, the W&B handshake - is caught, warned
about once, and the run continues. A measurement that was not recorded is a lost figure;
a measurement that crashed the process is a lost batch.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path, PurePath
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

OFFLINE = "offline"
ONLINE = "online"
DISABLED = "disabled"
MODES = (OFFLINE, DISABLED, ONLINE)
STREAM_NAME = "progress"

# Documented W&B switches for the metadata this project does not want shipped. CONSOLE is
# the one that matters most in practice: left on, W&B captures stdout, and the training log
# carries absolute paths of the data and the checkpoints.
BATCH_SUFFIX = re.compile(r"-(\d+)$")

PRIVACY_ENVIRONMENT = {
    "WANDB_DISABLE_CODE": "true",
    "WANDB_DISABLE_GIT": "true",
    "WANDB_CONSOLE": "off",
}


def flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """The scalars of one record, as `validation/regression/mae` style keys.

    W&B charts scalars, and a record is nested two levels deep. Anything that is not a
    number is dropped rather than stringified: a chart of a string is not a chart, and an
    UNDEFINED metric must stay absent instead of arriving as 0.
    """
    flat: dict[str, float] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}/"))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


class ProgressStream:
    """Append-only JSON lines, plus W&B when it is configured and importable."""

    def __init__(self, path: Path | None, run=None) -> None:
        self.path = path
        self.run = run
        self.count = 0
        self._warned = False

    # ------------------------------------------------------------------ construction
    @classmethod
    def open(cls, runs_dir: Path, stamp: str, settings: Mapping[str, Any] | None = None,
             metadata: Mapping[str, Any] | None = None) -> "ProgressStream":
        """A stream for one run; never raises.

        `settings` is `stage3.train.wandb` (absent or `{"mode": "disabled"}` switches W&B
        off and leaves the file). `metadata` is recorded as the W&B run config, so it must
        hold the run's own settings and nothing about the person running it.
        """
        path = None
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            path = runs_dir / f"{STREAM_NAME}_{stamp}.jsonl"
            path.write_text("", encoding="utf-8")
        except OSError as error:
            LOGGER.warning("không mở được dòng tiến trình (%s); lần chạy vẫn tiếp tục",
                           error)
            path = None
        return cls(path, _wandb_run(settings, stamp, metadata))

    # ------------------------------------------------------------------ use
    def record(self, payload: Mapping[str, Any]) -> None:
        """One measurement. Flushed, so a reader sees it before the next one."""
        self.count += 1
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
            except OSError as error:
                self._warn(f"không ghi được dòng tiến trình: {error}")
        if self.run is not None:
            try:
                self.run.log(flatten(payload), step=self.count)
            except Exception as error:   # noqa: BLE001 - a chart must not kill a batch
                self._warn(f"wandb.log thất bại: {error}")
                self.run = None

    def close(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as error:   # noqa: BLE001
                self._warn(f"wandb.finish thất bại: {error}")
            self.run = None

    def _warn(self, message: str) -> None:
        """Warn once per stream: one failing write usually means every write fails, and a
        warning per measurement would bury the training log it is printed into."""
        if not self._warned:
            self._warned = True
            LOGGER.warning("%s; lần chạy vẫn tiếp tục và manifest cuối vẫn đầy đủ",
                           message)


def file_name(value: Any) -> str:
    """The base name of a path-like setting, for a run configuration that leaves the machine.

    A split is identified by its file name; the directories above it are the machine's own
    business. The project's scripts pass relative paths, but a run started by hand with an
    absolute path would otherwise upload the whole tree - user name, project name - to
    W&B. Non-paths come back unchanged, an empty value as "".
    """
    text = str(value or "")
    if not text:
        return ""
    return PurePath(text.replace("\\", "/")).name or text


def batch_group(stamp: str) -> str | None:
    """The batch a run belongs to, from its stamp, or None.

    A stamp is `<date>_<time>_<task>-<pid>` because the batch passes `--run-id`, so the
    pid is the one thing every task of one batch shares. Using it as the W&B group puts
    the eleven runs of a diagnostics batch together instead of eleven unrelated rows, and
    it needs nothing from the caller. A stamp without that suffix (a hand-started run)
    simply has no group.
    """
    found = BATCH_SUFFIX.search(stamp)
    return f"batch-{found.group(1)}" if found else None


def _wandb_run(settings: Mapping[str, Any] | None, stamp: str,
               metadata: Mapping[str, Any] | None):
    """A W&B run, or None. Never raises, and never goes online by accident."""
    if not settings:
        return None
    mode = str(settings.get("mode") or OFFLINE)
    if mode == DISABLED or not bool(settings.get("enabled", True)):
        return None
    if mode not in MODES:
        LOGGER.warning("wandb.mode %r không hợp lệ (chỉ %s); bỏ qua wandb", mode,
                       ", ".join(MODES))
        return None
    try:
        import wandb
    except Exception as error:   # noqa: BLE001 - any import problem is non-fatal
        LOGGER.warning("không import được wandb (%s); chỉ ghi dòng tiến trình ra file",
                       error)
        return None
    # Set through the environment as well as the argument: wandb reads WANDB_MODE on its
    # own in places, and an 'offline' run that silently became online would ship the data
    # this project is not allowed to ship.
    os.environ["WANDB_MODE"] = mode
    # extra_environment FIRST and the privacy switches LAST. The other order let a config
    # entry - or a typo in one - turn console or code capture back on, which is the one
    # thing this block exists to prevent; console capture would ship the training log with
    # its absolute paths. A variable that must override one of the three therefore cannot
    # be set from the config at all, which is the intended answer.
    for name, value in (settings.get("extra_environment") or {}).items():
        if str(name) in PRIVACY_ENVIRONMENT:
            LOGGER.warning("extra_environment.%s bị bỏ qua: %s là cờ riêng tư bắt buộc "
                           "và luôn được đặt là %r", name, name,
                           PRIVACY_ENVIRONMENT[str(name)])
            continue
        os.environ[str(name)] = str(value)
    # Applied in every mode, so switching to offline and back cannot leave capture on.
    for name, value in PRIVACY_ENVIRONMENT.items():
        os.environ[name] = value
    # The W&B version goes into the run's own config: a chart somebody opens in a year
    # should say which client produced it.
    config = dict(metadata or {})
    config["wandb_version"] = getattr(wandb, "__version__", "unknown")
    try:
        run = wandb.init(project=str(settings.get("project") or "molopt-stage3"),
                         name=str(settings.get("run_name") or stamp),
                         group=str(settings.get("group") or "") or batch_group(stamp),
                         mode=mode, config=config,
                         dir=str(settings.get("dir") or "") or None,
                         reinit=True)
        LOGGER.info("wandb: chế độ %s, run %s, client %s", mode,
                    getattr(run, "name", "?"), config["wandb_version"])
        if mode == ONLINE:
            LOGGER.info("wandb ONLINE: số đo và cấu hình lần chạy được tải lên wandb.ai "
                        "trong lúc chạy (không có dữ liệu phân tử, không có nhãn)")
        return run
    except Exception as error:   # noqa: BLE001
        # Online mode fails for ordinary reasons - no `wandb login` on this machine, no
        # route out - and a run must not die of it. The stream still has every number.
        LOGGER.warning("wandb.init thất bại (%s); chỉ ghi dòng tiến trình ra file. "
                       "Nếu là chế độ online thì chạy `wandb login` trên máy này trước.",
                       error)
        return None
