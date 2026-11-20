"""Print the checkpoint path of ONE train run, named by its run id.

Why this exists. A batch script that wants to score or diagnose the checkpoint of the run
it has just finished has to name that checkpoint, and every candidate shortcut is wrong:

* "the newest .pt in runs/evaluator" can belong to a run that died halfway (`--mode train`
  saves its best checkpoint DURING the run), or to the previous task when this one died
  before writing anything;
* "the newest manifest since I started" still loses a race between two batches sharing the
  directory: job A writes its manifest and exits 0, job B writes its own a moment later,
  and A's next step then opens B's manifest. With the same split and shuffle seed every
  check inside the loader passes, so nothing announces it.

So the caller gives each train run a unique `--run-id`, which lands in the name of both the
manifest and the checkpoint (see `run_stamp` in run_evaluator.py), and this script opens
the manifest carrying that id:

    python scripts/pick_train_checkpoint.py <runs_dir> <run id>

Exit code 0 and the path on stdout, or exit code 1 and a reason on stderr. A manifest is
written only when the run reaches the end, so its existence is also the proof that the run
finished; the caller is still expected to check the train command's exit code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def pick(runs_dir: Path, run_id: str) -> Path:
    """Checkpoint of the train run whose files carry ``run_id``.

    Raises FileNotFoundError when no manifest carries that id, when more than one does
    (which would mean the id was not unique, and guessing is worse than stopping), or when
    the checkpoint the manifest names is missing.
    """
    if not run_id:
        raise FileNotFoundError("no run id given")
    manifests = sorted(runs_dir.glob(f"train_*_{run_id}.json"))
    if not manifests:
        raise FileNotFoundError(
            f"no train manifest with run id {run_id!r} in {runs_dir}: that run wrote no "
            f"manifest, so it did not finish")
    if len(manifests) > 1:
        names = ", ".join(path.name for path in manifests)
        raise FileNotFoundError(f"run id {run_id!r} is not unique, it matches {names}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    checkpoint = manifest.get("checkpoint")
    if not checkpoint:
        raise FileNotFoundError(f"{manifests[0]} records no checkpoint")
    path = Path(checkpoint)
    if not path.is_absolute():
        path = runs_dir.parent.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"{manifests[0]} names {checkpoint}, which does not exist")
    return path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python scripts/pick_train_checkpoint.py <runs_dir> <run id>",
              file=sys.stderr)
        return 2
    try:
        print(pick(Path(argv[1]), argv[2]))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"no usable checkpoint: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
