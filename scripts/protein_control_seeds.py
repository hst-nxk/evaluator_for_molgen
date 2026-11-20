"""Does the protein branch contribute anything, read against the noise of three pairs.

    python scripts/protein_control_seeds.py runs/evaluator runs/protein_seeds_2011_gpu1.log
    python scripts/protein_control_seeds.py runs/evaluator --pid 12345

Reads the six manifests `scripts/server_run_protein_seeds_2011.sh` writes (`t5r1..t5r3`
with the real protein, `t5s1..t5s3` with it shuffled) and prints, per pair, the two scores
and their difference, then the mean difference and the spread.

WHY A SPREAD. The 2026-11-19 batch measured +0.0027 for shuffling the protein column -
the shuffled run scored HIGHER. With one run per side that number has nothing to be read
against: "the protein contributes nothing" and "the protein contributes less than our
run-to-run noise" are different claims, and only the second one is supported by one pair.
The spread of three differences is what shows that variation. This script therefore refuses to
print a mean without also printing the spread, and it never calls a difference significant
nor a bound: three pairs are descriptive statistics, not a test, and the spread of this
sample is not an upper bound on any effect.

It also checks what the batch IS before putting a name on it: the split the manifests
name has to be PAIR75, the complete pairs have to carry one training seed each (or the
spread across pairs is not a spread across seeds), and pair 1 has to repeat the 2026-11-19
seed and permutation before it is called a reproducibility check of t3a/t3b. A sentence
whose condition does not hold is withheld, not reworded, and the exit code is non-zero.

It also refuses to pair two manifests that are not the two sides of ONE pair. A difference
is the protein's difference only if the seed, the split, the subset, the configuration and
the head all match and exactly one of the two sides was shuffled; otherwise there are two
valid manifests and a meaningless number between them.

Exit 0 when all six manifests are there, 1 when any is missing or carries the wrong
selection rule (then the differences printed are the ones that could be formed), 2 on a
usage error.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_diagnostics_manifests as gate  # noqa: E402

PAIRS = (1, 2, 3)
METRIC = "pairwise_accuracy_macro"

# What the report's two labels CLAIM, and therefore what has to be checked before either
# is printed. The split this batch is about, and the seed and permutation the 2026-11-19
# batch used for t3a/t3b - pair 1 repeats them, which is the only reason it may be called a
# reproduction of that pair. Both names were printed unconditionally until 2026-11-20: two
# manifests on the default split with seed 42 passed every check and were still reported as
# "PAIR75" and as a repetition of t3a/t3b.
EXPECTED_SPLIT = "pair_split_1411.parquet"
BATCH_1911_SEED = 20261023
BATCH_1911_PERMUTE = 20261119


def task_ids(pair: int) -> tuple[str, str]:
    """(real-protein task, shuffled-protein task) of one pair."""
    return f"t5r{pair}", f"t5s{pair}"


def score_of(manifest: Mapping[str, Any], slice_name: str) -> float | None:
    """The one number this comparison is about, or None when the manifest has no such."""
    scores = (manifest.get("validation_metrics") or {}).get(slice_name) or {}
    value = (scores.get("ranking") or {}).get(METRIC)
    return float(value) if isinstance(value, (int, float)) and not isinstance(
        value, bool) else None


def load(runs_dir: Path, task: str, pid: int) -> tuple[dict | None, str]:
    """(manifest, what is wrong with it) - exactly one file, selected the agreed way."""
    found = gate.manifests_for(runs_dir, task, pid)
    if len(found) != 1:
        names = [path.name for path in found] or "none"
        return None, f"expected exactly one manifest, found {len(found)} ({names})"
    try:
        manifest = gate.read(found[0])
    except ValueError as error:
        return None, str(error)
    problems = gate.selection_problems(manifest)
    if problems:
        return None, "; ".join(problems)
    return manifest, ""


def identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What has to match between the two sides of a pair, read from the manifest's own
    top-level fields rather than from `overrides` - a run started from a changed config.json
    carries the value without an override to show it.
    """
    return {"seed": manifest.get("seed"),
            "split": str(manifest.get("partition_override_pairs") or ""),
            "subset": str(manifest.get("train_subset") or ""),
            "permute": manifest.get("protein_permute_seed"),
            "configuration": manifest.get("configuration"),
            "head": manifest.get("head")}


def pair_problems(real: Mapping[str, Any], shuffled: Mapping[str, Any]) -> list[str]:
    """Why these two manifests are not the two sides of one pair.

    A difference is only the protein's difference if everything else about the two runs is
    the same. Two manifests with different seeds, different splits, or with the shuffle on
    (or off) on both sides can each be perfectly valid and still give a number that means
    nothing - and both would pass a check that only asked "are there two scores?".
    """
    left, right = identity(real), identity(shuffled)
    problems = []
    for field in ("seed", "split", "subset", "configuration", "head"):
        if left[field] != right[field]:
            problems.append(f"the two sides differ in {field} "
                            f"({left[field]!r} vs {right[field]!r}), so their difference "
                            f"is not the protein's difference")
    real_permute = left["permute"]
    shuffled_permute = right["permute"]
    if real_permute is None:
        problems.append("the real side records no protein_permute_seed, so it cannot be "
                        "told from a shuffled run (manifests written before 2026-11-19 "
                        "have no such field)")
    elif int(real_permute) >= 0:
        problems.append(f"the real side was itself SHUFFLED "
                        f"(protein_permute_seed={int(real_permute)})")
    if shuffled_permute is None:
        problems.append("the shuffled side records no protein_permute_seed")
    elif int(shuffled_permute) < 0:
        problems.append("the shuffled side was NOT shuffled "
                        f"(protein_permute_seed={int(shuffled_permute)})")
    return problems


def rows(runs_dir: Path, pid: int, slice_name: str) -> list[dict[str, Any]]:
    """One record per pair: both scores, the difference, and what is missing."""
    records = []
    for pair in PAIRS:
        real_task, shuffled_task = task_ids(pair)
        record: dict[str, Any] = {"pair": pair, "problems": []}
        manifests: dict[str, Mapping[str, Any] | None] = {}
        for side, task in (("real", real_task), ("shuffled", shuffled_task)):
            manifest, problem = load(runs_dir, task, pid)
            manifests[side] = manifest
            if manifest is None:
                record["problems"].append(f"{task}: {problem}")
                record[side] = None
                continue
            record[side] = score_of(manifest, slice_name)
            record[f"{side}_seed"] = manifest.get("seed")
            record[f"{side}_permute"] = manifest.get("protein_permute_seed")
            record[f"{side}_split"] = Path(
                str(manifest.get("partition_override_pairs") or "")).name
            if record[side] is None:
                record["problems"].append(
                    f"{task}: no {METRIC} on the {slice_name!r} slice")
        record["difference"] = None
        if manifests["real"] is not None and manifests["shuffled"] is not None:
            mismatches = pair_problems(manifests["real"], manifests["shuffled"])
            record["problems"].extend(f"pair {pair}: {problem}"
                                      for problem in mismatches)
            if not mismatches and record["real"] is not None \
                    and record["shuffled"] is not None:
                record["difference"] = record["shuffled"] - record["real"]
        records.append(record)
    return records


def label_problems(records: list[dict[str, Any]]) -> list[str]:
    """Why the report may not call this batch PAIR75, nor pair 1 a repetition of t3a/t3b.

    `pair_problems` only asks whether the two sides of a pair agree WITH EACH OTHER. Two
    manifests can agree perfectly and still be a different experiment: the default split
    instead of PAIR75, one seed used for all three pairs, or a pair 1 that does not repeat
    the 2026-11-19 configuration. Each of those makes one of the printed sentences false,
    so each is checked here and the sentence is withheld rather than reworded.
    """
    problems = []
    splits = {record.get("real_split") for record in records
              if record.get("real_split") is not None}
    unexpected = sorted(name for name in splits if name != EXPECTED_SPLIT)
    if unexpected:
        problems.append(f"this is not the PAIR75 batch: the manifests name the split(s) "
                        f"{unexpected or ['(none)']}, not {EXPECTED_SPLIT!r}")
    seeds = [record.get("real_seed") for record in records
             if record.get("difference") is not None]
    if len(seeds) > 1 and len(set(seeds)) != len(seeds):
        problems.append(f"the complete pairs do not have one training seed each "
                        f"({seeds}), so the spread across pairs is not the spread across "
                        f"seeds")
    first = next((record for record in records if record["pair"] == 1), None)
    if first is not None and first.get("real") is not None:
        if first.get("real_seed") != BATCH_1911_SEED \
                or first.get("shuffled_permute") != BATCH_1911_PERMUTE:
            problems.append(f"pair 1 does not repeat the 2026-11-19 configuration "
                            f"(seed {first.get('real_seed')}, permutation "
                            f"{first.get('shuffled_permute')}, expected "
                            f"{BATCH_1911_SEED} and {BATCH_1911_PERMUTE}), so it is not a "
                            f"reproducibility check of t3a/t3b")
    return problems


def report(records: list[dict[str, Any]], slice_name: str) -> tuple[bool, list[str]]:
    """(every pair complete, the lines to print)."""
    unverified = label_problems(records)
    where = ("PAIR75" if not any("not the PAIR75 batch" in problem
                                 for problem in unverified) else "the split below")
    lines = [f"Protein control over {len(PAIRS)} pairs on {where}, {METRIC} on the "
             f"{slice_name!r} slice.", "",
             "| Pair | seed | real protein | protein shuffled | difference |",
             "|---|---|---|---|---|"]
    good = True
    differences = []
    reals: list[float] = []   # the real side of the COMPLETE pairs, in the same order
    for record in records:
        if record["difference"] is None:
            good = False
            cells = [("-" if record[side] is None else f"{record[side]:.4f}")
                     for side in ("real", "shuffled")]
            lines.append(f"| {record['pair']} | {record.get('real_seed', '-')} | "
                         f"{cells[0]} | {cells[1]} | - |")
            continue
        differences.append(record["difference"])
        reals.append(record["real"])
        lines.append(f"| {record['pair']} | {record.get('real_seed', '-')} | "
                     f"{record['real']:.4f} | {record['shuffled']:.4f} | "
                     f"{record['difference']:+.4f} |")
    lines.append("")
    if not differences:
        lines.append("No pair is complete, so there is no difference to read and no "
                     "spread to read it against.")
    else:
        mean = statistics.fmean(differences)
        spread = max(differences) - min(differences)
        lines.append(f"Mean difference (shuffled - real): {mean:+.4f} over "
                     f"{len(differences)} pair(s).")
        lines.append(f"Spread of the differences: {spread:.4f} "
                     f"(from {min(differences):+.4f} to {max(differences):+.4f}).")
        if len(differences) > 1:
            lines.append(f"Standard deviation of the differences: "
                         f"{statistics.stdev(differences):.4f}.")
            # The SAME statistics for one side on its own, which is the scale the mean
            # difference has to be read against: how much the plain number moves when
            # only the training seed changes. The paired design exists to make the
            # difference steadier than either side, and printing both is what shows
            # whether it did. Computed here so no number in a report is arithmetic
            # somebody did by hand over the rows of the table above.
            real_only = reals
            lines.append(f"For scale, the REAL side alone across these "
                         f"{len(real_only)} seeds: spread "
                         f"{max(real_only) - min(real_only):.4f} "
                         f"(from {min(real_only):.4f} to {max(real_only):.4f}), "
                         f"standard deviation {statistics.stdev(real_only):.4f}.")
            lines.append("")
            # Descriptive statistics and nothing more. An earlier version of this script
            # printed that the effect was "bounded above by the spread", which does not
            # follow: the spread of three differences is a property of this sample, not a
            # bound on any effect. What can be said is what the four numbers are.
            lines.append(f"These are descriptive statistics over {len(differences)} "
                         f"pairs on {where}: the mean difference is {mean:+.4f} and the "
                         f"pair-to-pair spread of that same quantity is {spread:.4f}. "
                         f"That is the comparison to report - a mean of this size next to "
                         f"a spread of that size. It is NOT a bound on the effect, NOT a "
                         f"significance test, and three pairs cannot provide either.")
            lines.append("")
            if not unverified:
                lines.append("Pair 1 repeats the configuration of the 2026-11-19 batch "
                             "(same split, same seed, same permutation), so it is a "
                             "reproducibility check of t3a/t3b - not a fourth, "
                             "independent pair. Do not add t3a/t3b to this table as "
                             "another row.")
        else:
            lines.append("One pair only: this is the number that has no noise scale yet, "
                         "which is the reason this batch exists.")
    if unverified:
        good = False
        lines.append("")
        lines.append("WHAT THIS BATCH IS COULD NOT BE CONFIRMED FROM THE MANIFESTS:")
        lines.extend(f"  {problem}" for problem in unverified)
        lines.append("  Until these hold, the numbers above are a protein control of "
                     "whatever runs these manifests describe - not the PAIR75 control "
                     "this script is named after, and not something to put next to "
                     "t3a/t3b.")
    problems = [problem for record in records for problem in record["problems"]]
    if problems:
        good = False
        lines.append("")
        lines.append("Incomplete:")
        lines.extend(f"  {problem}" for problem in problems)
    return good, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise the three-pair protein control with its noise scale.")
    parser.add_argument("runs_dir", help="the directory holding the manifests")
    parser.add_argument("log", nargs="?", help="the batch's log, to read its pid from")
    parser.add_argument("--pid", type=int, help="the batch's pid, instead of its log")
    parser.add_argument("--slice", default=gate.WANTED_SLICE,
                        help=f"the slice to read (default {gate.WANTED_SLICE})")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"{runs_dir} is not a directory", file=sys.stderr)
        return 2
    if (args.log is None) == (args.pid is None):
        print("pass either the batch's log or --pid, not both and not neither",
              file=sys.stderr)
        return 2
    if args.pid is not None:
        pid = args.pid
    else:
        log = Path(args.log)
        if not log.is_file():
            print(f"{log} does not exist", file=sys.stderr)
            return 2
        try:
            pid = gate.pid_from_log(log.read_text(encoding="utf-8", errors="replace"))
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

    good, lines = report(rows(runs_dir, pid, args.slice), args.slice)
    for line in lines:
        print(line)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
