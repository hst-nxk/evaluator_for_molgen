"""The seen/unseen molecule breakdown across several seeds, with the scale to read it at.

    python scripts/seen_molecules_seeds.py runs/evaluator/seen_molecules_t7s*.json

WHY. The breakdown of one run put the MLP's macro for the group with NO molecule seen in
training at 0.6673 against the best classical baseline's 0.5831 - but that macro averages
215 queries, the thinnest number in the whole table, while the row-level measurement of
the same group (4,268 comparisons) puts the MLP only +0.0150 ahead of xgboost. A margin
taken from 215 queries cannot be quoted until it is known how far that number moves when
nothing changes but the training seed. This reads several per-seed reports and prints the
mean, the spread and the standard deviation of every group, in both units.

It refuses to average reports that are not comparable, and says which: a different split,
a different slice, a different `min_candidates`, a different train dump (the molecule set
"seen in training" would then mean something else), or a report whose own reconstruction
check did not agree. It also refuses what the 2026-11-20 review caught it accepting: the
SAME report twice, or two reports whose validation rows are not provably the same set. A
spread over three copies of one number is zero, and zero spread would have been read as a
rock-solid result. Identity is now proved from the reports themselves - equal
`validation_fingerprint` (a digest of the row keys, not the row count) and DISTINCT
`validation_checkpoint`. One seed is accepted and told it has no scale yet.

`--baseline <report>` is the other half of the review's point: the standard deviation of
the MLP alone is NOT the bar for "the MLP beats lasso". With a classical breakdown over
the same validation rows, this prints the three PAIRED differences MLP(seed i) - baseline
per group and whether they all have the same sign, which is the comparison that actually
carries the claim.

NOT A SIGNIFICANCE TEST. The mean of three seeds beside the spread of three seeds is a
description of three runs; three runs cannot give a confidence interval and this prints
none. What it does give is the comparison that matters: whether a difference between
models is larger or smaller than the movement caused by the seed alone.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

GROUPS = (0, 1, 2)
def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(report: dict[str, Any]) -> dict[str, Any]:
    """What has to MATCH before two reports may be averaged together.

    `validation_fingerprint` is the one that does the work: a digest of the validation row
    keys. `rows` is kept beside it because a report written before the fingerprint existed
    has no digest, and the count is then the only check available - stated as such rather
    than trusted.
    """
    return {"train": report.get("train"), "slice": report.get("slice"),
            "min_candidates": report.get("min_candidates"),
            "direction_tolerance": report.get("direction_tolerance"),
            "rows": report.get("rows"),
            "validation_fingerprint": report.get("validation_fingerprint"),
            # Two digests are only comparable over the same columns.
            "validation_fingerprint_columns":
                report.get("validation_fingerprint_columns")}


def distinctness(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    """Why these reports are not DIFFERENT RUNS AT DIFFERENT SEEDS.

    Two things have to hold, and the 2026-11-20 reviews broke each in turn: the reports
    must not be the same run (the same file twice gave a spread of zero), and they must
    not be the same SEED under two file names (two checkpoints with different names but
    one seed were accepted as two seeds - a rerun or a copy is not a replicate).

    `identity` is used rather than the checkpoint path, so a run that has no checkpoint -
    a classical fit, identified by its source manifest, model and alpha - can be checked
    by the same rule instead of being refused for lacking a `.pt`.
    """
    found = []
    resolved = [path.resolve() for path, _ in reports]
    if len(set(resolved)) != len(resolved):
        found.append("the same report file was passed more than once, so the spread "
                     "would be zero by construction")
    identities = [str(report.get("identity") or "") for _, report in reports]
    if all(identities):
        repeated = {value for value in identities if identities.count(value) > 1}
        for value in sorted(repeated):
            found.append(f"two reports came from the SAME run ({value}), so they are one "
                         f"run scored twice, not two seeds")
    else:
        found.append("some reports do not say which run produced them (written before "
                     "2026-11-20), so being different runs cannot be verified; rerun "
                     "scripts/seen_molecules.py on those dumps")

    seeds_seen = [report.get("validation_seed") for _, report in reports]
    if all(seed is not None for seed in seeds_seen):
        repeated_seeds = {seed for seed in seeds_seen if seeds_seen.count(seed) > 1}
        for seed in sorted(repeated_seeds, key=str):
            found.append(f"two reports were trained at the SAME SEED ({seed}); different "
                         f"checkpoint files of one seed are a rerun, not a replicate")
    else:
        found.append("the seed of at least one run could not be read from its train "
                     "manifest, so a repeated seed would not be noticed; the spread below "
                     "is reported WITHOUT that check")
    return found


def problems(reports: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    """Why this set of reports is not a seed sweep of one measurement."""
    found = []
    for path, report in reports:
        if report.get("stage") != "stage3_seen_molecules":
            found.append(f"{path.name} is not a seen-molecules report")
        if not (report.get("check") or {}).get("agrees", False):
            found.append(f"{path.name}: its own reconstruction check did not agree, so "
                         f"its breakdown must not be quoted, let alone averaged")
    if len(reports) < 2:
        return found
    found += distinctness(reports)
    first_path, first = reports[0]
    for path, report in reports[1:]:
        for key, value in identity(first).items():
            other = identity(report)[key]
            if value != other:
                if key == "validation_fingerprint":
                    found.append(
                        f"{path.name} scored a DIFFERENT SET of validation rows from "
                        f"{first_path.name} (row-key digests differ); a mean over them "
                        f"would mix two measurements")
                    continue
                found.append(f"{path.name} has {key} {other!r} where {first_path.name} "
                             f"has {value!r}; these are not the same measurement at a "
                             f"different seed")
    return found


def series(reports: list[dict[str, Any]], group: int, part: str, key: str
           ) -> list[float]:
    values = []
    for report in reports:
        for block in report.get(part) or []:
            if int(block.get("molecules_seen", -1)) == group:
                value = block.get(key)
                if value is not None:
                    values.append(float(value))
    return values


def describe(values: list[float], digits: int) -> str:
    """mean, spread and sd - never a mean on its own."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f} (one seed: no scale yet)"
    if min(values) == max(values):
        # The row and query counts are the same at every seed - the validation rows do not
        # move - so "6925 +-0 [6925..6925]" is noise around one number.
        return f"{values[0]:.{digits}f} (same at every seed)"
    return (f"{statistics.fmean(values):.{digits}f} "
            f"±{statistics.stdev(values):.{digits}f} "
            f"[{min(values):.{digits}f}..{max(values):.{digits}f}]")


def paired(reports: list[dict[str, Any]], baseline: dict[str, Any],
           baseline_name: str) -> list[str]:
    """MLP(seed i) - baseline, per group, on the same validation rows.

    WHY THIS AND NOT THE SPREAD. The spread of the MLP across seeds says how unstable the
    MLP is; it does not say whether the MLP is ahead. The quantity that carries the claim
    is the DIFFERENCE on the same rows, once per seed, and whether every one of them has
    the same sign. Three differences cannot give a p-value and none is printed.
    """
    out = ["", f"Paired against {baseline_name} "
           f"({baseline.get('identity') or 'unidentified fit'}), on the same validation "
           f"rows:", "",
           "| molecules seen | unit | baseline | MLP - baseline, per seed | mean | "
           "all one sign |", "|---|---|---|---|---|---|"]
    for group in GROUPS:
        for unit, part, key in (("macro", "by_seen_macro", "pairwise_accuracy_macro"),
                                ("micro (rows)", "by_row", "direction_accuracy_micro")):
            reference = series([baseline], group, part, key)
            values = series(reports, group, part, key)
            if not reference or not values:
                out.append(f"| {group} | {unit} | - | - | - | - |")
                continue
            differences = [value - reference[0] for value in values]
            same = (all(difference > 0 for difference in differences)
                    or all(difference < 0 for difference in differences))
            out.append(
                f"| {group} | {unit} | {reference[0]:.4f} | "
                + ", ".join(f"{difference:+.4f}" for difference in differences)
                + f" | {statistics.fmean(differences):+.4f} | "
                + ("yes" if same else "NO") + " |")
    out += ["",
            "A mean difference whose per-seed signs disagree is not a result, whatever "
            "its size. Three differences are a description of three runs: they are not a "
            "confidence interval and not a significance test.",
            "",
            "This comparison is only valid if the baseline scored THESE rows. The "
            "fingerprint check above covers the MLP reports; the baseline's own "
            "fingerprint is checked the same way."]
    return out


def lines(paths: list[Path], reports: list[dict[str, Any]],
          baseline: tuple[Path, dict[str, Any]] | None = None) -> list[str]:
    count = len(reports)
    first = reports[0]
    out = [f"Seen/unseen molecule breakdown across {count} seed(s)",
           f"split/train dump : {first.get('train')}",
           f"slice            : {first.get('slice')} | min_candidates "
           f"{first.get('min_candidates')} | direction tolerance "
           f"{first.get('direction_tolerance')}",
           f"reports          : " + ", ".join(path.name for path in paths), "",
           ("Each cell is mean ±sd [min..max] over the seeds." if count > 1
            else "One run: each cell is that run's own number, with no spread to read it "
                 "against."), "",
           "| molecules seen | rows | queries in macro | pairwise_accuracy_macro | "
           "direction accuracy (micro) |", "|---|---|---|---|---|"]
    for group in GROUPS:
        rows = series(reports, group, "by_seen_macro", "rows")
        ranked = series(reports, group, "by_seen_macro", "queries_ranked")
        macro = series(reports, group, "by_seen_macro", "pairwise_accuracy_macro")
        micro = series(reports, group, "by_row", "direction_accuracy_micro")
        out.append(f"| {group} | {describe(rows, 0)} | {describe(ranked, 0)} | "
                   f"{describe(macro, 4)} | {describe(micro, 4)} |")
    out.append("")
    if count < 2:
        out.append("ONE SEED ONLY: this is the number that has no scale yet. The point of "
                   "the table is the sd column, and one run cannot produce it.")
        return out
    worst = max((statistics.stdev(series(reports, group, "by_seen_macro",
                                         "pairwise_accuracy_macro")), group)
                for group in GROUPS
                if len(series(reports, group, "by_seen_macro",
                              "pairwise_accuracy_macro")) > 1)
    out += [f"The seed alone moves the macro of group {worst[1]} by sd {worst[0]:.4f}. "
            f"That is how unstable the MLP is on this group - it is NOT the bar for "
            f"beating a baseline. The quantity that carries such a claim is the paired "
            f"difference on the same rows, once per seed, which --baseline prints.",
            "",
            "These are descriptive statistics over the seeds listed above. They are NOT a "
            "confidence interval and NOT a significance test; this many runs cannot "
            "provide either. The row-level column has far more data behind it than the "
            "macro column - a group's macro averages the queries that survived the "
            "min_candidates bar, the micro column counts every comparable row - so where "
            "the two disagree, the row-level one is the better supported."]
    if baseline is not None:
        out += paired(reports, baseline[1], baseline[0].name)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", help="seen_molecules_*.json, one per seed")
    parser.add_argument("--baseline", default=None,
                        help="a seen_molecules report of a CLASSICAL baseline on the same "
                             "validation rows; prints MLP(seed i) - baseline per group")
    args = parser.parse_args(argv)

    paths = [Path(name) for name in args.reports]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        for path in missing:
            print(f"no such report: {path}", file=sys.stderr)
        return 2
    loaded = [(path, read(path)) for path in paths]
    found = problems(loaded)
    if found:
        print("THESE REPORTS ARE NOT A SEED SWEEP OF ONE MEASUREMENT:")
        for line in found:
            print(f"  - {line}")
        return 1

    baseline = None
    if args.baseline is not None:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_file():
            print(f"no such baseline report: {baseline_path}", file=sys.stderr)
            return 2
        baseline = (baseline_path, read(baseline_path))
        # The baseline must have scored the SAME ROWS. It must NOT be required to be
        # another seed of the MLP: a classical fit has no checkpoint and no seed sweep, so
        # `distinctness` does not apply to it and running it here is what made the CLI
        # refuse every real lasso dump (2026-11-20 review). What is required instead is
        # that it says which fit it is - source manifest, model, alpha.
        against = [line for line in problems([(paths[0], loaded[0][1]), baseline])
                   if "same report file" not in line
                   and "SAME run" not in line
                   and "SAME SEED" not in line
                   and "do not say which run" not in line
                   and "could not be read from its train manifest" not in line]
        if not str(baseline[1].get("identity") or ""):
            against.append(
                f"{baseline_path.name} does not say which fit produced it; rerun "
                f"scripts/seen_molecules.py on the classical dump so the report records "
                f"its source manifest, model and alpha")
        if against:
            print("THE BASELINE IS NOT COMPARABLE WITH THESE REPORTS:")
            for line in against:
                print(f"  - {line}")
            return 1

    print("\n".join(lines(paths, [report for _, report in loaded], baseline)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
