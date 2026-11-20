"""Does the model only rank pairs whose molecules it already met in training?

    python scripts/seen_molecules.py --train runs/evaluator/errors_frozen_direct_train_X \
        --validation runs/evaluator/errors_frozen_direct_validation_Y [--out seen.json]

WHY THIS EXISTS. On PAIR75 the MLP scores `pairwise_accuracy_macro` 0.7778 while the same
baselines on the same split reach 0.63-0.68; on SUB100 everything collapses towards 0.53.
The PAIR75 split is keyed on the PAIR, so no validation pair was in training - that is what
`unseen_pair` and the `strict` slice guarantee - but an individual MOLECULE of a validation
pair may well have been, inside some other pair. If the score lives on those rows, 0.7778 is
not a generalisation number, and the gap to SUB100 is a property of the split rather than of
the model.

WHAT IT MEASURES, in two units, because they answer different questions:

  * PER ROW (one MMP pair): the rows are split into three groups by how many of their two
    molecules appear anywhere in the train dump - 0, 1 or 2. Each group gets its row count,
    its MAE and its direction accuracy over rows (the MICRO quantity, from the dump's own
    `sign_counted`/`sign_correct` columns).
  * PER GROUP AS A SLICE: the same three groups, but with `pairwise_accuracy_macro`
    measured inside each - the rows of the group are selected first and the queries built
    afterwards, which is the order every slice of this project is measured in. This is
    what makes a macro over a 0/1/2 group well defined at all: one comparison inside a
    query involves THREE molecules (the reference and two candidates), so the count of
    seen molecules belongs to a row, and a slice is a set of rows. The three groups do not
    add up to the whole set - restricting the rows cuts a query into smaller queries - so
    they are to be compared with each other, not with the overall number.
  * PER QUERY OF THE WHOLE SET (one reference molecule, one protein, one endpoint - the
    unit `pairwise_accuracy_macro` averages over): every query is placed by whether its
    reference molecule was seen and by how many of its candidates were. This answers a
    different question from the one above: whether it matters which SIDE of the comparison
    the model had already met.

The query side is reconstructed from the row dump the way `metrics.ranking_report` builds
it: each pair becomes two directions (A->B and B->A with the sign flipped), queries are
groups of (reference molecule, protein, endpoint), and a query's accuracy is its agreeing
comparisons over its counted comparisons - a comparison being one PAIR OF CANDIDATES inside
that query, counted only when their two labels differ by more than the run's own direction
tolerance (`metrics.pairwise_direction`), so a query whose candidates all carry the same
label contributes nothing and `queries` and `ranked` are separate columns. Because
that is a reconstruction, it is CHECKED: the overall macro accuracy and the number of
queries with at least `min_candidates` candidates must match the `queries.parquet` of the
same dump, and this script exits non-zero when they do not. The bar matters: the evaluator
writes down only the queries that clear it (`analysis.query_table`), so comparing raw
totals would call a correct run wrong - two rows A-B and A-C rebuild three queries while
the file holds one. The full rebuilt total is reported beside it, as coverage. A breakdown that cannot reproduce the number it breaks down is not evidence.

WHAT IT DOES NOT DO. It does not say the model memorises. A higher score on rows whose
molecules were seen has at least two readings - the model recalls those molecules, or those
molecules are simply easier (better measured, more replicates, less extreme Δp) - and
telling them apart needs the label statistics of each group, which the last table prints
next to the scores rather than leaving to the reader's imagination.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from molopt.evaluator import metrics as metric_module  # noqa: E402

# The train-side lookup (which molecule was seen, how often, with which protein) already
# exists for the 2026-11-14 question and is reused rather than written again.
import error_analysis as errors  # noqa: E402

QUERIES_FILE = "queries.parquet"
REPORT_FILE = "report.json"
# How far the reconstruction may differ from queries.parquet before the result is refused.
TOLERANCE = 1e-9


def metrics_setting(validation_folder: Path, name: str) -> tuple[Any, str]:
    """(one `stage3.metrics` value of the run that wrote this dump, where it came from).

    WHERE IT IS READ. `report.json` carries `metrics_settings` (added 2026-11-20) and, for
    older dumps, a top-level `direction_tolerance`. It also carries `settings`, which is
    the stage3.ANALYSIS block and NOT this one - reading a metric threshold out of it
    would silently pick up a different setting that happens to share a name
    (`min_group_queries` and friends live there). So `settings` is never consulted here.

    When the dump says nothing, config.json is the fallback, and the source is returned
    with the value so the report can say which it used: today's config.json is not
    necessarily the config the dump was written with.
    """
    path = validation_folder / REPORT_FILE
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        block = report.get("metrics_settings") or {}
        if name in block:
            return block[name], REPORT_FILE
        if name == "direction_tolerance" and report.get(name) is not None:
            return report[name], REPORT_FILE
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    return config["stage3"]["metrics"][name], "config.json"


def min_candidates(validation_folder: Path) -> tuple[int, str]:
    """How many candidates a query needs before the evaluator writes it down at all.

    WHY THIS MATTERS HERE. `analysis.query_table` skips every query with fewer than
    `min_candidates` candidates, so `queries.parquet` is NOT the whole query set - it is
    the rankable part of it. The reconstruction below rebuilds every query, singletons
    included, so comparing raw totals calls a correct run wrong: two rows A-B and A-C give
    three rebuilt queries (A, B, C) and one written query (A). The counts are therefore
    compared on the same footing, and the full total is kept beside it as coverage.
    """
    value, source = metrics_setting(validation_folder, "min_candidates")
    return int(value), source


def direction_tolerance(validation_folder: Path) -> float:
    """The tolerance the dump itself was written with, not a number chosen here.

    `--mode errors` records it in report.json. Falling back to config.json is for a dump
    written before that field existed; a tolerance that does not match the run would make
    every reconstructed query disagree with queries.parquet, which the check below catches.
    """
    return float(metrics_setting(validation_folder, "direction_tolerance")[0])


def row_groups(table: Any, seen: np.ndarray) -> list[dict[str, Any]]:
    """One block per 0/1/2 group, in the dump's own row unit."""
    truth, prediction = errors.columns(table, "delta_p", "prediction")
    counted, correct = errors.columns(table, "sign_counted", "sign_correct")
    counted, correct = counted.astype(bool), correct.astype(bool)
    blocks = []
    for group in (0, 1, 2):
        mask = seen == group
        rows = int(np.count_nonzero(mask))
        if rows == 0:
            blocks.append({"molecules_seen": group, "rows": 0, "share": 0.0,
                           "mae": None, "direction_accuracy_micro": None,
                           "comparisons": 0, "label_mean": None, "label_sd": None,
                           "abs_label_mean": None})
            continue
        accuracy, denominator = errors.sign_accuracy(counted[mask], correct[mask])
        blocks.append({
            "molecules_seen": group, "rows": rows,
            "share": rows / float(table.num_rows),
            "mae": float(np.mean(np.abs(prediction[mask] - truth[mask]))),
            "direction_accuracy_micro": accuracy, "comparisons": denominator,
            # The label statistics of the group, so "these rows are easier" can be told
            # apart from "the model remembers these rows" instead of being assumed away.
            "label_mean": float(np.mean(truth[mask])),
            "label_sd": float(np.std(truth[mask])),
            "abs_label_mean": float(np.mean(np.abs(truth[mask]))),
        })
    return blocks


def doubled(table: Any) -> dict[str, Any]:
    """Every pair as its two directions, the way `metrics.expand_directions` does it."""
    first = table.column("smiles_a").to_pylist()
    second = table.column("smiles_b").to_pylist()
    proteins = table.column("protein_hash").to_pylist()
    endpoints = table.column("endpoint").to_pylist()
    truth, prediction = errors.columns(table, "delta_p", "prediction")
    return {"reference": first + second, "candidate": second + first,
            "protein": proteins + proteins, "endpoint": endpoints + endpoints,
            "truth": np.concatenate([truth, -truth]),
            "prediction": np.concatenate([prediction, -prediction]),
            "rows": len(first)}


def macro_over(table: Any, keep: np.ndarray, tolerance: float,
               needed: int) -> dict[str, Any]:
    """`pairwise_accuracy_macro` over the rows `keep` selects, with its query counts.

    The rows are selected FIRST and the directions expanded afterwards, which is the order
    `train.evaluate` uses for every slice (mask the rows, then build the queries). Treating
    "how many of this pair's molecules were seen in training" as a slice is what makes a
    macro over that group well defined: a comparison inside a query involves three
    molecules (the reference and two candidates), so the 0/1/2 count cannot be attached to
    a comparison, only to a row - and a slice is exactly a set of rows.
    """
    rows = np.flatnonzero(np.asarray(keep, dtype=bool))
    if rows.size == 0:
        return {"rows": 0, "queries": 0, "queries_eligible": 0, "queries_ranked": 0,
                "comparisons": 0, "pairwise_accuracy_macro": None}
    kept = table.take(rows)
    data = doubled(kept)
    queries: dict[tuple[str, str, str], list[int]] = {}
    for index, key in enumerate(zip(data["reference"], data["protein"],
                                    data["endpoint"])):
        queries.setdefault(key, []).append(index)
    accuracies, comparisons, eligible = [], 0, 0
    for members in queries.values():
        index = np.asarray(members, dtype=np.int64)
        # A query below the bar is SKIPPED, not merely left out of `eligible`:
        # `metrics.ranking_report` skips it before scoring anything
        # (queries_skipped_too_few_candidates), so scoring it here would make this macro a
        # different quantity from the one the manifests report. At min_candidates = 2 the
        # two behaviours happen to agree - a single candidate yields no comparison - which
        # is exactly why this went unnoticed until a run with 3 was tried.
        if index.size < needed:
            continue
        eligible += 1
        agree, counted = metric_module.pairwise_direction(
            data["truth"][index], data["prediction"][index], tolerance)
        comparisons += int(counted)
        if counted:
            accuracies.append(agree / counted)
    return {"rows": int(rows.size), "queries": len(queries),
            "queries_eligible": eligible, "queries_ranked": len(accuracies),
            "comparisons": comparisons,
            "pairwise_accuracy_macro": (float(np.mean(accuracies)) if accuracies
                                        else None)}


def seen_slices(table: Any, seen: np.ndarray, tolerance: float,
                needed: int) -> list[dict[str, Any]]:
    """The table the 2026-11-18 follow-up asked for: a macro per 0/1/2 group."""
    blocks = []
    for group in (0, 1, 2):
        block = {"molecules_seen": group}
        block.update(macro_over(table, seen == group, tolerance, needed))
        blocks.append(block)
    return blocks


def query_table(table: Any, seen_molecule: set[str], tolerance: float, needed: int
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(one block per query group, the overall reconstruction) in the macro's own unit.

    Queries are rebuilt exactly as `metrics.ranking_report` does: every pair becomes two
    directions, and a query is one (reference molecule, protein, endpoint).
    """
    first = table.column("smiles_a").to_pylist()
    second = table.column("smiles_b").to_pylist()
    proteins = table.column("protein_hash").to_pylist()
    endpoints = table.column("endpoint").to_pylist()
    truth, prediction = errors.columns(table, "delta_p", "prediction")

    reference = first + second
    candidate = second + first
    protein = proteins + proteins
    endpoint = endpoints + endpoints
    doubled_truth = np.concatenate([truth, -truth])
    doubled_prediction = np.concatenate([prediction, -prediction])

    queries: dict[tuple[str, str, str], list[int]] = {}
    for index, key in enumerate(zip(reference, protein, endpoint)):
        queries.setdefault(key, []).append(index)

    groups: dict[tuple[bool, str], dict[str, Any]] = {}
    accuracies_all = []
    for (molecule, _, _), rows in queries.items():
        index = np.asarray(rows, dtype=np.int64)
        agree, counted = (metric_module.pairwise_direction(
            doubled_truth[index], doubled_prediction[index], tolerance)
            if index.size >= needed else (0, 0))
        candidates_seen = sum(1 for row in rows if candidate[row] in seen_molecule)
        if candidates_seen == 0:
            share = "no candidate seen"
        elif candidates_seen == len(rows):
            share = "every candidate seen"
        else:
            share = "some candidates seen"
        key = (molecule in seen_molecule, share)
        block = groups.setdefault(key, {"reference_seen": key[0], "candidates": key[1],
                                        "queries": 0, "queries_eligible": 0,
                                        "queries_ranked": 0,
                                        "accuracies": [], "comparisons": 0})
        block["queries"] += 1
        if index.size < needed:
            # Counted as coverage only; the evaluator never scored it, so neither does
            # this table. See macro_over for why skipping and not-counting differ.
            continue
        block["queries_eligible"] += 1
        block["comparisons"] += int(counted)
        if counted:
            block["queries_ranked"] += 1
            block["accuracies"].append(agree / counted)
            accuracies_all.append(agree / counted)

    blocks = []
    for block in groups.values():
        accuracies = block.pop("accuracies")
        block["pairwise_accuracy_macro"] = (float(np.mean(accuracies)) if accuracies
                                            else None)
        blocks.append(block)
    blocks.sort(key=lambda block: (block["reference_seen"], block["candidates"]))
    overall = {"queries": len(queries),
               "queries_eligible": sum(block["queries_eligible"] for block in blocks),
               "queries_ranked": len(accuracies_all),
               "pairwise_accuracy_macro": (float(np.mean(accuracies_all))
                                           if accuracies_all else None)}
    return blocks, overall


def reconstruction_check(validation_folder: Path, overall: dict[str, Any],
                         needed: int) -> dict[str, Any]:
    """Does the rebuilt query set reproduce the dump's own queries.parquet?

    The breakdown is only worth reading if the whole it decomposes is the number the run
    recorded. When the file is absent this says so instead of claiming a check happened.
    """
    path = validation_folder / QUERIES_FILE
    if not path.is_file():
        return {"checked": False,
                "note": f"no {QUERIES_FILE} in the dump, so the reconstruction could not "
                        f"be checked against the run's own queries"}
    recorded = pq.read_table(path)
    agree = np.asarray(recorded.column("agree").to_numpy(zero_copy_only=False))
    counted = np.asarray(recorded.column("counted").to_numpy(zero_copy_only=False))
    ranked = counted > 0
    macro = float(np.mean(agree[ranked] / counted[ranked])) if ranked.any() else None
    # `queries.parquet` holds only the queries with at least `needed` candidates
    # (analysis.query_table skips the rest), so the comparable number is the rebuilt
    # queries that clear the same bar - not every rebuilt query. Comparing raw totals
    # called a correct run wrong: two rows A-B and A-C give three rebuilt queries and one
    # written query.
    same_queries = int(recorded.num_rows) == overall["queries_eligible"]
    same_ranked = int(np.count_nonzero(ranked)) == overall["queries_ranked"]
    if macro is None or overall["pairwise_accuracy_macro"] is None:
        same_macro = macro is None and overall["pairwise_accuracy_macro"] is None
        difference = None
    else:
        difference = abs(macro - overall["pairwise_accuracy_macro"])
        same_macro = difference <= TOLERANCE
    return {"checked": True, "min_candidates": needed,
            "queries_recorded": int(recorded.num_rows),
            "queries_rebuilt": overall["queries_eligible"],
            "queries_rebuilt_including_singletons": overall["queries"],
            "queries_ranked_recorded": int(np.count_nonzero(ranked)),
            "queries_ranked_rebuilt": overall["queries_ranked"],
            "macro_recorded": macro, "macro_rebuilt": overall["pairwise_accuracy_macro"],
            "difference": difference,
            "agrees": bool(same_queries and same_ranked and same_macro)}


def cell(value: Any, digits: int = 4) -> str:
    """One table cell: a number at a fixed width, or `-` when there is none."""
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def lines(report: dict[str, Any]) -> list[str]:
    """The report as text; every number comes from `report`."""
    out = [f"Molecules seen in training - {report['label'] or report['validation']}",
           f"train dump      : {report['train']}",
           f"validation dump : {report['validation']} (slice {report['slice']}, "
           f"{report['rows']:,} rows)",
           f"train molecules : {report['train_molecules']:,} distinct",
           f"direction tolerance: {report['direction_tolerance']}",
           f"min_candidates     : {report['min_candidates']} (read from "
           f"{report['min_candidates_source']})", "",
           "Per ROW (one MMP pair). The direction accuracy here is the MICRO quantity, "
           "over rows:", "",
           "| molecules seen | rows | share | MAE | direction acc. | comparisons | "
           "mean |label| |",
           "|---|---|---|---|---|---|---|"]
    for block in report["by_row"]:
        out.append(f"| {block['molecules_seen']} | {block['rows']:,} | "
                   f"{100.0 * block['share']:.1f} % | {cell(block['mae'])} | "
                   f"{cell(block['direction_accuracy_micro'])} | "
                   f"{block['comparisons']:,} | {cell(block['abs_label_mean'])} |")
    out += ["", "pairwise_accuracy_macro PER GROUP, the groups taken as slices (the rows "
            "are selected first, the queries built afterwards - the order every slice of "
            "this project is measured in):", "",
            f"| molecules seen | rows | queries | with >= "
            f"{report['min_candidates']} candidates | ranked | comparisons | "
            f"pairwise_accuracy_macro |", "|---|---|---|---|---|---|---|"]
    for block in report["by_seen_macro"]:
        out.append(f"| {block['molecules_seen']} | {block['rows']:,} | "
                   f"{block['queries']:,} | {block['queries_eligible']:,} | "
                   f"{block['queries_ranked']:,} | {block['comparisons']:,} | "
                   f"{cell(block['pairwise_accuracy_macro'])} |")
    out += ["", "Those three groups do not add up to the number of the whole set: a query "
            "of the whole set can hold candidates from two different groups, and "
            "restricting the rows splits it into smaller queries. Compare the groups with "
            "each other, not with the overall macro below.",
            "", "Per QUERY of the WHOLE set (reference molecule, protein, endpoint), "
            "placed by which SIDE was seen:", "",
            "| reference | candidates | queries | ranked | pairwise_accuracy_macro |",
            "|---|---|---|---|---|"]
    for block in report["by_query"]:
        out.append(f"| {'seen' if block['reference_seen'] else 'unseen'} | "
                   f"{block['candidates']} | {block['queries']:,} | "
                   f"{block['queries_ranked']:,} | "
                   f"{cell(block['pairwise_accuracy_macro'])} |")
    overall = report["overall"]
    out += ["", f"All queries: {overall['queries']:,} rebuilt, of which "
            f"{overall['queries_eligible']:,} have at least "
            f"{report['min_candidates']} candidates and {overall['queries_ranked']:,} "
            f"could be ranked; pairwise_accuracy_macro "
            f"{cell(overall['pairwise_accuracy_macro'])}"]
    check = report["check"]
    if not check["checked"]:
        out.append(f"RECONSTRUCTION NOT CHECKED: {check['note']}")
    elif check["agrees"]:
        out.append(f"Reconstruction agrees with {QUERIES_FILE}: "
                   f"{check['queries_recorded']:,} queries with at least "
                   f"{check['min_candidates']} candidates (of "
                   f"{check['queries_rebuilt_including_singletons']:,} rebuilt in all), "
                   f"macro {cell(check['macro_recorded'], 6)}.")
    else:
        out.append(f"RECONSTRUCTION DISAGREES with {QUERIES_FILE}: "
                   f"{check['queries_rebuilt']:,} queries rebuilt against "
                   f"{check['queries_recorded']:,} recorded, macro "
                   f"{cell(check['macro_rebuilt'], 6)} against "
                   f"{cell(check['macro_recorded'], 6)}. The breakdown above does not "
                   f"decompose the run's own number and must not be quoted.")
    return out


def build(train_folder: Path, validation_folder: Path, slice_name: str, label: str
          ) -> dict[str, Any]:
    """The whole report, from two `--mode errors` dumps of the SAME checkpoint."""
    train_table = errors.read_rows(train_folder)
    validation_table = errors.select_slice(errors.read_rows(validation_folder),
                                           slice_name)
    for name, table in (("train", train_table), ("validation", validation_table)):
        missing = sorted({"smiles_a", "smiles_b", "endpoint"} - set(table.column_names))
        if missing:
            raise ValueError(f"the {name} dump has no {missing}; rerun --mode errors "
                             f"(these columns were added on 2026-11-10)")

    counts, _ = errors.molecule_occurrences(train_table)
    molecules = set(counts)
    first = validation_table.column("smiles_a").to_pylist()
    second = validation_table.column("smiles_b").to_pylist()
    seen = np.array([int(a in molecules) + int(b in molecules)
                     for a, b in zip(first, second)], dtype=np.int64)

    tolerance = direction_tolerance(validation_folder)
    needed, needed_source = min_candidates(validation_folder)
    by_query, overall = query_table(validation_table, molecules, tolerance, needed)
    by_seen = seen_slices(validation_table, seen, tolerance, needed)
    return {"stage": "stage3_seen_molecules",
            "train": train_folder.name, "validation": validation_folder.name,
            "slice": slice_name, "label": label,
            "rows": int(validation_table.num_rows),
            "train_molecules": len(molecules),
            "direction_tolerance": tolerance, "min_candidates": needed,
            "min_candidates_source": needed_source,
            "by_row": row_groups(validation_table, seen),
            "by_seen_macro": by_seen,
            "by_query": by_query, "overall": overall,
            "check": reconstruction_check(validation_folder, overall, needed)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split the validation rows and queries by whether their molecules "
                    "were seen in training, and score each group.")
    parser.add_argument("--train", required=True,
                        help="an errors_*_train_* folder - the reference, never sliced")
    parser.add_argument("--validation", required=True,
                        help="the errors_*_validation_* folder of the SAME checkpoint")
    parser.add_argument("--slice", default="all", choices=errors.SLICE_NAMES,
                        help="the slice of the validation dump; on a pair-keyed split "
                             "strict equals all, since every validation pair is unseen")
    parser.add_argument("--label", default="", help="a name to print and store")
    parser.add_argument("--out", default="", help="a .json file to write; default print")
    args = parser.parse_args(argv)

    report = build(Path(args.train), Path(args.validation), args.slice, args.label)
    for line in lines(report):
        print(line)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nwritten: {args.out}")
    return 0 if report["check"].get("agrees", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
