"""Every run id a server batch mints must be readable by the manifest readers.

check_diagnostics_manifests.RUN_ID allows a hyphen only before 'eval'. A batch that mints
'<task>-curve-<pid>' therefore trains perfectly, writes its manifests, and is then skipped
by tasks_from_log() and by the figure step - which is what happened to t13full's curve run
on 2026-11-21. This test scans the shell of every server batch so the next new name fails
here instead of being discovered in a log a day later.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_diagnostics_manifests import RUN_ID                       # noqa: E402

LITERAL = re.compile(r'--run-id "([^"]+)"')
# What the shell will have put there by the time the line runs. Placeholders become text
# of the same SHAPE, never the same value: the test is about the PATTERN. A literal hyphen
# is left alone, which is the whole point - that is what the readers reject.
PID = re.compile(r"\$\$")
ARITHMETIC = re.compile(r"\$\(\([^)]*\)\)")
VARIABLE = re.compile(r"\$\{?[A-Za-z_][A-Za-z_0-9]*\}?")


def expand(run_id: str) -> str:
    """Every shell expansion replaced by lowercase letters or digits, nothing else."""
    run_id = PID.sub("1349898", run_id)
    run_id = ARITHMETIC.sub("1", run_id)
    return VARIABLE.sub("t99x", run_id)


def test_every_server_batch_mints_readable_run_ids() -> None:
    scripts = sorted((ROOT / "scripts").glob("server_run_*.sh"))
    assert scripts, "no server batch found; has the directory moved?"
    checked = 0
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        for literal in LITERAL.findall(text):
            if literal == "$RUN_ID":
                continue           # built by new_run_id(), covered by its own batch's tests
            expanded = expand(literal)
            assert "$" not in expanded, f"{script.name}: unexpanded {literal!r}"
            found = RUN_ID.fullmatch(f"--run-id {expanded}")
            assert found is not None, (
                f"{script.name}: run id {literal!r} -> {expanded!r} is invisible to "
                f"tasks_from_log(); a hyphen is allowed only before 'eval'")
            checked += 1
    assert checked >= 4, checked


def test_the_pattern_really_does_reject_the_shape_that_failed() -> None:
    assert RUN_ID.fullmatch("--run-id t13fullcurve-1349898") is not None
    assert RUN_ID.fullmatch("--run-id t13full-eval-1349898") is not None
    assert RUN_ID.fullmatch("--run-id t13full-curve-1349898") is None
