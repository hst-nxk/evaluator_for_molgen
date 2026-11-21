"""The free-VRAM pre-flight: the check that 2026-11-21's wasted GPU slot asked for.

gpu_memory_limit_mib caps this process; it reserves nothing. Two FULL seeds were handed
4.39 GiB on a card with 20 MiB free and died at their first backward pass. These tests run
the shell function against a stub nvidia-smi, because the only thing worth testing here is
what it does with the number it reads - including the two ways of reading no number at all.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "scripts" / "gpu_batch_lock.sh"
BASH = shutil.which("bash")
HEAD = shutil.which("head")
COREUTILS = str(Path(HEAD).parent) if HEAD else ""
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not available")


def call(tmp_path: Path, stub: str | None, needed: str = "4500",
         headroom: str = "1024") -> subprocess.CompletedProcess:
    """Run require_free_vram with `stub` standing in for nvidia-smi on PATH."""
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    if stub is not None:
        script = directory / "nvidia-smi"
        script.write_text("#!/usr/bin/env bash\n" + stub + "\n", encoding="utf-8")
        script.chmod(0o755)
    # The stub goes IN FRONT of the real PATH, never instead of it: replacing PATH takes
    # head and tr away too, and the function would then fail for the wrong reason.
    environment = dict(os.environ, PATH=str(directory) + os.pathsep + COREUTILS)
    return subprocess.run(
        [BASH, "-c", f'source "{LOCK.as_posix()}"; '
                     f'require_free_vram 6 {needed} {headroom}'],
        capture_output=True, text=True, env=environment)


def test_a_card_with_room_is_accepted(tmp_path: Path) -> None:
    done = call(tmp_path, 'echo " 9242 "')
    assert done.returncode == 0, done.stderr
    assert "9242 MiB free" in done.stdout


def test_the_card_that_failed_is_refused_with_code_5(tmp_path: Path) -> None:
    done = call(tmp_path, "echo 20")
    assert done.returncode == 5
    assert "20 MiB free" in done.stderr
    assert "4500 MiB plus 1024" in done.stderr


def test_the_headroom_is_required_not_just_the_cap(tmp_path: Path) -> None:
    """5000 MiB free is more than the 4500 cap and still not enough to run."""
    assert call(tmp_path, "echo 5000").returncode == 5
    assert call(tmp_path, "echo 5524").returncode == 0


def test_an_unreadable_card_is_not_treated_as_free(tmp_path: Path) -> None:
    for stub in ("echo '[N/A]'", "exit 9", "echo"):
        done = call(tmp_path, stub)
        assert done.returncode == 1, stub
        assert "could not read the free memory" in done.stderr, stub


def test_a_missing_nvidia_smi_is_an_error_not_a_pass(tmp_path: Path) -> None:
    if shutil.which("nvidia-smi", path=COREUTILS) is not None:
        pytest.skip("this machine has a real nvidia-smi beside head and tr")
    done = call(tmp_path, None)
    assert done.returncode == 1
    assert "not on PATH" in done.stderr


def test_only_the_first_line_is_read(tmp_path: Path) -> None:
    # nvidia-smi prints one line per GPU when -i is ignored by an old driver.
    done = call(tmp_path, 'printf "9242\n20\n"')
    assert done.returncode == 0, done.stderr


def test_the_batch_checks_after_taking_the_lock(tmp_path: Path) -> None:
    text = (ROOT / "scripts" / "server_run_full_seeds_2211.sh").read_text(encoding="utf-8")
    assert 'require_free_vram "$GPU" "${MEMORY_LIMIT:-4500}" || exit $?' in text
    assert text.index("claim_gpu") < text.index("require_free_vram")
    assert text.index("require_free_vram") < text.index("--mode train")
    assert "not a reservation" in text
