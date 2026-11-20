"""Every command of the 2026-11-19 batch must name its own run.

Why this file exists. Each mode writes its manifest to
``<mode>_<configuration>_<head>_<stamp>.json`` and the stamp is accurate only to the
second, so two batches sharing ``runs/evaluator/`` — which is what happens as soon as a
second GPU is used — can have two commands of the same mode and configuration start in the
same second and write the same path. One result then silently overwrites the other, even
though each command loaded the right checkpoint. ``--run-id`` is what keeps them apart (see
``run_stamp`` in run_evaluator.py), so the batch has to pass it EVERYWHERE, not only on the
train commands whose checkpoint is picked up afterwards.

These are text checks on the shell script: they cannot run it (it trains on a GPU for
hours), but a forgotten flag is exactly the kind of omission that silently costs a result.
"""

from __future__ import annotations

import os
import re
import time
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "server_run_diagnostics_1911.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")
LOCK_SCRIPT = SCRIPTS / "gpu_batch_lock.sh"
LOCK = LOCK_SCRIPT.read_text(encoding="utf-8")


def script_bytes(body: str) -> bytes:
    """The shell body as bytes with Unix line endings.

    Text mode would turn every "\\n" into "\\r\\n" on Windows, and bash then reads the
    carriage return as part of the last word on the line: a quoted path ends up with a CR
    inside it and "No such file or directory" names a file that plainly exists.
    """
    return body.replace("\r\n", "\n").encode()


def in_bash(body: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in the bash these tests launch, fed through stdin.

    Through stdin, and never as a path argument, because the bash that answers `bash` on a
    developer machine is not the shell this file was launched from: here it is WSL's, which
    reads E:\\... as neither a file nor /e/..., and that is the whole reason paths and
    probes both go through it rather than being guessed in Python.
    """
    done = subprocess.run(["bash", "-s", "--", *arguments], input=script_bytes(body),
                          capture_output=True, timeout=120)
    return subprocess.CompletedProcess(done.args, done.returncode,
                                       done.stdout.decode(errors="replace"),
                                       done.stderr.decode(errors="replace"))


def bash_has(command: str) -> bool:
    """Whether THAT bash can run `command` — not whether Windows' PATH holds it."""
    try:
        return in_bash(f'command -v {command} > /dev/null', ).returncode == 0
    except OSError:
        return False


HAS_FLOCK = bash_has("flock")


def commands() -> list[str]:
    """Every `run python run_evaluator.py ...` command, continuation lines joined."""
    joined = TEXT.replace("\\\n", " ")
    return [line.strip() for line in joined.splitlines()
            if line.strip().startswith("run python run_evaluator.py")]


def variable(name: str) -> str:
    """The value of a shell variable built with `NAME="..."` and `NAME+=" ..."` lines."""
    pattern = re.compile(rf'^{name}\+?="(.*)"$', re.MULTILINE)
    parts = pattern.findall(TEXT)
    assert parts, f"{name} is not assigned in the script"
    return "".join(parts)


def expanded(command: str) -> str:
    """One command with the script's own variables put back in."""
    for name in ("REF", "RESIDUAL", "SELECTION"):
        command = command.replace(f"${name}", variable(name))
    # $SELECTION is itself used inside $REF, so one more pass finishes the job.
    return command.replace("$SELECTION", variable("SELECTION"))


# The flags that decide WHICH result of several is the one quoted.
SELECTION_FLAGS = ("--set selection_metric=pairwise_accuracy_macro",
                   "--set selection_slice=strict")


def test_the_alpha_and_the_checkpoint_are_chosen_by_the_same_metric() -> None:
    """Review 2026-11-20: the comparison the meeting asked for is only a comparison if both
    sides choose their own winner the same way.

    --mode classical picks the alpha of ridge and lasso on validation through
    train.selection_score, exactly as --mode train picks a checkpoint. But the two flags
    that say WHICH metric and WHICH slice used to live inside $REF, and the classical tasks
    did not take $REF: ridge and lasso would have chosen by mae on 'all' while the MLP chose
    by direction accuracy on 'strict'. Nothing would have failed, and the manifest of each
    side would have been internally consistent - the table built from them would have
    compared two different questions.

    --mode capacity is left out on purpose: its verdict is a threshold on train R2
    (fit_train_r2), so no selection metric enters it. The one --mode eval command gets its
    flags from score_train's "$@", so the call sites are what carries them; they are
    checked in the test below.
    """
    for command in commands():
        if "--mode capacity" in command:
            continue
        if '"$@"' in command:
            continue
        text = expanded(command)
        missing = [flag for flag in SELECTION_FLAGS if flag not in text]
        assert not missing, (command, missing)


def scoring_calls() -> list[str]:
    """Every `score_train ...` call, continuation lines joined."""
    joined = TEXT.replace("\\\n", " ")
    return [line.strip() for line in joined.splitlines()
            if line.strip().startswith("score_train ")]


def test_the_scoring_runs_are_given_the_same_selection_flags() -> None:
    # score_train forwards "$@" to --mode eval, so a call that forgets the flags would
    # score the checkpoint under a different slice than the one its own run was chosen on.
    calls = scoring_calls()
    assert len(calls) == 4, calls
    for call in calls:
        missing = [flag for flag in SELECTION_FLAGS if flag not in expanded(call)]
        assert not missing, (call, missing)


def test_the_scoring_helper_forwards_its_extra_arguments() -> None:
    # The flags only reach --mode eval because the helper passes "$@" through; without it
    # the four call sites above would be checking something that goes nowhere.
    eval_commands = [command for command in commands() if "--mode eval" in command]
    assert len(eval_commands) == 1, eval_commands
    assert '"$@"' in eval_commands[0]


def test_the_classical_tasks_do_not_take_the_mlp_hyper_parameters() -> None:
    # They take $SELECTION, not $REF: a scikit-learn fit has no dropout, no learning rate
    # and no GPU memory limit, and passing them would put parameters into the manifest's
    # `overrides` that had no effect on the numbers above them.
    for command in commands():
        if "--mode classical" not in command:
            continue
        assert "$SELECTION" in command, command
        assert "$REF" not in command, command


def test_the_script_still_runs_the_expected_number_of_commands() -> None:
    # Tasks 1a-1c, 2a-2d, 3a-3d, the eval inside score_train, and task 4.
    assert len(commands()) == 13


def test_every_command_passes_a_run_id() -> None:
    missing = [command for command in commands() if '--run-id "$RUN_ID"' not in command]
    assert not missing, missing


def test_every_task_mints_its_own_id_before_the_command() -> None:
    labels = re.findall(r"new_run_id (\S+)", TEXT)
    # One per task, plus the one score_train mints from its label argument.
    assert labels.count('"$label"') == 1
    fixed = [label for label in labels if label != '"$label"']
    assert len(fixed) == len(set(fixed)), fixed
    assert set(fixed) == {"t1a", "t1b", "t1c", "t2a", "t2b", "t2c", "t2d",
                          "t3a", "t3b", "t3c", "t3d", "t4"}


def test_the_ids_carry_the_process_id() -> None:
    # Without $$ two batches would mint the same id for the same task, which is the race
    # the id exists to close.
    assert 'new_run_id() { RUN_ID="$1-$$"; }' in TEXT


def test_the_scoring_runs_have_ids_of_their_own() -> None:
    # score_train is called with the label of the SCORING run, not of the train run, so the
    # eval manifest cannot be confused with the train manifest either.
    calls = re.findall(r"score_train (\S+)", TEXT)
    assert calls == ["t3a-eval", "t3b-eval", "t3c-eval", "t3d-eval"]


def test_the_checkpoint_is_taken_by_run_id_and_not_by_file_age() -> None:
    assert '"$ROOT/runs/evaluator" "$RUN_ID"' in TEXT
    assert "MARKER" not in TEXT
    assert "mark_train" not in TEXT


# --------------------------------------------------- one batch, one pid file, one log
# Review 2026-11-19: both batches wrote `.cache/diagnostics_1911.pid`, so the documented
# stop command aimed at whichever batch started last, and the batch that finished first
# deleted the file while the other was still running. One shared log file is the same
# problem one step later, since the numbers are read out of the log.


def test_the_pid_file_and_the_log_are_named_after_the_gpu() -> None:
    assert 'LOG="$ROOT/runs/diagnostics_1911_gpu$GPU.log"' in TEXT
    assert 'PID_FILE="$directory/${prefix}_gpu$gpu.pid"' in LOCK
    assert 'LOCK_FILE="$directory/${prefix}_gpu$gpu.lock"' in LOCK
    # No path without the GPU in it is left anywhere, including the header's instructions.
    assert "diagnostics_1911.pid" not in TEXT
    assert "diagnostics_1911.log" not in TEXT


def test_the_gpu_is_claimed_before_anything_else_happens() -> None:
    # Two batches on ONE GPU would fight for its memory and append to one log, so the
    # second one has to be refused BEFORE the first task starts.
    claim = TEXT.index('claim_gpu "$GPU"')
    assert claim < TEXT.index('say "TASK 1a')
    assert 'claim_gpu "$GPU" "$ROOT/.cache" "diagnostics_1911" || exit $?' in TEXT
    # Checking a pid and then writing it is two steps, which is the hole the lock closes.
    assert "kill -0" not in TEXT


def test_the_pid_file_is_removed_on_every_exit_path() -> None:
    # It used to be a line at the end of the batch, which the early exits (a missing split
    # file, exit 3) skipped — leaving a pid that the operating system may later reuse.
    assert not [line for line in TEXT.splitlines()
                if line.strip().startswith('rm -f "$PID_FILE"')]
    assert 'trap "rm -f \'$PID_FILE\'" EXIT' in LOCK


def test_the_header_sends_the_reader_to_the_stop_script() -> None:
    # Not a `kill` one-liner: a pid alone cannot be checked, see stop_gpu_batch.sh. (The
    # header may still MENTION that one-liner to say why it is wrong, hence the line test
    # rather than a substring test.)
    assert "bash scripts/stop_gpu_batch.sh 1" in TEXT
    assert not [line for line in TEXT.splitlines()
                if "kill -- -$(cat" in line and not line.lstrip().startswith("#")]


# ----------------------------------------------------- the race itself, not its text
# The text checks above catch a forgotten GPU suffix or a dropped guard, but not the bug
# that made the guard necessary: reading a pid file and then writing it is two steps, so two
# batches launched in the same second both read "nobody there" before either writes. The
# test that settles it starts two processes that race for one lock and asserts that exactly
# one of them reaches the work while the other exits 4. It needs no GPU: the work is a
# `sleep`.


def bash_path(path: Path) -> str:
    """The path as the bash these tests launch reads it.

    On the server that is the path itself. On a developer machine it is not: Git Bash wants
    `/e/eval_training/...` while WSL wants `/mnt/e/eval_training/...`, so the translation is
    asked OF THAT SAME bash rather than guessed here.
    """
    probe = ('if command -v cygpath > /dev/null; then cygpath -u "$1";'
             ' elif command -v wslpath > /dev/null; then wslpath -u "$1";'
             ' else printf %s "$1"; fi')
    try:
        # as_posix(), never str(): a Windows path's backslashes do not survive the trip
        # into WSL as an argument, and `E:eval_training...` is then translated to nothing.
        return in_bash(probe, path.as_posix()).stdout.strip() or path.as_posix()
    except OSError:
        return path.as_posix()


def claimant(tmp_path: Path, holder_seconds: str) -> str:
    """A fake batch: claim GPU 0, announce task 1, hold the lock for a while."""
    return (f'source "{bash_path(LOCK_SCRIPT)}"\n'
            f'claim_gpu 0 "{bash_path(tmp_path)}" batch || exit $?\n'
            f'echo "TASK 1 pid $$"\n'
            f'sleep {holder_seconds}\n')


def start_claimant(tmp_path: Path, holder_seconds: str) -> subprocess.Popen[bytes]:
    """A claimant of GPU 0 in a session OF ITS OWN, as the run instructions' setsid gives.

    start_new_session matters for every test that then calls the stop script. Without it, on
    POSIX, the shell inherits pytest's process group, `claim_gpu` records THAT group, and a
    stop would signal it — pytest included. (On Windows the flag is ignored, which is why
    this had to be read rather than observed.)
    """
    started = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=os.name == "posix")
    assert started.stdin is not None
    started.stdin.write(script_bytes(claimant(tmp_path, holder_seconds)))
    started.stdin.close()
    started.stdin = None       # see the comment in race()
    return started


def race(tmp_path: Path, holder_seconds: str = "2") -> tuple[int, int, str, str]:
    """Start two claimants of GPU 0 at once; return (winners, losers, stdout, stderr)."""
    body = claimant(tmp_path, holder_seconds)
    # Fed through stdin so no path has to survive as a bash argument.
    started = [subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
               for _ in range(2)]
    # Both are started before either is fed, so they really do race for the lock.
    for process in started:
        assert process.stdin is not None
        process.stdin.write(script_bytes(body))
        process.stdin.close()
        # And then forgotten, because `communicate()` on POSIX flushes `self.stdin` before
        # closing it and catches only BrokenPipeError: on an already closed pipe it raises
        # `ValueError: flush of closed file` instead of returning the exit codes. (On
        # Windows it happens to survive, which is why this needed reading, not running.)
        process.stdin = None
    out, err = "", ""
    codes = []
    for process in started:
        process_out, process_err = process.communicate(timeout=60)
        out += process_out.decode(errors="replace")
        err += process_err.decode(errors="replace")
        codes.append(process.returncode)
    return codes.count(0), codes.count(4), out, err


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_exactly_one_of_two_simultaneous_batches_reaches_the_first_task(
        tmp_path: Path) -> None:
    winners, losers, out, err = race(tmp_path)
    assert (winners, losers) == (1, 1), (winners, losers, out, err)
    assert out.count("TASK 1") == 1, out
    # The loser says which GPU is taken and how to stop the batch that holds it, instead
    # of just failing.
    assert "already running on GPU 0" in err
    assert "stop_gpu_batch.sh 0" in err


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_the_gpu_is_free_again_once_the_winner_has_finished(tmp_path: Path) -> None:
    # The lock lives on an open descriptor, so the kernel releases it when the process ends
    # however it ends: a crashed batch must not block the GPU for good.
    assert race(tmp_path, holder_seconds="0")[0] >= 1
    winners, _, out, _ = race(tmp_path, holder_seconds="0")
    assert winners >= 1 and "TASK 1" in out


@pytest.mark.skipif(HAS_FLOCK,
                    reason="this is the behaviour on a machine WITHOUT flock; this bash "
                           "has it")
def test_a_missing_flock_says_so_instead_of_claiming_the_gpu_is_busy(tmp_path: Path) -> None:
    """Without flock, `! flock -n 9` is true (command not found), which reads exactly like
    "another batch holds the GPU". The message has to name the real problem."""
    # Fed through stdin, because a Windows-style path as a bash argument is not found by
    # Git Bash, which is the very environment this test is about.
    done = in_bash(claimant(tmp_path, "0"))
    assert done.returncode == 1, done
    assert "flock is not installed" in done.stderr
    assert "TASK 1" not in done.stdout


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_the_pid_file_is_gone_after_the_batch_ends(tmp_path: Path) -> None:
    race(tmp_path, holder_seconds="0")
    assert not (tmp_path / "batch_gpu0.pid").exists()


# ------------------------------------------------------- stopping ONE batch, and only it
# Review 2026-11-20: a pid alone cannot be checked. A batch killed with SIGKILL leaves its
# record behind, the operating system hands that number to something else later, and that
# something else may be a batch of the SAME script on another GPU — where even
# `ps | grep server_run_diagnostics_1911` matches, so the check that was in the header would
# have passed and the stop would have signalled the wrong GPU's process group.

STOP_SCRIPT = SCRIPTS / "stop_gpu_batch.sh"


def stop(tmp_path: Path, gpu: str = "0") -> subprocess.CompletedProcess[str]:
    return in_bash(f'bash "{bash_path(STOP_SCRIPT)}" {gpu} "{bash_path(tmp_path)}" batch')


def record(tmp_path: Path, gpu: str = "0") -> dict[str, str]:
    lines = (tmp_path / f"batch_gpu{gpu}.pid").read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def test_the_record_holds_what_a_stop_has_to_check() -> None:
    # The pid is not enough; the start time is what a reused pid cannot inherit.
    for key in ("pid=", "pgid=", "gpu=", "prefix=", "started="):
        assert key in LOCK, key
    assert "start_token" in LOCK


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_the_record_describes_the_running_batch(tmp_path: Path) -> None:
    held = start_claimant(tmp_path, "30")
    try:
        written = wait_for_record(tmp_path)
        assert written["gpu"] == "0" and written["prefix"] == "batch"
        assert int(written["pid"]) > 0 and int(written["started"]) >= 0
    finally:
        held.kill()
        held.communicate(timeout=30)


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_refuses_a_record_whose_pid_was_handed_to_something_else(
        tmp_path: Path) -> None:
    """THE CASE FROM THE REVIEW. The batch was SIGKILLed, so its record survived; the pid now
    belongs to another process (here: this very shell). Nothing may be signalled."""
    directory = tmp_path
    (directory / "batch_gpu0.lock").write_text("", encoding="utf-8")
    done = in_bash(f'echo "pid=$$" > "{bash_path(directory)}/batch_gpu0.pid"\n'
                   f'echo "pgid=$$" >> "{bash_path(directory)}/batch_gpu0.pid"\n'
                   f'echo "gpu=0" >> "{bash_path(directory)}/batch_gpu0.pid"\n'
                   f'echo "prefix=batch" >> "{bash_path(directory)}/batch_gpu0.pid"\n'
                   f'echo "started=1" >> "{bash_path(directory)}/batch_gpu0.pid"\n')
    assert done.returncode == 0, done
    # `started=1` cannot be this pid's real start time, so the pid is not that batch.
    result = stop(tmp_path)
    assert result.returncode == 1, result
    assert "no batch is running on GPU 0" in result.stdout
    # And the stale record is cleared, so the next reader is not misled by it again.
    assert not (tmp_path / "batch_gpu0.pid").exists()


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_refuses_a_record_written_for_another_gpu(tmp_path: Path) -> None:
    # newline="\n": written on Windows, read by bash, which would otherwise see a
    # carriage return as part of the value ("GPU 0\r" is not "GPU 0").
    (tmp_path / "batch_gpu1.pid").write_text("pid=1\npgid=1\ngpu=0\nprefix=batch\n"
                                             "started=1\n", encoding="utf-8",
                                             newline="\n")
    result = stop(tmp_path, gpu="1")
    assert result.returncode == 3, result
    assert "records GPU 0, not GPU 1" in result.stderr


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_on_a_free_gpu_says_so(tmp_path: Path) -> None:
    result = stop(tmp_path)
    assert result.returncode == 1, result
    assert "no batch is running on GPU 0" in result.stdout


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_ends_the_batch_and_frees_the_gpu(tmp_path: Path) -> None:
    held = start_claimant(tmp_path, "60")
    try:
        written = wait_for_record(tmp_path)
        # The batch must lead its own group before anything is stopped: that is what the
        # stop script requires, and what keeps this test from signalling pytest.
        assert written["pgid"] == written["pid"], written
        result = stop(tmp_path)
        assert result.returncode == 0, result
        assert "stopping the batch on GPU 0" in result.stdout
        held.communicate(timeout=30)
        # The GPU is claimable again, which is the only proof that matters.
        assert race(tmp_path, holder_seconds="0")[0] >= 1
    finally:
        held.kill()


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_finds_the_work_when_the_record_no_longer_matches(tmp_path: Path) -> None:
    """The other case from the review: the shell was SIGKILLed while the python run it
    started kept descriptor 9, so the GPU is still claimed although the recorded pid is no
    longer that batch. The record cannot identify the live work by itself, so the stop shows
    what is still alive in the process group before signalling it.

    The state is built by holding the lock and then spoiling the record's start time, which
    is exactly what the script sees in that case: lock held, record not matching. (A real
    SIGKILL of the shell cannot be used here - in this environment the background child does
    not survive its parent, so there would be nothing left holding the lock.)
    """
    held = start_claimant(tmp_path, "120")
    try:
        written = wait_for_record(tmp_path)
        assert written["pgid"] == written["pid"], written
        path = tmp_path / "batch_gpu0.pid"
        path.write_text(f"pid={written['pid']}\npgid={written['pgid']}\ngpu=0\n"
                        "prefix=batch\nstarted=1\n", encoding="utf-8", newline="\n")
        result = stop(tmp_path)
        assert result.returncode == 0, result
        assert "is gone, but GPU 0 is still claimed" in result.stdout
        assert "sleep 120" in result.stdout        # it names the work before killing it
        held.communicate(timeout=30)
        assert race(tmp_path, holder_seconds="0")[0] >= 1
    finally:
        held.kill()



def wait_for_record(tmp_path: Path, gpu: str = "0", seconds: float = 20) -> dict[str, str]:
    """Wait until the claimant has written its record, then return it."""
    path = tmp_path / f"batch_gpu{gpu}.pid"
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and "started=" in path.read_text(encoding="utf-8"):
            return record(tmp_path, gpu)
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


# --------------------------------------- the group number in the record is never trusted
# Review 2026-11-20: the record is the one thing a stop acts on, and it can be out of date.
# A new batch takes the lock a moment before it replaces the old record, so the lock belongs
# to the new batch while the number on disk still names the old one's group — and that number
# may since have been handed to something else. Both branches of the stop therefore require
# that a member of the named group is holding THIS GPU's lock file open.


def sacrificial_group(tmp_path: Path) -> tuple[subprocess.Popen[bytes], str]:
    """A harmless process leading its own group, to put into a record in place of the batch.

    It exists so that a regression in the stop script kills THIS and not pytest: every test
    below asserts the stop refuses, and if it wrongly signals instead, the damage is one
    sleep. Returns the process and the pid AS BASH SEES IT — `Popen.pid` on Windows is the
    pid of the Windows-side wrapper, which means nothing to /proc.
    """
    victim = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              start_new_session=os.name == "posix")
    assert victim.stdin is not None and victim.stdout is not None
    victim.stdin.write(script_bytes('echo $$\nsleep 120\n'))
    victim.stdin.close()
    victim.stdin = None
    pid = victim.stdout.readline().decode().strip()
    assert pid.isdigit(), pid
    return victim, pid


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_refuses_a_group_that_does_not_hold_the_gpu_lock(tmp_path: Path) -> None:
    """The live branch: the recorded pid is alive, leads its own group and its start time
    matches — every check the earlier version made — but no member of that group holds this
    GPU's lock file, so it is not this batch."""
    held = start_claimant(tmp_path, "120")
    victim, victim_pid = sacrificial_group(tmp_path)
    try:
        written = wait_for_record(tmp_path)
        rewrite_record(tmp_path, pid=victim_pid, pgid=victim_pid,
                       started=start_token(victim_pid))
        result = stop(tmp_path)
        assert result.returncode == 3, result
        assert "holds no descriptor" in result.stderr
        assert alive(victim_pid), "the stop signalled a group it had no business in"
        assert written["pid"] != victim_pid
    finally:
        victim.kill()
        held.kill()


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_refuses_when_the_record_names_a_group_that_is_not_the_lock_holder(
        tmp_path: Path) -> None:
    """The dead-shell branch, and the case from the review: the lock is held (by a NEWER
    batch), while the record still names an older batch's group. Nothing may be signalled."""
    held = start_claimant(tmp_path, "120")
    victim, victim_pid = sacrificial_group(tmp_path)
    try:
        wait_for_record(tmp_path)
        # started=1 cannot be anyone's real start time, so the recorded shell counts as gone;
        # the lock, however, is held — by `held`, which is not in the victim's group.
        rewrite_record(tmp_path, pid=victim_pid, pgid=victim_pid, started="1")
        result = stop(tmp_path)
        assert result.returncode == 3, result
        assert "NOT in process group" in result.stderr
        assert alive(victim_pid), "the stop signalled a group it had no business in"
    finally:
        victim.kill()
        held.kill()


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_refuses_when_the_recorded_group_is_not_the_processs_own(
        tmp_path: Path) -> None:
    """A record whose group no longer matches the process: without setsid a batch sits in
    someone else's group (pytest's, in these tests), and signalling that group would hit
    everything in it."""
    held = start_claimant(tmp_path, "120")
    try:
        written = wait_for_record(tmp_path)
        rewrite_record(tmp_path, pid=written["pid"], pgid="999999",
                       started=written["started"])
        result = stop(tmp_path)
        assert result.returncode == 3, result
        assert "not the recorded 999999" in result.stderr
        assert alive(written["pid"])
    finally:
        held.kill()


def alive(pid: str) -> bool:
    """Whether that pid is still running, asked of the bash the tests use."""
    return in_bash(f'kill -0 {pid} 2> /dev/null').returncode == 0


def start_token(pid: str) -> str:
    """The same token the lock script records, asked of the same bash."""
    return in_bash(f'source "{bash_path(LOCK_SCRIPT)}"\nstart_token {pid}').stdout.strip()


def rewrite_record(tmp_path: Path, *, pid: str, pgid: str, started: str,
                   gpu: str = "0") -> None:
    (tmp_path / f"batch_gpu{gpu}.pid").write_text(
        f"pid={pid}\npgid={pgid}\ngpu={gpu}\nprefix=batch\nstarted={started}\n",
        encoding="utf-8", newline="\n")


# ------------------------------------------------------- a stop must not report a zombie
# Review 2026-11-20: a process that has exited but whose parent has not reaped it yet is
# still listed by `ps`, so "is the group still alive?" answered with a plain `ps` would say
# yes for work that has already stopped — and the stop would wait ten seconds and then report
# failure (code 5) for a batch it had just stopped correctly. The window is short on the
# server, where a setsid'ed batch is reaped by init, and wide whenever the parent waits before
# reaping, which is what these tests do: `held.communicate()` is only called AFTER the stop
# has returned, exactly so that this case is covered rather than avoided.


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_group_members_leaves_out_a_process_nobody_has_reaped_yet() -> None:
    """A real zombie, made the way the stop tests make one: a process whose parent (this
    pytest run) has not called wait on it yet.

    It is NOT made with `sleep 0 &` inside bash, because bash reaps its own background jobs
    quickly and the test would then pass without ever seeing a zombie.
    """
    child = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=os.name == "posix")
    assert child.stdin is not None and child.stdout is not None
    child.stdin.write(script_bytes("echo $$\nexit 0\n"))
    child.stdin.close()
    child.stdin = None
    pid = child.stdout.readline().decode().strip()
    assert pid.isdigit(), pid
    try:
        state = ""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = in_bash(f'ps -o stat= -p {pid} | tr -d " "').stdout.strip()
            if state.startswith("Z"):
                break
            time.sleep(0.05)
        if not state.startswith("Z"):
            pytest.skip(f"no zombie appeared on this machine (ps says {state!r}); the "
                        f"parent's wait is not what holds the entry here")
        listed = in_bash(f'source "{bash_path(LOCK_SCRIPT)}"\n'
                         f'group_members $(ps -o pgid= -p {pid} | tr -d " ")')
        assert pid not in listed.stdout.split(), (pid, listed.stdout)
    finally:
        child.kill()
        child.communicate(timeout=30)


@pytest.mark.skipif(not HAS_FLOCK, reason="this bash has no flock")
def test_a_stop_reports_success_before_the_parent_reaps_the_shell(tmp_path: Path) -> None:
    """The ordering from the review: the stop is asked for its verdict while pytest, the
    parent of the batch shell, has not reaped it yet."""
    held = start_claimant(tmp_path, "60")
    try:
        written = wait_for_record(tmp_path)
        assert written["pgid"] == written["pid"], written
        result = stop(tmp_path)
        # Nothing has been reaped at this point - held.communicate() comes after.
        assert result.returncode == 0, result
        assert "still alive after 10 seconds" not in result.stderr
        held.communicate(timeout=30)
    finally:
        held.kill()


def test_both_endings_point_at_the_manifest_checker() -> None:
    # "No command failed" is not "the output is fit to quote", so the hint has to be on the
    # failing path too - that is the run most likely to be read in a hurry.
    assert TEXT.count("check_hint") == 3          # the definition and both exit paths
    assert "scripts/check_diagnostics_manifests.py runs/evaluator $LOG" in TEXT
