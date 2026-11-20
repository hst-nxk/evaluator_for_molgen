#!/usr/bin/env bash
# Stop the diagnostics batch on one GPU - and nothing else.
#
#   bash scripts/stop_gpu_batch.sh <GPU index> [cache directory] [prefix]
#
# WHY THIS IS A SCRIPT AND NOT A `kill` ONE-LINER. `kill -- -$(cat ....pid)` trusts a number
# that cannot be trusted. A batch killed with SIGKILL leaves its record behind; the operating
# system hands that pid to something else later, and that something else may be a batch of
# this very script on ANOTHER GPU - where `ps | grep server_run_diagnostics_1911` matches
# too, so even a careful-looking check passes and the stop hits the wrong GPU's whole process
# group. The record written by claim_gpu therefore holds the GPU, the process group and the
# process's start time, and this script checks all three before signalling anything.
#
# It also handles the case where only the parent shell died: the lock is held through a
# descriptor the python run inherited, so the GPU is still busy although the recorded shell
# is gone. Then there IS work to stop, the record cannot identify it by itself, and what this
# script does is show what is still alive in that process group before signalling it.
#
# WHY THE PROCESS GROUP IS CHECKED AND NOT TRUSTED. The group number in the record is the
# one thing a stop acts on, and the record can be out of date: a new batch may have taken the
# lock a moment before it replaces the old record, so the lock belongs to the new batch while
# the number on disk still points at the old one's group - and that number may since have
# been handed to something else entirely. So before signalling, this script requires BOTH:
#
#   * the group is the batch's own (`pgid == pid`, which is what `setsid` in the run
#     instructions produces) - signalling a group the batch merely happens to sit in would
#     hit whatever else is in it;
#   * some member of that group is holding THIS GPU's lock file open, checked through
#     /proc/<pid>/fd. That is the only fact that ties a process group to this GPU; the pid,
#     the group number and the start time can all go stale or be reused.
#
# Exit codes: 0 stopped, 1 nothing is running on that GPU (a stale record is removed), 2 bad
# usage, 3 the record cannot be trusted, 4 flock missing, 5 the signal was sent but the group
# is still alive.

set -u -o pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/stop_gpu_batch.sh <GPU index> [cache directory] [prefix]" >&2
  exit 2
fi
GPU="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIRECTORY="${2:-$ROOT/.cache}"
PREFIX="${3:-diagnostics_1911}"
LOCK_FILE="$DIRECTORY/${PREFIX}_gpu$GPU.lock"
PID_FILE="$DIRECTORY/${PREFIX}_gpu$GPU.pid"

if ! command -v flock > /dev/null; then
  echo "flock is not installed, so whether a batch holds GPU $GPU cannot be told" >&2
  exit 4
fi
source "$ROOT/scripts/gpu_batch_lock.sh"

# Is the GPU still claimed? Taking the lock means nobody holds it; the lock is dropped again
# as soon as this script ends, so asking costs nothing. No lock file at all means no batch
# has ever claimed this GPU here - "free", not "claimed by someone unknown".
held=0
if [[ -e "$LOCK_FILE" ]]; then
  exec 9>> "$LOCK_FILE" || exit 3
  if ! flock -n 9; then
    held=1
  fi
fi

if [[ ! -f "$PID_FILE" ]]; then
  if [[ $held -eq 1 ]]; then
    echo "GPU $GPU is claimed but nothing recorded who claimed it; look for the python run"\
         "with: ps -ef | grep run_evaluator" >&2
    exit 3
  fi
  echo "no batch is running on GPU $GPU"
  exit 1
fi

# group_members comes from scripts/gpu_batch_lock.sh, sourced above.
#
# Does any member of process group $1 hold the file $2 open? This is what proves that a group
# really is the batch of this GPU. /proc is the only way to ask; without it (not Linux) the
# question cannot be answered, and the caller is told so rather than being given a guess.
group_holds() {
  local pgid="$1" target="$2" member descriptor
  if [[ ! -d /proc/$$/fd ]]; then
    return 2
  fi
  for member in $(group_members "$pgid"); do
    for descriptor in /proc/"$member"/fd/*; do
      [[ -e "$descriptor" ]] || continue
      if [[ "$(readlink -f "$descriptor" 2> /dev/null)" == "$target" ]]; then
        return 0
      fi
    done
  done
  return 1
}

# Send SIGTERM to a process group and say whether it actually went away. Returning 0 right
# after `kill` would report success when the group had already vanished, or when the signal
# was refused.
stop_group() {
  local pgid="$1" waited=0
  if ! kill -- "-$pgid" 2> /dev/null; then
    echo "could not signal process group $pgid (gone already, or not ours to signal)" >&2
    return 5
  fi
  while [[ $waited -lt 100 ]]; do
    if [[ -z "$(group_members "$pgid")" ]]; then
      return 0
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  echo "process group $pgid was signalled but is still alive after 10 seconds; look at it"\
       "with: ps -o pid=,args= -g $pgid" >&2
  return 5
}

pid=""; pgid=""; gpu=""; prefix=""; started=""
while IFS="=" read -r key value; do
  case "$key" in
    pid) pid="$value" ;;
    pgid) pgid="$value" ;;
    gpu) gpu="$value" ;;
    prefix) prefix="$value" ;;
    started) started="$value" ;;
  esac
done < "$PID_FILE"

if [[ -z "$pid" || -z "$pgid" || -z "$started" ]]; then
  echo "$PID_FILE is not a record this script can check (written by an older version?);"\
       "stop the batch by hand after checking what it is: ps -o pid=,args= -p '$pid'" >&2
  exit 3
fi
# The GPU in the record must be the GPU asked for. They can only differ if the file was
# copied or edited, and acting on it anyway is how one GPU's stop reaches another's batch.
if [[ "$gpu" != "$GPU" ]]; then
  echo "$PID_FILE records GPU $gpu, not GPU $GPU; refusing to signal anything" >&2
  exit 3
fi

LOCK_TARGET="$(readlink -f "$LOCK_FILE" 2> /dev/null || echo "$LOCK_FILE")"

now="$(start_token "$pid")"
if [[ -n "$now" && "$now" == "$started" ]]; then
  # The recorded shell is alive and is the one that wrote the record. Still not enough to
  # signal its group: check the group is its OWN, and that the group holds this GPU's lock.
  current="$(ps -o pgid= -p "$pid" 2> /dev/null | tr -d " ")"
  if [[ "$current" != "$pgid" ]]; then
    echo "pid $pid is now in process group $current, not the recorded $pgid; refusing to"\
         "signal either. Look at it with: ps -o pid=,pgid=,args= -p $pid" >&2
    exit 3
  fi
  if [[ "$pgid" != "$pid" ]]; then
    echo "pid $pid does not lead its own process group ($pgid does), so stopping that group"\
         "would hit whatever else is in it. Start the batch with setsid, as the run"\
         "instructions say, or stop this one by hand: kill $pid" >&2
    exit 3
  fi
  group_holds "$pgid" "$LOCK_TARGET"
  case $? in
    0) ;;
    1) echo "process group $pgid holds no descriptor on $LOCK_FILE, so it is not the batch"\
            "of GPU $GPU; refusing to signal it" >&2
       exit 3 ;;
    *) echo "cannot read /proc, so it cannot be checked that process group $pgid is the"\
            "batch of GPU $GPU; stop it by hand after looking: ps -o pid=,args= -g $pgid" >&2
       exit 3 ;;
  esac
  echo "stopping the batch on GPU $GPU: pid $pid, process group $pgid"
  stop_group "$pgid"
  exit $?
fi

# The recorded shell is gone (or the pid now belongs to something else, which the start time
# just proved). Whether there is anything left to stop is decided by the LOCK, not by the pid.
if [[ $held -eq 0 ]]; then
  echo "no batch is running on GPU $GPU; removing the record left by the one that died"
  rm -f "$PID_FILE"
  exit 1
fi
# The shell is gone but the GPU is still claimed: the python run it started inherited the
# lock. The record cannot identify that work - the pid in it is dead or reused - so the group
# is only signalled when a member of it is actually holding this GPU's lock file.
group_holds "$pgid" "$LOCK_TARGET"
case $? in
  0) ;;
  1) echo "the batch shell (pid $pid) is gone and GPU $GPU is claimed by something that is"\
          "NOT in process group $pgid (the record is out of date: a newer batch probably"\
          "holds the lock). Refusing to signal. Find the holder with:"\
          "ls -l /proc/*/fd 2> /dev/null | grep $(basename "$LOCK_FILE")" >&2
     exit 3 ;;
  *) echo "cannot read /proc, so the holder of GPU $GPU cannot be identified; stop it by"\
          "hand after looking: ps -o pid=,args= -g $pgid" >&2
     exit 3 ;;
esac
echo "the batch shell (pid $pid) is gone, but GPU $GPU is still claimed: its python run"\
     "inherited the lock. Still alive in process group $pgid:"
ps -o pid=,args= -g "$pgid" 2> /dev/null || true
echo "stopping process group $pgid"
stop_group "$pgid"
exit $?
