# One batch per GPU, claimed atomically. Sourced by the diagnostics batches; not run alone.
#
#   source scripts/gpu_batch_lock.sh
#   claim_gpu "$GPU" "$ROOT/.cache" "diagnostics_1911" || exit $?
#
# Stopping a batch goes through scripts/stop_gpu_batch.sh, which reads the record this
# function writes.
#
# WHY A LOCK AND NOT A PID FILE. "Read the pid file, and write it if nobody is alive" is two
# steps, so two batches launched at the same second can both read "nobody there" before
# either writes: both then run on one GPU, fight for its memory, append to one log, and the
# one that finishes first deletes the other's pid file. `flock -n` is one step - the kernel
# hands the lock to exactly one of them - so the loser cannot get past this function.
#
# The lock lives on a file descriptor this shell keeps open, so the kernel releases it when
# the process ends HOWEVER it ends, `kill -9` included: a crashed batch can never leave a
# lock that blocks the GPU for good. That is also why the pid record is written only under
# the lock and removed by a trap on EXIT, instead of by a line at the end of the batch: the
# early exit for a missing split file used to leave the pid behind.
#
# WHY THE RECORD HOLDS MORE THAN A NUMBER. A pid alone cannot be checked. If a batch is
# SIGKILLed, its record survives; the operating system then hands that number to something
# else - possibly to a batch of this very script on ANOTHER GPU, where even
# `ps | grep server_run_diagnostics_1911` matches - and a stop command aimed at the first GPU
# would signal the second one's whole process group. So the record also holds the GPU, the
# process group, and the process's start time, which the kernel never repeats for a reused
# pid. See start_token and scripts/stop_gpu_batch.sh.
#
# Sets PID_FILE and LOCK_FILE in the caller. Returns 4 when another batch holds the GPU
# (the same code the batch exits with), 1 when flock is missing or the files cannot be
# written.

# The start time of a process, in clock ticks since boot: the one part of a process's
# identity that a reused pid cannot inherit. Prints nothing when the process is gone.
#
# Field 22 of /proc/<pid>/stat, counted AFTER the ')' that closes the command name - the
# name itself may hold spaces and brackets, so the fields cannot simply be counted from the
# start of the line. `ps -o lstart=` is the fallback for a kernel without /proc; it is only
# accurate to the second, which is still enough to catch a pid handed out later.
start_token() {
  local pid="$1"
  if [[ -r "/proc/$pid/stat" ]]; then
    awk -F')' '{split($NF, fields, " "); print fields[20]}' "/proc/$pid/stat"
  else
    ps -o lstart= -p "$pid" 2> /dev/null | tr -s ' ' '_' | tr -d '\n'
  fi
}

# Every LIVING pid in a process group, one per line. Empty when the group is gone.
#
# Zombies are left out on purpose. A process that has exited but whose parent has not reaped
# it yet still appears in `ps`, so counting it would mean "the group is still alive" for work
# that has already stopped - and the stop would report failure (code 5) for a batch it had
# just stopped correctly. The window is short on the server, where a setsid'ed batch is
# reaped by init, and wide whenever the parent is a program that waits before reaping, which
# is exactly what the tests do.
group_members() {
  ps -o pid=,stat= -g "$1" 2> /dev/null | awk '$2 !~ /^[Zz]/ {print $1}' || true
}

claim_gpu() {
  local gpu="$1"
  local directory="$2"
  local prefix="$3"
  # Named after the GPU and NOTHING else. The prefix used to be part of both names, which
  # made the claim per SCRIPT instead of per GPU: scripts/server_run_diagnostics_1911.sh
  # took diagnostics_1911_gpu1.lock while scripts/server_run_protein_seeds_2011.sh took
  # protein_seeds_2011_gpu1.lock, both succeeded, and two batches then shared one GPU -
  # exactly what this file exists to prevent. The prefix is still recorded INSIDE the pid
  # file, where it names the batch without keying the lock.
  LOCK_FILE="$directory/gpu$gpu.lock"
  PID_FILE="$directory/gpu$gpu.pid"
  mkdir -p "$directory" || return 1
  # Without flock there is no atomic claim at all, and `! flock -n 9` would be TRUE
  # (command not found), which reads exactly like "another batch holds the GPU". Say what
  # is actually wrong instead. flock ships with util-linux, so it is there on the server;
  # it is missing in Git Bash on Windows.
  if ! command -v flock > /dev/null; then
    echo "flock is not installed, so one batch per GPU cannot be enforced; install"\
         "util-linux or run the batch on the server" >&2
    return 1
  fi
  # Descriptor 9 stays open for the life of this shell; see the header.
  exec 9>> "$LOCK_FILE" || return 1
  if ! flock -n 9; then
    echo "a batch is already running on GPU $gpu; stop it with:" >&2
    echo "  bash scripts/stop_gpu_batch.sh $gpu" >&2
    return 4
  fi
  # One `key=value` per line, read by scripts/stop_gpu_batch.sh. `pgid` is what a stop
  # signals: the batch is launched with setsid, so the shell leads its own group and the
  # python run beneath it is in that group.
  {
    echo "pid=$$"
    echo "pgid=$(ps -o pgid= -p $$ | tr -d ' ')"
    echo "gpu=$gpu"
    echo "prefix=$prefix"
    echo "started=$(start_token $$)"
  } > "$PID_FILE"
  trap "rm -f '$PID_FILE'" EXIT
  return 0
}

# Refuses the batch when the GPU does not physically have the memory the run will ask for.
#
#   require_free_vram "$GPU" "${MEMORY_LIMIT:-4500}" || exit $?
#
# WHY. gpu_memory_limit_mib is a CAP on this process, not a reservation: the kernel hands
# out what is there, and another user's job can take the card between the moment a GPU is
# chosen and the moment training starts. On 2026-11-21 GPU 6 had 9,242 MiB free when it was
# picked and 20 MiB by the time the batch ran, so two FULL seeds died at their first
# backward pass - after 30 s, two W&B runs and a lock that then had nothing to protect.
# Reading the free memory here costs one nvidia-smi call and fails in a second instead.
#
# The headroom is not decoration: the cap counts this process's allocator, while CUDA
# context, cuBLAS workspaces and the driver itself also need room on the card, and that is
# several hundred MiB before a single tensor exists.
#
# Returns 5 when the card is too full (a code no other early exit uses), 1 when nvidia-smi
# cannot be read - unreadable is NOT treated as free.
require_free_vram() {
  local gpu="$1"
  local needed="$2"
  local headroom="${3:-1024}"
  if ! command -v nvidia-smi > /dev/null; then
    echo "nvidia-smi is not on PATH, so free VRAM cannot be checked; run this on the"\
         "server" >&2
    return 1
  fi
  local free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" \
    2> /dev/null | head -1 | tr -d ' \r')"
  if [[ ! $free =~ ^[0-9]+$ ]]; then
    echo "could not read the free memory of GPU $gpu (nvidia-smi said '${free}')" >&2
    return 1
  fi
  if (( free < needed + headroom )); then
    echo "GPU $gpu has ${free} MiB free; this batch needs ${needed} MiB plus ${headroom}"\
         "MiB of headroom for the CUDA context. Another job is using the card - pick"\
         "another GPU or wait." >&2
    return 5
  fi
  echo "GPU $gpu: ${free} MiB free, batch capped at ${needed} MiB"
  return 0
}
