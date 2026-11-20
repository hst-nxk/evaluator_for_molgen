#!/usr/bin/env bash
# Batch of 2026-11-19: THE THREE DIAGNOSTICS ASSIGNED AT THE 2026-11-18 MEETING.
#
# How to run (on the Brev server, in a shell where `conda activate nxk` has been done):
#   cd /data/khainx/evaluator_for_molgen
#   python -m pip install scikit-learn scipy xgboost      # once only, for task 1
#   setsid nohup bash scripts/server_run_diagnostics_1911.sh 1 > /dev/null 2>&1 &
#   tail -f /data/khainx/evaluator_for_molgen/runs/diagnostics_1911_gpu1.log
#
# The single argument is the GPU INDEX (required; it is never auto-detected). The log and
# the pid file are named after it, so two batches on two GPUs do not write over each
# other: with one shared pid file, `kill` aimed at whichever batch started last, and the
# batch that finished first deleted the file while the other was still running. One
# shared log also interleaved the two batches' lines, which is the same problem one step
# later - the numbers are read from the log.
#
# To stop the batch on GPU 1 part-way:
#   bash scripts/stop_gpu_batch.sh 1
# Not `kill -- -$(cat ....pid)`: a pid alone cannot be checked. A batch killed with SIGKILL
# leaves its record behind, the operating system hands that number to something else later,
# and that something else may be a batch of THIS script on another GPU - where even
# `ps | grep server_run_diagnostics_1911` matches. The stop script compares the GPU, the
# process group and the process start time (which a reused pid cannot inherit) before it
# signals anything, and it also recognises the case where only the shell died while the
# python run still holds the GPU through the inherited lock.
#
# ------------------------------------------------------------------------------------
# THREE TASKS, CHEAPEST FIRST
#
# TASK 1 — CLASSICAL BASELINES (2026-11-18 meeting):
#   "if this complicated 3.7-million-parameter deep learning model of yours cannot even
#    beat a linear regression, then it is not solving anything."
#   Run Linear, Ridge, Lasso and XGBoost on EXACTLY the features, EXACTLY the split and
#   EXACTLY the metrics of a real training run, scoring both train and validation.
#
# TASK 2 — CAPACITY CHECK (two readings of the same observation collided at the meeting):
#   one that "an architecture this size, with this little data, cannot train to a good
#   result"; the other that "the point is it is not even overfitting". Give the model a
#   1,024-row sample, switch off weight decay and every dropout, then measure on THAT
#   SAME sample.
#
# TASK 4 — GRADIENT NORMS PER BLOCK: after loss.backward() on train rows, the norm of each
#   block's own parameter gradients. A block can have non-zero activations and still
#   receive exactly zero gradient, which the ReLU count and the input-sensitivity measure
#   cannot see.
#
# TASK 3 — PROTEIN CONTROL (from the question "why would feeding in the protein branch
#   not make sense when we are not splitting by protein"): keep the model, the parameter
#   count and the distribution of protein embeddings, and only SHUFFLE the row -> protein
#   link. Every shuffled run is paired with ONE control run on the same seed without the
#   shuffle, because only the difference between them is a meaningful number.
#
# ------------------------------------------------------------------------------------
# READ THIS BEFORE COMPARING NUMBERS
#
# * TWO SPLITS, both run on purpose. The question from the meeting is about the case where
#   the split is NOT by protein, i.e. the PAIR75 scope (`inputs.partition_override_pairs`:
#   proteins seen, pairs unseen). The SUB100 scope (split by protein cluster) is kept
#   alongside because every earlier number in this project was measured there. A SUB100 run
#   does NOT answer the meeting's question, and vice versa.
# * HYPER-PARAMETERS are taken from the 2026-11-14 reference batch
#   (`server_run_pairsplit_1411.sh`): residual 7x102 with factor 1.849, dropout 0.5 /
#   hidden 0.3936, learning rate 9.86e-05, weight decay 0, checkpoint selected on
#   `pairwise_accuracy_macro` over the `strict` slice. Run with the config defaults instead
#   (head `plain`, lr 1e-3, selection on MAE) and the results compare with NO table this
#   project has ever reported.
# * Task 2 runs TWO architectures: the default `plain` and the reference `residual`. A
#   capacity verdict applies only to the exact architecture of that run. Tasks 2b and 2c
#   differ in ONE thing only - 2c keeps dropout - so that their difference measures
#   dropout. Both therefore take the same $REF hyper-parameters; an earlier version of this
#   script gave 2b the config default learning rate (1e-3) and 2c the reference one
#   (9.86e-05), which made the pair unreadable.
# * Task 2a-2c fit a 1,024-row sample drawn from the first 200,000 train rows, and those
#   rows are ALL `Ki` (measured 2026-11-19). Their verdict is about fitting a small Ki
#   sample, nothing wider. Task 2d is the 100k question: it fits the whole SUB100 subset.
# * Task 3 scores the best checkpoint on TRAIN as well (tasks 3a'-3d'), because the
#   requirement from the meeting is to read train and validation side by side, and a
#   training run only records validation at its selection point. Those steps and task 4
#   take the checkpoint from the manifest of the train run that just succeeded, found by
#   the UNIQUE --run-id that run was given (scripts/pick_train_checkpoint.py), never "the
#   newest .pt" and never "the newest manifest": a dead run, a checkpoint saved mid-run, or
#   a second batch writing into the same directory would otherwise hand them another model.
# * Task 4 measures gradients with the labels put back into the space the checkpoint was
#   trained in, and its manifest records which loss and which label scaling were used.
# * Task 1 does NOT standardise its features; fingerprints are 0/1 bits while protein
#   embeddings are real numbers, so one alpha lands on the two blocks with two different
#   effective strengths. See the docstring of src/molopt/evaluator/classical.py.
# * Task 3 is read as a DIFFERENCE, on one seed. This project's seed-noise floor is 0.0021
#   on macro direction accuracy: a gap smaller than that concludes nothing, and even a
#   large gap only says "the real protein link helps under these conditions" — it does not
#   prove the original model ignores the protein entirely (a retrained model can
#   compensate).
# ------------------------------------------------------------------------------------

set -u -o pipefail

if [[ $# -lt 1 ]]; then
  echo "a GPU index is required, for example: bash scripts/server_run_diagnostics_1911.sh 1" >&2
  exit 2
fi
GPU="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Named after the GPU, so two batches never share it: see the header. The pid file and the
# lock file are named the same way, by claim_gpu below.
LOG="$ROOT/runs/diagnostics_1911_gpu$GPU.log"
SUBSET="work/splits/train_subset_c200_r500.parquet"
PAIR_SPLIT="work/splits/pair_split_1411.parquet"
SEED=20261023
PERMUTE_SEED=20261119
FAILED=0

mkdir -p "$ROOT/runs" "$ROOT/.cache"
# ONE batch per GPU, claimed in a single step. Reading a pid file and then writing it is two
# steps, and two batches launched in the same second both read "nobody there" before either
# writes - so both would run on one GPU, fight for its memory and append to one log. The
# lock also owns the pid file: it is written under the lock and removed by a trap on EXIT,
# including the early exit for a missing split file below. See scripts/gpu_batch_lock.sh,
# which sets PID_FILE.
source "$ROOT/scripts/gpu_batch_lock.sh"
claim_gpu "$GPU" "$ROOT/.cache" "diagnostics_1911" || exit $?
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

for file in "$SUBSET" "$PAIR_SPLIT"; do
  if [[ ! -f "$file" ]]; then
    echo "missing $file; build it first with scripts/build_train_subset.py or scripts/build_pair_split.py" | tee -a "$LOG"
    exit 3
  fi
done

# HOW A RESULT IS CHOSEN. Every command that picks one result out of several has to pick it
# the same way, or the comparison is between two different questions. The MLP picks a
# checkpoint by this metric on this slice; --mode classical picks the alpha of ridge and
# lasso with the same code path (train.selection_score, see selected_alpha in the manifest).
# Until 2026-11-20 these two flags lived inside $REF, which the classical tasks did not
# take: ridge and lasso would then have chosen their alpha by mae on the 'all' slice while
# the MLP chose its checkpoint by direction accuracy on 'strict'. Their own variable keeps
# the shared part visible, and the classical tasks take THIS and not $REF, because the rest
# of $REF (dropout, learning rate, GPU memory) means nothing to a scikit-learn fit.
SELECTION="--set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"

# Hyper-parameters of the 2026-11-14 reference batch. The variable names are kept the same
# as in that batch's script so the two can still be read side by side.
REF="$SELECTION"
REF+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"
REF+=" --set gpu_memory_limit_mib=4500 --set eval_batch_size=2048"
RESIDUAL="--set model.head_block=residual --set model.residual_blocks=7"
RESIDUAL+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936 --set model.residual_dropout=0.0"

say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }

# Exit code of the last command `run` executed. Every dependent step (scoring a
# checkpoint on train, measuring its gradients) must run ONLY when that train command
# succeeded, and `run` itself always returns 0 so one dead task cannot stop the batch.
LAST_STATUS=0

# Run one command, sending both stdout and stderr to the log. A failing task does NOT stop
# the batch (the three tasks are independent of each other), but failures are COUNTED and
# the script exits non-zero at the end — otherwise a batch that lost task 1 entirely still
# finishes with the word "DONE" and exit code 0.
# `status=$?` has to be captured RIGHT AFTER the command: writing `if ! "$@"; then echo $?`
# prints the status of the negation (always 0), i.e. it reports "failed with code 0".
run() {
  echo "+ $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local status=$?
  LAST_STATUS=$status
  if [[ $status -ne 0 ]]; then
    FAILED=$((FAILED + 1))
    echo "!! FAILED (code $status) at: $*" | tee -a "$LOG"
  fi
  return 0
}

# EVERY command here is given a unique run id, which lands in the name of every file that
# command writes (see run_stamp in run_evaluator.py). Two reasons, both about two batches
# sharing runs/evaluator/ - which is exactly what happens when a second GPU is used:
#
# * output manifests are named `<mode>_<configuration>_<head>_<stamp>.json` and the stamp is
#   accurate only to the SECOND, so two runs of the same mode and configuration started in
#   the same second write the same path and one silently overwrites the other - including
#   the eval manifests, although each of them loaded the right checkpoint;
# * a step that wants the checkpoint of the train run it just finished cannot otherwise tell
#   its own manifest from the other batch's: "the newest manifest" belongs to whichever job
#   finished last, and with the same split and shuffle seed every check inside the loader
#   passes, so nothing announces it. Nor is "the newest .pt" any better: `--mode train`
#   saves its best checkpoint DURING the run, so that file can belong to a run that died.
#
# $$ is this script's pid, so two batches never mint the same id even for the same task, and
# the task label keeps the file names readable.
RUN_ID=""
new_run_id() { RUN_ID="$1-$$"; }

# The checkpoint of the train run that just finished, or empty when it failed or wrote no
# manifest. Checking LAST_STATUS here is what makes every dependent step conditional on the
# train run itself: `run` deliberately returns 0 so one dead task cannot stop the batch.
checkpoint_of_last_train() {
  if [[ ${LAST_STATUS:-1} -ne 0 ]]; then
    return 0
  fi
  python "$ROOT/scripts/pick_train_checkpoint.py" "$ROOT/runs/evaluator" "$RUN_ID" \
    2>> "$LOG" || true
}

say "TASK 1a - classical baselines, SUB100, with the protein block"
new_run_id t1a
run python run_evaluator.py --run-id "$RUN_ID" --mode classical --config frozen \
  --baseline all --set "inputs.train_subset=$SUBSET" \
  $SELECTION

say "TASK 1b - classical baselines, SUB100, protein block DROPPED (feature control)"
new_run_id t1b
run python run_evaluator.py --run-id "$RUN_ID" --mode classical --config no_protein \
  --baseline all --set "inputs.train_subset=$SUBSET" \
  $SELECTION

say "TASK 1c - classical baselines, PAIR75 (proteins seen, pairs unseen)"
new_run_id t1c
run python run_evaluator.py --run-id "$RUN_ID" --mode classical --config frozen \
  --baseline all --set "inputs.partition_override_pairs=$PAIR_SPLIT" \
  $SELECTION

say "TASK 2a - capacity, DEFAULT architecture (plain 512-256), regularisation off"
new_run_id t2a
run python run_evaluator.py --run-id "$RUN_ID" --mode capacity --config frozen --head direct

# 2b and 2c take the SAME $REF $RESIDUAL; the only difference is --regularized, so their
# difference is the effect of dropout at the reference architecture. Note that $REF sets
# weight_decay=0.0, so even 2c runs without decay - the pair isolates dropout alone.
say "TASK 2b - capacity, REFERENCE architecture (residual 7x102), regularisation off"
new_run_id t2b
run python run_evaluator.py --run-id "$RUN_ID" --mode capacity --config frozen \
  --head direct $REF $RESIDUAL

say "TASK 2c - same architecture, same hyper-parameters, but KEEPING dropout"
new_run_id t2c
run python run_evaluator.py --run-id "$RUN_ID" --mode capacity --config frozen \
  --head direct --regularized $REF $RESIDUAL

# The 100k question. Fitting all 100,000 rows of SUB100 leaves no sampling window, so this
# is the only capacity run whose verdict is about the real training set rather than about a
# 1,024-row Ki sample. Last of the capacity runs because it is by far the most expensive.
say "TASK 2d - capacity on the WHOLE SUB100 subset (100,000 rows), regularisation off"
new_run_id t2d
run python run_evaluator.py --run-id "$RUN_ID" --mode capacity --config frozen \
  --head direct $REF $RESIDUAL --set "inputs.train_subset=$SUBSET" \
  --set capacity.rows=100000 --set capacity.read_rows=100000

# Score a checkpoint on TRAIN. First argument: the label of THIS scoring run, so its own
# manifest carries an id too. The flags MUST repeat the split and the shuffle seed of the
# training run: `--mode eval` rebuilds the rows from the command it is given, and since
# 2026-11-19 both are written into the .pt and the loader refuses a mismatch - so a missing
# flag stops the command instead of quietly scoring the wrong rows.
score_train() {
  local label="$1"
  local checkpoint="$2"
  shift 2
  if [[ -z "$checkpoint" ]]; then
    echo "!! that training run produced no usable checkpoint, skipping the train half" | tee -a "$LOG"
    FAILED=$((FAILED + 1))
    return 0
  fi
  new_run_id "$label"
  run python run_evaluator.py --mode eval --run-id "$RUN_ID" --checkpoint "$checkpoint" \
    --partition train "$@"
}

say "TASK 3a - PAIR75, CONTROL: protein not shuffled (this is the meeting's question)"
new_run_id t3a
run python run_evaluator.py --mode train --run-id "$RUN_ID" --config frozen --head direct \
  $REF $RESIDUAL --set "inputs.partition_override_pairs=$PAIR_SPLIT" --set "seed=$SEED"
# This one is the control of the meeting's own scope, so task 4 measures its gradients.
# Empty when task 3a failed, and every step that uses it is then skipped rather than
# silently aimed at another run's file.
CONTROL_CHECKPOINT="$(checkpoint_of_last_train)"
say "TASK 3a' - the same checkpoint scored on TRAIN"
score_train t3a-eval "$CONTROL_CHECKPOINT" $REF \
  --set "inputs.partition_override_pairs=$PAIR_SPLIT"

say "TASK 3b - PAIR75, protein SHUFFLED across rows (seed $PERMUTE_SEED)"
new_run_id t3b
run python run_evaluator.py --mode train --run-id "$RUN_ID" --config frozen --head direct \
  $REF $RESIDUAL --set "inputs.partition_override_pairs=$PAIR_SPLIT" --set "seed=$SEED" \
  --set "protein_permute_seed=$PERMUTE_SEED"
say "TASK 3b' - the same checkpoint scored on TRAIN (still shuffled)"
score_train t3b-eval "$(checkpoint_of_last_train)" $REF \
  --set "inputs.partition_override_pairs=$PAIR_SPLIT" \
  --set "protein_permute_seed=$PERMUTE_SEED"

say "TASK 3c - SUB100 (split by protein cluster), CONTROL: not shuffled"
new_run_id t3c
run python run_evaluator.py --mode train --run-id "$RUN_ID" --config frozen --head direct \
  $REF $RESIDUAL --set "inputs.train_subset=$SUBSET" --set "seed=$SEED"
say "TASK 3c' - the same checkpoint scored on TRAIN"
score_train t3c-eval "$(checkpoint_of_last_train)" $REF --set "inputs.train_subset=$SUBSET"

say "TASK 3d - SUB100, protein SHUFFLED across rows"
new_run_id t3d
run python run_evaluator.py --mode train --run-id "$RUN_ID" --config frozen --head direct \
  $REF $RESIDUAL --set "inputs.train_subset=$SUBSET" --set "seed=$SEED" \
  --set "protein_permute_seed=$PERMUTE_SEED"
say "TASK 3d' - the same checkpoint scored on TRAIN (still shuffled)"
score_train t3d-eval "$(checkpoint_of_last_train)" $REF --set "inputs.train_subset=$SUBSET" \
  --set "protein_permute_seed=$PERMUTE_SEED"

# Per-block gradient norms on the control checkpoint of the meeting's own scope (PAIR75,
# unshuffled). This is the direct answer to "do all 7 residual blocks receive gradient",
# which the ReLU and sensitivity measurements cannot give. Measured on TRAIN rows, the only
# place a training gradient means anything.
say "TASK 4 - gradient norms per block + dead ReLU units, PAIR75 control checkpoint"
if [[ -n "${CONTROL_CHECKPOINT:-}" ]]; then
  new_run_id t4
  run python run_evaluator.py --run-id "$RUN_ID" --mode diagnose \
    --checkpoint "$CONTROL_CHECKPOINT" --partition train $REF \
    --set "inputs.partition_override_pairs=$PAIR_SPLIT"
else
  echo "!! task 3a produced no usable checkpoint, skipping task 4" | tee -a "$LOG"
  FAILED=$((FAILED + 1))
fi

# No `rm -f "$PID_FILE"` here: claim_gpu's trap removes it on EVERY exit path, which is the
# point - this line used to be skipped by the early exits.
# Whether the batch finished or not, say how to check what it actually left behind. "No
# command failed" is not the same as "the seven manifests the three tables need are there
# and were all chosen the same way": a classical manifest is rewritten after every fit, a
# dead task leaves no file at all, and the directory also holds manifests from earlier
# months. The checker decides that, by run id, and exits non-zero when a table must not be
# built yet.
check_hint() {
  echo "" | tee -a "$LOG"
  echo "Before building any table, run:" | tee -a "$LOG"
  echo "  python scripts/check_diagnostics_manifests.py runs/evaluator $LOG" | tee -a "$LOG"
}

if [[ $FAILED -ne 0 ]]; then
  say "FINISHED WITH $FAILED FAILED COMMAND(S) - read the '!! FAILED' lines in $LOG before using any number"
  check_hint
  exit 1
fi
say "DONE, no command failed. Manifests in runs/evaluator/: classical_*.json, capacity_*.json, train_*.json"
check_hint
echo "Compare the two runs of task 3 with: python scripts/group_runs.py --slice strict --metric pairwise_accuracy_macro --min-seeds 1" | tee -a "$LOG"
