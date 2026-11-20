#!/usr/bin/env bash
#
# Everything still owed from 2026-11-20, in one detached sequence. Start it and go away.
#
#   setsid nohup bash scripts/server_run_overnight_2011.sh 1 > /dev/null 2>&1 &
#   tail -F runs/overnight_2011.log
#
# WHAT IT RUNS, in order, on ONE GPU so nothing contends. Analysis first, because it is
# minutes and answers questions already asked; training after, because it is the part that
# needs the night.
#
#   1. Figures for the 2026-11-19 batch. No GPU. Also NAMES every result with no figure.
#   2. Validation row dumps for the three seeds of the protein batch (t5r1/t5r2/t5r3),
#      the seen/unseen breakdown of each, and the across-seed summary. This is the number
#      the whole seen/unseen argument is currently missing: the MLP's macro for the group
#      with NO molecule seen in training is 0.6673 over 215 queries, against the best
#      classical 0.5831, while the row-level measurement of the same group puts the gap at
#      only +0.0150. A margin taken from 215 queries cannot be quoted until the seed's own
#      movement is known.
#   3. The same breakdown on SUB100 (t3c), where validation holds UNSEEN PROTEIN CLUSTERS.
#      PAIR75 keeps pairs unseen but not molecules; SUB100 is the split where the model
#      scores 0.5492, and whether the seen/unseen effect explains that is the question.
#   4. TRAINING, three experiments, each answering a question that is open right now:
#      4a. SUB100 at THREE seeds. SUB100 is the only split that puts unseen protein
#          clusters in validation, it is where the generalisation claim has to be made,
#          and it has ONE seed - the MLP's 0.5492 against lasso's 0.5423 is +0.0069, and
#          nobody knows whether that survives a change of seed. The train side is measured
#          at every point, so these runs also produce the two-line learning curve.
#      4b. PAIR75 WITHOUT the protein branch (--config no_protein), three seeds matched to
#          t5r1/t5r2/t5r3. Shuffling the protein column left the score +0.0030 BETTER, but
#          a shuffle keeps the branch, its parameters and the input size, while removing it
#          changes capacity too. Both measurements are what a reviewer needs before the
#          branch is dropped or kept.
#      4c. SUB100 without the protein branch. Proteins are what SUB100 splits on, so if
#          the branch ever matters it matters here - and it has never been measured.
#   5. Figures and tables again, for everything produced tonight.
#
# The train dump of a split is reused across every model and seed on that split: it only
# supplies the SET OF MOLECULES seen in training, which is a property of the split.
#
# NINE training runs. A PAIR75 run took 2m50s and a SUB100 run about 17 minutes while
# OTHER PEOPLE's jobs held the same GPU at 100% utilisation, so this is roughly two hours,
# not a night - and it is sequential, which is the actual reason to leave it unattended.

set -u -o pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/server_run_overnight_2011.sh <GPU index>" >&2
  exit 2
fi
GPU="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/runs/overnight_2011.log"
PAIR_SPLIT="work/splits/pair_split_1411.parquet"
SUB_SPLIT="work/splits/train_subset_c200_r500.parquet"
DIAGNOSTICS_LOG="runs/diagnostics_1911_gpu1.log"
PROTEIN_LOG="runs/protein_seeds_2011_gpu1.log"
PAIR_TRAIN_DUMP="${PAIR_TRAIN_DUMP:-}"
# A classical seen_molecules report over the SAME validation rows, for the paired
# MLP - baseline table. Optional, and checked by the summary itself: a baseline on other
# rows is refused rather than differenced.
BASELINE_REPORT="${BASELINE_REPORT:-}"
FAILED=0

mkdir -p "$ROOT/runs" "$ROOT/.cache"
export PYTHONUNBUFFERED=1

say() { echo "" | tee -a "$LOG"; echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

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

source "$ROOT/scripts/gpu_batch_lock.sh"
claim_gpu "$GPU" "$ROOT/.cache" "overnight_2011" || exit $?
export CUDA_VISIBLE_DEVICES="$GPU"

# `--mode errors` and `--mode eval` rebuild the model from the checkpoint, so the
# architecture flags below do not steer them; they are here because `--mode train` needs
# them and one variable keeps the two halves of every pair identical.
SETTINGS="--set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"
# MEMORY_LIMIT caps PyTorch's caching allocator, NOT the process: the CUDA context itself
# takes a few hundred MiB on top, outside this budget. The 250 MiB peak measured on
# 2026-11-20 was measured at width 102, which is also what the flags below now pin, so it
# applies to these runs - it would NOT have applied to config.json's width 512.
SETTINGS+=" --set gpu_memory_limit_mib=${MEMORY_LIMIT:-4500} --set eval_batch_size=2048"

# THE WHOLE REFERENCE FLAG SET, copied from scripts/server_run_protein_seeds_2011.sh.
#
# This is not pedantry. config.json's own defaults are width 512, dropout 0.1, learning
# rate 1e-3, weight decay 0.01, while every run these nine must be compared with - t3a,
# t3c, t5r1-t5r3 - was trained at width 102, dropout 0.5, learning rate 9.86e-05, weight
# decay 0, with residual_hidden_factor 1.849 and residual_hidden_dropout 0.3936. Passing
# only head_block and residual_blocks (as the first version of this script did) would have
# changed the architecture AND the optimiser at the same time as the protein branch, so
# step 4b would have measured all three at once and answered nothing. Nine training runs
# would have been spent for no comparable number.
SETTINGS+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"
RESIDUAL="--set model.head_block=residual --set model.residual_blocks=7"
RESIDUAL+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936 --set model.residual_dropout=0.0"
SEEDS=(20261023 20261024 20261025)
CONFIGURATION=frozen
RUN_ID=""
LAST_STATUS=1

new_run_id() { RUN_ID="$1-$$"; }

checkpoint_of_last_train() {
  if [[ ${LAST_STATUS:-1} -ne 0 ]]; then
    return 0
  fi
  python "$ROOT/scripts/pick_train_checkpoint.py" "$ROOT/runs/evaluator" "$RUN_ID" \
    2>> "$LOG" || true
}

# One training run, then the SAME checkpoint scored on TRAIN. The second pass is what makes
# a train-versus-validation figure possible: the gap has to come from one checkpoint on one
# split, and the selected checkpoint is usually not the end of the last epoch.
train_and_score() {
  local label="$1"
  shift
  say "$label - training ($CONFIGURATION)"
  new_run_id "$label"
  run python run_evaluator.py --mode train --run-id "$RUN_ID" --config "$CONFIGURATION" \
    --head direct $SETTINGS $RESIDUAL "$@"
  local checkpoint
  checkpoint="$(checkpoint_of_last_train)"
  if [[ -z "$checkpoint" ]]; then
    echo "!! $label produced no usable checkpoint; its train half is skipped" \
      | tee -a "$LOG"
    return 0
  fi
  say "$label - the SAME checkpoint scored on TRAIN"
  new_run_id "$label-eval"
  run python run_evaluator.py --mode eval --run-id "$RUN_ID" --checkpoint "$checkpoint" \
    --partition train $SETTINGS "$@"
}

# ------------------------------------------------------------------ 1. figures, 19/11
say "step 1 - figures for the 2026-11-19 batch"
if [[ -f "$DIAGNOSTICS_LOG" ]]; then
  run python scripts/figures_for_batch.py runs/evaluator "$DIAGNOSTICS_LOG"
else
  echo "no $DIAGNOSTICS_LOG; skipping its figures" | tee -a "$LOG"
fi

# ------------------------------------------------------------------ 2. three seeds, PAIR75
# The train dump is reused: on one split the set of molecules seen in training is the same
# whatever model or seed produced the predictions.
if [[ -z "$PAIR_TRAIN_DUMP" ]]; then
  PAIR_TRAIN_DUMP="$(ls -dt runs/evaluator/errors_*_train_* 2> /dev/null | head -1)"
fi
if [[ -z "$PAIR_TRAIN_DUMP" || ! -f "$PAIR_TRAIN_DUMP/rows.parquet" ]]; then
  say "step 2 SKIPPED - no PAIR75 train row dump found"
  echo "pass one in: PAIR_TRAIN_DUMP=<dir> bash scripts/server_run_overnight_2011.sh $GPU" \
    | tee -a "$LOG"
else
  say "step 2 - the three seeds of the protein batch, on PAIR75"
  echo "train dump (molecule set): $PAIR_TRAIN_DUMP" | tee -a "$LOG"
  PROTEIN_PID=""
  if [[ -f "$PROTEIN_LOG" ]]; then
    PROTEIN_PID="$(grep -oE '\-\-run-id t5r1-[0-9]+' "$PROTEIN_LOG" | head -1 \
      | grep -oE '[0-9]+$')"
  fi
  if [[ -z "$PROTEIN_PID" ]]; then
    echo "could not read the protein batch pid from $PROTEIN_LOG; skipping step 2" \
      | tee -a "$LOG"
  else
    echo "protein batch pid: $PROTEIN_PID" | tee -a "$LOG"
    REPORTS=()
    for index in 1 2 3; do
      TASK="t5r$index"
      CKPT="$(python scripts/pick_train_checkpoint.py runs/evaluator \
        "$TASK-$PROTEIN_PID" 2> /dev/null)"
      if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
        echo "no checkpoint for $TASK-$PROTEIN_PID; skipping this seed" | tee -a "$LOG"
        continue
      fi
      say "step 2.$index - validation dump, $TASK ($(basename "$CKPT"))"
      run python run_evaluator.py --mode errors --run-id "t7s$index-$$" \
        --checkpoint "$CKPT" --partition validation $SETTINGS \
        --set inputs.partition_override_pairs=$PAIR_SPLIT
      DUMP="$(ls -d runs/evaluator/errors_*_validation_*t7s$index-$$ 2> /dev/null | head -1)"
      if [[ -z "$DUMP" ]]; then
        echo "the dump of t7s$index-$$ is not where it was expected" | tee -a "$LOG"
        continue
      fi
      REPORT="runs/evaluator/seen_molecules_t7s$index-$$.json"
      run python scripts/seen_molecules.py --train "$PAIR_TRAIN_DUMP" \
        --validation "$DUMP" --slice all --label "PAIR75 $TASK" --out "$REPORT"
      [[ -f "$REPORT" ]] && REPORTS+=("$REPORT")
    done
    if [[ ${#REPORTS[@]} -ge 1 ]]; then
      say "step 2 summary - across the seeds"
      # --baseline is the comparison that carries the claim: the spread of the MLP across
      # seeds says how unstable the MLP is, NOT whether it is ahead of lasso. With a
      # classical breakdown over the same rows this prints MLP(seed i) - lasso once per
      # seed. BASELINE_REPORT is optional; without it the summary is printed on its own.
      if [[ -n "${BASELINE_REPORT:-}" && -f "${BASELINE_REPORT:-}" ]]; then
        run python scripts/seen_molecules_seeds.py "${REPORTS[@]}" \
          --baseline "$BASELINE_REPORT"
      else
        echo "no BASELINE_REPORT passed, so there is no MLP - baseline table; pass a " \
             "classical seen_molecules report to get one" | tee -a "$LOG"
        run python scripts/seen_molecules_seeds.py "${REPORTS[@]}"
      fi
    else
      echo "no per-seed report was produced, so there is nothing to summarise" \
        | tee -a "$LOG"
    fi
  fi
fi

# ------------------------------------------------------------------ 3. SUB100
if [[ ! -f "$SUB_SPLIT" ]]; then
  say "step 3 SKIPPED - $SUB_SPLIT is not on this machine"
else
  DIAG_PID=""
  if [[ -f "$DIAGNOSTICS_LOG" ]]; then
    DIAG_PID="$(grep -oE '\-\-run-id t3c-[0-9]+' "$DIAGNOSTICS_LOG" | head -1 \
      | grep -oE '[0-9]+$')"
  fi
  CKPT=""
  if [[ -n "$DIAG_PID" ]]; then
    CKPT="$(python scripts/pick_train_checkpoint.py runs/evaluator "t3c-$DIAG_PID" \
      2> /dev/null)"
  fi
  if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    say "step 3 SKIPPED - no t3c checkpoint found"
  else
    say "step 3 - seen/unseen on SUB100, where validation holds unseen protein clusters"
    echo "checkpoint: $CKPT" | tee -a "$LOG"
    SUB_SETTINGS="$SETTINGS --set inputs.train_subset=$SUB_SPLIT"
    for partition in train validation; do
      run python run_evaluator.py --mode errors --run-id "t8$partition-$$" \
        --checkpoint "$CKPT" --partition "$partition" $SUB_SETTINGS
    done
    SUB_TRAIN="$(ls -d runs/evaluator/errors_*_train_*t8train-$$ 2> /dev/null | head -1)"
    SUB_VALIDATION="$(ls -d runs/evaluator/errors_*_validation_*t8validation-$$ \
      2> /dev/null | head -1)"
    if [[ -z "$SUB_TRAIN" || -z "$SUB_VALIDATION" ]]; then
      echo "the SUB100 dumps are not where they were expected; nothing was analysed" \
        | tee -a "$LOG"
    else
      # `all` and not strict: the slice to compare with the PAIR75 reading, and on SUB100
      # the strict slice is a real subset, so the slice is stated rather than assumed.
      run python scripts/seen_molecules.py --train "$SUB_TRAIN" \
        --validation "$SUB_VALIDATION" --slice all --label "SUB100 t3c" \
        --out "runs/evaluator/seen_molecules_sub100_$$.json"
    fi
  fi
fi

# ------------------------------------------------------------------ 4. training
# measure_train_metrics: a forward pass over the train rows at EVERY measurement, so the
# learning curve carries two lines of the SAME metric instead of validation beside a loss
# that is not comparable with it. Seconds on these splits; not on FULL, which is why it is
# a flag and not the default.
MEASURE="--set measure_train_metrics=true"

if [[ ! -f "$SUB_SPLIT" ]]; then
  say "step 4a SKIPPED - $SUB_SPLIT is not on this machine"
else
  say "step 4a - SUB100 at three seeds: the split with UNSEEN PROTEIN CLUSTERS, run once so far"
  CONFIGURATION=frozen
  for index in 0 1 2; do
    train_and_score "t9a$((index + 1))" --set "inputs.train_subset=$SUB_SPLIT" \
      --set "seed=${SEEDS[$index]}" $MEASURE
  done
fi

if [[ ! -f "$PAIR_SPLIT" ]]; then
  say "step 4b SKIPPED - $PAIR_SPLIT is not on this machine"
else
  say "step 4b - PAIR75 WITHOUT the protein branch, seeds matched to t5r1-t5r3"
  CONFIGURATION=no_protein
  for index in 0 1 2; do
    train_and_score "t9b$((index + 1))" \
      --set "inputs.partition_override_pairs=$PAIR_SPLIT" \
      --set "seed=${SEEDS[$index]}" $MEASURE
  done
fi

if [[ ! -f "$SUB_SPLIT" ]]; then
  say "step 4c SKIPPED - $SUB_SPLIT is not on this machine"
else
  say "step 4c - SUB100 without the protein branch: proteins are what this split splits on"
  CONFIGURATION=no_protein
  for index in 0 1 2; do
    train_and_score "t9c$((index + 1))" --set "inputs.train_subset=$SUB_SPLIT" \
      --set "seed=${SEEDS[$index]}" $MEASURE
  done
fi

# ------------------------------------------------------------------ 5. figures and tables
say "step 5 - figures and tables again, for everything produced tonight"
if [[ -f "$PROTEIN_LOG" ]]; then
  run python scripts/figures_for_batch.py runs/evaluator "$PROTEIN_LOG"
fi
run python scripts/figures_for_batch.py runs/evaluator "$LOG"
run python scripts/diagnostics_tables.py runs/evaluator "$LOG"

say "FINISHED with $FAILED failed command(s)"
echo "read with:" | tee -a "$LOG"
echo "  grep -n 'FAILED\\|SKIPPED\\|NO figure' runs/overnight_2011.log" | tee -a "$LOG"
echo "  sed -n '/step 2 summary/,\$p' runs/overnight_2011.log" | tee -a "$LOG"
echo "  grep -n 'macro\\|FINISHED' runs/overnight_2011.log | tail -40" | tee -a "$LOG"
[[ $FAILED -ne 0 ]] && exit 1
exit 0
