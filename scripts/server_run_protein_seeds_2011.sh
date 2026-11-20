#!/usr/bin/env bash
#
# Does the protein branch contribute anything? - the same question as task 3 of the
# 2026-11-19 batch, asked three times so the answer has a noise scale.
#
#   setsid nohup bash scripts/server_run_protein_seeds_2011.sh 1 > /dev/null 2>&1 &
#   tail -F runs/protein_seeds_2011_gpu1.log
#   python scripts/protein_control_seeds.py runs/evaluator runs/protein_seeds_2011_gpu1.log
#
# WHY. The 2026-11-19 batch measured pairwise_accuracy_macro 0.7778 with the real protein
# column (t3a) and 0.7805 with it shuffled across rows (t3b): +0.0027, i.e. destroying
# every row's protein identity made the model slightly BETTER. One run per side cannot tell
# that apart from run-to-run noise, and a null result without a noise scale cannot be
# written down - "no difference" and "a difference smaller than our noise" are different
# claims. Three pairs give the spread that the one number has to be read against.
#
# WHAT IS HELD FIXED. Everything the 2026-11-19 batch used: the PAIR75 split, the
# 2026-11-14 hyper-parameters, and the same selection metric and slice, so these six runs
# are comparable with t3a/t3b and not only with each other. What varies is the training
# seed, and - on the shuffled side only - the permutation draw. Pair i differs in ONE
# thing: whether the protein column was shuffled. Across pairs both the training seed and
# the permutation draw change, so the spread of the three differences covers both sources
# of noise rather than only the initialisation.
#
# COST. Six train runs on PAIR75 (75,000 train rows, 10 epochs, about 150 s each on an
# H100) - roughly 20 minutes including the data reads. No classical fit, no capacity run.

set -u -o pipefail

if [[ $# -lt 1 ]]; then
  echo "a GPU index is required, for example: bash scripts/server_run_protein_seeds_2011.sh 1" >&2
  exit 2
fi
GPU="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/runs/protein_seeds_2011_gpu$GPU.log"
PAIR_SPLIT="work/splits/pair_split_1411.parquet"
FAILED=0

# Pair i is (SEEDS[i] with the real protein, SEEDS[i] with the protein shuffled by
# PERMUTE[i]). The first seed is the one the 2026-11-19 batch used, so pair 1 reproduces
# t3a/t3b and a mismatch there is a signal in itself - something else changed.
SEEDS=(20261023 20261024 20261025)
PERMUTE=(20261119 20261219 20261319)

mkdir -p "$ROOT/runs" "$ROOT/.cache"
# One batch per GPU, claimed in one step; the lock also owns the pid file and a trap
# removes it on every exit path. See scripts/gpu_batch_lock.sh.
source "$ROOT/scripts/gpu_batch_lock.sh"
claim_gpu "$GPU" "$ROOT/.cache" "protein_seeds_2011" || exit $?
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

if [[ ! -f "$PAIR_SPLIT" ]]; then
  echo "missing $PAIR_SPLIT; build it first with scripts/build_pair_split.py" | tee -a "$LOG"
  exit 3
fi

# The same two flags both sides of the 2026-11-19 batch were given. A pair whose two runs
# selected their checkpoint by different rules would measure the rule, not the protein.
SELECTION="--set selection_metric=pairwise_accuracy_macro --set selection_slice=strict"

REF="$SELECTION"
REF+=" --set model.dropout=0.5 --set learning_rate=9.86e-05 --set weight_decay=0.0"
REF+=" --set gpu_memory_limit_mib=4500 --set eval_batch_size=2048"
RESIDUAL="--set model.head_block=residual --set model.residual_blocks=7"
RESIDUAL+=" --set model.residual_width=102 --set model.residual_hidden_factor=1.849"
RESIDUAL+=" --set model.residual_hidden_dropout=0.3936 --set model.residual_dropout=0.0"

say() { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }

run() {
  echo "+ $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  local status=$?
  if [[ $status -ne 0 ]]; then
    FAILED=$((FAILED + 1))
    echo "!! FAILED (code $status) at: $*" | tee -a "$LOG"
  fi
  return 0
}

# Every command carries its own run id, which lands in the name of every file it writes:
# the timestamp in a manifest name is accurate to the second only, and the summary script
# finds a run by this suffix. t5r<i> is pair i with the real protein, t5s<i> with it
# shuffled.
say "protein control, 3 pairs on PAIR75, pid $$"

for index in 0 1 2; do
  pair=$((index + 1))
  seed="${SEEDS[$index]}"
  permute="${PERMUTE[$index]}"

  say "PAIR $pair - real protein, seed $seed"
  run python run_evaluator.py --mode train --run-id "t5r$pair-$$" --config frozen \
    --head direct $REF $RESIDUAL \
    --set "inputs.partition_override_pairs=$PAIR_SPLIT" \
    --set "seed=$seed"

  say "PAIR $pair - protein SHUFFLED, seed $seed, permutation $permute"
  run python run_evaluator.py --mode train --run-id "t5s$pair-$$" --config frozen \
    --head direct $REF $RESIDUAL \
    --set "inputs.partition_override_pairs=$PAIR_SPLIT" \
    --set "seed=$seed" --set "protein_permute_seed=$permute"
done

summary_hint() {
  echo "" | tee -a "$LOG"
  echo "To read the result:" | tee -a "$LOG"
  echo "  python scripts/protein_control_seeds.py runs/evaluator $LOG" | tee -a "$LOG"
}

if [[ $FAILED -ne 0 ]]; then
  say "FINISHED WITH $FAILED FAILED COMMAND(S) - read the '!! FAILED' lines in $LOG before using any number"
  summary_hint
  exit 1
fi
say "DONE, no command failed"
summary_hint
