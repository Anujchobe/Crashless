#!/bin/bash
# Run on the CARLA machine, server already up. Fine-tune + scratch for the given seeds,
# then 100-episode evaluations of zero-shot, fine-tuned and scratch policies.
# Usage: bash scripts/carla_seeds.sh 1 2
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=runs/transfer
for s in "$@"; do
  [ -f $T/sac_finetune_s$s/eval.json ] || $PY train.py --algo sac --env carla --seed $s \
      --init-from $T/sac_2d_s$s/model.pt --learning-starts 1000 --steps 30000 \
      --out $T/sac_finetune_s$s > $T/finetune_s$s.log 2>&1
  [ -f $T/sac_scratch_s$s/eval.json ] || $PY train.py --algo sac --env carla --seed $s \
      --steps 30000 --out $T/sac_scratch_s$s > $T/scratch_s$s.log 2>&1
done
for s in "$@"; do
  for d in sac_2d_s$s sac_finetune_s$s sac_scratch_s$s; do
    [ -f $T/$d/carla_eval.json ] && grep -q '"episodes": 100' $T/$d/carla_eval.json && continue
    $PY scripts/carla_eval.py $T/$d --episodes 100 > $T/$d.eval.txt 2>&1
  done
done
echo ALL_DONE > $T/seeds_done.txt
