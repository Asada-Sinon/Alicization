#!/usr/bin/env bash
# R14 的剂量探针：**探针，非结论**（单种子）。
# 找「实测 frugivory_frac 达到 30% / 50%」需要的 fruit_regrow_baseline / regrow_baseline 剂量。
#
# 为什么不能解析算（program.md §11.3 已实测）：抵消剂量非单调，只能实测扫。
# 为什么换旋钮：plant_max/fruit_max 同时是 capacity 和 regrow 的 ref_max，
# baseline 通量对它们**恒定不变**（实测四臂全是 94.14/36.92），
# 所以「等总量再分配」守恒的是承载量不是供给通量，净效果是砍草 33% ——
# 那是 experiments.md 已关闭的「降资源总量」那根轴。
set -u
RUN=outputs/20260804-dose
SCRIPT=explorations/20260804-readouts/trajectory.py
CONC=${CONC:-5}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_water_frac=0.40 --set fruit_energy=4.0 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
   --set forage_curvature=1.0 --set eat_rate=0.5)
# (fruit_regrow_baseline, regrow_baseline) 的剂量对：抬果 / 压草，两侧都扫
DOSES=("0.25:0.010" "0.50:0.010" "1.00:0.010" "0.25:0.005" "0.50:0.005" "1.00:0.005")
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260804-dose  (**探针，非结论**，单种子)"
  echo "git HEAD: $(git rev-parse HEAD)"; echo "目的: 找达到实测 frugivory_frac 30%/50% 的剂量"
  echo "doses (fruit_regrow_baseline:regrow_baseline): ${DOSES[*]}"
  echo "started: $(date -Is)"; } >> "$RUN/provenance.txt"
for d in "${DOSES[@]}"; do
  IFS=: read -r frb rb <<< "$d"
  out="$RUN/frb${frb}_rb${rb}.log"
  grep -q '^JSON ' "$out" 2>/dev/null && continue
  while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
  while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" 22000 --json --seed 0 \
    --checkpoints 1 "${W[@]}" --set "fruit_regrow_baseline=$frb" \
    --set "regrow_baseline=$rb" > "$out" 2>&1 &
  sleep 5
done
wait
echo "done: $(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)/${#DOSES[@]}"
