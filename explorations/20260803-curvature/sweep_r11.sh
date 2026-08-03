#!/usr/bin/env bash
# R11：降需求造出的双峰，是饱和松开还是兑换率变了。判据见 docs/multispecies_program.md §15
# （先写后跑，commit b73b2dc）。
#
# 只跑 eat_rate=0.5 那半边——N15/T15 复用 R10 的 N/K10 臂（同世界、同 forage_curvature=1.0、
# 同步数），dip_ratio 从它们的 bin_n 逐位重算。省 48 个 run。
#
# 归因闸是本轮重点：N05m/T05m 把两层需求同比降 3 倍，饱和照样松开而兑换率不变。
# 只有 T05 塌而 T05m 不塌 ⇒ 双峰是操作产物，必须写成负结果。
set -u
RUN=outputs/20260804-demand
SCRIPT=explorations/20260803-curvature/fitness_surface.py
CONC=${CONC:-5}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
   --set regrow_baseline=0.010 --set plant_max=2.0 --set water_sea_dist=1
   --set grass_shade=1.3 --set forage_curvature=1.0)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
# 臂名:forage_tradeoff:eat_rate:fruit_eat_rate
ARMS=(N05:0.0:0.5:1.0 T05:1.0:0.5:1.0 N05m:0.0:0.5:0.333 T05m:1.0:0.5:0.333)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260804-demand"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §15（先写后跑，commit b73b2dc）"
  echo "world: ${W[*]}"; echo "arms: ${ARMS[*]}"
  echo "N15/T15 复用: outputs/20260803-curvature/r10/{N,K10}_*"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]}"; echo "started: $(date -Is) free: $(free_mib)"
} >> "$RUN/provenance.txt"
for a in "${ARMS[@]}"; do
  IFS=: read -r tag t er fer <<< "$a"
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${tag}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" 20000 --json \
      --seed "$s" "${W[@]}" --set "forage_tradeoff=$t" --set "eat_rate=$er" \
      --set "fruit_eat_rate=$fer" > "$out" 2>&1 &
    sleep 6
  done; done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/96" >> "$RUN/provenance.txt"; echo "done: $ok/96"
