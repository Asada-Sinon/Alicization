#!/usr/bin/env bash
# §12.8 的判决性探针：**基线凹度是不是供给饱和造成的？**
#
# R10 判决（feasibility.md §12）算出：曲率杠杆能注入 +0.087，而实测基线 W'' = −0.137，
# 所以 k 调到哪都翻不了号。要让曲率扫描有意义，得先把基线凹度打下来。
# analyst 的 [提案] 机制是供给饱和：gain = demand·min(1, plant_cell/demand_cell)，
# 「提高自己偏好层的 demand」买不到成比例的摄入，而「放弃另一层」的损失全额兑现
# ⇒ 实现摄入是自身乘子的凹函数，与 k 无关。
#
# **用两种互斥的方式降饱和，这是本探针的设计要点**：
#   S1 降 eat_rate  —— 降需求，**一点食物都没加**
#   S2 抬 regrow_baseline —— 加供给，**但那是 experiments.md §5 已关闭的「加食物」**
# 若只有 S2 让 W''→0，那就是「加食物」的老故事，不是饱和；
# **只有 S1 也让 W''→0，饱和机制才算坐实**，而且它给出一条不加承载力的活路。
#
# 全部 k=1.0（不动曲率）——本探针只问基线凹度，不问手术。
set -u
RUN=outputs/20260803-curvature/sat
SCRIPT=explorations/20260803-curvature/fitness_surface.py
CONC=${CONC:-5}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
   --set plant_max=2.0 --set water_sea_dist=1 --set grass_shade=1.3
   --set forage_tradeoff=1.0 --set forage_curvature=1.0)
SEEDS=(0 1 2 3); REPS=(1 2)
# 臂名:额外 --set
ARMS=("S0:regrow_baseline=0.010" "S1:regrow_baseline=0.010:eat_rate=0.5"
      "S2:regrow_baseline=0.060")
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260803-curvature/sat  (探针，非结论)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "问题: 基线 W'' 是不是供给饱和造成的？两种互斥降饱和方式"
  echo "arms: ${ARMS[*]}"; echo "seeds: ${SEEDS[*]} reps: ${REPS[*]}"
  echo "started: $(date -Is)"; } >> "$RUN/provenance.txt"
for a in "${ARMS[@]}"; do
  tag="${a%%:*}"; rest="${a#*:}"
  EXTRA=(); IFS=: read -ra parts <<< "$rest"
  for kv in "${parts[@]}"; do EXTRA+=(--set "$kv"); done
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${tag}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" 20000 --json \
      --seed "$s" "${W[@]}" "${EXTRA[@]}" > "$out" 2>&1 &
    sleep 5
  done; done
done
wait
echo "done: $(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)/24"
