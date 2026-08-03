#!/usr/bin/env bash
# §12.3 C 阶段：定标探针（**非结论**）。在 niche（厚果层）世界里扫 fruit_dry_weight。
#
# 找什么：**「分离了但果实还吃得到」的工作点**。§12.3 C 明写这一阶段的核心不是找「分离最大」
# ——把果层挪到没人去的地方等于把果层删了，不是造生态位。
#
# 为什么基线是 niche 不是默认配置（§12.3.A' 判决）：默认世界里果层与种群的重叠是**纯几何**的
# （sel_ratio_water 1.009，扣掉水距后落回零模型），可下降空间 ÷ MDE 只有 0.52——手术做对了
# 也测不出下降。niche 世界里是行为性的（3.712，12/12，p=0.00049），同一比值 6.62。
#
# 为什么每臂 3 个种子而不是 §12.3 C 写的「单种子」：A 阶段实测了噪声，niche 世界里
# sel_ratio_water 的 σ̂_W=0.439（CV 12%）、frugivory_frac 0.037（CV 13%），而
# carnivore_frac 的 2×噪声占基线 71%、population 的 σ_B=0——**单 run 读不出护栏有没有变差，
# 只能读有没有塌到 0**。3 个种子仍是探针（远低于 12×2 的判决协议，不产出结论），
# 但至少让主口径的读数不是一次抽样。
set -u
RUN=outputs/20260803-dryC
SCRIPT=explorations/20260803-overlapA/measure_overlap.py
COMMON=(20000 --json)
NICHE=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
       --set regrow_baseline=0.010 --set plant_max=2.0)
# w=0 也跑 3 个：不直接引用 A 的 niche 臂，因为 death_thirst_frac 是 A 跑完之后才加进
# measure_overlap.py 的，A 的日志里没这个字段；而且像对像（同脚本版本、同种子、同重复数）
# 比省 3 个 run 值钱。A 的 niche 臂仍留作交叉核对。
WEIGHTS=(0.0 0.25 0.5 0.75 1.0)
SEEDS=(0 1 2)

mkdir -p "$RUN"
for w in "${WEIGHTS[@]}"; do
  tag="w${w//./}"
  for s in "${SEEDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" \
      "${COMMON[@]}" "${NICHE[@]}" --set "fruit_dry_weight=$w" \
      --seed "$s" > "$RUN/${tag}_s${s}_r1.log" 2>&1 &
  done
done
wait
echo "done: $(ls "$RUN"/w*_s*_r1.log | wc -l) logs"
