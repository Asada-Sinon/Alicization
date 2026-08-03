#!/usr/bin/env bash
# R9：结构化草层下的资源分割。判据先写后跑，见 docs/multispecies_program.md §13.5。
#
# 2×2 因子（grass_shade × forage_tradeoff），12 种子 × 2 重复 = 96 run。
#
# 为什么 forage_tradeoff 是因子而不是常量：§9.6 的基线不是「方差多大」而是「处理臂比
# 中性漂变窄 38.3%」。没有同世界的中性对照，「方差变大了」这句话无法成立——不同世界的
# 漂变方差本来就不同。主判据因此是**交互项** d_s = (T1−N1) − (T0−N0)。
#
# 为什么 water_sea_dist 钉在 ON 跑遍四臂而不是当因子：认海会把果层总承载抬 +10%，
# 而「加厚果层」是已关闭的路线（experiments.md §5）。钉成常量之后它对任何跨臂对比都是
# 常量，混杂消失。代价是本轮不回答「认海本身有没有用」，那留给 R9-E。
#
# ⚠️ 这个脚本在跑的时候，它读的每一个文件都不许动——脚本自身、underworld/、
# measure_ecotype.py。两种机制都不报错（bash 按偏移量增量读脚本；sweep 是逐批启动进程，
# 改 underworld/ 会让后面几批 import 到新代码）。见 MEMORY.md [LEARN:tooling]。
set -u
RUN=outputs/20260803-shade/r9
SCRIPT=explorations/20260803-shade/measure_ecotype.py
STEPS=20000
CONC=6

# 全部四臂共用的世界：niche 厚果层（出处 outputs/20260803-partition/provenance.txt）
# + 认海。§12.3.A' 判定 C/D 的基线必须是 niche 臂——默认世界里可下降空间 ÷ MDE 只有 0.52。
WORLD=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
       --set regrow_baseline=0.010 --set plant_max=2.0 --set water_sea_dist=1)

SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11)
REPS=(1 2)
# 臂名:grass_shade:forage_tradeoff
ARMS=(N0:0.0:0.0 T0:0.0:1.0 N1:1.3:0.0 T1:1.3:1.0)

mkdir -p "$RUN"
{
  echo "run_id: 20260803-shade/r9"
  echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §13.5（先写后跑）"
  echo "world: ${WORLD[*]}"
  echo "arms: ${ARMS[*]}"
  echo "seeds: ${SEEDS[*]}  reps: ${REPS[*]}  steps: $STEPS  concurrency: $CONC"
  echo "measure: $SCRIPT"
  echo "started: $(date -Is)"
} > "$RUN/provenance.txt"

for a in "${ARMS[@]}"; do
  IFS=: read -r tag shade tradeoff <<< "$a"
  for s in "${SEEDS[@]}"; do
    for r in "${REPS[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n; done
      XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" \
        "$STEPS" --json --seed "$s" "${WORLD[@]}" \
        --set "grass_shade=$shade" --set "forage_tradeoff=$tradeoff" \
        > "$RUN/${tag}_s${s}_r${r}.log" 2>&1 &
    done
  done
done
wait
echo "finished: $(date -Is)" >> "$RUN/provenance.txt"
echo "done: $(ls "$RUN"/*_s*_r*.log | wc -l) logs (expect 96)"
