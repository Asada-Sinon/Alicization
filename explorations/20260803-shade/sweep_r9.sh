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
#
# ⚠️ 这张卡是共用的。第一次启动时它上面正跑着一个占 18.4 GiB 的
# `bench_train_v21.py --slug v21-throughput-rerun`（不是本会话的东西），于是 6 并发直接
# 把卡撑爆，头 11 个 run 里有 3 个死于 RESOURCE_EXHAUSTED，而且**剩下的还在污染对方的
# 吞吐量读数**。所以本脚本现在做两件事：
#   1. **显存闸**：每启动一个 run 之前查一次空闲显存，不够就等。单进程实测峰值 ~880 MiB。
#   2. **断点续跑**：已经有 `JSON ` 行的日志直接跳过。撞上 OOM 或被打断之后重跑本脚本
#      即可补齐，不必从头再来。
set -u
RUN=outputs/20260803-shade/r9
SCRIPT=explorations/20260803-shade/measure_ecotype.py
STEPS=20000
CONC=${CONC:-3}            # 默认 3：单进程 ~880 MiB，给共用的卡留足余量
NEED_MIB=${NEED_MIB:-2600} # 启动前要求的空闲显存

# 全部四臂共用的世界：niche 厚果层（出处 outputs/20260803-partition/provenance.txt）
# + 认海。§12.3.A' 判定 C/D 的基线必须是 niche 臂——默认世界里可下降空间 ÷ MDE 只有 0.52。
WORLD=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
       --set regrow_baseline=0.010 --set plant_max=2.0 --set water_sea_dist=1)

SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11)
REPS=(1 2)
# 臂名:grass_shade:forage_tradeoff
ARMS=(N0:0.0:0.0 T0:0.0:1.0 N1:1.3:0.0 T1:1.3:1.0)

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

mkdir -p "$RUN"
{
  echo "run_id: 20260803-shade/r9"
  echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §13.5（先写后跑）"
  echo "world: ${WORLD[*]}"
  echo "arms: ${ARMS[*]}"
  echo "seeds: ${SEEDS[*]}  reps: ${REPS[*]}  steps: $STEPS"
  echo "concurrency: $CONC  gpu gate: ${NEED_MIB} MiB free"
  echo "measure: $SCRIPT"
  echo "started: $(date -Is)  free_mib_at_start: $(free_mib)"
} >> "$RUN/provenance.txt"

launched=0; skipped=0
for a in "${ARMS[@]}"; do
  IFS=: read -r tag shade tradeoff <<< "$a"
  for s in "${SEEDS[@]}"; do
    for r in "${REPS[@]}"; do
      out="$RUN/${tag}_s${s}_r${r}.log"
      # 断点续跑：判据是「文件里已经有 JSON 行」，不是「文件存在」——进程一启动文件就存在，
      # 按存在判会把半截日志（或一段 OOM traceback）当成完整数据。
      if grep -q '^JSON ' "$out" 2>/dev/null; then skipped=$((skipped+1)); continue; fi
      while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n; done
      while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
      XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" \
        "$STEPS" --json --seed "$s" "${WORLD[@]}" \
        --set "grass_shade=$shade" --set "forage_tradeoff=$tradeoff" \
        > "$out" 2>&1 &
      launched=$((launched+1))
      sleep 6      # 错开编译期的显存尖峰，别让两个进程同时爬到峰值
    done
  done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*_s*_r*.log 2>/dev/null | wc -l)
echo "finished: $(date -Is)  launched=$launched skipped=$skipped ok=$ok/96" >> "$RUN/provenance.txt"
echo "done: $ok/96 logs carry a JSON line (launched $launched, skipped $skipped)"
