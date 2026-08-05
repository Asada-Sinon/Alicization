#!/usr/bin/env bash
# R18-B：那个漂移停在哪。判据见 `multispecies_program.md` §23（先写后跑）。
# 与 R18 同一个世界、同一个臂（R38n），只把步数从 350k 加到 800k：
# 12 种子 × r=2 = 24 run × 800k 步 × 6 检查点。**用户已拍板全卡**，CONC=8。
#
# 为什么必须从头重跑：`trajectory.py` 没有 resume，R18 的 run 也没留下可续的检查点。
# 预计跨代数 690–730（实测为准，§21.2：不许写成「700 代」）。约 16 小时。
#
# ⚠️ 跑的时候源码树冻结：`underworld/**`、`trajectory.py`、`split_score.py`、
#    `neutral_null.py`、以及**正在执行的本脚本**都不许动（bash 边读边执行）。
#    分析脚本 `analyze_r18.py` / `20260805-r18-verdict/*.py` 不在冻结名单里。
# ⚠️ python 加 `-u`：日志块缓冲会让人以为卡住了。
set -u
RUN=outputs/20260806-gen700
SCRIPT=explorations/20260804-readouts/trajectory.py
STEPS=${STEPS:-800000}
CONC=${CONC:-8}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
   --set forage_curvature=1.0 --set eat_rate=0.5 --set ridge_wavenumber=1
   --set fruit_regrow_baseline=0.25 --set regrow_baseline=0.010 --set diet_delta=1.5)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260806-gen700  (R18-B 那个漂移停在哪)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §23（先写后跑）"
  echo "world: ${W[*]}"; echo "arm: R38n only (diet_delta=1.5 由构造无捕食)"
  echo "⚠️ 不是默认世界：9 项与 Config() 不同（forage_tradeoff 默认 0.0）"
  echo "⚠️ 与 R18 完全同一个世界同一个臂，只加步数 350k -> $STEPS"
  echo "δ 已跑前算死：末段增量 σ̂_W=0.00953 ⇒ 等价性阈值 δ=0.0095（用 R18 的 24 run 自估）"
  echo "预测：若几何衰减 r=0.575 继续，700 代时末段增量 +0.00044（比 δ 小 21 倍）⇒ 功效充足"
  echo "⚠️ 判决按**实测**跨代数写，不许写成「700 代」（§21.2）"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: $STEPS checkpoints: 6  CONC=$CONC"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
  out="$RUN/R38n_s${s}_r${r}.log"
  grep -q '^JSON ' "$out" 2>/dev/null && continue
  while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
  while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u "$SCRIPT" "$STEPS" --json \
    --seed "$s" --checkpoints 6 "${W[@]}" > "$out" 2>&1 &
  sleep 5
done; done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/24" >> "$RUN/provenance.txt"; echo "done: $ok/24"
