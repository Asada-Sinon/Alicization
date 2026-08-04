#!/usr/bin/env bash
# R17：生殖隔离。判据见 `docs/multispecies_program.md` §20（先写后跑，commit 21832d0）。
# 2 臂 × 12 种子 × r=2 = 48 run × 100k 步 × 4 检查点，CONC=3 约 4 小时。
# 两臂只差 `mate_forage_weight`：0（旧行为，对照）vs 0.2（主臂）。
# ⚠️ **必须跑在 R38 世界上**：默认世界 forage_tradeoff=0.0 ⇒ forage_pref 是完全中性的基因
#    ⇒ 任何下游读数必然是 null，而那个 null 与「机制无效」分不开（§20.5）。
# ⚠️ 主判据是 LD（两簇在非 forage_pref 基因上的 mean|Cohen's d|），不是「双峰更尖」——
#    后者按构造是 null（单基因座 + 均匀交叉从不产生中间值）。
# ⚠️ python 加 `-u`：上一轮日志块缓冲，跑到一半看不到任何进度。
# ⚠️ 共用卡：显存闸 + 断点续跑。跑的时候它读的文件全部冻结（含 underworld/**）。
set -u
RUN=outputs/20260805-isolation
SCRIPT=explorations/20260804-readouts/trajectory.py
CONC=${CONC:-3}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
   --set forage_curvature=1.0 --set eat_rate=0.5 --set ridge_wavenumber=1
   --set fruit_regrow_baseline=0.25 --set regrow_baseline=0.010 --set diet_delta=1.5)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
ARMS=(W00:0.0 W20:0.2)          # 臂名:mate_forage_weight
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260805-isolation  (R17 生殖隔离)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §20（先写后跑，commit 21832d0）"
  echo "world: ${W[*]}"; echo "arms: ${ARMS[*]}"
  echo "⚠️ 不是默认世界：9 项与 Config() 不同（forage_tradeoff 默认 0.0）"
  echo "主判据 H1 = 两簇在非 forage_pref 基因上的 mean|Cohen's d|（ld_mean_abs_d）"
  echo "零点来自 W00 对照臂，不是 0——冒烟实测 w=0 时 mean|d| 已有 0.36–0.40"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: 100000 checkpoints: 4  CONC=$CONC"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for a in "${ARMS[@]}"; do
  IFS=: read -r tag mw <<< "$a"
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${tag}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u "$SCRIPT" 100000 --json \
      --seed "$s" --checkpoints 4 "${W[@]}" --set "mate_forage_weight=$mw" > "$out" 2>&1 &
    sleep 5
  done; done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/48" >> "$RUN/provenance.txt"; echo "done: $ok/48"
