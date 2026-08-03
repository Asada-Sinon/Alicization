#!/usr/bin/env bash
# R10：权衡前沿的曲率。判据先写后跑，见 docs/multispecies_program.md §14。
#
# 主判据是 quad_intake（个体级回归 last_food ~ 1+z+z²），不是 sd —— 后者的 MDE 是基线的
# 126%，R9 已经用 96 个 run 证明它测不出中等效应。噪声标定见 §14.2。
#
# ⚠️ 跑的时候它读的每一个文件都不许动（脚本自身、underworld/、fitness_surface.py）。
# ⚠️ 这张卡是共用的：显存闸 + 断点续跑（判据是日志里已有 JSON 行，不是文件存在）。
set -u
RUN=outputs/20260803-curvature/r10
SCRIPT=explorations/20260803-curvature/fitness_surface.py
STEPS=20000
CONC=${CONC:-5}
NEED_MIB=${NEED_MIB:-2600}
WORLD=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
       --set regrow_baseline=0.010 --set plant_max=2.0 --set water_sea_dist=1
       --set grass_shade=1.3)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
# 臂名:forage_tradeoff:forage_curvature
ARMS=(N:0.0:1.0 K10:1.0:1.0 K05:1.0:0.5 K035:1.0:0.35)
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260803-curvature/r10"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §14（先写后跑，commit f44cc8b）"
  echo "world: ${WORLD[*]}"; echo "arms: ${ARMS[*]}"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: $STEPS conc: $CONC"
  echo "started: $(date -Is) free_mib: $(free_mib)"; } >> "$RUN/provenance.txt"
launched=0; skipped=0
for a in "${ARMS[@]}"; do
  IFS=: read -r tag t k <<< "$a"
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${tag}_s${s}_r${r}.log"
    if grep -q '^JSON ' "$out" 2>/dev/null; then skipped=$((skipped+1)); continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" \
      "$STEPS" --json --seed "$s" "${WORLD[@]}" \
      --set "forage_tradeoff=$t" --set "forage_curvature=$k" > "$out" 2>&1 &
    launched=$((launched+1)); sleep 6
  done; done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*_s*_r*.log 2>/dev/null | wc -l)
echo "finished: $(date -Is) launched=$launched skipped=$skipped ok=$ok/96" >> "$RUN/provenance.txt"
echo "done: $ok/96"
