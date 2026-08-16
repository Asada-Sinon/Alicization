#!/usr/bin/env bash
# R21：捕食压力的剂量曲线——临界点在哪。判据见 `docs/multispecies_program.md` §28（先写后跑）。
#
# R20 只测了两个端点（无捕食 / 默认强度），中间一片空白，而 §27.4 的禁令之一正是
# 「不许说捕食者让分化不可能——本轮只有两档」。本轮补中间 4 档。
#
# `diet_delta` 语义（dynamics.py:307）：`d_i − d_j > diet_delta` 才能捕食
# ⇒ **值越大捕食越难**。食草者 diet≈0.2、食肉者≈0.8 ⇒ 有效范围约 [0.15, 0.6]。
# 端点 0.15（全捕食）与 1.5（无捕食）**直接复用 R20 的 wn1on / wn1off**，不重跑。
#
# 96 run × 100k 步，地形固定 wn1，其余与 R20 完全一致。CONC=8，约 8 小时。
# ⚠️ 跑的时候源码树冻结：underworld/**、trajectory.py、split_score.py、neutral_null.py、
#    以及正在执行的本脚本。
set -u
RUN=outputs/20260816-r21b
SCRIPT=explorations/20260804-readouts/trajectory.py
STEPS=${STEPS:-100000}
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
CONC=${CONC:-8}; NEED_MIB=${NEED_MIB:-2600}
BASE=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
      --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
      --set forage_curvature=1.0 --set eat_rate=0.5
      --set fruit_regrow_baseline=0.25 --set regrow_baseline=0.010
      --set ridge_wavenumber=1)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260816-r21b (R21b 补过渡带 0.70/0.80/0.90)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §28（先写后跑）"
  echo "档位: diet_delta 0.25 / 0.35 / 0.45 / 0.60"
  echo "端点复用 R20：wn1on(diet_delta=0.15) 与 wn1off(1.5)，见 outputs/20260813-r20"
  echo "地形固定 wn1；其余 10 项覆盖与 R20 一致"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: $STEPS CONC=$CONC ⇒ 72 run"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
for dd in 0.70 0.80 0.90; do
  cell="dd${dd/./}"
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    out="$RUN/${cell}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u "$SCRIPT" "$STEPS" --json \
      --seed "$s" --checkpoints 4 "${BASE[@]}" --set diet_delta=$dd > "$out" 2>&1 &
    sleep 3
  done; done
done
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/72" >> "$RUN/provenance.txt"; echo "done: $ok/96"
