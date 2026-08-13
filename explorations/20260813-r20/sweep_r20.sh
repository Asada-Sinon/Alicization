#!/usr/bin/env bash
# R20：换张地图、把捕食者放回去——那两个生态型还在不在。
# 判据见 `docs/multispecies_program.md` §26（先写后跑）。
#
# 为什么是这一轮：R9→R18-B 立住的全部结论都带同一对脚注——**单地形**（伪重复，
# terrain.build 不吃 RNG ⇒ 换种子换不了地图）与**无捕食者**（diet_delta=1.5 由构造关掉）。
# 这是它们最大的局限，而且能直接验。**答案不管是什么都是硬结论**，
# 不像可演化的隔离那样会卡在测量上（R19 已经卡了两次）。
#
# 3 地形 × 2 捕食 = 6 格 × 12 种子 × r=2 = **144 run × 100k 步**，CONC=8，约 5.5 小时。
# 其余 9 项世界覆盖与 R18-B 完全一致，**只动这两个因子**。
#
# ⚠️ 有捕食者的臂 R16 实测会出现处理失效（48 run 里 4 个）——崩溃 run 分开报，
#    断点续跑已内建。
# ⚠️ 跑的时候源码树冻结：underworld/**、trajectory.py、split_score.py、
#    neutral_null.py、以及正在执行的本脚本。
set -u
RUN=outputs/20260813-r20
SCRIPT=explorations/20260804-readouts/trajectory.py
STEPS=${STEPS:-100000}
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
CONC=${CONC:-8}; NEED_MIB=${NEED_MIB:-2600}
# 与 R18-B 相同的 9 项世界覆盖（不含 ridge_wavenumber / diet_delta，那两个是本轮的因子）
BASE=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
      --set water_sea_dist=1 --set grass_shade=1.3 --set forage_tradeoff=1.0
      --set forage_curvature=1.0 --set eat_rate=0.5
      --set fruit_regrow_baseline=0.25 --set regrow_baseline=0.010)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260813-r20  (R20 换地图 + 加捕食者)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §26（先写后跑）"
  echo "base: ${BASE[*]}"
  echo "因子: 地形 wn1 / wn2 / wn1_base35   ×   捕食 off(diet_delta=1.5) / on(默认)"
  echo "基准: R18-B 的 wn1+off 实测 low_mass 0.5820、两峰占比 1.0000"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: $STEPS CONC=$CONC ⇒ 144 run"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
run_cell() {   # $1=格名  $2..=该格的 --set
  local cell=$1; shift
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    local out="$RUN/${cell}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u "$SCRIPT" "$STEPS" --json \
      --seed "$s" --checkpoints 4 "${BASE[@]}" "$@" > "$out" 2>&1 &
    sleep 3
  done; done
}
# 捕食 off = diet_delta 1.5（与 R18-B 同）；on = 不设，用默认
run_cell wn1off  --set ridge_wavenumber=1 --set diet_delta=1.5
run_cell wn1on   --set ridge_wavenumber=1
run_cell wn2off  --set ridge_wavenumber=2 --set diet_delta=1.5
run_cell wn2on   --set ridge_wavenumber=2
run_cell b35off  --set ridge_wavenumber=1 --set ridge_base_frac=0.35 --set diet_delta=1.5
run_cell b35on   --set ridge_wavenumber=1 --set ridge_base_frac=0.35
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/144" >> "$RUN/provenance.txt"; echo "done: $ok/144"
