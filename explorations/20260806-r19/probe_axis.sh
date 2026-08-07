#!/usr/bin/env bash
# R19 的诊断探针（§22.4d）：对照臂为什么也在涨？
#
# 跑前探针发现 `forage_tradeoff=0` 的 H2 对照臂 `mean gene` 也涨了 +0.207，
# 而它按设计不该涨——没有觅食权衡就没有中间型劣势，隔离没有收益可言。
# 在这件事查清之前，R19 的任何阳性结果都无法归因于隔离。
#
# 四个臂，只差一个开关，都开 `mate_forage_heritable=True`：
#   A 主臂        forage_tradeoff=1.0                        有谷、真实轴
#   B H2 对照     forage_tradeoff=0.0                        无谷、真实轴  ← 它涨了
#   C 随机轴      forage_tradeoff=1.0 + random_axis=True     有谷、随机轴
#   D 无同型交配  forage_tradeoff=1.0 + assortative=False    w 无从表达
#
# 读法：
#   若 C 不涨而 B 涨  ⇒ 涨的是「按同一条**可遗传**轴排队」的 by-construction 正反馈
#   若 C 也涨         ⇒ 与轴是否可遗传无关，问题在「按 w 分层排队」这件事本身
#   若 D 也涨         ⇒ 连排序都不需要，那是别的机制（基因搭连？）
set -u
RUN=outputs/20260807-r19probe
SCRIPT=explorations/20260804-readouts/trajectory.py
STEPS=${STEPS:-20000}          # ~40 代，足以看到探针里那个 14 代就出现的信号
SEEDS=(0 1 2 3)
CONC=${CONC:-8}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_curvature=1.0
   --set eat_rate=0.5 --set ridge_wavenumber=1 --set fruit_regrow_baseline=0.25
   --set regrow_baseline=0.010 --set diet_delta=1.5 --set mate_forage_heritable=True)
mkdir -p "$RUN"
{ echo "run_id: 20260807-r19probe  (R19 诊断探针：对照臂为什么也涨)"
  echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §22.4d（先写后跑）"
  echo "steps: $STEPS  seeds: ${SEEDS[*]}  CONC=$CONC"
  echo "started: $(date -Is)"; } >> "$RUN/provenance.txt"
run_arm() {  # $1=臂名  $2..=额外 --set
  local arm=$1; shift
  for s in "${SEEDS[@]}"; do
    local out="$RUN/${arm}_s${s}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u "$SCRIPT" "$STEPS" --json \
      --seed "$s" --checkpoints 2 "${W[@]}" "$@" > "$out" 2>&1 &
    sleep 2
  done
}
run_arm A --set forage_tradeoff=1.0
run_arm B --set forage_tradeoff=0.0
run_arm C --set forage_tradeoff=1.0 --set mate_forage_random_axis=True
run_arm D --set forage_tradeoff=1.0 --set assortative_mating=False
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/16" >> "$RUN/provenance.txt"; echo "done: $ok/16"
