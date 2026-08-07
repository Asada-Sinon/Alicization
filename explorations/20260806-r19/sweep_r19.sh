#!/usr/bin/env bash
# R19：隔离能不能自发演化。判据见 `docs/multispecies_program.md` §22 + §22.4b–e（先写后跑）。
#
# **四个臂，每个回答一个问题**（§22.4e，三轮跑前探针之后重新设计的）：
#   A 主臂        forage_tradeoff=1.0                      有谷、可遗传轴
#   B 无谷        forage_tradeoff=0.0                      无谷、可遗传轴
#   C 随机轴      forage_tradeoff=1.0 + random_axis=True   **真正的零点**（既无搭车也无收益）
#   D 无同型交配  forage_tradeoff=1.0 + assortative=False  阴性对照，w 无从表达
#
# 判据：**C 是零点**；**A−C = 总效应**（搭车 + 收益）；**A−B = 收益的净效应**；
#       **B−C = 搭车扩散单独的贡献**。判决必须把总效应与净效应分开报——
#       只报 A−C 会把搭车说成「隔离被选择」。
#
# **为什么是 100k 步**：§22.4b 实测选择压在分裂完成后掉 26,000 倍，跑长了后面全是噪声；
# 100k ≈ 150 代，覆盖分裂形成期（~63 代）+ 给基因充分响应时间，且与 R16 同长、
# 可直接引用它的中性零分布。
#
# **为什么没有「初始 w 在阈值之上」那一臂**：§22.4c 曾担心入侵障碍（小 w 的选择系数
# 为负）挡住第一步，但 §22.4d 的探针实测主臂 40 代就涨到 +0.597、远超阈值
# ⇒ **入侵障碍不是瓶颈**，那一臂的信息量因此下降，省掉。
#
# ⚠️ 跑的时候源码树冻结：`underworld/**`、`trajectory.py`、`split_score.py`、
#    `neutral_null.py`、以及正在执行的本脚本。
set -u
RUN=outputs/20260807-r19
SCRIPT=explorations/20260804-readouts/trajectory.py
STEPS=${STEPS:-100000}
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11); REPS=(1 2)
CONC=${CONC:-8}; NEED_MIB=${NEED_MIB:-2600}
W=(--set fruit_energy=4.0 --set fruit_water_frac=0.40 --set plant_max=2.0
   --set water_sea_dist=1 --set grass_shade=1.3 --set forage_curvature=1.0
   --set eat_rate=0.5 --set ridge_wavenumber=1 --set fruit_regrow_baseline=0.25
   --set regrow_baseline=0.010 --set diet_delta=1.5 --set mate_forage_heritable=True)
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1; }
mkdir -p "$RUN"
{ echo "run_id: 20260807-r19  (R19 隔离能不能自发演化)"; echo "git HEAD: $(git rev-parse HEAD)"
  echo "dirty: $([ -z "$(git status --porcelain)" ] && echo false || echo true)"
  echo "判据: docs/multispecies_program.md §22 + §22.4b–e（先写后跑）"
  echo "world: ${W[*]}"
  echo "arms: A=主臂 B=无谷 C=随机轴(零点) D=无同型交配"
  echo "MDE 已跑前算死: mean gene 0.176（≈ w 0.044）；探针 A +0.597 / B +0.396 / C −0.007 / D −0.072"
  echo "seeds: ${SEEDS[*]} reps: ${REPS[*]} steps: $STEPS  CONC=$CONC  ⇒ 96 run"
  echo "started: $(date -Is) free: $(free_mib)"; } >> "$RUN/provenance.txt"
run_arm() {
  local arm=$1; shift
  for s in "${SEEDS[@]}"; do for r in "${REPS[@]}"; do
    local out="$RUN/${arm}_s${s}_r${r}.log"
    grep -q '^JSON ' "$out" 2>/dev/null && continue
    while [ "$(jobs -rp|wc -l)" -ge "$CONC" ]; do wait -n; done
    while [ "$(free_mib)" -lt "$NEED_MIB" ]; do sleep 30; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -u "$SCRIPT" "$STEPS" --json \
      --seed "$s" --checkpoints 4 "${W[@]}" "$@" > "$out" 2>&1 &
    sleep 3
  done; done
}
run_arm A --set forage_tradeoff=1.0
run_arm B --set forage_tradeoff=0.0
run_arm C --set forage_tradeoff=1.0 --set mate_forage_random_axis=True
run_arm D --set forage_tradeoff=1.0 --set assortative_mating=False
wait
ok=$(grep -l '^JSON ' "$RUN"/*.log 2>/dev/null|wc -l)
echo "finished: $(date -Is) ok=$ok/96" >> "$RUN/provenance.txt"; echo "done: $ok/96"
