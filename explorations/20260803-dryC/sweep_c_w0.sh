#!/usr/bin/env bash
# C 阶段的 w=0 参照臂，3 个种子。
#
# 为什么单独一个文件而不是加进 sweep_c.sh：那次 sweep 已经在跑，而 **bash 是按文件偏移量
# 增量读脚本的**——改一个正在执行的脚本可能让它读到错位的字节。本轮已经犯过一次
# （给 sweep_c.sh 的 WEIGHTS 加了 0.0），侥幸没坏是因为 for 是复合命令、bash 在执行前已整段
# 读完。不要再赌第二次。sweep_c.sh 里保留了 0.0，所以从头重跑那个脚本能复现全部 5 臂；
# 本文件只是给这一次补上缺的那臂。
#
# 为什么不直接引用 A 的 niche 臂：death_thirst_frac 是 A 跑完之后才加进 measure_overlap.py
# 的，A 的日志里没有这个字段。像对像（同脚本版本、同种子、同重复数）比省 3 个 run 值钱。
set -u
RUN=outputs/20260803-dryC
SCRIPT=explorations/20260803-overlapA/measure_overlap.py
COMMON=(20000 --json)
NICHE=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
       --set regrow_baseline=0.010 --set plant_max=2.0)

mkdir -p "$RUN"
for s in 0 1 2; do
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" \
    "${COMMON[@]}" "${NICHE[@]}" --set fruit_dry_weight=0.0 \
    --seed "$s" > "$RUN/w00_s${s}_r1.log" 2>&1 &
done
wait
echo "done: $(ls "$RUN"/w00_s*_r1.log | wc -l) logs"
