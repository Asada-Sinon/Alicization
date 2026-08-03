#!/usr/bin/env bash
# §12 A 阶段：果层 × 种群 重叠度基线。12 种子 × 2 重复 = 24 run，20000 步。
# 并发上限 6 —— 瓶颈是主机 CPU 上的 XLA 编译，不是显存也不是 GPU（MEMORY.md [LEARN:env]）。
# _r1/_r2 传的是同一个 --seed：重复的差异来自 GPU 原子重排本身，那正是 σ̂_W 要估的噪声。
# 文件名格式 = exp_stats.RunSet.load 的默认解析格式。
set -u
RUN=outputs/20260803-overlapA
SCRIPT=explorations/20260803-overlapA/measure_overlap.py
COMMON=(20000 --json)

mkdir -p "$RUN"
for s in $(seq 0 11); do
  for r in 1 2; do
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" \
      "${COMMON[@]}" --seed "$s" > "$RUN/base_s${s}_r${r}.log" 2>&1 &
  done
done
wait
echo "done: $(ls "$RUN"/base_s*_r*.log | wc -l) logs"
