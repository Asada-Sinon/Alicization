#!/usr/bin/env bash
# §12 A 阶段第二个臂：**厚果层**（第二生态位）配置下的重叠度基线。
#
# 为什么必须有这个臂：R2/P2/P3/P4 全跑在这套参数上（果层真供约 28% 能量），D 阶段的参照系
# §9.6 的 −38.3% 就是在这个世界里测的。默认配置下 frugivory_frac ≈ 0.0013，果实几乎没人吃，
# forage_pref 没有可作用的资源——基线量在默认世界、判决发生在厚果层世界，两者读不到一起。
#
# 参数出处：outputs/20260803-partition/provenance.txt:8（逐字相同，未改一项）。
set -u
RUN=outputs/20260803-overlapA
SCRIPT=explorations/20260803-overlapA/measure_overlap.py
COMMON=(20000 --json)
NICHE=(--set fruit_regrow_baseline=0.25 --set fruit_energy=4.0 --set fruit_water_frac=0.40
       --set regrow_baseline=0.010 --set plant_max=2.0)

mkdir -p "$RUN"
for s in $(seq 0 11); do
  for r in 1 2; do
    while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python "$SCRIPT" \
      "${COMMON[@]}" "${NICHE[@]}" --seed "$s" > "$RUN/niche_s${s}_r${r}.log" 2>&1 &
  done
done
wait
echo "done: $(ls "$RUN"/niche_s*_r*.log | wc -l) logs"
