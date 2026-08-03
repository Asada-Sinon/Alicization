"""C 阶段定标探针的读数表（**非结论**）。

回答什么：`docs/multispecies_program.md` §12.3.C —— 在 `niche` 世界里把果层往干旱带挪，
**哪个 `fruit_dry_weight` 上「分离了但果实还吃得到」**。找的不是「分离最大」：把果层挪到
没人去的地方等于把果层删了，不是造生态位。

读什么：`outputs/20260803-dryC/w{00,025,05,075,10}_s{0,1,2}_r1.log`（15 个）。

**`w=0` 参照是用同一版脚本重跑的，不是直接引用 A 的 `niche` 臂**，理由有二：
`death_thirst_frac` 是 A 跑完之后才加进 `measure_overlap.py` 的，A 的日志里没有这个字段；
而且像对像（同脚本版本、同种子、同重复数）比省 3 个 run 值钱。A 的 `niche` 臂仍作交叉核对
——`fruit_dry_weight=0` 时 `terrain.build` 的分支不执行，两批 `w=0` 的重叠度应当同量级。

**这一阶段不做假设检验。** 每臂只有 3 个种子 × 1 重复，远低于判决协议（12×2），
`r=1` 也无法就地自估 `σ̂_W`。所以：

- 容差用 **A 阶段在同一个 `niche` 世界里自估的 2×配对差噪声**（§12.3.A' 那张表）。
  引用别处的噪声在判决里是不允许的，**在探针里可以，但必须写明**——本脚本每次都打印
  「这些容差来自 A，不是本轮自估」。
- 输出是一张读数表 + 每个读数「越没越过容差」的标记，供人挑工作点，不是 p 值。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-dryC/analyze_c.py
"""
import sys

sys.path.insert(0, "scripts")

import numpy as np

from exp_stats import RunSet

SEEDS = [0, 1, 2]
REF = "w00"                                    # 同版本脚本重跑的 w=0 参照
ARMS = [("w025", 0.25), ("w05", 0.5), ("w075", 0.75), ("w10", 1.0)]

# §12.3.A' 的容差表：A 阶段在 niche 世界自估的 2 × 配对差噪声。**不是本轮自估的。**
BASE_TOL = {
    "sel_ratio_water": 0.879,
    "population": 592.5,
    "carnivore_frac": 0.0943,
    "frugivory_frac": 0.0737,
    "herb_water_dist": 5.682,
    "forest_frac": 0.0611,
}
# 五个读数（A 判决把三个扩成五个）+ 护栏
READOUTS = ["sel_ratio_water", "frugivory_frac", "fruit_cells_eff",
            "herb_water_dist", "death_thirst_frac"]
GUARDS = ["population", "carnivore_frac", "forest_frac", "fruit_cap_total"]

print("=" * 104)
print("§0 载入")
print("=" * 104)
probe = RunSet.load("outputs/20260803-dryC", [REF] + [a for a, _ in ARMS], SEEDS, [1])
assert not probe.problems, probe.problems
print(f"  探针 (5 臂 × {len(SEEDS)} 种子 × 1 重复): {len(probe.records)} run")
print(f"  各臂 overrides 差异字段 = {sorted(probe.overrides_diff())}  (期望只有 fruit_dry_weight)")

# 交叉核对：A 的 niche 臂（12 种子 × 2 重复）与本轮 w=0（3 种子 × 1 重复）应当同量级。
cross = RunSet.load("outputs/20260803-overlapA", ["niche"], SEEDS, [1, 2])
if not cross.problems:
    a_sel = cross.cell_means("niche", "sel_ratio_water")
    c_sel = probe.cell_means(REF, "sel_ratio_water")
    print(f"  交叉核对 sel_ratio_water: A 的 niche(同 3 种子) {a_sel.mean():.4f} "
          f"vs 本轮 w=0 {c_sel.mean():.4f}  "
          f"（差 {c_sel.mean()-a_sel.mean():+.4f}，A 自估 σ̂_W=0.439，同量级即可）")
print("\n  ** 容差来自 A 阶段（§12.3.A'）在同一个 niche 世界的自估，不是本轮自估。")
print("     本轮 r=1，无法就地估 σ̂_W。这是探针不是判决，不出 p 值。 **")


print()
print("=" * 104)
print("§1 五个读数（§12.3.C 预注册）：均值 ± 跨种子 SD，n=3")
print("=" * 104)
hdr = f'  {"读数":<20} {"w=0":>16}'
for _, w in ARMS:
    hdr += f' {"w=" + str(w):>16}'
print(hdr)
rows = {}
for m in READOUTS + GUARDS:
    v0 = probe.cell_means(REF, m)
    line = f"  {m:<20} {v0.mean():>10.4f}±{v0.std(ddof=1):<5.3f}"
    rows[m] = [v0.mean()]
    for arm, _ in ARMS:
        v = probe.cell_means(arm, m)
        rows[m].append(v.mean())
        line += f" {v.mean():>10.4f}±{v.std(ddof=1):<5.3f}"
    print(line)

print()
print("=" * 104)
print("§2 相对 w=0 的变化，以及是否越过 A 的容差（2×噪声）")
print("=" * 104)
for m in READOUTS + GUARDS:
    tol = BASE_TOL.get(m)
    base = rows[m][0]
    line = f"  {m:<20} 基线 {base:>10.4f}  容差 " + (f"±{tol:<8.4f}" if tol else "  （无）   ")
    print(line)
    for i, (arm, w) in enumerate(ARMS):
        d = rows[m][i + 1] - base
        pct = 100 * d / base if base else float("nan")
        mark = ""
        if tol:
            mark = "  ** 越过容差 **" if abs(d) > tol else "  (容差内)"
        print(f"      w={w:<5} {rows[m][i+1]:>10.4f}   Δ={d:>+10.4f} ({pct:>+7.1f}%){mark}")

print()
print("=" * 104)
print("§3 工作点判读（§12.3.C 的三重张力）")
print("=" * 104)
print("  要的是「分离了 **且** 果实还吃得到 **且** 不把人赶去渴死」：")
print("    ① sel_ratio_water 下去了  ② frugivory_frac 没塌  ③ death_thirst_frac 没暴涨")
print("    ④ fruit_cells_eff 没碎    ⑤ herb_water_dist 才是「离没离开河岸」的答案")
print()
f0 = rows["frugivory_frac"][0]
s0 = rows["sel_ratio_water"][0]
t0 = rows["death_thirst_frac"][0]
print(f'  {"w":>6} {"重叠↓":>10} {"果实留存":>10} {"渴死Δ":>10} {"有效格":>9} {"离岸Δ":>9}  判读')
for i, (arm, w) in enumerate(ARMS):
    s = rows["sel_ratio_water"][i + 1]
    f = rows["frugivory_frac"][i + 1]
    t = rows["death_thirst_frac"][i + 1]
    e = rows["fruit_cells_eff"][i + 1]
    hw = rows["herb_water_dist"][i + 1] - rows["herb_water_dist"][0]
    keep = f / f0 if f0 else float("nan")
    verdict = []
    if s < s0 - BASE_TOL["sel_ratio_water"]:
        verdict.append("分离✓")
    else:
        verdict.append("分离✗")
    if keep >= 0.5:
        verdict.append("果实留存✓")
    else:
        verdict.append("果实塌✗")
    if t - t0 > 0.10:
        verdict.append("渴死↑⚠")
    print(f"  {w:>6} {100*(s/s0-1):>9.1f}% {100*keep:>9.1f}% {t-t0:>+10.4f} "
          f"{e:>9.0f} {hw:>+9.2f}  {' '.join(verdict)}")
print()
print("  注：「分离✓」的门槛是降幅超过 A 的容差 0.879（≈基线的 23.7%）。")
print("  「果实留存✓」的门槛是 frugivory_frac 保住 w=0 的一半以上——这是我在本脚本里定的")
print("  操作性阈值，§12.3.C 只写了「不能塌回 0」，没给数。**挑工作点时要连这句一起读。**")
