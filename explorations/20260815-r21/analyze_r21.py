"""R21 判决：捕食压力的剂量曲线。判据 `docs/multispecies_program.md` §28（先写后跑）。

**本脚本必须入库**——§28.4 的护栏之一。R19 与 R20 的判决都没留分析脚本，
导致已发表数字只能靠「哪个口径能复现」反推（§19.2 记过这个流程洞，§27 又犯一次）。

六档曲线：两个端点（`diet_delta` 0.15 全捕食 / 1.5 无捕食）**复用 R20 的
`wn1on`/`wn1off`**，中间四档（0.25/0.35/0.45/0.60）来自本轮。
两批的世界配置除 `diet_delta` 外完全一致（BASE 10 项 + `ridge_wavenumber=1`）。

三条判据（跑之前写死）
--------------------
- **H1 曲线形状**：相邻档最大跌幅 ÷ 其余相邻跌幅的中位数，**≥3 记为「有阈值」**。
- **H2 临界档**：首个使 `low_mass` 跌破 0.25（无捕食基准的一半）的 `diet_delta`。
- **H3 处理失效（必报）**：每档末帧 `carn_frac < 0.005` 的 run 数——它们等于变成了
  无捕食臂。R20 漏报 5/72，而那 4 个恰是该格 `low_mass` 最高的种子。

口径（§28.4）
------------
- 分析单位是**格均值**（先把 r=2 平均），不是 run。
- 主读数 **`low_mass`**；两峰占比只作方向佐证（R20 复核实测它两臂饱和、功效不足）。
- **同时给「对齐世代」与「对齐步数」两种**——R20 复核证明两种都成立才算夹住混杂。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260815-r21/analyze_r21.py
"""
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")
sys.path.insert(0, "explorations/20260805-r18-verdict")

import numpy as np
from diag_h2_degenerate import parts, CTR
from exp_stats import bootstrap_ci
from neutral_null import gen_weights

LOW = CTR < 0.35
# (diet_delta, 目录, 文件前缀)。端点复用 R20。
ARMS = [(0.15, "outputs/20260813-r20", "wn1on"),
        (0.25, "outputs/20260815-r21", "dd025"),
        (0.35, "outputs/20260815-r21", "dd035"),
        (0.45, "outputs/20260815-r21", "dd045"),
        (0.60, "outputs/20260815-r21", "dd060"),
        # R21b 补的过渡带——R21 的五档全落在「捕食仍充分发生」的一侧（§29.2）
        (0.70, "outputs/20260816-r21b", "dd070"),
        (0.80, "outputs/20260816-r21b", "dd080"),
        (0.90, "outputs/20260816-r21b", "dd090"),
        (1.50, "outputs/20260813-r20", "wn1off")]
# 相邻档距不等（0.15→0.60 每档 0.10–0.15，0.90→1.50 跨 0.60），
# **H1 的「相邻跃升」只在档距可比的区间内比**——R21 就是被这个坑判出了假阈值。
COMPARABLE = (0.15, 0.25, 0.35, 0.45, 0.60, 0.70, 0.80, 0.90)
CRIT = 0.25          # H2 的阈值（无捕食基准约 0.51 的一半）


def wmean(v, g, m):
    w = gen_weights(g, "iso") * m
    s = w.sum()
    return float((w * np.asarray(v, float)).sum() / s) if s > 1e-9 else np.nan


def load(d, pre):
    """回传 {seed: (low_mass_末1/4, 两峰占比, 末帧carn, 跨代数, low_mass@gen100, low_mass@t20k)}"""
    per, dead = {}, []
    for f in sorted(glob.glob(f"{d}/{pre}_s*_r*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        js = json.loads(txt.split("JSON ")[1].split("\n")[0])
        if js["collapsed"]:
            continue
        tr = js["traj"]
        g = np.array([q["generation"] for q in tr], float)
        t = np.array([q["t"] for q in tr], float)
        H = np.array([q["hist"] for q in tr], float)
        lm_all = np.array([h[LOW].sum() / max(h.sum(), 1.0) for h in H])
        late = g >= g.max() * 0.75
        lm = wmean(lm_all, g, late)
        pk = wmean([1.0 if (parts(h, CTR) and "gap" in parts(h, CTR)) else 0.0 for h in H],
                   g, late)
        # 末帧 carn_frac：traj 末点的 carnivore_frac 常为 NaN，取最后一个有限值
        cf = [q.get("carnivore_frac") for q in tr if q.get("carnivore_frac") is not None]
        cf = [x for x in cf if np.isfinite(x)]
        carn = float(cf[-1]) if cf else np.nan
        # 两种对齐
        m100 = (g >= 95) & (g <= 105)
        lm100 = float(lm_all[m100].mean()) if m100.any() else np.nan
        m20k = (t >= 19000) & (t <= 23000)
        lm20k = float(lm_all[m20k].mean()) if m20k.any() else np.nan
        s = int(f.split("/")[-1].split("_")[1][1:])
        per.setdefault(s, []).append((lm, pk, carn, g.max(), lm100, lm20k))
        if np.isfinite(carn) and carn < 0.005:
            dead.append(f.split("/")[-1][:-4])
    cells = {s: tuple(np.nanmean([x[i] for x in v]) for i in range(6)) for s, v in per.items()}
    return cells, dead


def main():
    print("=" * 92)
    print("R21 判决：捕食压力的剂量曲线（判据 `multispecies_program.md` §28）")
    print("=" * 92)
    rows, deads = [], {}
    for dd, d, pre in ARMS:
        C, dead = load(d, pre)
        deads[dd] = dead
        if not C:
            print(f"  diet_delta={dd}: 无数据"); continue
        a = np.array([list(v) for v in C.values()], float)   # [格, 6]
        rows.append((dd, a, sorted(C)))
    print(f"\n{'diet_delta':>10} {'low_mass':>9} {'95%CI':>20} {'两峰占比':>9} "
          f"{'carn末帧':>9} {'代数':>6} {'格数':>5} {'失效':>5}")
    for dd, a, ks in rows:
        lo, hi = bootstrap_ci(a[:, 0])
        print(f"{dd:>10.2f} {a[:,0].mean():>9.4f} {f'[{lo:.4f}, {hi:.4f}]':>20} "
              f"{a[:,1].mean():>9.4f} {np.nanmean(a[:,2]):>9.4f} {a[:,3].mean():>6.0f} "
              f"{len(ks):>5} {len(deads[dd]):>5}")
    print("\n（`diet_delta` 越大 ⇒ 捕食越难发生；0.15=默认强度，1.5=由构造无捕食。"
          "\n  端点两档复用 R20 的 `wn1on`/`wn1off`，其余配置逐项相同。）")

    # ---- H1：曲线形状 ----
    print("\n  **H1 曲线是突变还是渐变**（§28.3：最大跌幅 ÷ 其余中位数 ≥3 ⇒ 有阈值）")
    dd = np.array([r[0] for r in rows if r[0] in COMPARABLE])
    lm = np.array([r[1][:, 0].mean() for r in rows if r[0] in COMPARABLE])
    print(f"    （只在档距可比的区间内比：{[f'{v:.2f}' for v in dd]}——"
          f"1.50 那档与 0.90 相隔 0.60，纳进来会造出假阈值，§29.2 的教训）")
    # 按 diet_delta 升序 = 捕食压力递减，low_mass 应递增
    step = np.diff(lm)
    print(f"    逐档 `low_mass`  {[f'{v:.4f}' for v in lm]}")
    print(f"    相邻档跃升      {[f'{v:+.4f}' for v in step]}")
    biggest = np.argmax(np.abs(step))
    others = np.delete(np.abs(step), biggest)
    ratio = np.abs(step[biggest]) / max(np.median(others), 1e-9)
    print(f"    最大跃升在 {dd[biggest]:.2f}→{dd[biggest+1]:.2f}：{step[biggest]:+.4f}，"
          f"其余中位数 {np.median(others):.4f}  ⇒ **比值 {ratio:.2f}**")
    print(f"    ⇒ {'**有阈值（突变）**' if ratio >= 3 else '**无阈值（渐变，连续压制）**'}")

    # ---- H2：临界档 ----
    print(f"\n  **H2 首个使 `low_mass` 跌破 {CRIT} 的档**")
    below = [d_ for d_, v in zip(dd, lm) if v < CRIT]
    if below:
        print(f"    捕食压力从大到小：{[f'{d_:.2f}' for d_ in dd]}")
        print(f"    低于 {CRIT} 的档：{[f'{d_:.2f}' for d_ in below]}  "
              f"⇒ **临界点落在 {max(below):.2f} 与 "
              f"{min([d_ for d_ in dd if d_ > max(below)], default=float('nan')):.2f} 之间**")
    else:
        print(f"    没有一档低于 {CRIT}")

    # ---- 两种对齐 ----
    print("\n  **两种对齐口径**（§28.4：两种都成立才算夹住世代周转的混杂）")
    print(f"{'diet_delta':>10} {'末1/4':>9} {'@gen100':>9} {'@t≈21k':>9}")
    for dd_, a, _ in rows:
        print(f"{dd_:>10.2f} {a[:,0].mean():>9.4f} {np.nanmean(a[:,4]):>9.4f} "
              f"{np.nanmean(a[:,5]):>9.4f}")

    # ---- H3：处理失效 ----
    print("\n  **H3 处理失效（末帧 `carn_frac` < 0.005 ⇒ 该 run 等于变成了无捕食臂）**")
    for dd_, a, ks in rows:
        if dd_ >= 1.4:
            print(f"    diet_delta={dd_:.2f}: **由构造无捕食，`carn_frac=0` 是设计不是失效，跳过**")
            continue
        d_ = deads[dd_]
        print(f"    diet_delta={dd_:.2f}: {len(d_)} 个" + (f"  {d_}" if d_ else ""))
    print("    ⚠️ R20 漏报了 5/72，而那 4 个恰是该格 `low_mass` 最高的种子——"
          "本轮逐档核对，避免同一失误。")


if __name__ == "__main__":
    main()
