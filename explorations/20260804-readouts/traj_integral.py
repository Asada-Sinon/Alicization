"""R15-A：把 R13 的**轨迹**再分析一遍。零 GPU 成本。

为什么这批数据还能用
--------------------
作废 R13 主判据的是直方图 range 截断（`MEMORY.md [LEARN:tooling]`、
`feasibility.md §15`）。但那个 bug **只污染了检查点路径**：

    git show d3c3029:explorations/20260804-readouts/trajectory.py
      第 79 行（轨迹，逐帧）：idx = jnp.clip(((p - lo) / w).astype(jnp.int32), 0, nb - 1)
      第 147 行（检查点）  ：idx = np.digitize(p, BINS) - 1     ← 丢掉窗外个体

`clip` 把窗外个体压进端点箱，**一个都不丢**。逐帧核过 24 run × 373 帧的
`sum(hist) == n_herb`，**全等**。所以轨迹是干净的**全员普查**，而且：

- **373 个点**，不是 5 个检查点；
- 跨 **118–613 代**，而 §14.4 自己承认原主判据那个 2000 步窗只有 **9.7 代**、
  按设计欠功效约 **40 倍**。

分箱几何（clip 口径下）
-----------------------
`BINS = linspace(0.15, 0.85, 15)`，宽 0.05，14 箱。
bin0 = (−∞, 0.20)、bin_k = [0.15+0.05k, 0.20+0.05k)、bin13 = [0.80, +∞)。
箱心 0.175 … 0.825。`low_mass` 取箱心 < 0.35 ⇒ bin0..bin3 ⇒ **恰好是 `pref < 0.35`**，
完整、无截断。

两趟制
------
**第一趟（`--pass noise`）只算噪声量**——帧间增量的方差、自相关、有效样本量、
每个 run 内 OLS 的 SE。**不打印任何效应**。拿它把 MDE 与「预测效应 ÷ MDE」
写进预注册并提交之后，才跑第二趟。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260804-readouts/traj_integral.py --pass noise
"""
import argparse
import glob
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np

RUN_DIR = "outputs/20260804-traj"
# R13 版的分箱几何。**写死在这里而不是 import**，因为现版 `trajectory.py` 的 BINS
# 已经换成 linspace(0,1,21) 了——拿新常数解读旧数据正是本轮要避免的那类错。
BINS_R13 = np.linspace(0.15, 0.85, 15)
CTR_R13 = 0.5 * (BINS_R13[:-1] + BINS_R13[1:])
LOW = CTR_R13 < 0.35          # bin0..bin3 ⇒ pref < 0.35
MIN_HERB = 100                # 有效性闸：n_herb 太小时 x 的量化太粗


def load():
    """读 24 个 run 的轨迹，并把「hist 是全员普查」这件事**每次都重新核一遍**。"""
    runs = []
    for f in sorted(glob.glob(f"{RUN_DIR}/*.log")):
        txt = open(f).read()
        if "JSON " not in txt:
            continue
        d = json.loads(txt.split("JSON ")[1].split("\n")[0])
        tr = d["traj"]
        h = np.array([t["hist"] for t in tr], float)          # [n_frames, 14]
        nh = np.array([t["n_herb"] for t in tr], float)
        assert np.allclose(h.sum(1), nh, rtol=1e-6, atol=1e-3), f"{f}: hist 不是全员普查"
        runs.append({
            "name": f.split("/")[-1][:-4],
            "wn": d["overrides"]["ridge_wavenumber"],
            "seed": d["seed"],
            "t": np.array([t["t"] for t in tr], float),
            "gen": np.array([t["generation"] for t in tr], float),
            "n_herb": nh,
            "hist": h,
            "carn": np.array([t["carnivore_frac"] for t in tr], float),
            "pop": np.array([t["population"] for t in tr], float),
            # **坏版本**的 `split_score`，只留作「证明它坏」的对照，不参与任何判据
            "ss_field": np.array([t["split_score"] for t in tr], float),
            "checkpoints": d["checkpoints"],
            "collapsed": d["collapsed"],
        })
    return runs


def low_mass(h):
    """逐帧 `low_mass`。分母是该帧的全体食草者，**不是窗内个体**。"""
    return h[:, LOW].sum(1) / np.maximum(h.sum(1), 1.0)


def gate(r):
    """有效性闸：只保留 `n_herb >= MIN_HERB` 的帧。返回布尔掩码。"""
    return r["n_herb"] >= MIN_HERB


def acf1(v):
    """滞后 1 的自相关。"""
    v = v - v.mean()
    d = float((v * v).sum())
    return float((v[:-1] * v[1:]).sum() / d) if d > 0 else float("nan")


def noise_pass(runs):
    """**只报噪声量。一个效应都不打印。**

    要的是三件事：
    ① 帧间增量 `Δx` 的 SD——它定 P1 斜率的分母；
    ② `Δx` 的自相关——鞅的增量应当不相关，若显著非零则 OLS 的 SE 要打折；
    ③ 每个 run 内 OLS 斜率的 SE，以及跨 run 的 SD——免费的方差分解。
    """
    print("=" * 92)
    print("R15-A 第一趟：**只算噪声**（效应留到预注册提交之后再算）")
    print("=" * 92)
    print(f"  {len(runs)} 个 run；有效性闸 n_herb >= {MIN_HERB}")

    rows = []
    for r in runs:
        m = gate(r)
        x = low_mass(r["hist"])[m]
        g = r["gen"][m]
        dx = np.diff(x)
        # 每帧的平均世代增量——把「每帧」换算成「每代」的换算因子
        dg = float(np.median(np.diff(g)))
        # P1 的回归量：Δx_k 对滞后一期的 x_{k-1}
        y, z = dx[1:], x[:-2]
        n = len(y)
        # OLS 的残差 SD 与斜率 SE（**只用残差，不看斜率本身**）
        zc = z - z.mean()
        ssz = float((zc * zc).sum())
        if n > 2 and ssz > 0:
            b = float((zc * (y - y.mean())).sum() / ssz)
            resid = y - y.mean() - b * zc
            se = float(np.sqrt((resid ** 2).sum() / (n - 2) / ssz))
        else:
            se = float("nan")
        rows.append({
            "name": r["name"], "kept": int(m.sum()), "dropped": int((~m).sum()),
            "sd_dx": float(dx.std(ddof=1)), "acf1_dx": acf1(dx),
            "sd_x": float(x.std(ddof=1)), "dg": dg,
            "gen_span": float(g[-1] - g[0]), "se_b_per_gen": se / dg if dg > 0 else float("nan"),
            "carn_lost": bool(r["carn"][len(r["carn"]) // 2:].min() <= 0.0),
        })

    print()
    print(f'  {"run":<16}{"入算帧":>7}{"丢帧":>6}{"跨代":>8}{"SD(Δx)":>10}'
          f'{"ACF1(Δx)":>10}{"SD(x)":>9}{"SE(β)/代":>12}{"失捕":>6}')
    for q in rows:
        print(f'  {q["name"]:<16}{q["kept"]:>7}{q["dropped"]:>6}{q["gen_span"]:>8.0f}'
              f'{q["sd_dx"]:>10.5f}{q["acf1_dx"]:>10.3f}{q["sd_x"]:>9.4f}'
              f'{q["se_b_per_gen"]:>12.6f}{"是" if q["carn_lost"] else "":>6}')

    se = np.array([q["se_b_per_gen"] for q in rows])
    acf = np.array([q["acf1_dx"] for q in rows])
    print()
    print("  汇总")
    print(f'    丢帧合计 {sum(q["dropped"] for q in rows)} / {sum(q["kept"] + q["dropped"] for q in rows)}')
    print(f'    Δx 的滞后-1 自相关：均值 {acf.mean():+.3f}，范围 [{acf.min():+.3f}, {acf.max():+.3f}]')
    print(f'      （鞅的增量应当 ≈0。显著为负 ⇒ 存在均值回复或测量噪声；显著为正 ⇒ OLS 的 SE 偏小）')
    print(f'    run 内 OLS 的 SE(β)：中位 {np.median(se):.6f}/代，范围 [{se.min():.6f}, {se.max():.6f}]')
    print()
    print("  ** MDE（跨 24 run 的单样本 t，双侧 α=0.05、功效 0.80）**")
    print("     跨 run SD 要等第二趟才知道；这里给出**若跨 run 变异等于 run 内 SE 中位**")
    print("     时的下界，真实 MDE 只会更大：")
    mde_lb = 2.87 * float(np.median(se)) / np.sqrt(len(rows))   # (1.96+0.84)/√n，t 修正约 2.87
    print(f'     MDE_lower = 2.87 × {np.median(se):.6f} / √{len(rows)} = {mde_lb:.6f} /代')
    print()
    print("  参照：§16.4 修正后反解的 s = 0.00136/代（`Ne·s` = 0.40）。")
    print(f'     预测效应 ÷ MDE_lower = {0.00136 / mde_lb:.1f}（**这是上界，第二趟要用真 SD 重算**）')


def describe_pass(runs):
    """描述性一趟：**不做任何检验**。三件事，每件都不依赖零假设。

    ① traj 里的 `split_score` 字段是**坏版本**算的（补零平滑，`803dc60` 才修，
       而这批 run 完于该提交之前）——实测它与修好版在 **46% 的帧**上不一致。
       **一律丢弃，从 `hist` 用修好的 `split_score.py` 重算。**
    ② `bin0_mass` 作第二读数：clip 口径下 bin0 = (−∞,0.20)，宽度是别的箱的 4 倍，
       而在少数簇真存在的 run 里它占 `low_mass` 的 **55–91%**
       ⇒ `low_mass` 在「有簇」和「无簇」两层里量的其实是两样东西。
    ③ `x` 恰为 0 的帧占比——这一条直接改变问题的提法。
    """
    from split_score import dip_ratio, retained, split_score

    print("=" * 100)
    print("R15-A 描述趟：**不做检验**。traj 的 `split_score` 字段已丢弃，从 hist 重算")
    print("=" * 100)
    print(f'  {"run":<15}{"跨代":>6}{"mean(x)":>9}{"bin0/x":>8}{"x=0 帧":>8}'
          f'{"留存占空比":>11}{"前半":>8}{"后半":>8}{"坏SS一致":>9}')
    rows = []
    for r in runs:
        m = gate(r)
        h, g = r["hist"][m], r["gen"][m]
        x = low_mass(h)
        b0 = h[:, 0] / np.maximum(h.sum(1), 1.0)
        ret = np.array([retained(h[i], CTR_R13) for i in range(len(h))], float)
        # 世代加权：每帧的权重是它代表的代数跨度
        w = np.gradient(g)
        w = np.clip(w, 0.0, None)
        w = w / max(w.sum(), 1e-12)
        half = len(x) // 2
        wa, wb = w[:half] / max(w[:half].sum(), 1e-12), w[half:] / max(w[half:].sum(), 1e-12)
        # 坏版本 `split_score` 与重算值的一致率（判定 >0 是否同号）
        ss_new = np.array([split_score(h[i], CTR_R13)[0] for i in range(len(h))])
        ss_old = np.array([t for t in r["ss_field"]])[m]
        agree = float(((ss_new > 0) == (ss_old > 0)).mean())
        rows.append({
            "name": r["name"], "seed": r["seed"], "wn": r["wn"],
            "gen_span": float(g[-1] - g[0]),
            "mean_x": float((w * x).sum()), "b0_frac": float(b0.sum() / max(x.sum(), 1e-9)),
            "zero_frac": float((x == 0).mean()),
            "occ": float((w * ret).sum()), "occ_a": float((wa * ret[:half]).sum()),
            "occ_b": float((wb * ret[half:]).sum()),
            "x_a": float((wa * x[:half]).sum()), "x_b": float((wb * x[half:]).sum()),
            "agree": agree,
        })
    for q in rows:
        print(f'  {q["name"]:<15}{q["gen_span"]:>6.0f}{q["mean_x"]:>9.4f}{q["b0_frac"]:>8.3f}'
              f'{q["zero_frac"]:>8.3f}{q["occ"]:>11.3f}{q["occ_a"]:>8.3f}{q["occ_b"]:>8.3f}'
              f'{q["agree"]:>9.3f}')

    print()
    print("  汇总（**全部描述性，一个检验都没做**）")
    z = np.array([q["zero_frac"] for q in rows])
    print(f'    `x` 恰为 0 的帧占比：>30% 的 run **{int((z > 0.30).sum())}/24**，'
          f'>75% 的 **{int((z > 0.75).sum())}/24**')
    print(f'    ⇒ **一半以上的 run 里果专精簇大部分时间根本不存在。**')
    print(f'       「x 平均而言降不降」因此是问错了问题——见 §19.5 的 T3。')
    ag = np.array([q["agree"] for q in rows])
    print(f'    坏版 `split_score` 与重算版的同号一致率：均值 {ag.mean():.3f}，'
          f'最低 {ag.min():.3f} ⇒ **必须丢弃字段值**')
    b0 = np.array([q["b0_frac"] for q in rows])
    mx = np.array([q["mean_x"] for q in rows])
    sel = mx > 0.05
    print(f'    bin0 在 low_mass 里的占比（mean(x)>0.05 的 {int(sel.sum())} 个 run）：'
          f'{b0[sel].min():.3f}–{b0[sel].max():.3f}')
    print(f'    ⇒ `low_mass` 主要由 (−∞,0.20) 这一个箱决定，**0.20 以下零分辨率**')
    return rows


def checkpoint_vs_traj(runs):
    """**本轮最硬的一条**：同一批个体，截断口径与全员普查给出相反的历史。

    R13 的检查点用 `np.digitize` 把窗外个体从分子分母同时剔除。而果专精簇**成功的方式
    就是往更极端走**（`pref` 掉到 0.15 以下）⇒ **读数的失效模式与它要测的现象完全反相关**：
    簇越成功，被删得越干净。
    """
    from split_score import retained, split_score
    print()
    print("=" * 100)
    print("检查点（截断）vs 轨迹（全员普查）：同一批个体，两段相反的历史")
    print("=" * 100)
    for r in runs:
        # 只列检查点 low_mass 掉到 ~0 而轨迹 x 没掉的 run
        cps = r["checkpoints"]
        i_last = int(np.argmin(np.abs(r["t"] - cps[-1]["t"])))
        x = low_mass(r["hist"])
        if cps[-1]["low_mass"] > 0.02 or x[i_last] < 0.10:
            continue
        print(f'  === {r["name"]}（跨 {r["gen"][-1] - r["gen"][0]:.0f} 代，'
              f'捕食者均值 {np.nanmean(r["carn"]):.4f}）')
        print(f'    {"t":>8}{"检查点 low_mass":>16}{"检查点 in_window":>17}'
              f'{"轨迹 x":>10}{"轨迹 split_score":>17}{"retained":>10}')
        for cp in cps:
            i = int(np.argmin(np.abs(r["t"] - cp["t"])))
            n = np.array(cp["bin_n"], float)
            iw = n.sum() / max(cp["n_samples"], 1)
            ss, _ = split_score(r["hist"][i], CTR_R13)
            print(f'    {cp["t"]:>8}{cp["low_mass"]:>16.4f}{iw:>17.4f}'
                  f'{x[i]:>10.4f}{ss:>17.4f}{str(retained(r["hist"][i], CTR_R13)):>10}')


def correlate_pass(runs):
    """跨 run 相关。**后验生成的假说，标为描述性**——它决定下一轮烧卡扫什么，不当结论。"""
    from scipy.stats import spearmanr
    print()
    print("=" * 100)
    print("跨 run 相关（**后验、描述性**：这是数据生成的假说，不是本轮的判据）")
    print("=" * 100)
    v = {}
    for k, fn in (("mean_x", lambda r: float(np.mean(low_mass(r["hist"])))),
                  ("carn", lambda r: float(np.nanmean(r["carn"]))),
                  ("gen_span", lambda r: float(r["gen"][-1] - r["gen"][0])),
                  ("pop", lambda r: float(np.nanmean(r["pop"]))),
                  ("n_herb", lambda r: float(np.mean(r["n_herb"])))):
        v[k] = np.array([fn(r) for r in runs])
    for k in ("carn", "pop", "n_herb", "gen_span"):
        s = spearmanr(v["mean_x"], v[k])
        print(f'  Spearman(mean(x), {k:<9}) = {s.statistic:+.3f}   p={s.pvalue:.4g}'
              f'{"   ** 被排除 **" if s.pvalue > 0.05 else ""}')
    print()
    print("  ⚠️ `carn` 与 `pop` 彼此耦合（关掉捕食者 ⇒ 种群翻倍），本数据分不开。")
    print("     `gen_span` 若不显著，则「世代钟慢所以簇没被漂掉」这个解释被排除。")
    print("  ⚠️ 24 个 run 只有 12 个独立种子（`wn1_sN` 与 `wn2_sN` 同 founder），")
    print("     所以上面的 p 值偏乐观，只作定性用。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", default="noise",
                    choices=["noise", "describe", "all"])
    a = ap.parse_args()
    runs = load()
    if a.which in ("noise", "all"):
        noise_pass(runs)
    if a.which in ("describe", "all"):
        describe_pass(runs)
        checkpoint_vs_traj(runs)
        correlate_pass(runs)


if __name__ == "__main__":
    main()
