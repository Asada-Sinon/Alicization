"""R13：R11 那个少数簇是**稳定多态**，还是**正在走向固定的暂态**？

一个 run 给出整条轨迹，而不是靠不同 run 比较。判据见 `docs/multispecies_program.md` §16。

为什么问法要变（`plan-critic` 拦下两个 BLOCKER）
-----------------------------------------------
最初的设计是「`low_mass` 对检查点的逐种子斜率，检验它是否 <0」。**那个设计做完无法判断对错**：

- `low_mass` 在 `T05` 的臂均是 0.1264 而池化格内 SD 是 0.1251（**噪声/均值 = 0.99**），
  ICC 只有 0.22；
- **24 个 run 里 8 个 `low_mass` 恰为 0**。而该设计的「阳性答案」是「不拒绝斜率 = 0」——
  在一个 1/3 样本已贴地板的读数上，「斜率≈0」既是稳定多态的预测，**也是「早就没了」的预测**。

改成**二值留存 + 群体遗传学零假设**：`retained = split_score > 0`，零点可解析算出，
**MDE 问题随之消失**。且 t=20k 的 `split_score` 是**入组条件**（写在跑之前）——
那 8 个从一开始就没有少数簇的 run 会贡献「斜率恰好 0」，与稳定多态同号同值。

零假设的数（已独立复核）
------------------------
`forage_pref` 是单基因、sigmoid 映射、`mutation_sigma=0.02`、参与 crossover ⇒ 单倍型 WF。
中性臂 `N05` 实测 `sd(pref)=0.0862`、`mean_pref=0.4867` ⇒ 基因位方差 0.1189 ⇒
突变—漂变平衡 `V* = 2·Ne·σ_m²` 解出 **`Ne ≈ 149`**（**不是普查种群 2300，差 15 倍**）。
于是 p=0.13 的中性丢失期望是 `2Ne·p/(1−p)·|ln p| ≈ 91 代 ≈ 1.6 万步`。
**旁证**：`T05` 跑了 26000 步 ≈ 恰好一个丢失时间，而 **8/24 已丢** —— 正是中性预期。

采样结构（`plan-critic` S-2）
-----------------------------
- **便宜的轨迹**：每 `TRAJ_EVERY` 步在**设备上**算一次食草者 `forage_pref` 直方图
  （只传 14 个浮点，不把 90 MiB 的 genome 拉回主机），外加 `population` / `carnivore_frac`
  / 平均世代数。**世代长因此是读数而不是假设。**
- **贵的读数**：只在 5 个检查点各取一个 `WINDOW` 步的窗（逐箱摄入/饮水、`dlog`、二次曲面）。
- **BLRT 只在最后一个检查点**：199 次自助 EM 是主机 CPU 成本，5 并发下会互抢
  （`MEMORY.md [LEARN:env]`），而它已被两次预声明功效不足。

崩溃检测（修一个真 bug）
------------------------
`fitness_surface.py` 在种群崩溃时**不报错也不打印 `collapsed to zero`**：`herb` 全 False ⇒
`p.mean()` 返回 NaN，照样打一行 JSON，于是 `exp_stats.RunSet.load` 的崩溃自检**不会触发**。
本脚本每个采样点检查存活食草者数，低于阈值就打印 `collapsed to zero` 并中止。

    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260804-readouts/trajectory.py 100000 --seed 0 --json --set ...
"""
import argparse
import dataclasses
import json
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from split_score import split_score
from underworld import Config, new_world
from underworld import state as state_mod

# **值域必须覆盖 [0,1]，不能只量中间 70%。** `forage_pref_of` 是 sigmoid，值域就是 [0,1]；
# 而 R14 用 `linspace(0.15,0.85,15)` + `np.digitize` 把窗外个体从**分子和分母同时剔除**，
# 实测 `R50` 臂在 cp1 有 **58% 的个体落在标尺之外**，于是 `Δ(窗内比例)` 与 `Δlow_mass` 的
# Spearman 高达 +0.729（p=5e-5）——**「衰减」几乎就是截断本身**，主判据整个作废。
# 换成不截断的口径后符号翻转且不显著（+0.026，p=0.266）。
# 这是同型失误的**第三次**（固定带的双峰检测器、边界平滑补零、直方图 range），
# 见 `MEMORY.md [LEARN:tooling]`。
BINS = np.linspace(0.0, 1.0, 21)
CTR = 0.5 * (BINS[:-1] + BINS[1:])
TRAJ_EVERY = 250           # 便宜轨迹的采样间隔
WINDOW = 2000              # 每个检查点的贵读数窗口
WIN_EVERY = 10
MIN_HERB = 30              # 低于这个数判为崩溃


# R19（`multispecies_program.md` §22）：`mate_forage` 基因的读数。
#
# **必须记原始基因，不是映射后的 `w`**（§22.4）：`mate_forage_of` 是单边
# `clip(sigmoid(g)−0.5, 0, None)`，纯漂变下 `mean w` 会 by-construction 从 0 往上漂
# （实测：基因完全不被读的臂上 `mean w` 已是 0.0346）。`mean gene` 才是鞅。
#
# **阈值 0.847 = 入侵正收益区的下界**（§22.4c 实测：`w≤0.10` 的选择系数 ≈0 甚至 −0.003，
# `w=0.20` 跳到 +0.047；`w=0.2 ⇒ gene=0.847`）。记「越过它的个体占比」是因为
# **`mean gene` 会把「10% 冲上去、90% 不动」平均成「几乎没动」**——而那恰恰是
# 正反馈点火的样子。
MF_THRESH = 0.847
MF_BINS = np.linspace(-3.0, 3.0, 21)      # 基因是实数；初始 N(0,0.4)、669 代后 SD≈0.67


def _hist_fn(cfg):
    """设备上算食草者的 `forage_pref` 直方图 + R19 的 `mate_forage` 基因读数。"""
    lo, w, nb = float(BINS[0]), float(BINS[1] - BINS[0]), len(CTR)
    mlo, mw, mnb = float(MF_BINS[0]), float(MF_BINS[1] - MF_BINS[0]), len(MF_BINS) - 1

    @jax.jit
    def f(genome, alive, diet, generation):
        p = state_mod.forage_pref_of(genome, cfg)
        herb = alive & (diet < 0.35)
        idx = jnp.clip(((p - lo) / w).astype(jnp.int32), 0, nb - 1)
        h = jnp.zeros(nb).at[idx].add(herb.astype(jnp.float32))
        gen = jnp.sum(jnp.where(alive, generation, 0.0)) / jnp.maximum(jnp.sum(alive), 1)
        # --- R19 读数（食草者谱系，生态型住在那里）---
        mg = genome[:, cfg.mate_forage_index]
        nh = jnp.maximum(jnp.sum(herb), 1)
        mf_mean = jnp.sum(jnp.where(herb, mg, 0.0)) / nh          # H1 的主读数
        mf_above = jnp.sum(jnp.where(herb & (mg > MF_THRESH), 1.0, 0.0)) / nh
        midx = jnp.clip(((mg - mlo) / mw).astype(jnp.int32), 0, mnb - 1)
        mf_hist = jnp.zeros(mnb).at[midx].add(herb.astype(jnp.float32))
        return h, jnp.sum(herb), gen, mf_mean, mf_above, mf_hist
    return f


def _ld_fn(cfg):
    """两个觅食簇在**非 `forage_pref` 基因**上的分化（`multispecies_program.md` §20 的 H1）。

    **为什么主判据不是「分裂会不会变得更尖」**：`forage_pref` 是**单基因座**，
    而 `genome.py` 的均匀交叉**从不产生中间值**——一个位点非此即彼。
    中间型只来自 `σ=0.02` 的突变，与配对无关。**所以重组本来就压不平那个双峰，
    同型交配也就不可能让它更尖**；那个设计按构造是个 null（审查抓出来的）。
    同型交配真正改变的是 `forage_pref` 与**其余基因**的连锁不平衡。

    统计量 = 两簇（`pref<0.35` / `pref>0.65`）之间，**每个非 `forage_pref` 位点**的
    Cohen's d，取 `mean |d|`。**零假设不是 0**：随机交配下漂变本身就产生非零值，
    所以必须有 `w=0` 对照臂，不能拿 0 当零点。

    全部在设备上算，**只传回 9 个浮点**（全基因组 mean|d| + 7 个具名性状 + 两簇计数），
    不把 `[n_max, 1386]`（90 MiB）的 genome 拉回主机。
    """
    fp = cfg.forage_pref_index
    bp = cfg.brain_params
    # 屏蔽 forage_pref 自己：它按构造两簇完全分开，算进去只会稀释别的位点的信号
    keep = jnp.ones(cfg.genome_size, jnp.float32).at[fp].set(0.0)

    @jax.jit
    def f(genome, alive, diet):
        pref = jax.nn.sigmoid(genome[:, fp])
        herb = alive & (diet < 0.35)
        lo = (herb & (pref < 0.35)).astype(jnp.float32)[:, None]
        hi = (herb & (pref > 0.65)).astype(jnp.float32)[:, None]
        nlo, nhi = jnp.sum(lo), jnp.sum(hi)

        def moments(w, n):
            mu = jnp.sum(genome * w, 0) / jnp.maximum(n, 1.0)
            var = jnp.sum(w * (genome - mu) ** 2, 0) / jnp.maximum(n - 1.0, 1.0)
            return mu, var

        mlo, vlo = moments(lo, nlo)
        mhi, vhi = moments(hi, nhi)
        sp = jnp.sqrt(jnp.maximum(0.5 * (vlo + vhi), 1e-12))
        d = jnp.abs((mhi - mlo) / sp)
        # 两簇都要有足够个体，否则 d 是噪声；不够就返回 NaN 而不是一个小数字
        ok = (nlo >= 30.0) & (nhi >= 30.0)
        mean_d = jnp.where(ok, jnp.sum(d * keep) / jnp.sum(keep), jnp.nan)
        traits = jnp.where(ok, d[bp:bp + 7], jnp.nan)   # diet/invest/size/attack/escape/armor/spike
        # **分子与分母各回传一个**（R17 §20.1 留下的 undecidable）：`d = |Δμ| / σ_pooled`，
        # 只存比值就分不开「两簇真的分化」与「簇内方差被压小」。两个浮点就能补掉。
        num = jnp.where(ok, jnp.sum(jnp.abs(mhi - mlo) * keep) / jnp.sum(keep), jnp.nan)
        den = jnp.where(ok, jnp.sum(sp * keep) / jnp.sum(keep), jnp.nan)
        return mean_d, traits, nlo, nhi, num, den
    return f


def dip_ratio(n):
    p = np.asarray(n, float)
    p = p / max(p.sum(), 1.0)
    mid = (CTR >= 0.40) & (CTR <= 0.60)
    m = float((p * CTR).sum())
    sd = math.sqrt(max(float((p * (CTR - m) ** 2).sum()), 1e-12))
    e = float(norm.cdf(0.60, m, sd) - norm.cdf(0.40, m, sd))
    return float(p[mid].sum()) / max(e, 1e-9)


def _fmt(v):
    return "崩溃" if v is None else f"{v:.3f}"


def main(steps, seed, overrides, as_json, checkpoints):
    cfg = dataclasses.replace(Config(), seed=seed, **(overrides or {}))
    state, key, _step, scan_fn, _terrain = new_world(cfg)
    hist_fn = _hist_fn(cfg)
    ld_fn = _ld_fn(cfg)
    cps = sorted(int(round(steps * (i + 1) / checkpoints)) for i in range(checkpoints))

    row = {}
    traj, out_cps = [], []
    fired = set()                 # 显式记账：窗口本身推进 WINDOW 步，会越过后面的检查点
    done = 0
    collapsed = False

    def snapshot():
        h, nh, gen, mfm, mfa, mfh = hist_fn(state.genome, state.alive, state.diet,
                                            state.generation)
        return (np.asarray(h), int(nh), float(gen),
                float(mfm), float(mfa), np.asarray(mfh))

    while done < steps and not collapsed:
        nxt = min(done + TRAJ_EVERY, steps)
        due = [c for c in cps if c not in fired and c <= nxt]
        at_cp = bool(due)
        state, key, ms = scan_fn(state, key, nxt - done)
        done = nxt
        h, nh, gen, mf_mean, mf_above, mf_hist = snapshot()
        d = ms._asdict()
        traj.append({"t": done, "hist": h.tolist(), "n_herb": nh, "generation": gen,
                     # R19（§22）：append，不 insert——旧的读数脚本按 key 取，不受影响
                     "mf_mean": mf_mean, "mf_above": mf_above, "mf_hist": mf_hist.tolist(),
                     "population": float(np.asarray(d["population"])[-1]),
                     "carnivore_frac": float(np.asarray(d["carnivore_frac"])[-1]),
                     "split_score": split_score(h, CTR)[0]})
        if nh < MIN_HERB:
            print(f"!! population collapsed to zero (n_herb={nh} at t={done})")
            collapsed = True
            break

        if at_cp:
            # 贵读数：一个 WINDOW 步的窗，收 pref / food / drink
            pa, fa, da = [], [], []
            h0 = None
            for i in range(WINDOW // WIN_EVERY):
                state, key, _ms = scan_fn(state, key, WIN_EVERY)
                row = {k: float(np.asarray(v)[-1]) for k, v in _ms._asdict().items()}
                done += WIN_EVERY
                alive = np.asarray(state.alive)
                diet = np.asarray(state.diet)
                herb = alive & (diet < 0.35)
                pa.append(np.asarray(state_mod.forage_pref_of(state.genome, cfg))[herb])
                fa.append(np.asarray(state.last_food)[herb])
                da.append(np.asarray(state.last_drink)[herb])
            q = max(len(pa) // 4, 1)
            h0 = np.histogram(np.concatenate(pa[:q]), bins=BINS)[0].astype(float) / q
            h1 = np.histogram(np.concatenate(pa[-q:]), bins=BINS)[0].astype(float) / q
            p = np.concatenate(pa)
            f = np.concatenate(fa)
            dr = np.concatenate(da)
            idx = np.digitize(p, BINS) - 1
            n = np.array([(idx == i).sum() for i in range(len(CTR))], float)
            nn = n / max(n.sum(), 1)
            ok = (h0 > 5) & (h1 > 5)
            dlog = np.where(ok, np.log(np.maximum(h1, 1e-9) / np.maximum(h0, 1e-9)), np.nan)
            ss, sat = split_score(n, CTR)
            cp = {
                "t": done, "n_samples": int(len(p)),
                "bin_n": n.tolist(),
                "intake": [float(f[idx == i].mean()) if (idx == i).sum() >= 30 else None
                           for i in range(len(CTR))],
                "drink": [float(dr[idx == i].mean()) if (idx == i).sum() >= 30 else None
                          for i in range(len(CTR))],
                "dlog_per_bin": [None if not np.isfinite(v) else float(v) for v in dlog],
                "split_score": ss, "split_at": sat,
                "low_mass": float(nn[CTR < 0.35].sum()),
                "high_mass": float(nn[CTR > 0.65].sum()),
                "dip_ratio": dip_ratio(n),
                "sd": float(np.mean([x.std() for x in pa])),
                "mean_pref": float(p.mean()),
                # **窗内比例是一等读数，不是诊断信息。** 低于 0.95 直接判读数失效——
                # 这一条能让上面那类截断失误不可能再静默发生。
                "in_window": float(n.sum() / max(len(p), 1)),
                "readout_valid": bool(n.sum() / max(len(p), 1) >= 0.95),
            }
            # 连锁不平衡：两个觅食簇在非 `forage_pref` 基因上的分化（§20 的 H1）。
            # 设备上算，只回传 9 个浮点。
            _md, _tr, _nlo, _nhi, _num, _den = ld_fn(state.genome, state.alive, state.diet)
            cp["ld_mean_abs_d"] = float(_md)
            cp["ld_trait_d"] = [float(v) for v in np.asarray(_tr)]
            cp["ld_n_lo"], cp["ld_n_hi"] = int(_nlo), int(_nhi)
            cp["ld_num"], cp["ld_den"] = float(_num), float(_den)
            # **通量三件套：操作检查（manipulation check）用的**。
            # `§13.7`/`§16.3` 说「资源比固定在 11%」是**承载**口径；而这个世界实测的
            # **供给通量**配比早就是 ~1:2（`T05` 各 run 的 `frugivory_frac` = 0.31–0.44）。
            # 没有这三个读数，「配比没动」和「配比动了但没用」分不开——R14 的臂达标与否
            # 就由它判，不由承载表判。三行，`ms._asdict()` 里本来就有。
            for _k in ("graze_gain", "fruit_gain", "frugivory_frac"):
                cp[_k] = row.get(_k, float("nan"))
            # BLRT 只在最后一个检查点（199 次自助 EM 是主机 CPU 成本，且已两次预声明功效不足）
            if done >= steps - WINDOW - TRAJ_EVERY and len(p) >= 30:
                from probe_trait_dist import blrt_two_components
                rng = np.random.default_rng(seed)
                smp = p if len(p) <= 2000 else p[rng.choice(len(p), 2000, replace=False)]
                cp["blrt_lr_per_n"] = blrt_two_components(
                    smp.astype(np.float64), n_boot=199, seed=seed)["lr_per_n"]
            out_cps.append(cp)
            fired.update(due)
            # 窗口后补一个轨迹点，否则 `gen_total` 会漏掉窗口里走过的世代
            hw, nhw, genw, mfm_w, mfa_w, mfh_w = snapshot()
            # R19：**末检查点是 H1 的主判据**（§22.4），这一条 append 少了这三个字段
            # 就等于跑完没有数据——差点漏掉。
            traj.append({"t": done, "hist": hw.tolist(), "n_herb": nhw, "generation": genw,
                         "mf_mean": mfm_w, "mf_above": mfa_w, "mf_hist": mfh_w.tolist(),
                         "population": float("nan"), "carnivore_frac": float("nan"),
                         "split_score": split_score(hw, CTR)[0]})
            print(f"  t={done:>7}  split_score={ss:.4f}  low_mass={cp['low_mass']:.4f}  "
                  f"dip={cp['dip_ratio']:.3f}  n_herb={int(n.sum() // (WINDOW // WIN_EVERY))}")

    jax.block_until_ready(state.genome)
    out = {"seed": seed, "steps": steps, "collapsed": collapsed,
           "checkpoints": out_cps, "traj": traj, "overrides": overrides or {}}
    # 入组条件与主判据用的量，提到顶层方便分析脚本读
    if out_cps:
        out["entry_split_score"] = out_cps[0]["split_score"]      # t = 第一个检查点
        out["final_split_score"] = out_cps[-1]["split_score"]
        out["retained"] = int(out_cps[-1]["split_score"] > 0.0)
        out["gen_total"] = float(traj[-1]["generation"] - traj[0]["generation"])
    # **轨迹级读数**（`feasibility.md` §17.9/§18 的判据，收编进常规输出）。
    # 判据是「后半程世代加权的 `retained` 占空比」，而**中性零分布留在分析时算**——
    # 它的单位是「跨 run 的计数」，per-run 的零分布不是对的单位，
    # 而且每个 run 都跑 200 次模拟会把 sweep 的 CPU 成本翻几倍。
    # 零假设需要的输入（帧 0 的直方图、跨代数、帧数）本来就在 `traj` 里，分析时现取。
    #
    # **三个一起报，不许单看占空比**（§18.4 的代价换来的）：Stage 2 的四个臂里
    # `R50n` 的占空比 0.9512 与 `R38n` 的 0.9870 几乎一样，但前者是
    # `low_mass=0.7492 / split_score=0.2506`（整个分布搬到果实侧），
    # 后者是 `0.4501 / 0.4423`（近 50/50 的平衡双簇）——**是两件不同的事。**
    # **崩溃的 run 不吐这组读数。** 崩溃时上面 `break`，它的「后半程」是崩溃前的半段——
    # 那会产出一个形状正常的错误数字，而不是异常（这正是本项目反复踩的那类失误）。
    # 字段一律写出来（崩溃时为 None），免得下游 `d["retained_occ"]` 时有时无地 KeyError。
    out["retained_occ"] = out["retained_occ_first"] = None
    out["low_mass_second"] = out["split_score_second"] = None
    if len(traj) >= 4 and not collapsed:
        from neutral_null import occupancy, weighted_second_half
        _h = [np.asarray(q["hist"], float) for q in traj]
        _g = np.array([q["generation"] for q in traj], float)
        # **三个读数必须过同一把尺子**（同一个 `mode`），否则加权口径的差异会被读成效应。
        out["retained_occ"] = occupancy(_h, _g, CTR)                       # 后半程
        out["retained_occ_first"] = occupancy(_h, _g, CTR, second_half=False)
        out["low_mass_second"] = weighted_second_half(
            [h[CTR < 0.35].sum() / max(h.sum(), 1.0) for h in _h], _g)
        out["split_score_second"] = weighted_second_half(
            [split_score(h, CTR)[0] for h in _h], _g)

    if as_json:
        print("JSON " + json.dumps(out))
    print(f"\n[轨迹。总世代数 {out.get('gen_total', float('nan')):.0f}；"
          f"后半程 retained 占空比 {_fmt(out['retained_occ'])}"
          f"（前半 {_fmt(out['retained_occ_first'])}）；"
          f"后半 low_mass {_fmt(out['low_mass_second'])} / "
          f"split_score {_fmt(out['split_score_second'])}。"
          f"中性零分布分位由分析脚本算，见 feasibility.md §17.9/§18]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="?", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--checkpoints", type=int, default=5)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="F=V")
    a = ap.parse_args()
    from underworld.config import parse_overrides
    main(a.steps, a.seed, parse_overrides(a.sets), a.json, a.checkpoints)
