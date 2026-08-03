"""实验统计口径的**唯一实现**——`docs/run_to_run_variance.md` §7.1 那三条分析口径。

为什么它存在：新协议（12 种子 × 2 重复）比旧协议多一层「格均值」和一次「噪声自估」，
而在此之前每一轮都由 result-analyst 现写一份 `analyze.py`。不共享实现，就等于每轮
把同一段算术重推一次、每轮都有推错的机会——`explorations/20260803-partition/` 下三份
脚本里这段代码就已经被抄了三遍。

三条口径（`docs/run_to_run_variance.md` §7.1，缺一不可）：

1. **先把每格 `r` 次重复取均值，再对 `s` 个格均值做配对检验。** 不得把 `s×r` 个 run 当成
   `s×r` 个独立样本——重复是同种子复跑，只含混沌噪声（`--seed` 在 20000 步上控制不住这个
   世界，见该文档 §4）。→ `RunSet.cell_means()`
2. **必报「效应量 ÷ 配对差噪声」**，噪声 `= √2·σ̂_W/√r`，`σ̂_W` 由本次实验的格内散度自估，
   不许引用别处的数。比值 <1 一律标注功效不足，哪怕 p 过线。→ `PairedResult.ratio` /
   `.underpowered`
3. **护栏容差必须对着 `√2·σ̂_W` 定**，不许凭直觉写 ±10% / +5pp。→ `RunSet.pair_noise()`

用法（在 `explorations/<run_id>/analyze.py` 里）::

    import sys; sys.path.insert(0, 'scripts')
    from exp_stats import RunSet, paired, mde_sign_consistent

    rs = RunSet.load('outputs/20260803-partition',
                     arms=['P_tradeoff0', 'Q_tradeoff1'], seeds=range(6), reps=(1, 2, 3))
    r = paired(rs, 'sd', 'Q_tradeoff1', 'P_tradeoff0')
    print(r.format())

回归验收：`explorations/20260803-partition/verify_exp_stats.py` 用本模块从 90 个 log 复算
P4，逐位核对 `docs/multispecies_program.md` §9.11 里已发表的每一个数。改动本模块后必须重跑它。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import norm, rankdata, wilcoxon

# 新协议的默认设计（§7.1）。改这两个数之前先读 §7.0 的算术：SE 只取决于乘积 s·r，
# 但 p 下限与伪重复只由 s 决定，所以预算优先加种子。
DEFAULT_SEEDS = 12
DEFAULT_REPS = 2

# bootstrap 的默认重抽次数与随机种子。固定种子是为了让同一份日志每次复算逐位一致
# ——分析结果本身不该带随机性。
BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20260803


def wilcoxon_p_floor(s: int) -> float:
    """`n=s` 配对符号秩检验能达到的**最小双侧 p**。

    全部 s 个差同号时，双侧 p = 2 / 2**s。这就是本项目结论常年顶在 0.031 的机械原因
    （s=6 → 0.03125）；s=12 → 0.00049。
    """
    if s < 1:
        return float("nan")
    return min(1.0, 2.0 / (2.0 ** s))


@dataclass
class RunSet:
    """一批 `(arm, seed, rep)` 三元组索引的 run，外加它们的出处。

    `records[(arm, seed, rep)]` 是 `run_headless.py --json` / `probe_trait_dist.py` 末尾那行
    `JSON {...}` 解析出来的 dict；`sources[...]` 是 `文件名:行号`，用来满足
    `/exp` skill 硬约束 2「每个数字要能指到某个文件」。
    """

    records: dict[tuple[str, int, int], dict]
    sources: dict[tuple[str, int, int], str]
    arms: list[str]
    seeds: list[int]
    reps: list[int]
    problems: list[str] = field(default_factory=list)

    # ---------------- 载入 ----------------
    @classmethod
    def load(
        cls,
        directory: str,
        arms: Sequence[str],
        seeds: Iterable[int],
        reps: Iterable[int] = (1, 2),
        name_fmt: str = "{arm}_s{seed}_r{rep}.log",
        json_prefix: str = "JSON ",
        check: Callable[[dict, str, int, int], str | None] | None = None,
    ) -> "RunSet":
        """从 `directory` 读 `arms × seeds × reps` 个日志，每个取唯一一行 `JSON {...}`。

        载入期就把能机械核对的都核了，结果进 `.problems`——**调用方必须自己检查它**
        （`assert not rs.problems` 或打印出来）。已核：文件缺失、JSON 行不唯一、
        记录里的 `seed` 与文件名不符、日志中出现 `collapsed to zero`。
        `check` 可再加一条按臂归因的校验（例如「`overrides.forage_tradeoff` 等于臂名剂量」），
        返回错误字符串或 None。
        """
        arms, seeds, reps = list(arms), list(seeds), list(reps)
        records: dict[tuple[str, int, int], dict] = {}
        sources: dict[tuple[str, int, int], str] = {}
        problems: list[str] = []
        for arm in arms:
            for s in seeds:
                for r in reps:
                    path = os.path.join(directory, name_fmt.format(arm=arm, seed=s, rep=r))
                    if not os.path.exists(path):
                        problems.append(f"{path}: 文件不存在")
                        continue
                    lines = open(path).read().splitlines()
                    hits = [(i + 1, ln) for i, ln in enumerate(lines) if ln.startswith(json_prefix)]
                    if len(hits) != 1:
                        problems.append(f"{path}: 找到 {len(hits)} 行 JSON，期望 1")
                        continue
                    lineno, raw = hits[0]
                    rec = json.loads(raw[len(json_prefix):])
                    if rec.get("seed") != s:
                        problems.append(f"{path}: JSON seed={rec.get('seed')} != 文件名 s{s}")
                    if "collapsed to zero" in "\n".join(lines):
                        problems.append(f"{path}: 出现 population collapsed to zero")
                    if check is not None:
                        msg = check(rec, arm, s, r)
                        if msg:
                            problems.append(f"{path}: {msg}")
                    records[(arm, s, r)] = rec
                    sources[(arm, s, r)] = f"{os.path.basename(path)}:{lineno}"
        return cls(records, sources, arms, seeds, reps, problems)

    # ---------------- 口径 1：格均值 ----------------
    def cell(self, arm: str, seed: int, metric: str) -> tuple[float, float, np.ndarray]:
        """一个格（同臂同种子的 `r` 次重复）的 `(均值, 格内 SD, 原始值)`。"""
        v = np.array([self.records[(arm, seed, r)][metric] for r in self.reps], float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        return float(v.mean()), sd, v

    def cell_means(self, arm: str, metric: str) -> np.ndarray:
        """一个臂的 `s` 个格均值——**这才是配对检验的样本**（口径 1）。"""
        return np.array([self.cell(arm, s, metric)[0] for s in self.seeds], float)

    def raw(self, arm: str, metric: str) -> np.ndarray:
        """一个臂全部 `s×r` 个 run 的原始值。只用于描述性统计与极值追溯——
        **拿它做检验就是伪重复**（重复是同种子复跑）。"""
        return np.array(
            [self.records[(arm, s, r)][metric] for s in self.seeds for r in self.reps], float
        )

    # ---------------- 口径 2/3：噪声自估 ----------------
    def pooled_within_sd(self, metric: str, arms: Sequence[str] | None = None) -> float:
        """`σ̂_W`：跨格池化的**格内**（同种子复跑）SD，即本世界的复现噪声。

        池化用平方平均（等格大小下即合并方差的平方根）。`arms=None` 时用全部臂；
        判决时若某臂本身就是被处理改动方差的对象，只传对照/中性臂更干净。
        """
        arms = list(arms) if arms is not None else self.arms
        w = np.array([self.cell(a, s, metric)[1] for a in arms for s in self.seeds], float)
        return float(np.sqrt((w ** 2).mean()))

    def pair_noise(self, metric: str, arms: Sequence[str] | None = None) -> float:
        """配对差的噪声 `√2·σ̂_W/√r`（口径 2 的分母，也是口径 3 的护栏尺）。

        `√2` 来自两臂各贡献一份独立噪声；`/√r` 来自格均值已经把 `r` 次重复平均掉了。
        **护栏容差要对着这个数定，不许凭直觉写 ±10% / +5pp**——那大约就是 1 个噪声 SD，
        零效应也会经常撞线。
        """
        return self.pooled_within_sd(metric, arms) * math.sqrt(2) / math.sqrt(len(self.reps))

    # ---------------- 载入自检 ----------------
    def overrides_diff(self) -> list[str]:
        """各臂 `overrides` 里**取值不同**的字段名。

        只有一个字段不同才谈得上把差异归因给它。返回多于一项 → 多变量混杂，判决作废。
        """
        per_arm: dict[str, dict] = {}
        for (arm, _, _), rec in self.records.items():
            per_arm.setdefault(arm, rec.get("overrides", {}))
        keys = sorted(set().union(*[set(d) for d in per_arm.values()])) if per_arm else []
        return [
            k for k in keys
            if len({str(per_arm[a].get(k, "<缺>")) for a in per_arm}) > 1
        ]


def bootstrap_ci(
    d: Sequence[float],
    B: int = BOOTSTRAP_B,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """配对差均值的百分位 bootstrap 区间。默认固定种子，同一份数据复算逐位一致。"""
    rng = rng if rng is not None else np.random.default_rng(BOOTSTRAP_SEED)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(axis=1)
    return float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


@dataclass
class PairedResult:
    """一次配对检验的全部报告字段。`format()` 直接给出可贴进 docs 的多行文本。"""

    metric: str
    label: str
    arm_a: str
    arm_b: str
    mean_a: float
    mean_b: float
    diff: np.ndarray          # 逐种子配对差 a - b（口径 1：格均值之差）
    n_pos: int
    n_neg: int
    W: float
    p: float
    p_floor: float
    ci: tuple[float, float]
    dz: float                 # Cohen dz
    r_rb: float               # 匹配对秩双列相关
    noise: float              # √2·σ̂_W/√r（口径 2）
    ratio: float              # 效应量 ÷ 噪声
    observed_sd: float        # 实测配对差 SD

    @property
    def underpowered(self) -> bool:
        """比值 <1 → 功效不足，**哪怕 p 过线也必须标注**（口径 2）。"""
        return abs(self.ratio) < 1.0

    @property
    def at_floor(self) -> bool:
        """p 顶在该样本量的机械下限上——是「地板值」不是强证据。"""
        return abs(self.p - self.p_floor) < 1e-9

    def format(self) -> str:
        n = len(self.diff)
        pct = 100 * self.diff.mean() / self.mean_b if self.mean_b else float("nan")
        lo, hi = self.ci
        lines = [
            f"--- {self.metric} : {self.label} ---",
            f"  {self.arm_a} 均值 = {self.mean_a:.5f}   {self.arm_b} 均值 = {self.mean_b:.5f}   "
            f"差 = {self.diff.mean():+.5f} ({pct:+.2f}% of {self.arm_b})",
            f"  逐种子差 = {np.array2string(self.diff, precision=5, sign='+')}",
            f"  同向种子数: 差>0 {self.n_pos}/{n}, 差<0 {self.n_neg}/{n}",
            f"  配对 Wilcoxon 双侧: W={self.W:.1f}  p={self.p:.5f}"
            f"   [n={n} 地板 p={self.p_floor:.5f}]{'  ** 顶在地板 **' if self.at_floor else ''}",
            f"  95% bootstrap CI (B={BOOTSTRAP_B}, 配对差均值) = [{lo:+.5f}, {hi:+.5f}]  "
            f"{'不含 0' if lo * hi > 0 else '**含 0**'}",
            f"  效应量: Cohen dz = {self.dz:+.3f}   匹配对秩双列 r_rb = {self.r_rb:+.3f}",
            f"  效应/噪声比 (自估配对差噪声 {self.noise:.6f}) = {self.ratio:+.3f}"
            f"{'   ** <1，功效不足 **' if self.underpowered else ''}",
            f"  配对差实测 SD = {self.observed_sd:.6f} vs 仅噪声预测 {self.noise:.6f} "
            f"-> 实测/预测 = {self.observed_sd / self.noise:.2f}",
        ]
        return "\n".join(lines)


def paired(
    rs: RunSet,
    metric: str,
    arm_a: str,
    arm_b: str,
    label: str | None = None,
    noise_arms: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> PairedResult:
    """口径 1+2 的完整实现：格均值 → 配对 Wilcoxon → 效应量 → bootstrap CI → 效应/噪声比。

    `arm_a - arm_b` 是报告的差值方向（一般 a=处理、b=对照）。`noise_arms` 指定拿哪些臂
    自估 `σ̂_W`（默认全部臂）。
    """
    x, y = rs.cell_means(arm_a, metric), rs.cell_means(arm_b, metric)
    d = x - y
    W, p = wilcoxon(x, y, alternative="two-sided")
    lo, hi = bootstrap_ci(d, rng=rng)
    sd_d = float(d.std(ddof=1))
    dz = float(d.mean() / sd_d) if sd_d > 0 else float("nan")
    rk = rankdata(np.abs(d))
    r_rb = float((rk[d > 0].sum() - rk[d < 0].sum()) / (len(d) * (len(d) + 1) / 2))
    noise = rs.pair_noise(metric, noise_arms)
    return PairedResult(
        metric=metric,
        label=label if label is not None else f"{arm_a} vs {arm_b}",
        arm_a=arm_a,
        arm_b=arm_b,
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        diff=d,
        n_pos=int((d > 0).sum()),
        n_neg=int((d < 0).sum()),
        W=float(W),
        p=float(p),
        p_floor=wilcoxon_p_floor(len(d)),
        ci=(lo, hi),
        dz=dz,
        r_rb=r_rb,
        noise=noise,
        ratio=float(d.mean() / noise) if noise > 0 else float("nan"),
        observed_sd=sd_d,
    )


def mde_sign_consistent(sigma_d: float, s: int, power: float = 0.80) -> float:
    """**全 `s` 个种子同向**（即 p 顶到地板 `2/2ˢ`）所需的最小可检出效应。

    功效 = P(s 个差同号) = Φ(δ/σ)^s，令它等于 `power` 解出 δ。

    **注意这是「达到地板 p」的口径，不是「达到 p≤0.05」的口径。** n=6 时两者等价
    （地板恰好是 0.031，且只有 6/6 同向才够得着），这也是它当初被写出来的场景；
    n=12 时地板是 0.00049，远严于 0.05，所以这个数对 n=12 **偏保守**。要 p≤0.05 口径的
    功效，用 `power_paired_wilcoxon()`。

    报零结果时必须把它写出来：零结果只能排除**大于这个数**的效应，更小的窗口本设计看不见。
    """
    z = float(norm.ppf(power ** (1.0 / s)))
    return z * sigma_d


def _signed_rank_null(n: int) -> np.ndarray:
    """`W+`（正差的秩和）的**精确**零分布计数：`counts[k]` = 有多少种符号组合给出 `W+=k`。

    多项式卷积 DP：每个秩 `i` 要么计入要么不计入，总质量 `2ⁿ`。连续分布下 `d` 不会有
    0 值或 `|d|` 并列，所以不需要 scipy 那套 zero/tie 修正。
    """
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for i in range(1, n + 1):
        counts = counts + np.concatenate((np.zeros(i), counts[:-i]))
    return counts


def power_paired_wilcoxon(
    effect: float,
    noise: float,
    s: int,
    alpha: float = 0.05,
    B: int = 20000,
    rng: np.random.Generator | None = None,
) -> float:
    """配对 Wilcoxon 双侧检验在 `d ~ N(effect, noise²)` 下的功效，蒙特卡洛估计。

    `noise` 是**配对差**的噪声（即 `√2·σ̂_W/√r`，`RunSet.pair_noise()` 给的那个数），
    不是 `σ_W`。零分布用 `_signed_rank_null` 精确枚举，所以小 n 下没有正态近似误差
    ——而小 n 正是本项目关心的区间。

    为什么不用「全部同向」的概率代替：那只在 n=6 附近成立。n=12 时 p≤0.05 允许有反向
    种子，用同向概率会把所需样本量报得离谱地大；而同向概率**随 n 单调下降**，直接拿它
    解样本量还会解出方向相反的答案。
    """
    if s < 6:
        return 0.0  # n<6 时双侧最小可达 p 是 2/2^5=0.0625 > 0.05，任何数据都过不了线
    rng = rng if rng is not None else np.random.default_rng(BOOTSTRAP_SEED)
    d = rng.normal(effect, noise, size=(B, s))
    order = np.argsort(np.abs(d), axis=1)
    ranks = np.empty(order.shape, dtype=np.int64)
    np.put_along_axis(ranks, order, np.tile(np.arange(1, s + 1), (B, 1)), axis=1)
    w_plus = (ranks * (d > 0)).sum(axis=1)

    counts = _signed_rank_null(s)
    cdf = np.cumsum(counts) / counts.sum()                 # P(W+ <= k)
    sf = 1.0 - np.concatenate(([0.0], cdf[:-1]))           # P(W+ >= k)
    p = np.minimum(1.0, 2.0 * np.minimum(cdf[w_plus], sf[w_plus]))
    return float((p <= alpha).mean())


def required_seeds(
    effect: float,
    sigma_w: float,
    reps: int = DEFAULT_REPS,
    power: float = 0.80,
    alpha: float = 0.05,
    max_seeds: int = 60,
    rng: np.random.Generator | None = None,
) -> int | None:
    """要在 `power` 功效下把效应 `effect` 检出到 `p≤alpha`，需要多少个种子。

    设计阶段用它回答「这个预算够不够」，而不是事后解释为什么没跑出来。
    返回 `None` 表示 `max_seeds` 之内都不够——那时该重新想的是效应量或指标，不是加机器。

    `sigma_w` 是**同种子重跑**的 SD（不是跨种子 SD——用错这个正是 `conventions.md` §5.1
    更正的那个错误，它把所需样本量低估了约 27 倍）。参考量级见
    `docs/run_to_run_variance.md` §4。
    """
    noise = sigma_w * math.sqrt(2) / math.sqrt(reps)
    if noise <= 0:
        return 6
    for s in range(6, max_seeds + 1):
        if power_paired_wilcoxon(effect, noise, s, alpha, rng=rng) >= power:
            return s
    return None
