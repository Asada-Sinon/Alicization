"""`scripts/exp_stats.py` 的回归验收：用共享库复算 P4，逐位核对已发表的数。

回答什么：`docs/run_to_run_variance.md` §7.2 第 4 项的验收条件——「用
`outputs/20260803-partition/` 复算 P4，数字与 §9.11 逐位一致」。P4 当初是由
`analyze_curve.py`（26k，那段统计算术手抄了第三遍）算出来的；本脚本换成 import
`scripts/exp_stats.py` 重算同样的量，两边对不上就说明库写错了。

读什么：`outputs/20260803-partition/{P_tradeoff0,L_tradeoff0125,N_tradeoff025,
M_tradeoff05,Q_tradeoff1}_s{0..5}_r{1..3}.log`（90 个，每个一行 `JSON {...}`）。
期望值硬编码在下面 EXPECTED 里，出处逐条标了 `docs/multispecies_program.md` §9.11 的行。

注意这是**旧协议的数据**（s=6, r=3），不是新协议的 s=12/r=2——它在这里只作回归夹具，
用来证明库的算术与已发表结论一致。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-partition/verify_exp_stats.py
"""
import sys

sys.path.insert(0, "scripts")

import numpy as np
from scipy.stats import friedmanchisquare

from exp_stats import RunSet, mde_sign_consistent, paired, wilcoxon_p_floor

DIR = "outputs/20260803-partition"
ARMS = ["P_tradeoff0", "L_tradeoff0125", "N_tradeoff025", "M_tradeoff05", "Q_tradeoff1"]
DOSE = dict(zip(ARMS, [0.0, 0.125, 0.25, 0.5, 1.0]))
SHORT = dict(zip(ARMS, ["P(0.0)", "L(.125)", "N(.25)", "M(0.5)", "Q(1.0)"]))
P, L, N, M, Q = ARMS
SEEDS, REPS = list(range(6)), [1, 2, 3]

# §9.11 已发表的数字。左边是本脚本算出来的量，右边是文档里印着的值 + 出处行号。
EXPECTED = {
    # docs/multispecies_program.md:811 —— 五点 sd 臂均值
    "mean_sd": {P: 0.09508, L: 0.07953, N: 0.09294, M: 0.07025, Q: 0.05866},
    # :812 —— 各点对 P 的配对差、同向数、p
    "diff": {"L-P": -0.01555, "N-P": -0.00214, "M-P": -0.02483, "Q-P": -0.03643},
    "p": {"L-P": 0.03125, "N-P": 1.00, "M-P": 0.03125, "Q-P": 0.03125},
    "n_pos": {"L-P": 0, "N-P": 4, "M-P": 0, "Q-P": 0},   # :814 「L>P 的种子数 0/6, N>P 4/6」
    # :819 —— N 相对 L 的真实抬头
    "NL_diff": 0.01341, "NL_pos": 5, "NL_p": 0.0625, "NL_ci": (0.005, 0.021),
    # :823 —— 剔除 seed 0 后 N-P 翻正
    "NP_drop_s0": 0.00440,
    # :839 —— 最强的 L-P 效应/噪声比也只有 0.62；实测配对差 SD 是噪声预测的 0.51 倍
    "LP_ratio": -0.62, "LP_obs_over_pred": 0.51,
    # :826 —— 本设计的 80% 功效 MDE
    "mde_pct": {"L-P": 24.2, "N-P": 40.9},
    # :831 —— bimodality_coefficient > 0.555 的 run 数
    "bimodal": {P: 10, L: 10, N: 10, M: 5, Q: 3},
    # :834-835 —— 五臂 Friedman 护栏
    "friedman": {"carnivore_frac": 0.83, "population": 0.78, "min_pop": 0.059},
}

FAIL: list[str] = []


def chk(name, got, want, tol, unit=""):
    ok = abs(got - want) <= tol
    if not ok:
        FAIL.append(f"{name}: 算得 {got:.6f}, 文档 {want:.6f} (容差 {tol})")
    print(f"  {'OK ' if ok else '**差异**'} {name:<44} 算得 {got:>+10.5f}{unit}  "
          f"文档 {want:>+9.5f}{unit}")


def check_dose(rec, arm, s, r):
    got = rec["overrides"].get("forage_tradeoff", 0.0)
    if abs(float(got) - DOSE[arm]) > 1e-12:
        return f"overrides.forage_tradeoff={got} != 臂名剂量 {DOSE[arm]}"
    return None


print("=" * 92)
print("§0 载入（走 exp_stats.RunSet.load，自带缺失/JSON/seed/collapse/归因五项自检）")
print("=" * 92)
rs = RunSet.load(DIR, ARMS, SEEDS, REPS, check=check_dose)
print(f"  期望 {len(ARMS) * len(SEEDS) * len(REPS)} 个 run，实际载入 {len(rs.records)} 个")
print(f"  载入自检 problems = {rs.problems if rs.problems else '无'}")
print(f"  各臂 overrides 差异字段 = {rs.overrides_diff()}  "
      f"{'[单变量，可归因]' if rs.overrides_diff() == ['forage_tradeoff'] else '[多变量，不可归因!]'}")
assert not rs.problems, rs.problems
assert rs.overrides_diff() == ["forage_tradeoff"]
print(f"  n=6 配对 Wilcoxon 双侧 p 下限 = {wilcoxon_p_floor(6):.5f}  (文档 :812 印的 0.031)")

print()
print("=" * 92)
print("§1 口径 1：格均值（3 次重复先平均）→ 五点 sd 臂均值 vs §9.11:811")
print("=" * 92)
for a in ARMS:
    chk(f"mean sd {SHORT[a]}", float(rs.cell_means(a, "sd").mean()), EXPECTED["mean_sd"][a], 5e-6)

print()
print("=" * 92)
print("§2 口径 2：配对检验 + 效应/噪声比 vs §9.11:812/814/839")
print("=" * 92)
# §9.11 的噪声自估只用 L、N 两臂（analyze_curve.py §3），这里照搬同一口径
NOISE_ARMS = [L, N]
print(f"  σ̂_W 自估口径：仅用 {[SHORT[a] for a in NOISE_ARMS]} 两臂 36 个 run 的格内散度")
print(f"  池化格内 SD σ̂_W = {rs.pooled_within_sd('sd', NOISE_ARMS):.6f}   "
      f"配对差噪声 √2·σ̂_W/√3 = {rs.pair_noise('sd', NOISE_ARMS):.6f}")
res = {}
for arm, tag in [(L, "L-P"), (N, "N-P"), (M, "M-P"), (Q, "Q-P")]:
    r = paired(rs, "sd", arm, P, label=f"{SHORT[arm]} vs {SHORT[P]}", noise_arms=NOISE_ARMS)
    res[tag] = r
    print()
    print(r.format())
    chk(f"{tag} 配对差", float(r.diff.mean()), EXPECTED["diff"][tag], 5e-6)
    chk(f"{tag} p 值", r.p, EXPECTED["p"][tag], 5e-3)
    if r.n_pos != EXPECTED["n_pos"][tag]:
        FAIL.append(f"{tag} 同向数: 算得 {r.n_pos}/6, 文档 {EXPECTED['n_pos'][tag]}/6")
    print(f"  OK  {tag} 差>0 种子数 {r.n_pos}/6 == 文档 {EXPECTED['n_pos'][tag]}/6")

print()
chk("L-P 效应/噪声比", res["L-P"].ratio, EXPECTED["LP_ratio"], 5e-3)
chk("L-P 实测SD/噪声预测", res["L-P"].observed_sd / res["L-P"].noise,
    EXPECTED["LP_obs_over_pred"], 5e-3)
assert res["L-P"].underpowered, "L-P 比值 <1，应被标为功效不足"
assert res["N-P"].ci[0] * res["N-P"].ci[1] < 0, "N-P 的 CI 应当含 0（文档 :812）"
print(f"  OK  L-P 被标注 underpowered = {res['L-P'].underpowered}（口径 2 要求 <1 必标）")
print(f"  OK  N-P bootstrap CI = [{res['N-P'].ci[0]:+.5f}, {res['N-P'].ci[1]:+.5f}] 含 0")
print(f"  OK  L/M/Q-P 顶在地板 p: {[res[t].at_floor for t in ['L-P', 'M-P', 'Q-P']]}")

print()
print("=" * 92)
print("§3 两处「必须写准」的措辞 vs §9.11:819/823")
print("=" * 92)
rNL = paired(rs, "sd", N, L, label=f"{SHORT[N]} vs {SHORT[L]}", noise_arms=NOISE_ARMS)
print(rNL.format())
chk("N-L 配对差（0.25 相对 0.125 的抬头）", float(rNL.diff.mean()), EXPECTED["NL_diff"], 5e-6)
chk("N-L p 值", rNL.p, EXPECTED["NL_p"], 5e-5)
if rNL.n_pos != EXPECTED["NL_pos"]:
    FAIL.append(f"N-L 同向数: 算得 {rNL.n_pos}/6, 文档 {EXPECTED['NL_pos']}/6")
print(f"  OK  N-L 差>0 种子数 {rNL.n_pos}/6 == 文档 {EXPECTED['NL_pos']}/6")
chk("N-L CI 下界", rNL.ci[0], EXPECTED["NL_ci"][0], 5e-4)
chk("N-L CI 上界", rNL.ci[1], EXPECTED["NL_ci"][1], 5e-4)
print("  => 「不能写五点单调」成立：N 相对 L 有真实抬头，CI 不含 0。")

d_np = rs.cell_means(N, "sd") - rs.cell_means(P, "sd")
chk("剔除 seed 0 后 N-P（应翻正）", float(d_np[1:].mean()), EXPECTED["NP_drop_s0"], 5e-6)
print(f"  => 「不能写 N 比漂变更窄」成立：负号来自 seed 0 那个离群 run "
      f"(P_tradeoff0_s0_r3 sd={rs.records[(P, 0, 3)]['sd']:.5f}，"
      f"全 90 run 最大值 {max(rs.raw(a, 'sd').max() for a in ARMS):.5f})。")

print()
print("=" * 92)
print("§4 零结果的边界：MDE vs §9.11:826")
print("=" * 92)
base = float(rs.cell_means(P, "sd").mean())
for tag in ["L-P", "N-P"]:
    mde = mde_sign_consistent(res[tag].observed_sd, len(SEEDS))
    chk(f"{tag} 80%功效 MDE（占 P 均值 %）", 100 * mde / base, EXPECTED["mde_pct"][tag], 0.05, "%")

print()
print("=" * 92)
print("§5 护栏与形状复核（不经库，直接从同一批记录算，核 §9.11:831/834-835）")
print("=" * 92)
for a in ARMS:
    got = int((rs.raw(a, "bimodality_coefficient") > 0.555).sum())
    want = EXPECTED["bimodal"][a]
    if got != want:
        FAIL.append(f"bimodal {SHORT[a]}: 算得 {got}/18, 文档 {want}/18")
    print(f"  {'OK ' if got == want else '**差异**'} bimodality>0.555 {SHORT[a]:<9} "
          f"算得 {got:>2}/18   文档 {want:>2}/18")
for m, want in EXPECTED["friedman"].items():
    _, p = friedmanchisquare(*[rs.cell_means(a, m) for a in ARMS])
    chk(f"Friedman {m} p", float(p), want, 5e-3)
n_cells = len(ARMS) * len(SEEDS)
ok_carn = sum(1 for a in ARMS for s in SEEDS if rs.cell(a, s, "carnivore_frac")[0] >= 0.05)
ok_pop = sum(1 for a in ARMS for s in SEEDS if rs.cell(a, s, "min_pop")[0] >= 1)
print(f"  OK  护栏: carn_frac>=0.05 的格 {ok_carn}/{n_cells}，min_pop>=1 的格 {ok_pop}/{n_cells}"
      f"   (文档 :835 「30 个格全部」)")
assert ok_carn == n_cells == ok_pop

print()
print("=" * 92)
if FAIL:
    print(f"验收**未通过**，{len(FAIL)} 处与 §9.11 不符：")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("验收通过：exp_stats.py 复算 P4 的每一个数都与 docs/multispecies_program.md §9.11 一致。")
print("=" * 92)
