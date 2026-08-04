"""[事后分组，非预注册] H1 的零假设检验分两组重跑：R38p 里**丢失捕食者的 4 个 run**
与**捕食者存活的 20 个 run**。

为什么要跑：§19.5 要求丢失捕食者的 run「必须单列，不做事后筛」。读数表照做了 min 值，
但 H1 的计数 7/24 里有 4 个正是丢失组 —— 那 4 个 run 在处理意义上已经变成了 `R38n`。
**这不改 H1 的预注册判决（那是 24/24 与 7/24）**，只是把 7 拆开看它由什么组成。

口径与 `analyze_r16.py` 的 H1 完全一致（同一个 `sim_run` / `occ_slice(0.75,1.0)` /
`(1+k)/(reps+1)` / reps=200 / 同一颗种子 20260805），只换 run 集合。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r16-verdict/null_subgroup.py
"""
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "explorations/20260804-readouts")

import numpy as np
from analyze_r16 import BINS, load, occ_slice
from neutral_null import DEFAULT_NS, OCC_THRESH, genes_from_hist, sim_run

REPS_NULL = 200


def run_null(runs, label):
    occ4 = lambda d: occ_slice(d["hist"], d["gen"], 0.75, 1.0)
    T = sum(1 for r in runs if occ4(r) > OCC_THRESH)
    print(f"  {label}: 实测 **{T}/{len(runs)}**  "
          f"（占空比 = {np.round([occ4(r) for r in runs], 3).tolist()}）")
    rng = np.random.default_rng(20260805)
    for N in DEFAULT_NS:
        cnt = []
        for _ in range(REPS_NULL):
            c = 0
            for r in runs:
                g0 = genes_from_hist(r["hist"][0], BINS, rng)
                span = int(max(r["gen"][-1] - r["gen"][0], 5))
                hs = sim_run(g0, span, len(r["hist"]), N, BINS, rng)
                if occ_slice(hs, r["gen"], 0.75, 1.0) > OCC_THRESH:
                    c += 1
            cnt.append(c)
        cnt = np.array(cnt)
        k = int((cnt >= T).sum())
        p = (1.0 + k) / (REPS_NULL + 1.0)
        print(f"    N={N:<5} 零均值 {cnt.mean():>5.2f}  5–95% [{np.percentile(cnt,5):.0f},"
              f"{np.percentile(cnt,95):.0f}]  最大 {cnt.max():>3.0f}  "
              f"p={'<' if k == 0 else '='}{p:.4f}{'   ** 不拒绝 **' if p >= 0.05 else ''}")


def main():
    R, _ = load()
    lost, kept = [], []
    for k in sorted(R):
        if k[0] != "R38p":
            continue
        (lost if float(np.nanmin(R[k]["carn"])) <= 0.0 else kept).append(R[k])
    print("H1 事后分组（**不是预注册判决**，§19.5 只准分开报、不准筛）")
    print(f"  丢失捕食者 {len(lost)} run: {[r['name'] for r in lost]}")
    run_null(kept, "R38p 捕食者存活组")
    run_null(lost, "R38p 丢失捕食者组")


if __name__ == "__main__":
    main()
