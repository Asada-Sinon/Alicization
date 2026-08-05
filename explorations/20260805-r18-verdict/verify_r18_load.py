"""`analyze_r18.load()` 的回归验收：崩溃 run 的那条路从没被走过。

为什么写这个
------------
`analyze_r18.py` 在跑之前就提交了（commit `1dfb822`），而它的 `load()` 里有一行

    (bad if rec["collapsed"] else R).setdefault(0, None)

**`bad` 是 list，list 没有 `setdefault`** ⇒ 只要有任何一个 run `collapsed=True`，
整个判决脚本抛 `AttributeError`。而 §21.5 恰恰把「崩溃 run 分开报，不池化」写成护栏
——**这条护栏的实现本身是崩的**。R16 同臂实测 0/24 崩溃，所以它一次都没被触发；
R18 跑 3.5 倍长（350k 步），触发概率不是零。

本脚本按 `MEMORY.md` 的教训办：**先证明它抓得住那个 bug，再证明修好之后通过**。
一个在修复前也「通过」的验收，等于没有验收。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260805-r18-verdict/verify_r18_load.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "explorations/20260804-readouts")

import analyze_r18
import numpy as np

NB = 20


def synth_log(path, *, collapsed, npts=40, gen_end=450.0):
    """造一条和 trajectory.py 输出同构的日志：进度行 + 末尾一条 `JSON {...}`。"""
    gen = np.linspace(1.2, gen_end, npts)
    traj = []
    for g in gen:
        h = np.zeros(NB)
        h[3] = 300.0          # 低食性簇
        h[15] = 300.0         # 高食性簇
        traj.append({"hist": h.tolist(), "n_herb": float(h.sum()),
                     "generation": float(g), "t": int(g * 750)})
    d = {"traj": traj, "collapsed": collapsed}
    path.write_text("  t=  60500  split_score=0.4746  n_herb=5203\n"
                    + "JSON " + json.dumps(d) + "\n")


def build(tmp, n_ok, n_collapsed):
    """n_ok 个正常 run（占满 seed×rep），外加 n_collapsed 个崩溃 run。"""
    tmp.mkdir(parents=True, exist_ok=True)
    made = 0
    for s in range(12):
        for r in (1, 2):
            if made >= n_ok:
                break
            synth_log(tmp / f"R38n_s{s}_r{r}.log", collapsed=False)
            made += 1
    for i in range(n_collapsed):
        s = 11 - i // 2
        synth_log(tmp / f"R38n_s{s}_r{2 - i % 2}.log", collapsed=True, gen_end=60.0)
    return made


def old_load_line(collapsed):
    """修复前那一行的原样复现——用来证明本验收确实针对一个真 bug。"""
    bad, R = [], {}
    (bad if collapsed else R).setdefault(0, None)


def main():
    fails = []

    # ---- 0. 先证明这个验收抓得住修复前的 bug ----------------------------------
    old_load_line(False)                      # 正常 run：旧代码走 dict，通过
    try:
        old_load_line(True)
        fails.append("修复前的那一行竟然没抛异常 —— 本验收失去意义，先查它是不是被改过")
        print("  0. 旧代码在 collapsed=True 上 **没有** 抛异常 ⇒ 本验收无效")
    except AttributeError as e:
        print(f"  0. 旧代码在 collapsed=True 上如预期抛 {type(e).__name__}: {e}")
        print("     ⇒ 本验收针对的是一个真 bug，不是摆设")

    tmp = Path(tempfile.mkdtemp(prefix="r18load-")) / "runs"
    orig = analyze_r18.RUN_DIR
    try:
        # ---- 1. 全正常：24 run 全部装进 R，bad 为空 --------------------------
        n = build(tmp, n_ok=24, n_collapsed=0)
        analyze_r18.RUN_DIR = str(tmp)
        R, bad = analyze_r18.load()
        print(f"  1. 全正常 {n} run  →  载入 {len(R)}，崩溃 {len(bad)}")
        if len(R) != 24 or len(bad) != 0:
            fails.append(f"全正常应为 24/0，实得 {len(R)}/{len(bad)}")
        if sorted(R.keys())[:2] != [(0, 1), (0, 2)]:
            fails.append(f"键解析错了：{sorted(R.keys())[:3]}")

        # ---- 2. 含崩溃 run：不抛异常，且崩溃的不进 R ------------------------
        shutil.rmtree(tmp)
        build(tmp, n_ok=20, n_collapsed=4)
        try:
            R, bad = analyze_r18.load()
            print(f"  2. 20 正常 + 4 崩溃  →  载入 {len(R)}，崩溃 {len(bad)}"
                  f"：{[b['name'] for b in bad]}")
            if len(bad) != 4:
                fails.append(f"崩溃 run 应为 4，实得 {len(bad)}")
            if len(R) != 20:
                fails.append(f"正常 run 应为 20，实得 {len(R)}（崩溃的被池化进去了？）")
            if any(rec["collapsed"] for rec in R.values()):
                fails.append("崩溃 run 混进了 R —— §21.5 要求分开报，不池化")
        except Exception as e:
            fails.append(f"含崩溃 run 时 load() 抛了 {type(e).__name__}: {e}")
            print(f"  2. **load() 抛异常：{type(e).__name__}: {e}**")

        # ---- 3. 空目录：优雅退出，不抛 --------------------------------------
        shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            R, bad = analyze_r18.load()
            print(f"  3. 空目录  →  载入 {len(R)}，崩溃 {len(bad)}（不抛异常）")
            if R or bad:
                fails.append("空目录竟然载入了东西")
        except Exception as e:
            fails.append(f"空目录时 load() 抛了 {type(e).__name__}: {e}")

        # ---- 4. 全崩溃：R 为空，bad 满 --------------------------------------
        shutil.rmtree(tmp)
        build(tmp, n_ok=0, n_collapsed=6)
        try:
            R, bad = analyze_r18.load()
            print(f"  4. 全崩溃 6 run  →  载入 {len(R)}，崩溃 {len(bad)}")
            if len(R) != 0 or len(bad) != 6:
                fails.append(f"全崩溃应为 0/6，实得 {len(R)}/{len(bad)}")
        except Exception as e:
            fails.append(f"全崩溃时 load() 抛了 {type(e).__name__}: {e}")
    finally:
        analyze_r18.RUN_DIR = orig
        shutil.rmtree(tmp.parent, ignore_errors=True)

    print()
    if fails:
        print("**未通过**：")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("全部通过：崩溃 run 分开报（§21.5），空目录优雅退出。")


if __name__ == "__main__":
    main()
