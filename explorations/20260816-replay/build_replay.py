"""把已跑完的 run 导出成可回放的数据，供 `replay.html` 用。

**零 GPU**——只读 `outputs/` 里的 JSON 行，不重跑任何模拟。

导出什么
--------
每个 run 的 `traj` 里已有：每 ~250 步一帧的 `forage_pref` 20 箱直方图、世代数、
种群、`carnivore_frac`。降采样到约 400 帧就足够流畅回放，而 20 箱直方图正好
能让人**直接看到一个峰裂成两个**——那是这条纲领最核心的那件事。

并排哪两个
----------
- **左：无捕食**（`R38n`，`outputs/20260806-gen700`）——两簇分开并维持数百代
- **右：有捕食**（`wn1on`，`outputs/20260813-r20`）——果侧那一簇被压掉

这正是 R20 判决（`feasibility.md` §27）的内容，摆成并排就不用解释了。

跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
      explorations/20260816-replay/build_replay.py
"""
import glob
import json
import os

import numpy as np

TARGET_FRAMES = 400
OUT = "explorations/20260816-replay/replay_data.json"
PANELS = [
    {"key": "nopred", "title": "没有捕食者",
     "log": "outputs/20260806-gen700/R38n_s0_r1.log",
     "note": "两簇分开，并维持数百代"},
    {"key": "pred", "title": "有捕食者",
     "log": "outputs/20260813-r20/wn1on_s0_r1.log",
     "note": "吃果子那一簇被压掉"},
]


def load(path):
    txt = open(path).read()
    assert "JSON " in txt, f"{path} 没有 JSON 行（run 没跑完？）"
    return json.loads(txt.split("JSON ")[1].split("\n")[0])["traj"]


def downsample(tr, n):
    """按**世代**等距取帧——各 run 的世代钟不同，按帧号取会切到不同的生命阶段。"""
    g = np.array([q["generation"] for q in tr], float)
    want = np.linspace(g.min(), g.max(), n)
    idx = [int(np.argmin(np.abs(g - w))) for w in want]
    # 去重但保持顺序
    seen, keep = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i); keep.append(i)
    return [tr[i] for i in keep]


def main():
    out = {"bins": 20, "panels": []}
    for p in PANELS:
        tr = downsample(load(p["log"]), TARGET_FRAMES)
        frames = []
        for q in tr:
            h = np.asarray(q["hist"], float)
            tot = max(h.sum(), 1.0)
            lo = float(h[:7].sum() / tot)          # forage_pref < 0.35 = 果专精侧
            frames.append({
                "g": round(float(q["generation"]), 1),
                "h": [round(float(x / tot), 5) for x in h],   # 归一化，画的是占比
                "n": int(q["n_herb"]),
                "lo": round(lo, 4),
            })
        out["panels"].append({"key": p["key"], "title": p["title"], "note": p["note"],
                              "frames": frames,
                              "gmax": frames[-1]["g"], "src": os.path.basename(p["log"])})
        print(f"  {p['title']:<8} {len(frames):>4} 帧   世代 {frames[0]['g']:.1f} → {frames[-1]['g']:.1f}"
              f"   果侧占比 {frames[0]['lo']:.3f} → {frames[-1]['lo']:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\n✓ 写出 {OUT}（{os.path.getsize(OUT)/1024:.0f} KB）")
    # 两个面板的帧数可能不同（世代跨度不同），页面按各自的进度条比例对齐
    ns = [len(p["frames"]) for p in out["panels"]]
    print(f"  两个面板帧数 {ns} —— 页面按**世代比例**对齐，不是按帧号")


if __name__ == "__main__":
    main()
