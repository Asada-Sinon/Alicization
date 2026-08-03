"""把饱和探针的 `forage_pref` 占据分布画出来。**十个回合里第一次出现真双峰，要看一眼。**

`bimodality_coefficient` 与 `sd` 都区分不了「两个分开的众数」和「一个变宽的单峰」——
`BC` 是没有零分布的描述量（`scripts/probe_trait_dist.py` 的 docstring 明写），
`sd` 对两者一视同仁。直方图能区分，而这正是本项目「GPU/canvas 输出要看，不能只推理」
那条纪律的同一件事（`CLAUDE.md`）。

环境里没有 matplotlib，PNG 用 `zlib` + `struct` 直接写（与
`explorations/20260803-shade/render_worlds.py` 同一个最小实现）。

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-curvature/render_bimodality.py
"""
import glob
import json
import struct
import sys
import zlib

import numpy as np

OUT = "outputs/20260803-curvature/bimodality.png"
W_BAR, H_PANEL, PAD, GAP = 26, 150, 34, 10
ARMS = [("S0", "eat_rate 1.5  (基线, 饱和)", (150, 150, 160)),
        ("S1", "eat_rate 0.5  (松开饱和, 不加食物)", (120, 220, 140)),
        ("S2", "regrow_baseline 0.06  (加食物)", (220, 160, 110))]


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    open(path, "wb").write(png)


rec = {}
for f in sorted(glob.glob("outputs/20260803-curvature/sat/*.log")):
    for ln in open(f):
        if ln.startswith("JSON "):
            b = f.split("/")[-1][:-4].split("_")
            rec[(b[0], int(b[1][1:]), int(b[2][1:]))] = json.loads(ln[5:])

ctr = np.array(rec[("S0", 0, 1)]["bin_centers"])
nb = len(ctr)
seeds = [0, 1, 2, 3]
panel_w = nb * W_BAR
img_w = PAD + len(seeds) * (panel_w + GAP) + PAD
img_h = PAD + len(ARMS) * (H_PANEL + GAP + 18) + PAD
img = np.full((img_h, img_w, 3), 18, dtype=np.uint8)

# 中间带 [0.40, 0.60] —— 分裂时这一段应当是空的
mid = (ctr >= 0.40) & (ctr <= 0.60)

for ai, (arm, _label, colour) in enumerate(ARMS):
    y0 = PAD + ai * (H_PANEL + GAP + 18)
    for si, s in enumerate(seeds):
        x0 = PAD + si * (panel_w + GAP)
        n = np.array(rec[(arm, s, 1)]["bin_n"], float)
        n = n / max(n.sum(), 1)
        hmax = max(n.max(), 1e-9)
        # 中间带底色：一眼看出「谷」在哪
        img[y0:y0 + H_PANEL, x0 + int(mid.argmax()) * W_BAR:
            x0 + (int(mid.argmax()) + int(mid.sum())) * W_BAR] = (34, 30, 44)
        for i, v in enumerate(n):
            h = int(round(v / hmax * (H_PANEL - 4)))
            if h <= 0:
                continue
            img[y0 + H_PANEL - h: y0 + H_PANEL,
                x0 + i * W_BAR + 2: x0 + (i + 1) * W_BAR - 2] = colour
        img[y0 + H_PANEL: y0 + H_PANEL + 2, x0: x0 + panel_w] = (90, 90, 100)

write_png(OUT, img)
print(f"{OUT}  {img_w}x{img_h}")
print("  行 = 三个臂（上→下：S0 基线 / S1 松开饱和 / S2 加食物），列 = 4 个种子")
print("  横轴 = forage_pref 0.17→0.82（高 = 草专精），深色竖带 = 中间带 [0.40,0.60]")
print("  **S1 那一行：两侧有柱、深色带里空** —— 那就是双峰。S0/S2 的质量都堆在深色带里或紧邻。")
for arm, label, _ in ARMS:
    fm = [float(np.array(rec[(arm, s, 1)]["bin_n"], float)[mid].sum()
                / max(np.array(rec[(arm, s, 1)]["bin_n"], float).sum(), 1)) for s in seeds]
    print(f"  {arm}  中间带占比 {[round(x, 4) for x in fm]}   {label}")
