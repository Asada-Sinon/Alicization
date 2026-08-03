"""把「当前世界」和「R9 提案世界」并排画出来，用眼睛看，不只是读数字。

本项目的纪律之一：**GPU / canvas 的输出要看，不能只推理**（`CLAUDE.md`，那个把植被场渲成
一整块饱和色板的 shader bug 就是只推理漏掉的）。地形也一样——「草层动态范围封顶 1.538×、
铺满整张地图」是一句话，但那句话长什么样，看一眼就懂了。

环境里没有 matplotlib，也不打算为一张图装依赖：PNG 用 `zlib` + `struct` 直接写
（真彩 8-bit RGB，每行一个 filter 0 字节），三十行，标准库。

产出 `outputs/20260803-shade/*.png`，每张是 2×3 的贴图：
  行 1 = 当前世界（sea=off, shade=0）
  行 2 = R9 提案（sea=ON, shade=1.3）
  列   = 草层承载 / 果层承载 / 水距

重跑：XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
        explorations/20260803-shade/render_worlds.py
"""
import struct
import sys
import zlib

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

from underworld import Config
from underworld import terrain as terrain_mod

OUT = "outputs/20260803-shade"
SCALE = 3            # pixels per cell -- 128*3 = 384 px per panel
GAP = 8


def write_png(path: str, rgb: np.ndarray) -> None:
    """Minimal truecolour PNG. `rgb` is uint8 [h, w, 3]."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def ramp(v: np.ndarray, lo: float, hi: float, stops) -> np.ndarray:
    """Piecewise-linear colour ramp over normalised `v`. `stops` = [(t, (r,g,b)), ...]."""
    t = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    ts = np.array([s[0] for s in stops])
    cs = np.array([s[1] for s in stops], dtype=np.float64)
    out = np.empty(t.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(t, ts, cs[:, c])
    return out.astype(np.uint8)


# ---- terrain variants (same arithmetic as terrain_cross.py) ----------------
cfg = Config()
T0 = terrain_mod.build(cfg)
centers = np.asarray(terrain_mod._cell_centers(cfg))
height, rock = np.asarray(T0.height), np.asarray(T0.rock)
wd_river = np.asarray(T0.water_dist)
is_sea = height < cfg.sea_level


def _wrap(d, size):
    return (d + size / 2.0) % size - size / 2.0


@jax.jit
def _dist_to(points, src):
    def body(i, best):
        blk = jax.lax.dynamic_slice(src, (i * 256, 0), (256, 2))
        dd = _wrap(points[:, None, :] - blk[None, :, :], cfg.world_size)
        return jnp.minimum(best, jnp.min(jnp.sqrt(jnp.sum(dd * dd, 2) + 1e-12), 1))
    return jax.lax.fori_loop(0, src.shape[0] // 256, body,
                             jnp.full((points.shape[0],), jnp.inf))


n_trim = (int(is_sea.sum()) // 256) * 256
d_sea = np.asarray(_dist_to(jnp.asarray(centers), jnp.asarray(centers[is_sea][:n_trim])))
wd_full = np.minimum(wd_river, np.where(is_sea, 0.0, d_sea))

px = np.sin(2.0 * np.pi * cfg.fruit_wavenumber_x * centers[:, 0] / cfg.world_size)
py = np.sin(2.0 * np.pi * cfg.fruit_wavenumber_y * centers[:, 1] / cfg.world_size)
patch = np.clip((px * py - cfg.fruit_patch_threshold) / (1.0 - cfg.fruit_patch_threshold),
                0.0, 1.0)


def variant(sea_fix: bool, shade: float, renorm=True):
    wd = wd_full if sea_fix else wd_river
    eb = np.exp(-((height - cfg.forest_elev) ** 2) / (2.0 * cfg.forest_elev_sigma ** 2))
    forest = np.where(is_sea, 0.0, eb * np.exp(-wd / cfg.forest_water_scale))
    fert = np.clip(cfg.grass_base + cfg.forest_bonus * forest - shade * forest, 0.0, None)
    grass = np.where(is_sea, 0.0, cfg.plant_max * fert * (1.0 - rock))
    fruit = np.where(is_sea, 0.0, cfg.fruit_max * patch * forest ** 2 * (1.0 - rock))
    if renorm and shade > 0:
        grass = grass * (variant(sea_fix, 0.0, False)[1].sum() / grass.sum())
    return wd, grass, fruit


g = cfg.grid


def tile(field: np.ndarray, kind: str) -> np.ndarray:
    img = field.reshape(g, g)
    if kind == "grass":
        rgbf = ramp(img, 0.0, 3.1, [(0.0, (26, 22, 18)), (0.25, (74, 62, 34)),
                                    (0.6, (108, 148, 52)), (1.0, (196, 230, 120))])
    elif kind == "fruit":
        rgbf = ramp(img, 0.0, 2.2, [(0.0, (18, 18, 22)), (0.15, (60, 34, 52)),
                                    (0.55, (176, 60, 92)), (1.0, (255, 176, 96))])
    else:  # water distance -- near water bright
        rgbf = ramp(img, 0.0, 110.0, [(0.0, (120, 208, 255)), (0.12, (36, 110, 172)),
                                      (0.5, (44, 46, 54)), (1.0, (140, 132, 120))])
    sea_mask = is_sea.reshape(g, g)
    rgbf[sea_mask] = np.array([14, 32, 58], dtype=np.uint8)
    return np.kron(rgbf, np.ones((SCALE, SCALE, 1), dtype=np.uint8))


rows = []
for sea_fix, shade in ((False, 0.0), (True, 1.3)):
    wd, grass, fruit = variant(sea_fix, shade)
    cols = [tile(grass, "grass"), tile(fruit, "fruit"), tile(wd, "water")]
    sep = np.full((cols[0].shape[0], GAP, 3), 20, dtype=np.uint8)
    rows.append(np.concatenate([cols[0], sep, cols[1], sep, cols[2]], axis=1))
sep_h = np.full((GAP, rows[0].shape[1], 3), 20, dtype=np.uint8)
sheet = np.concatenate([rows[0], sep_h, rows[1]], axis=0)
write_png(f"{OUT}/worlds.png", sheet)
print(f"{OUT}/worlds.png  {sheet.shape[1]}x{sheet.shape[0]}")
print("  行1 = 当前世界 (sea=off, shade=0)    行2 = R9 提案 (sea=ON, shade=1.3)")
print("  列  = 草层承载 | 果层承载 | 水距（亮蓝=近水）")

# a second sheet that answers the criterion directly: where do the two bowls sit?
for name, (sea_fix, shade) in (("bowls_base", (False, 0.0)), ("bowls_r9", (True, 1.3))):
    wd, grass, fruit = variant(sea_fix, shade)
    land = ~is_sea
    gn = grass / max(grass[land].max(), 1e-9)
    fn = fruit / max(fruit[land].max(), 1e-9)
    rgbf = np.zeros((g * g, 3))
    rgbf[:, 1] = 235 * np.clip(gn, 0, 1) ** 0.7      # grass -> green
    rgbf[:, 0] = 245 * np.clip(fn, 0, 1) ** 0.5      # fruit -> red
    rgbf[:, 2] = 60 * np.clip(fn, 0, 1) ** 0.5
    rgbf[is_sea] = np.array([14, 32, 58])
    img = np.kron(rgbf.reshape(g, g, 3).astype(np.uint8),
                  np.ones((SCALE, SCALE, 1), dtype=np.uint8))
    write_png(f"{OUT}/{name}.png", img)
    print(f"{OUT}/{name}.png   绿=草 红=果 黄=两者重叠")
