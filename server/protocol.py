"""Binary snapshot encoding for the dashboard websocket.

One message per frame, little-endian:

    header (72 bytes):
        magic             4s   b"UNDW"
        frame             u32
        n_agents          u32
        grid              u32
        world_size        f32
        mean_energy       f32
        plant_total       f32
        mean_age          f32
        mean_diet         f32
        carn_frac         f32
        mean_water        f32
        diet_std          f32
        carn_speed        f32
        herb_speed        f32
        mean_elevation    f32
        forest_frac       f32
        fruit_total       f32
        phase             f32
    agents  (n_agents * 20 bytes): x f32, y f32, diet f32, energy f32, id f32
    plant   (grid*grid bytes):     u8, plant energy scaled to [0,255]
    fruit   (grid*grid bytes):     u8, fruit scaled by fruit_max

v6 added `fruit_total` and the fruit plane. Both went on the *end* of their
section, which is why no existing client offset moved: the header grew 64 -> 68
with the new float last, and the fruit plane follows the plant plane. Inserting
either one further up would have silently shifted every read after it.

v7 appended `phase` (the day-night clock in [0,1), 0/1=midnight, 0.5=midday) as
the last header float (offset 68), header 68 -> 72. Same discipline: last field,
so every prior offset held. The dashboard uses it to dim the world at night and
show a clock, so the diel predation (default-on, docs/day_night.md) is visible.

Terrain is static for a whole run and travels in its own one-shot message
(`encode_terrain`, magic b"UNTR"), sent on connect and after a reset, rather than
being re-sent 30 times a second. The old sine-stream parameters are gone from the
per-frame header with it -- the client reads the water layer off the terrain
message now instead of recomputing a formula that had to be kept in sync.

`id` is the agent's fixed slot index -- stable across frames while it lives, so
the dashboard can follow / inspect a selected individual. Only living agents are
sent, so payload tracks the population. The stream geometry fields are constant
per run but sent every frame anyway -- cheap, and keeps the client from having to
duplicate `Config`'s stream constants.

`diet_std` / `carn_speed` / `herb_speed` are the evolution telemetry the dashboard
plots as time series: a high diet_std means the herbivore/carnivore split is still
cleanly bimodal, and carn_speed climbing toward herb_speed means carnivores are
evolving active pursuit rather than sitting still and ambushing. `mean_elevation`
and `forest_frac` say where the population has chosen to live.
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"UNDW"
TERRAIN_MAGIC = b"UNTR"
_HEADER = struct.Struct("<4sIII ffffffffffffff")
_TERRAIN_HEADER = struct.Struct("<4sIf")


def encode_terrain(height: np.ndarray, forest: np.ndarray, water_dist: np.ndarray,
                   grid: int, world_size: float, river_half_width: float) -> bytes:
    """One-shot static terrain: 12-byte header then three grid*grid u8 planes.

    Sent once per world rather than per frame -- terrain never changes, and at
    grid=128 this is ~49 KB that would otherwise be re-sent 30 times a second.

    Planes are normalised to [0,255] for transport:
        height  -- remapped from [min,max] to full range, so relief shading has
                   contrast whatever the configured ridge height
        forest  -- canopy density, already [0,1]
        water   -- proximity, 255 at the waterline falling off over one bite of
                   land, so the client can draw a soft shoreline
    """
    h = np.asarray(height, dtype=np.float32)
    lo, hi = float(h.min()), float(h.max())
    h_u8 = ((h - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    f_u8 = (np.clip(np.asarray(forest), 0.0, 1.0) * 255.0).astype(np.uint8)
    w = 1.0 - np.clip(np.asarray(water_dist) / max(river_half_width, 1e-6), 0.0, 1.0)
    w_u8 = (w * 255.0).astype(np.uint8)

    header = _TERRAIN_HEADER.pack(TERRAIN_MAGIC, int(grid), float(world_size))
    return header + h_u8.tobytes() + f_u8.tobytes() + w_u8.tobytes()


def encode(frame: int, alive: np.ndarray, pos: np.ndarray, diet: np.ndarray,
           energy: np.ndarray, plant: np.ndarray, fruit: np.ndarray, grid: int,
           world_size: float, plant_max: float, fruit_max: float,
           metrics: dict) -> bytes:
    idx = np.nonzero(alive)[0]
    n = int(idx.size)

    header = _HEADER.pack(
        MAGIC, int(frame) & 0xFFFFFFFF, n, int(grid),
        float(world_size),
        float(metrics.get("mean_energy", 0.0)),
        float(metrics.get("plant_total", 0.0)),
        float(metrics.get("mean_age", 0.0)),
        float(metrics.get("mean_diet", 0.0)),
        float(metrics.get("carnivore_frac", 0.0)),
        float(metrics.get("mean_water", 0.0)),
        float(metrics.get("diet_std", 0.0)),
        float(metrics.get("carn_speed", 0.0)),
        float(metrics.get("herb_speed", 0.0)),
        float(metrics.get("mean_elevation", 0.0)),
        float(metrics.get("forest_frac", 0.0)),
        float(metrics.get("fruit_total", 0.0)),
        float(metrics.get("phase", 0.0)),
    )

    agents = np.empty((n, 5), dtype="<f4")
    agents[:, 0] = pos[idx, 0]
    agents[:, 1] = pos[idx, 1]
    agents[:, 2] = diet[idx]
    agents[:, 3] = energy[idx]
    agents[:, 4] = idx                       # stable slot id

    plant_u8 = np.clip(plant / max(plant_max, 1e-6), 0.0, 1.0)
    plant_u8 = (plant_u8 * 255.0).astype(np.uint8)
    # Scaled by fruit_max, not by the local capacity: the client should see where
    # fruit is actually dense, not where each cell happens to be full relative to
    # its own small ceiling.
    fruit_u8 = np.clip(fruit / max(fruit_max, 1e-6), 0.0, 1.0)
    fruit_u8 = (fruit_u8 * 255.0).astype(np.uint8)

    return header + agents.tobytes() + plant_u8.tobytes() + fruit_u8.tobytes()
