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
        stream_amplitude  f32
        stream_wavenumber f32
        stream_base_y     f32
        stream_half_width f32
    agents  (n_agents * 20 bytes): x f32, y f32, diet f32, energy f32, id f32
    plant   (grid*grid bytes):     u8, plant energy scaled to [0,255]

`id` is the agent's fixed slot index -- stable across frames while it lives, so
the dashboard can follow / inspect a selected individual. Only living agents are
sent, so payload tracks the population. The stream geometry fields are constant
per run but sent every frame anyway -- cheap, and keeps the client from having to
duplicate `Config`'s stream constants.

`diet_std` / `carn_speed` / `herb_speed` are the evolution telemetry the dashboard
plots as time series: a high diet_std means the herbivore/carnivore split is still
cleanly bimodal, and carn_speed climbing toward herb_speed means carnivores are
evolving active pursuit rather than sitting still and ambushing.
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"UNDW"
_HEADER = struct.Struct("<4sIII ffffffffffffff")


def encode(frame: int, alive: np.ndarray, pos: np.ndarray, diet: np.ndarray,
           energy: np.ndarray, plant: np.ndarray, grid: int, world_size: float,
           plant_max: float, metrics: dict, stream_amplitude: float,
           stream_wavenumber: float, stream_base_y: float,
           stream_half_width: float) -> bytes:
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
        float(stream_amplitude), float(stream_wavenumber),
        float(stream_base_y), float(stream_half_width),
    )

    agents = np.empty((n, 5), dtype="<f4")
    agents[:, 0] = pos[idx, 0]
    agents[:, 1] = pos[idx, 1]
    agents[:, 2] = diet[idx]
    agents[:, 3] = energy[idx]
    agents[:, 4] = idx                       # stable slot id

    plant_u8 = np.clip(plant / max(plant_max, 1e-6), 0.0, 1.0)
    plant_u8 = (plant_u8 * 255.0).astype(np.uint8)

    return header + agents.tobytes() + plant_u8.tobytes()
