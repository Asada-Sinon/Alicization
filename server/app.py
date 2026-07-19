"""FastAPI backend: runs the Underworld sim in a background loop and streams
compact binary snapshots to any connected dashboard over a websocket.

The sim (GPU) and the viewers are decoupled: stepping happens in an executor
thread so websocket I/O never blocks on JAX, and slow clients simply receive the
latest frame (dropped intermediates = backpressure for free).

Run:  .venv/bin/python scripts/run_live.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager

sys.path.insert(0, ".")

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server import protocol
from underworld import Config, new_world


class Simulation:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state, self.key, self.step_fn, self.scan_fn = new_world(cfg)
        self.playing = True
        self.speed = 4            # sim steps per rendered frame (the FLA dial)
        self.frame = 0
        self.total_steps = 0
        self.metrics: dict = {}
        self.snapshot: bytes = b""
        self.selected: int | None = None   # slot id being inspected
        self.detail: str | None = None      # JSON detail for the selected agent
        self._reset = False

    def _advance(self):
        """Runs in an executor thread (JAX releases the GIL during device work)."""
        self.state, self.key, ms = self.scan_fn(self.state, self.key, self.speed)
        self.metrics = {k: float(np.asarray(v)[-1]) for k, v in ms._asdict().items()}
        self.total_steps += self.speed

    def _build_snapshot(self):
        s = self.state
        self.frame += 1
        self.snapshot = protocol.encode(
            self.frame,
            np.asarray(s.alive), np.asarray(s.pos), np.asarray(s.diet),
            np.asarray(s.energy), np.asarray(s.plant),
            self.cfg.grid, self.cfg.world_size, self.cfg.plant_max, self.metrics,
            self.cfg.stream_amplitude, self.cfg.stream_wavenumber,
            self.cfg.stream_base_y, self.cfg.stream_half_width,
        )
        self.detail = self._build_detail() if self.selected is not None else None

    def _build_detail(self) -> str:
        """Full inspector readout for the selected agent (one slot -> a few
        scalar device->host reads per frame)."""
        i = self.selected
        s = self.state
        if i is None or not (0 <= i < self.cfg.n_max) or not bool(s.alive[i]):
            return json.dumps({"id": i, "alive": False})
        vx, vy = float(s.vel[i, 0]), float(s.vel[i, 1])
        # 摇光 view: the agent's retina input, hidden neurons, and outputs.
        r = self.cfg.retina_sectors
        li = np.asarray(s.last_input[i])
        hid = np.asarray(s.hidden[i])
        out = np.asarray(s.last_output[i])
        return json.dumps({
            "id": int(i),
            "alive": True,
            "diet": float(s.diet[i]),
            "energy": float(s.energy[i]),
            "water": float(s.water[i]),
            "age": float(s.age[i]),
            "generation": float(s.generation[i]),
            "x": float(s.pos[i, 0]),
            "y": float(s.pos[i, 1]),
            "heading": float(s.heading[i]),
            "speed": float((vx * vx + vy * vy) ** 0.5),
            "food": float(s.last_food[i]),
            "meat": float(s.last_meat[i]),
            "damage": float(s.last_damage[i]),
            "drink": float(s.last_drink[i]),
            "sectors": r,
            "retina_food": li[0:r].round(3).tolist(),
            "retina_prey": li[r:2 * r].round(3).tolist(),
            "retina_pred": li[2 * r:3 * r].round(3).tolist(),
            "retina_water": li[3 * r:4 * r].round(3).tolist(),
            "hidden": hid.round(3).tolist(),
            "turn": float(out[0]),
            "thrust": float(out[1]),
        })

    async def loop(self):
        loop = asyncio.get_event_loop()
        self._build_snapshot()
        target_dt = 1.0 / 30.0
        while True:
            t0 = time.time()
            if self._reset:
                self.state, self.key, self.step_fn, self.scan_fn = new_world(self.cfg)
                self.total_steps = 0
                self._reset = False
                self._build_snapshot()
            if self.playing:
                await loop.run_in_executor(None, self._advance)
                self._build_snapshot()
            await asyncio.sleep(max(0.0, target_dt - (time.time() - t0)))

    def apply(self, msg: dict):
        t = msg.get("type")
        if t == "play":
            self.playing = True
        elif t == "pause":
            self.playing = False
        elif t == "speed":
            self.speed = int(max(1, min(2000, msg.get("value", 4))))
        elif t == "reset":
            self._reset = True
            self.selected = None
        elif t == "select":
            sid = int(msg.get("id", -1))
            self.selected = sid if sid >= 0 else None


sim = Simulation(Config())


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(sim.loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    async def receiver():
        try:
            while True:
                sim.apply(await websocket.receive_json())
        except Exception:
            pass

    recv_task = asyncio.create_task(receiver())
    last_frame = -1
    try:
        while True:
            if sim.frame != last_frame and sim.snapshot:
                last_frame = sim.frame
                await websocket.send_bytes(sim.snapshot)
                if sim.detail is not None:
                    await websocket.send_text(sim.detail)
            await asyncio.sleep(1.0 / 30.0)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        recv_task.cancel()


# Serve the dashboard (index.html + js) at the root.
app.mount("/", StaticFiles(directory="web", html=True), name="web")
