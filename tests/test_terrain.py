"""Terrain invariants: the elevation field, the traced rivers, and whether the
world is actually habitable. Run with: .venv/bin/python -m pytest
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import jax.numpy as jnp
import numpy as np
import pytest

from underworld import Config
from underworld import terrain as T


@pytest.fixture(scope="module")
def built():
    cfg = Config()
    return cfg, T.build(cfg)


def test_height_finite_and_bounded(built):
    cfg, tr = built
    h = np.asarray(tr.height)
    assert np.all(np.isfinite(h))
    # ridge term tops out at ridge_height, basin term bottoms at -basin_depth
    assert h.max() <= cfg.ridge_height + 1e-4
    assert h.min() >= -cfg.basin_depth - 1e-4


def test_derived_fields_in_range(built):
    cfg, tr = built
    for name in ("forest", "rock"):
        a = np.asarray(getattr(tr, name))
        assert np.all(np.isfinite(a))
        assert a.min() >= 0.0 and a.max() <= 1.0, name
    cap = np.asarray(tr.capacity)
    assert cap.min() >= 0.0 and cap.max() <= cfg.plant_max + 1e-4
    assert np.asarray(tr.water_dist).min() >= 0.0
    fcap = np.asarray(tr.fruit_capacity)
    assert np.all(np.isfinite(fcap))
    assert fcap.min() >= 0.0 and fcap.max() <= cfg.fruit_max + 1e-4


def test_rivers_never_flow_uphill(built):
    """The load-bearing property: a river must descend at every recorded step.

    Regression guard -- before rivers were frozen on reaching the sea, they spent
    their remaining step budget wandering the flat sea bottom where the gradient
    is numerical noise, which registered as flowing uphill.
    """
    cfg, tr = built
    rivers = np.asarray(tr.rivers)
    assert rivers.shape == (cfg.n_rivers, cfg.river_steps, 2)
    for i in range(cfg.n_rivers):
        h = np.asarray(T.height_at(jnp.asarray(rivers[i]), cfg))
        rises = np.sum(np.diff(h) > 1e-6)
        assert rises == 0, f"river {i} flows uphill on {rises} steps"


def test_rivers_start_high_and_reach_the_sea(built):
    cfg, tr = built
    rivers = np.asarray(tr.rivers)
    for i in range(cfg.n_rivers):
        h = np.asarray(T.height_at(jnp.asarray(rivers[i]), cfg))
        assert h[0] > 0.3, f"river {i} does not start in the mountains"
        assert h[-1] <= cfg.sea_level + 1e-3, f"river {i} never reaches the sea"


def test_nothing_grows_on_rock_or_open_water(built):
    """Open water grows exactly nothing; bare rock grows too little to live on.

    Rock uses a smoothstep, so the high peaks taper to near-zero capacity rather
    than hitting a hard boundary -- there is no sharp altitude where rock starts.
    The meaningful assertion is therefore economic, not an epsilon: a cell whose
    entire standing crop is worth less than one step of `base_cost` cannot
    sustain an agent even if it grazes the cell bare.
    """
    cfg, tr = built
    h = np.asarray(tr.height)
    cap = np.asarray(tr.capacity)
    assert np.all(cap[h < cfg.sea_level] == 0.0)
    peaks = np.asarray(tr.rock) > 0.99
    assert peaks.any(), "no bare rock anywhere -- the range is not tall enough"
    assert np.all(cap[peaks] < cfg.base_cost)
    fcap = np.asarray(tr.fruit_capacity)
    assert np.all(fcap[h < cfg.sea_level] == 0.0)
    # Rock tapers by smoothstep like the grass capacity above, so the peaks hold
    # a residue rather than an exact zero. Same economic bar: worth less than one
    # step of existing.
    assert np.all(fcap[peaks] * cfg.fruit_energy < cfg.base_cost)


def test_fruit_is_patchy_and_wooded(built):
    """Fruit has to be scarce and tied to real canopy, or it is just more grass.

    The point of the resource is that it is worth travelling to and worth
    remembering. A fruit layer smeared over most of the map would be neither.
    """
    cfg, tr = built
    fcap = np.asarray(tr.fruit_capacity)
    forest = np.asarray(tr.forest)

    present = fcap > 0.0
    assert present.any(), "no fruit anywhere"
    assert present.mean() < 0.25, (
        f"fruit covers {100*present.mean():.1f}% of the map -- not patchy")

    # Assert on capacity, not presence. Where fruit *can* occur is set by the
    # sine lattice, which knows nothing about trees; it is the `forest ** 2`
    # factor on the magnitude that ties the resource to canopy. Testing presence
    # measures the lattice and would pass for a fruit layer scattered over open
    # grassland.
    land = np.asarray(tr.height) >= cfg.sea_level
    weighted_forest = float((fcap * forest).sum() / fcap.sum())
    assert weighted_forest > 1.8 * float(forest[land].mean()), (
        f"capacity-weighted canopy {weighted_forest:.3f} vs land mean "
        f"{forest[land].mean():.3f} -- fruit is not a forest resource")


def test_world_is_habitable(built):
    """Water has to be reachable from the land worth living on.

    A full tank at full thrust covers a hard-limited distance; land beyond that
    is a death trap rather than a frontier. Asserted as a fraction of *liveable*
    land (capacity worth grazing) rather than a max over every cell, since a
    single arid pocket on the far plains is realistic, not a bug.
    """
    cfg, tr = built
    cap = np.asarray(tr.capacity)
    wd = np.asarray(tr.water_dist)
    tank = (cfg.water_max / (cfg.base_water_cost + cfg.move_water_cost)
            * cfg.max_speed * cfg.dt)

    liveable = cap > 0.2 * cfg.plant_max
    assert liveable.mean() > 0.4, "less than 40% of the map can feed anything"

    beyond_oneway = (wd[liveable] > tank * 0.75).mean()
    beyond_roundtrip = (wd[liveable] > tank * 0.5).mean()
    assert beyond_oneway == 0.0, (
        f"{100*beyond_oneway:.1f}% of liveable land is beyond one-way water range")
    assert beyond_roundtrip < 0.10, (
        f"{100*beyond_roundtrip:.1f}% of liveable land is beyond round-trip range")


def test_geography_scales_with_world_size():
    """World-scale terrain lengths are fractions, so halving the map halves the
    range rather than leaving a mountain that no longer fits on it."""
    big, small = Config(), Config(world_size=256.0, grid=64, sense_grid=12)
    assert small.ridge_sigma == pytest.approx(big.ridge_sigma / 2)
    assert small.ridge_base_y == pytest.approx(big.ridge_base_y / 2)
    tr = T.build(small)
    h = np.asarray(tr.height)
    assert np.all(np.isfinite(h)) and h.max() > 0.5


def test_sense_cell_covers_vision_radius():
    """If a sense cell is smaller than the vision radius, agents beyond the 3x3
    block are invisible; if cells get too big, k_neighbors overflows and drops
    agents silently. Either way the neighbour index stops being trustworthy."""
    cfg = Config()
    assert cfg.world_size / cfg.sense_grid >= cfg.vision_radius
