"""Static configuration for an Underworld run.

All values here are treated as compile-time constants: a `Config` is closed over
by `build_step`, so JAX bakes these numbers into the jitted world-step. Anything
that changes an array *shape* (n_max, grid size, brain dims) must live here.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    # --- world ---
    world_size: float = 512.0      # square torus edge length (world units)
    grid: int = 128                # plant field is grid x grid cells
    dt: float = 0.2                # seconds per step

    # --- spatial neighbour index (M1) ---
    sense_grid: int = 24           # binning grid for neighbour queries. MUST scale
    #                                with world_size: the sense cell must stay >=
    #                                vision_radius, or cells hold more agents than
    #                                k_neighbors and the overflow is silently
    #                                dropped -- invisible to both vision and predation.
    k_neighbors: int = 24          # max agents stored per sense-cell
    vision_radius: float = 21.0    # perception radius (<= sense cell size)
    attack_range: float = 6.0      # carnivore bite range (short = prey refuge)
    diet_delta: float = 0.15       # prey must be this much more herbivorous

    # --- population (fixed-capacity arrays) ---
    n_max: int = 16384             # hard cap on simultaneous agents
    n_init: int = 2000             # initial living agents. There is a sweet spot
    #                                 here and it is narrow in BOTH directions.
    #                                 Seed too close to the ~700 equilibrium and the
    #                                 carnivore founder pool is too small to survive
    #                                 its own stochasticity (an earlier build died
    #                                 out by step 800 at n_init=280). Seed far above
    #                                 it and the die-off itself does the killing:
    #                                 at n_init=4800 the 7x crash wiped carnivores
    #                                 on every seed tested. 2000 against ~700 is the
    #                                 gentle-crash regime.

    # --- brain (fixed-topology recurrent net; weights live in the genome) ---
    # inputs: per retina sector [food, prey, predator, water, slope]
    #       + [own_energy, own_diet, own_water]
    #       + per memory slot [sin bearing, cos bearing, distance, confidence].
    # See the `in_dim` property for the authoritative count.
    retina_sectors: int = 8        # directional vision resolution
    hidden: int = 16               # recurrent hidden units (the fluctlight's memory)
    out_dim: int = 2               # [turn, thrust]
    trait_dim: int = 1             # non-brain genes; [0] = diet
    genome_init_scale: float = 0.4
    food_sample_dist: float = 9.0  # how far ahead each sector samples the plant field

    # --- movement / metabolism ---
    max_speed: float = 8.0         # world units / second
    max_turn: float = 3.0          # radians / second
    base_cost: float = 0.02        # energy / step just to exist
    move_cost: float = 0.05        # extra energy / step at full thrust
    carn_cost: float = 0.10        # extra upkeep scaled by diet: idle predators
    #                                die back fast, coupling their numbers to prey.
    #                                Was 0.15 in the flat 256^2 world; the terrain
    #                                world charges predators for climbing and slows
    #                                them in cover, and at 0.15 they went extinct on
    #                                every seed tested. 0.10 restores a ~15% carnivore
    #                                fraction (measured over 3 seeds).

    # --- feeding: herbivory (plant field) ---
    eat_rate: float = 1.5          # max plant energy an agent drains / step
    eat_efficiency: float = 1.0
    energy_scale: float = 10.0     # normalizer for the energy sensor input
    carn_graze_cutoff: float = 0.75  # above this diet, grazing is hard-zeroed:
    #                                  true carnivores cannot subsist on plants at
    #                                  all. Sits past where the steep (1-diet)^6
    #                                  taper has already decayed to near-nothing,
    #                                  so there's no real energetic cliff there.

    # --- feeding: predation (carnivory) ---
    pred_rate: float = 0.5         # energy a full carnivore bites from its target / step
    pred_efficiency: float = 0.7   # trophic loss: predator gains < prey loses
    meat_water_frac: float = 0.3   # a bite also draws this fraction of the prey's
    #                                 *water* (as energy damage), transferred to
    #                                 the attacker at the same pred_efficiency --
    #                                 a kill hydrates as well as feeds.

    # --- terrain: one elevation field drives mountains, rivers and forest ---
    # h(x,y) = H_local(x) * exp(-d_ridge^2 / 2*sigma^2)          <- the range
    #          - basin_depth * ((1 - cos(pi*d/half_L)) / 2)^p    <- regional drainage
    # The ridge centerline reuses the sine-in-y-of-x form the old stream used, so
    # an integer wavenumber keeps it seamless across the torus seam.
    # World-scale lengths are stored as *fractions of world_size* so the whole
    # geography scales with the map: change world_size alone and the range stays
    # proportionate instead of silently degenerating. (Agent-scale lengths --
    # vision, river width, step length -- stay absolute, since those are set by
    # the creatures, not the map.)
    ridge_base_frac: float = 0.5      # centerline mean y, as a fraction of world
    ridge_amp_frac: float = 0.117     # meander amplitude, fraction of world
    ridge_wavenumber: int = 1         # whole periods across world_size (integer!)
    ridge_height: float = 1.0         # peak elevation
    ridge_sigma_frac: float = 0.066   # gaussian half-width, fraction of world
    ridge_peak_wavenumber: int = 5    # peaks along the range (integer!)
    ridge_peak_depth: float = 0.45    # how far passes drop below peak height.
    #                                   These passes are the cheap crossings --
    #                                   migration routes should emerge at them.
    basin_depth: float = 0.25         # elevation drop from ridge to the antipodal sea
    basin_power: float = 3.0          # >1 keeps the plains flat and the sea narrow
    sea_level: float = -0.20          # cells below this are open water (drinkable)

    # --- rivers: traced once at init by steepest descent from the ridge ---
    n_rivers: int = 6                 # NOT an aesthetic choice -- a full tank at
    #                                   full thrust only covers ~229 world units,
    #                                   so a single river would leave lethal dead
    #                                   zones at 512^2. Six sources spread along
    #                                   the range keep every cell within reach.
    river_steps: int = 400            # traced points per river (static shape)
    river_step_len: float = 2.0       # world units per descent step
    river_half_width: float = 8.0     # half-width of the drinkable band

    # --- forest: mid-elevation, near water ---
    forest_elev: float = 0.10         # elevation the canopy peaks at
    forest_elev_sigma: float = 0.18   # elevation band half-width
    forest_water_frac: float = 0.117  # canopy decay length away from water,
    #                                   as a fraction of world_size
    grass_base: float = 0.65          # open-ground fertility (fraction of plant_max)
    forest_bonus: float = 0.35        # extra fertility under full canopy
    forest_slow: float = 0.25         # speed penalty at full canopy
    forest_occlusion: float = 0.35    # vision-radius penalty at full canopy. Note
    #                                   attack_range is deliberately NOT reduced:
    #                                   short sight + unchanged reach is what makes
    #                                   forest genuine ambush cover.

    # --- bare rock: nothing grows on the peaks ---
    rock_h0: float = 0.45             # elevation where fertility starts to fail
    rock_h1: float = 0.75             # elevation where it reaches bare rock

    # --- movement over terrain ---
    climb_cost: float = 3.0           # energy per unit of elevation gained. Uphill
    #                                   only: downhill is free but never generates
    #                                   energy, which would be an exploit.

    # --- water (a separate resource from food) ---
    water_init: float = 8.0
    water_max: float = 10.0           # hydration cap
    water_scale: float = 10.0         # normalizer for the own-water sensor input
    base_water_cost: float = 0.02     # water / step just to exist
    move_water_cost: float = 0.05     # extra water / step at full thrust (panting)
    drink_rate: float = 2.0           # water gained / step while standing in the stream
    forage_water_frac: float = 0.10   # water drawn per unit of energy grazed.
    #                                   Measured: a herbivore at equilibrium takes
    #                                   mean(last_food)=0.144/step and pays
    #                                   0.0524/step for water at its realized thrust
    #                                   of 0.65, so 0.10 is a ~27% subsidy -- it
    #                                   stretches one-way range ~198 -> ~272 world
    #                                   units (round trip 99 -> 136, comfortably
    #                                   past the 35.5 median distance-to-water)
    #                                   without making water a non-issue. Note the
    #                                   strong invariant "grazing alone can never
    #                                   sustain an agent" is NOT achievable at any
    #                                   useful value: a forager crossing virgin
    #                                   ground strips ~0.37 energy/step and goes
    #                                   net-positive above frac~0.14. That is
    #                                   ecologically correct -- inland self-
    #                                   sufficiency is a low-density privilege that
    #                                   disappears as the interior fills and the
    #                                   field is drawn down. The invariant that
    #                                   does hold is stated against the
    #                                   *equilibrium* field; see
    #                                   test_forage_water_cannot_replace_drinking.

    # --- life cycle ---
    energy_init: float = 8.0
    repro_threshold: float = 16.0  # reproduce above this energy
    repro_cost_frac: float = 0.5   # fraction of energy handed to the child
    max_age: float = 3000.0        # steps before old age
    spawn_radius: float = 3.0      # child placed within this radius of parent

    # --- genetics ---
    mutation_sigma: float = 0.05      # gaussian noise on brain genes per birth
    diet_mutation_sigma: float = 0.015  # diet is strongly heritable (keeps types
    #                                     distinct instead of blurring to omnivore)
    hue_drift: float = 0.02           # lineage colour drift per birth
    carnivore_init_frac: float = 0.05  # fraction of founders seeded as carnivores

    # --- plants (logistic regrowth toward carrying capacity) ---
    plant_max: float = 2.2         # carrying capacity per cell: kept moderate on
    #                                purpose -- a dense plant field sustains a dense
    #                                population, and once mean neighbour spacing
    #                                drops well below attack_range, ambush (sit
    #                                still, bites arrive on their own) beats pursuit
    #                                on energy return. This value keeps spacing
    #                                close to attack_range so carnivores actually
    #                                have to move to eat (verified: carnivore mean
    #                                speed climbs toward herbivore speed over a 20k
    #                                step run, instead of decaying to ~0 as it did
    #                                at plant_max=5.0). Going much lower than this
    #                                (tried plant_max=1.2) starves carnivores faster
    #                                than they can find prey and drives them extinct
    #                                over the long run -- this is the lowest tested
    #                                value that still holds a stable carn_frac.
    plant_init: float = 1.5        # mean initial plant energy per cell
    regrow_rate: float = 0.06      # logistic growth rate
    regrow_baseline: float = 0.015 # spontaneous regrowth: a food floor that sets
    #                                the sustainable population & prevents
    #                                total-starvation extinction

    # --- fruit: the forest's high-value, patchy resource ---
    # Forests hold far more total biomass than grassland but far *less* that a
    # grazer can eat -- woody, tough, defended, with the canopy shading out the
    # herb layer. That is why savanna carries the large-herbivore biomass and
    # rainforest does not. What forest does offer is fruit: concentrated,
    # accessible, and patchy. So canopy is not simply "more food" here; it is a
    # low grazing floor with a high-value exception scattered through it.
    fruit_max: float = 4.0            # capacity per cell, ~1.8x plant_max: worth a
    #                                   detour and worth remembering, without
    #                                   displacing the plant field as the floor
    #                                   that sets sustainable population
    fruit_energy: float = 2.0         # energy per unit fruit -- a full fruit cell
    #                                   is worth ~3.6x a full grass cell
    fruit_eat_rate: float = 1.0       # below eat_rate=1.5: a patch takes a while to
    #                                   strip, so it supports a visit, not one bite
    fruit_regrow_rate: float = 0.008  # ~7x slower than regrow_rate. The slowness is
    #                                   the point: a patch that refilled quickly
    #                                   would give memory no edge over just
    #                                   searching, and remembering it is what this
    #                                   resource exists to reward
    fruit_regrow_baseline: float = 0.001  # small but nonzero -- at exactly zero a
    #                                   stripped cell is a logistic fixed point and
    #                                   stays dead forever
    fruit_wavenumber_x: int = 7       # coprime with _y so the patch lattice never
    fruit_wavenumber_y: int = 11      #   repeats within the torus; long beat period
    #                                   keeps the patches looking irregular
    fruit_patch_threshold: float = 0.55  # keeps roughly the top ~20% of the sine
    #                                   product; with forest**2 on top this lands
    #                                   fruit on a small percentage of the map

    # --- long-term spatial memory (see memory.py) ---
    memory_water_slots: int = 2       # one slot is fragile: a single stale entry
    #                                   blinds the agent. Two lets a lineage hold
    #                                   a home river and an outbound waypoint
    memory_fruit_slots: int = 2       # symmetric, for the patchy fruit layer
    memory_dist_scale: float = 60.0   # tanh normaliser for remembered distance.
    #                                   Median distance-to-water (35.5) lands
    #                                   mid-range; the post-forage-water round
    #                                   trip (~136) saturates it, which is where
    #                                   the journey stops being survivable anyway
    memory_decay: float = 0.998       # per step -> half-life 346 steps. A round
    #                                   trip to median water is ~190 steps, so a
    #                                   slot survives roughly two trips: long
    #                                   enough to be worth having, short enough
    #                                   that a stale one doesn't strand descendants
    memory_drift: float = 0.25        # dead-reckoning noise as a fraction of each
    #                                   step's displacement. Weaker than it looks:
    #                                   random-walk error grows as sqrt(n), so a
    #                                   200-step trip accumulates only ~3.7 units.
    #                                   memory_decay does the forgetting; do not
    #                                   crank this up to compensate

    # --- rng ---
    seed: int = 0

    # ----- derived (not fields) -----
    @property
    def n_cells(self) -> int:
        return self.grid * self.grid

    @property
    def n_sense_cells(self) -> int:
        return self.sense_grid * self.sense_grid

    @property
    def cell_size(self) -> float:
        return self.world_size / self.grid

    @property
    def half_world(self) -> float:
        """Greatest possible torus distance along one axis."""
        return self.world_size / 2.0

    # World-scale terrain lengths, resolved to world units.
    @property
    def ridge_base_y(self) -> float:
        return self.ridge_base_frac * self.world_size

    @property
    def ridge_amplitude(self) -> float:
        return self.ridge_amp_frac * self.world_size

    @property
    def ridge_sigma(self) -> float:
        return self.ridge_sigma_frac * self.world_size

    @property
    def forest_water_scale(self) -> float:
        return self.forest_water_frac * self.world_size

    @property
    def memory_slots(self) -> int:
        return self.memory_water_slots + self.memory_fruit_slots

    @property
    def in_dim(self) -> int:
        """Retina channels (food/prey/predator/water/slope per sector) + energy
        + diet + own water + four numbers per memory slot (sin/cos of bearing,
        squashed distance, confidence)."""
        return 5 * self.retina_sectors + 3 + 4 * self.memory_slots

    @property
    def brain_params(self) -> int:
        # recurrent net: W_in, W_rec, b_h, W_out, b_out
        i, h, o = self.in_dim, self.hidden, self.out_dim
        return i * h + h * h + h + h * o + o

    @property
    def diet_index(self) -> int:
        """Column in the genome holding the diet gene."""
        return self.brain_params

    @property
    def genome_size(self) -> int:
        return self.brain_params + self.trait_dim
