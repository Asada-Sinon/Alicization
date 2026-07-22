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
    #       + per memory slot [sin bearing, cos bearing, distance, confidence]
    #       + per retina sector [peer] (diet similarity, appended after
    #         memory -- see sensors.sense).
    # See the `in_dim` property for the authoritative count.
    retina_sectors: int = 8        # directional vision resolution
    hidden: int = 16               # recurrent hidden units (the fluctlight's memory)
    out_dim: int = 2               # [turn, thrust]
    trait_dim: int = 3             # non-brain genes; [0] = diet, [1] = investment,
    #                                 [2] = size
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
    water_deficit_buffer: float = 0.0  # how far below zero `water` can go before
    #                                    `reproduction.cull` calls it dehydration
    #                                    death (docs/water_fix_buffer.md). Real
    #                                    mammals don't die the instant water hits
    #                                    zero -- ordinary mammals tolerate ~10-12%
    #                                    body-mass water loss, camels 25-30%
    #                                    (Schmidt-Nielsen et al. 1956) -- so a
    #                                    zero-buffer instant-death rule is a
    #                                    simplification with no biological
    #                                    grounding. Default 0.0 reproduces the old
    #                                    behaviour exactly (`water <= 0` is the
    #                                    same test as `water <= -0.0`), same
    #                                    convention as `trample_impact`: this
    #                                    doesn't exist until an ablation arm turns
    #                                    it on with --set water_deficit_buffer=....
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
    # Per-offspring investment is a *gene*, not a constant -- see `invest_of`.
    # These bound it; the range is Polyworld's, which evolved bodies in a
    # comparable 2D world for years without it degenerating to an extreme.
    invest_min: float = 0.2        # floor on the energy/water fraction given away
    invest_span: float = 0.6       # so the gene maps onto [0.2, 0.8]
    #                                A gene of 0 sigmoids to 0.5 -- the old fixed
    #                                `repro_cost_frac` -- so a fresh population
    #                                starts at the previous behaviour and drifts
    #                                from there, which keeps the baseline clean.
    # Body size: NOT the gape-limited predation refuge that was originally
    # proposed (see docs/biology.md S8.2 -- juveniles die of thirst at a mean
    # age of 52.5 steps, long before predation risk at 170.7 steps applies, so
    # that premise is dead). Coupled here only to the water economy: bigger
    # storage (size^1.0, a volume) vs bigger metabolic/water loss (size^0.75,
    # Kleiber), giving desiccation tolerance ~ size^0.25 -- a real but very flat
    # lever (doubling requires a 16x size increase). Deliberately NOT coupled to
    # intake (eat_rate/drink_rate) or predation (attack_range/diet_delta): if
    # intake also scaled up, size would run away to size_max with no
    # countervailing cost, which would be an unfalsifiable "gene saturation"
    # result rather than a real trade-off between metabolic cost and thirst
    # tolerance.
    size_min: float = 0.4          # floor; a gene of 0 sigmoids to 0.5 -> size=1.0,
    size_span: float = 1.2         #   so a fresh population starts at the old
    #                                 unscaled behaviour, range [0.4, 1.6]
    max_age: float = 3000.0        # steps before old age
    spawn_radius: float = 3.0      # child placed within this radius of parent

    # --- genetics ---
    mutation_sigma: float = 0.05      # gaussian noise on brain genes per birth
    diet_mutation_sigma: float = 0.015  # diet is strongly heritable (keeps types
    #                                     distinct instead of blurring to omnivore)
    invest_mutation_sigma: float = 0.02  # 0.4x the brain rate. Trait genes are
    #                                      deliberately slower than brain genes:
    #                                      a body the brain cannot track is worse
    #                                      than a body that changes slowly, and
    #                                      the same 0.3x ratio on `diet` is what
    #                                      keeps the herbivore/carnivore split
    #                                      from blurring.
    size_mutation_sigma: float = 0.02  # same rate as investment -- the brain is
    #                                    adapted to a body's speed/reach, which
    #                                    do not depend on `size` here, but a
    #                                    slow-drifting body is still the safer
    #                                    default for any future trait that does.
    hue_drift: float = 0.02           # lineage colour drift per birth
    carnivore_init_frac: float = 0.05  # fraction of founders seeded as carnivores

    # --- ablation switches: none of these change any array shape, so an arm
    # with a switch flipped stays genome-compatible with the default arm ---
    # Diet speciation, four layers. `docs/biology.md` §10.1/§10.5: the
    # herbivore/carnivore split is currently
    # held apart by four independent layers, not by selection alone. Each flag
    # below defaults to the current (baked-in) behaviour, so flipping none of
    # them changes nothing; `--set NAME=0` turns one off for an ablation arm.
    diet_bimodal_init: bool = True     # False: founders start from a single
    #                                    neutral cluster (diet gene centred on
    #                                    0, i.e. omnivore) instead of two.
    diet_crossover_exempt: bool = True  # False: diet gene is a normal
    #                                    crossover site like any other gene,
    #                                    instead of always taken from parent A.
    diet_mutation_asymmetric: bool = True  # False: diet mutates at the same
    #                                    `mutation_sigma` as brain genes,
    #                                    instead of the slower `diet_mutation_sigma`.
    assortative_mating: bool = True    # False: second parents are paired
    #                                    uniformly at random instead of
    #                                    sorted by diet -- per Dieckmann &
    #                                    Doebeli (1999), this is the layer
    #                                    theory says *maintains* a branch
    #                                    rather than just seeding one, so it
    #                                    is tested separately from the other
    #                                    three (see docs/TODO.md priority 2).
    peer_channel_enabled: bool = True  # the `peer` retina channel (see
    #                                    sensors.sense) has no natural "off"
    #                                    position -- it is part of the sensory
    #                                    layer, so there is no pre-channel
    #                                    population to compare against. This
    #                                    flag forces peer_val to zero everywhere
    #                                    while leaving `in_dim`/`genome_size`
    #                                    unchanged, so an ablation arm and the
    #                                    full arm remain genome-compatible and
    #                                    directly comparable.
    los_occlusion_enabled: bool = False  # docs/three_d.md S5.1: block visibility
    #                                    of a candidate (for prey/pred/peer)
    #                                    when the terrain between observer and
    #                                    candidate rises above the straight-line
    #                                    interpolation of their two heights --
    #                                    "a mountain blocks sight". Default OFF,
    #                                    same convention as `trample_impact`:
    #                                    this doesn't exist unless an ablation
    #                                    arm explicitly turns it on. Does not
    #                                    change `in_dim`/`genome_size` -- it is
    #                                    a visibility judgement on the existing
    #                                    `closeness` term, not a new channel.
    los_samples: int = 4               # interior points sampled along the
    #                                    observer->candidate line (excluding
    #                                    both endpoints), evenly spaced. Only
    #                                    read when `los_occlusion_enabled`.
    los_margin: float = 0.1            # height a sample must exceed the
    #                                    observer/candidate interpolation by
    #                                    (same units as `terrain.height`, i.e.
    #                                    a fraction of `ridge_height`) before
    #                                    it counts as blocking -- a small
    #                                    positive margin so a perfectly flat
    #                                    line (interpolation error is exactly
    #                                    0 for the analytic height field) is
    #                                    never mistaken for occlusion.

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

    # --- niche construction: passive trampling (Stage 0 of docs/TODO.md
    # priority 3 -- a zero-genome-cost side effect of moving, not a brain
    # output). Agents deposit onto a [n_cells] field just like the existing
    # `demand_per_cell` scatter-add in `dynamics.graze`; the field then erodes
    # `ecology.regrow`'s plant carrying capacity, closing a real feedback loop
    # a population can walk itself into (or, if it disperses, out of).
    trample_decay: float = 0.99       # per-step decay of "recent foot traffic".
    #                                    Half-life ~69 steps: slower than
    #                                    regrow_rate=0.06 (~17-step timescale)
    #                                    so it reflects recent presence rather
    #                                    than instantaneous occupancy, but far
    #                                    short of the static terrain -- this is
    #                                    not permanent land-forming.
    trample_rate: float = 0.01        # deposit per agent occupying a cell, per
    #                                    step, before decay. Set equal to
    #                                    (1 - trample_decay) so a single
    #                                    continuously-occupied cell asymptotes
    #                                    to exactly the field's own cap of 1.0
    #                                    (geometric series rate/(1-decay) = 1);
    #                                    a crowd of agents saturates faster,
    #                                    a single pass-through leaves a small
    #                                    transient bump that decays away.
    trample_impact: float = 0.0       # fraction of plant carrying capacity
    #                                    eroded at maximum trample (1.0).
    #                                    Default OFF -- unlike fruit_max's
    #                                    "0 disables" convention, this
    #                                    mechanism doesn't exist unless an
    #                                    ablation arm explicitly turns it on
    #                                    with --set trample_impact=... .
    #                                    Applies to `plant` (grass/ground
    #                                    cover) only, not `fruit`: piosphere
    #                                    erosion is agents' feet wearing down
    #                                    the herb layer they walk on, not
    #                                    damage to the canopy fruit hangs from
    #                                    overhead.
    #                                    docs/biology.md SS11.1: measured to be
    #                                    a *negative* feedback (agents cluster
    #                                    -> capacity erodes -> they leave ->
    #                                    trample's own Moran's I falls). Kept
    #                                    for reproducibility of that published
    #                                    null result; do not delete.
    trample_path_gain: float = 0.0    # fraction of `forest_slow`'s canopy speed
    #                                    penalty cancelled at maximum trample
    #                                    (1.0). This is the sign-corrected
    #                                    Stage 0 mechanism from docs/biology.md
    #                                    SS11.1: real game trails are a
    #                                    *positive* feedback -- repeated
    #                                    passage compacts ground and clears
    #                                    undergrowth, making the same route
    #                                    cheaper to cross again, not scarcer of
    #                                    food. Applied to `forest_slow` (a
    #                                    movement cost) rather than
    #                                    `trample_impact` (a food-capacity
    #                                    cost) so the two mechanisms stay
    #                                    disentangled -- one records a
    #                                    published negative result, this one
    #                                    tests the corrected hypothesis. Only
    #                                    has anything to cancel where
    #                                    `forest_slow * terrain.forest` > 0,
    #                                    i.e. paths can only be measured
    #                                    forming through canopy, not open
    #                                    ground -- see `dynamics.act`. Default
    #                                    OFF, same convention as
    #                                    `trample_impact`.

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
        squashed distance, confidence) + a sixth retina channel (peer, diet
        similarity rather than difference -- appended after memory, not
        interleaved with the other five, so it doesn't renumber any existing
        `server/app.py` slice offset)."""
        return 5 * self.retina_sectors + 3 + 4 * self.memory_slots + self.retina_sectors

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
    def invest_index(self) -> int:
        """Column holding the per-offspring investment gene.

        Trait genes are appended *after* the brain block, so adding one leaves
        every brain weight at the same offset and only grows `genome_size` by
        one. That is why a new trait is cheap where a new sensory input is not.
        """
        return self.brain_params + 1

    @property
    def size_index(self) -> int:
        """Column holding the body-size gene. Appended after investment for the
        same reason investment was appended after diet."""
        return self.brain_params + 2

    @property
    def genome_size(self) -> int:
        return self.brain_params + self.trait_dim
