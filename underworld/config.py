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
    world_size: float = 256.0      # square torus edge length (world units)
    grid: int = 64                 # plant field is grid x grid cells
    dt: float = 0.2                # seconds per step

    # --- spatial neighbour index (M1) ---
    sense_grid: int = 12           # binning grid for neighbour queries
    k_neighbors: int = 24          # max agents stored per sense-cell
    vision_radius: float = 21.0    # perception radius (<= sense cell size)
    attack_range: float = 6.0      # carnivore bite range (short = prey refuge)
    diet_delta: float = 0.15       # prey must be this much more herbivorous

    # --- population (fixed-capacity arrays) ---
    n_max: int = 8192              # hard cap on simultaneous agents
    n_init: int = 1200             # initial living agents. Water is a second,
    #                                 geographically scarce resource (the stream
    #                                 covers ~8% of the map) on top of food, so the
    #                                 long-run equilibrium (~250-330) is much lower
    #                                 than this. Seeding directly at that lower
    #                                 count starves the carnivore *founder* pool
    #                                 (~5% of n_init) into early stochastic
    #                                 extinction -- tried n_init=280, carnivores
    #                                 died out by step 800. Seeding high and
    #                                 letting it cull down (verified stable to 30k
    #                                 steps) is safer than a snappier reset.

    # --- brain (fixed-topology recurrent net; weights live in the genome) ---
    # inputs: per retina sector [food, prey, predator] + [own_energy, own_diet].
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
    carn_cost: float = 0.15        # extra upkeep scaled by diet: idle predators
    #                                die back fast, coupling their numbers to prey

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

    # --- water (a meandering stream; a separate resource from food) ---
    stream_amplitude: float = 40.0    # meander amplitude, world units
    stream_wavenumber: int = 2        # whole sine periods across world_size (must
    #                                    be an integer so the torus seam is seamless)
    stream_base_y: float = 128.0      # centerline mean y
    stream_half_width: float = 10.0   # half-width of the drinkable band
    water_init: float = 8.0
    water_max: float = 10.0           # hydration cap
    water_scale: float = 10.0         # normalizer for the own-water sensor input
    base_water_cost: float = 0.02     # water / step just to exist
    move_water_cost: float = 0.05     # extra water / step at full thrust (panting)
    drink_rate: float = 2.0           # water gained / step while standing in the stream

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
    def in_dim(self) -> int:
        """Retina channels (food/prey/predator/water per sector) + energy + diet
        + own water."""
        return 4 * self.retina_sectors + 3

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
