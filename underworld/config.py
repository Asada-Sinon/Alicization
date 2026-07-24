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
    trait_dim: int = 7             # non-brain genes; [0] = diet, [1] = investment,
    #                                 [2] = size, [3] = attack_range (predator reach),
    #                                 [4] = escape (prey evasion), [5] = armor (bite-damage
    #                                 reduction), [6] = spike (damage reflected onto the
    #                                 attacker). [3][4] are the red-queen pair
    #                                 (docs/attack_range_redqueen.md); [5][6] are the visible
    #                                 morphological defences (docs/trait_defense_catalog.md,
    #                                 docs/trait_addition_feasibility.md). Every one costs
    #                                 energy, never water (docs/trait_roadmap.md §5) -- see
    #                                 `attack_cost`/`escape_cost`/`armor_cost`/`spike_cost`.
    genome_init_scale: float = 0.4
    food_sample_dist: float = 9.0  # how far ahead each sector samples the plant field

    # --- movement / metabolism ---
    max_speed: float = 8.0         # world units / second
    max_turn: float = 3.0          # radians / second
    base_cost: float = 0.02        # energy / step just to exist
    move_cost: float = 0.05        # extra energy / step at full thrust
    carn_cost: float = 0.15        # extra upkeep scaled by diet: idle predators
    #                                die back fast, coupling their numbers to prey.
    #                                History: 0.15 in the flat 256^2 world; the
    #                                terrain world charges predators for climbing and
    #                                slows them in cover, and at 0.15 under the OLD
    #                                (undoubled) water economy they went extinct on
    #                                every seed, so it was dropped to 0.10 (~15% carn).
    #                                Restored to 0.15 as the *compensation knob* of the
    #                                water retune (docs/water_fix_decision.md): with
    #                                base/move_water_cost halved below, 0.15 is a healthy
    #                                working point (carn_frac 0.216, 6 seeds, min 0.194,
    #                                none near extinction) that holds predators off the
    #                                river without inflating their share. NARROW window:
    #                                +0.05 is a cliff (0.20 -> 0.10, 0.25+ -> die-off).
    #                                Do NOT retune water amplitude or terrain without
    #                                re-calibrating this (docs/water_fix_retune.md 4.2).

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

    # --- red-queen co-evolution: heritable attack range vs prey escape ---
    # (docs/attack_range_redqueen.md, docs/trait_roadmap.md §7.3). The predator's
    # attack gene lengthens its bite reach; the prey's escape gene shortens the
    # attacker's *effective* reach against it. Each side pays an ENERGY tax for its
    # investment -- never a water tax, which would drag the trait into the juvenile-
    # thirst death-censoring trap that killed the body-size gene (§5 of the roadmap).
    # The tax is modelled on `carn_cost` (energy ledger, diet-scaled), the one
    # standing precedent proven safe against that trap.
    attack_min: float = 1.0        # attack_range_of maps the gene to
    attack_span: float = 10.0      #   [attack_min, attack_min+attack_span] = [1, 11];
    #                                a gene of 0 sigmoids to 0.5 -> 1 + 0.5*10 = 6.0 =
    #                                the pre-gene `attack_range` constant, so a fresh
    #                                population starts exactly at the old behaviour and
    #                                drifts from there. Upper bound 11 stays well under
    #                                the sense cell (world_size/sense_grid = 21.3): a
    #                                bite beyond that reaches prey the neighbour table
    #                                never gathered and would fail silently. Guarded by
    #                                check_config_invariants.
    attack_cost: float = 0.012     # energy/step per world-unit of reach ABOVE the 6.0
    #                                baseline, scaled by diet like carn_cost so
    #                                herbivores (who almost never clear the diet_delta
    #                                predation gate) pay ~0 and their attack gene drifts
    #                                neutrally. Tuned by short probe (docs/
    #                                attack_range_redqueen.md): heavy enough that reach
    #                                does not just peg at the 11 ceiling, light enough
    #                                that carnivores are not driven extinct.
    attack_mutation_sigma: float = 0.02  # same slow rate as size/invest trait genes.
    escape_span: float = 12.0      # escape_of maps the gene to [0, escape_span/2] = [0,6]
    #                                world units shaved off an attacker's effective reach.
    #                                Neutral (gene 0) = 0 EXACTLY (unlike attack's 6.0):
    #                                a fresh population has no evasion, so any escape is
    #                                evolved, not seeded -- the clean red-queen baseline.
    escape_cost: float = 0.012     # energy/step per world-unit of evasion, scaled by
    #                                (1-diet) so herbivores (the hunted) pay and
    #                                carnivores' escape gene drifts neutrally. Symmetric
    #                                to attack_cost.
    escape_mutation_sigma: float = 0.02
    attack_range_heritable: bool = True  # False: predation reads the fixed
    #                                `cfg.attack_range` constant; the attack gene still
    #                                exists, drifts and is reported, but has no functional
    #                                effect and levies no tax -- the clean control arm,
    #                                genome-compatible with the True arm (same layout).
    prey_escape_enabled: bool = True     # False: the escape gene has no effect on
    #                                predation and no tax -- the *unilateral* arm (only
    #                                the predator's reach evolves), used to tell a true
    #                                red-queen (both traits climb) from one-sided
    #                                optimisation (reach drifts up until its tax bites).

    # --- visible morphological defences (docs/trait_defense_catalog.md,
    # docs/trait_addition_feasibility.md): armour reduces a bite's damage, spikes
    # reflect damage back onto the attacker. Both are prey-side traits -- the tax is
    # scaled by (1-diet) so drifting carnivores pay ~0, gene=0 maps to *zero* defence
    # (a fresh population has none, so any defence is evolved, not seeded), and both
    # ride the ENERGY ledger in `metabolize`, NEVER thirst's water ledger: a water tax
    # would censor the trait through the juvenile-thirst bottleneck exactly as it
    # reversed the body-size gene (docs/trait_roadmap.md §5). They are the structural
    # twins of the escape gene -- same one-sided map, same (1-diet) gate, same slow
    # mutation, same energy tax. First landing of the defence line; the A/B judgement
    # and the falsifiable predictions are in docs/trait_addition_feasibility.md §B.
    armor_span: float = 1.0          # armor_of maps the gene to [0, 0.5]: the fraction
    #                                  of a bite's energy damage negated. 0 at gene=0.
    armor_cost: float = 0.012        # energy/step per unit armour, scaled by (1-diet).
    #                                  Same magnitude as escape_cost; tune with a short
    #                                  probe before a long run so it neither pegs at the
    #                                  ceiling nor starves the predators.
    armor_mutation_sigma: float = 0.02   # slow trait rate, same as size/escape.
    armor_heritable: bool = True     # False: predation ignores the armour gene and
    #                                  levies no tax; the gene still exists, drifts and
    #                                  is reported -- the clean control arm, genome-
    #                                  compatible with the True arm.
    # spike is DUAL-USE (docs/trait_defense_landing.md §7, redesign after the first
    # evolution run showed the old reflect-only spike was too kin-level to evolve):
    # the SAME gene is an OFFENSIVE weapon for a carnivore (extra bite damage, which
    # punches through evolved prey armour) and a DEFENSIVE retaliation for a herbivore
    # (a bitten prey ENVENOMS its attacker -- a decaying slow + energy drain over the
    # following steps, non-lethal). Which role applies is set by whether you are the
    # attacker or the prey in a bite, not by a diet branch. gene=0 -> no offense bonus
    # and no venom, so a fresh population reproduces the old predation exactly.
    spike_span: float = 1.0          # spike_of maps the gene to [0, 0.5]. 0 at gene=0.
    spike_offense_gain: float = 1.0  # attacker bite damage *= (1 + spike_offense_gain *
    #                                  attacker_spike). spike=0 -> x1 (neutral start).
    spike_venom_gain: float = 2.0    # venom deposited on the attacker when a bite lands
    #                                  = prey_spike * spike_venom_gain (prey_spike in
    #                                  [0,0.5], so up to ~1.0 per hit, accumulating).
    spike_cost: float = 0.012        # energy/step per unit spike -- UNIVERSAL now (not
    #                                  (1-diet)-gated): both lineages grow & pay for
    #                                  spikes, each benefiting in its own role.
    spike_mutation_sigma: float = 0.02
    spike_heritable: bool = True     # False: spikes have no offense, deposit no venom,
    #                                  and levy no tax -- the clean control arm.
    # Venom debuff field (per-agent, decays like fear/trample): a carnivore that bit a
    # spiked prey carries `venom` that saps its speed and energy for a few steps.
    venom_decay: float = 0.9         # per-step multiplicative decay (~7-step half-life).
    venom_slow: float = 0.6          # fraction of speed removed at venom>=1 (clipped).
    venom_drain: float = 0.15        # energy/step drained at venom>=1 (clipped).

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
    base_water_cost: float = 0.01     # water / step just to exist. Halved from 0.02
    #                                   as the "water valve" of the retune
    #                                   (docs/water_fix_decision.md): the master fix
    #                                   for the juvenile-thirst bottleneck. Paired with
    #                                   carn_cost=0.15 above so the extra prey it saves
    #                                   don't inflate carnivore share. Drops thirst
    #                                   deaths 0.829->0.540 and pushes predators off the
    #                                   river (carn_water_dist 11.4->25.3), 6 seeds.
    move_water_cost: float = 0.025    # extra water / step at full thrust (panting).
    #                                   Halved from 0.05 with base_water_cost (same
    #                                   retune). Cost: population ~+79% -- a known,
    #                                   separately-tracked tradeoff (docs/rebalance.md
    #                                   holds "shrink population" as an orthogonal axis).
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
    # Density-dependent reproduction (docs/herbivore_overpopulation.md L6): crowding
    # raises the effective energy an agent needs to breed, so dense cells self-limit --
    # a logistic negative feedback the world lacked (reproduction was a pure energy
    # gate, no density term). Because herbivores form the dense crowds and carnivores
    # are sparse, this throttles herbivores far more than carnivores WITHOUT touching
    # the water/plant carrying capacity (the levers docs/scale_and_density.md §5.3
    # falsified). Default 0.0 = OFF = a compile-time no-op branch, bit-exact old world.
    density_repro_penalty: float = 0.0  # eff_threshold = repro_threshold * (1 + penalty
    #                                     * min(crowd/cap, 1)); 0 disables it entirely.
    density_repro_cap: float = 6.0      # agents in one plant cell at which the penalty
    #                                     saturates (world/grid = 4-unit cells).
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

    # Neonatal water floor, decoupled from `invest_frac` (docs/water_system.md
    # SS2.3/3.3, arm_B: raising `invest_min` was measured to do almost nothing
    # to death_thirst_frac, because it raises the SAME shared fraction that
    # also sets the energy handover -- evolution absorbed the change without
    # newborn hydration moving much, since the floor and the thing it floors
    # were never independent). This floors the WATER fraction only:
    # `water_frac = max(invest_frac, water_lactation_floor_frac)`, while
    # `energy = parent_energy * invest_frac` stays untouched. Default 0.0 is a
    # true no-op -- invest_frac is always >= invest_min > 0, so
    # max(invest_frac, 0.0) == invest_frac and behaviour is bit-identical to
    # before this field existed. Modelled on lactation, which is
    # evolutionarily a channel separate from the quantity/quality dial that
    # sets clutch/egg provisioning (Oftedal 2002, J Mammary Gland Biol
    # Neoplasia 7:225-252 and 7:253-266) -- see
    # docs/water_fix_provisioning.md for the design rationale and measured
    # effect of turning this on. Not a gene: this is a Config constant, so
    # evolution has no heritable variation in it to compress back toward
    # invest_min, unlike raising invest_min itself. Sweep values should stay
    # <= invest_min + invest_span (0.8 by default) so a parent never hands
    # over more water than the existing gene's own upper bound would already
    # permit -- this keeps the mechanism inside a region of state space the
    # kernel already tolerates rather than opening an untested one.
    water_lactation_floor_frac: float = 0.0

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

    # --- landscape of fear (docs/landscape_of_fear.md S3.2, the #1-ranked path
    # to push carnivores off the river). A [n_cells] `fear` field in WorldState
    # takes a lagged, decaying imprint of where carnivores lurk; sensors.sense
    # folds it into the existing `pred` retina channel via max, so prey can learn
    # to skirt a predator's camping ground. A camped predator then carves a
    # dead-zone prey avoid -> its ambush stops paying -> it must roam to eat.
    # Zero in_dim/genome_size change (the fold reuses the pred channel), so an
    # evolved population still loads. DEFAULT ON at the working point below,
    # measured over 6 paired seeds (docs/landscape_of_fear.md "6.实测"): with the
    # fear field on, carnivores ROAM instead of camping the river (carn_speed
    # 1.5->2.4, up in 6/6 seeds, paired p=0.031) and their share drops modestly
    # (carn_frac 12.1%->10.2%, down in 5/6) WITHOUT worsening the juvenile-thirst
    # bottleneck (death_thirst_age rose, +4.9). The one thing it does NOT do is
    # relocate predators far from water (carn_water_dist +1.4 only, noisy) -- true
    # river-departure needs diel commuting, which this clockless world lacks.
    # Turn OFF (the ablation control / pre-fear behaviour) with --set fear_rate=0:
    # the deposit and the sensor fold are both compile-time branches on
    # fear_rate>0, so at 0 the field stays identically zero and behaviour is
    # bit-exact the pre-fear baseline.
    fear_decay: float = 0.99      # per-step decay of the fear trace. Half-life
    #                               ~69 steps, same timescale as trample_decay:
    #                               it reflects *recent* predator presence, not a
    #                               permanent territory map. A carnivore round of
    #                               ambush-then-move leaves a fading mark, long
    #                               enough for prey to react, short enough that a
    #                               vacated spot reopens.
    fear_rate: float = 0.05       # deposit per carnivore (diet>0.5) occupying a
    #                               cell, per step, before decay. 0.05 (=5x the
    #                               "start small" 0.01 probe) is the validated
    #                               working point: strong enough that carn_speed
    #                               moves in 6/6 seeds, gentle enough that thirst
    #                               structure did not degrade. Set 0 for the
    #                               ablation control. A single continuously-
    #                               occupied cell asymptotes to fear_rate/(1-decay)
    #                               = 0.05/0.01 = 5, clipped to the field cap 1.0,
    #                               so an actively-camped cell saturates -- which is
    #                               the point (a persistently-guarded spot reads as
    #                               maximally dangerous).
    fear_sense_scale: float = 3.0 # how strongly the sampled fear reads into the
    #                               pred channel before the max-fold. At the 6-seed
    #                               validated working point (3.0); 1.0 was too weak
    #                               to clear seed noise (docs/landscape_of_fear.md
    #                               6.实测: rate0.01/scale1.0 came back null). The
    #                               risk to watch when raising it is the juvenile-
    #                               thirst window (the river is both the highest-fear
    #                               and the only-water place) -- death_thirst_frac/age,
    #                               not just the spatial metrics; at 3.0 it held.
    #                               Only read when fear_rate>0.

    # --- day-night (diel) cycle (docs/day_night.md; the next step on the
    # landscape-of-fear line). docs/landscape_of_fear.md §6 named the missing
    # ingredient: the fear field made carnivores ROAM (carn_speed 1.5->2.4, 6/6,
    # p=0.031) but could not make them LEAVE the riverbank (carn_water_dist +1.4,
    # noisy) -- "water is a hard constraint, predators camp the water, and the
    # public-information signal cannot push them off it without pushing prey into
    # thirst death. What is really missing is DIEL COMMUTING (drink and leave),
    # which this clockless world cannot give." So this adds a clock.
    #
    # Deliberately NOT a brain input: no sin/cos phase channel is fed in (that
    # would hand evolution the answer). Instead a global `phase` scalar drives an
    # environmental rhythm whose sensory *consequences* -- the retina dimming at
    # night, own_water draining faster at midday -- are the zeitgebers a recurrent
    # brain can entrain to. Whether 16 hidden units can evolve that internal clock
    # is exactly the open question Phase 1 tests; if not, `hidden` grows (Phase 2).
    #
    # `day_length > 0` is a compile-time branch (cfg is closed over by build_step,
    # not traced): at the default 0 the clock never advances, `phase` stays 0, and
    # every downstream fold (sensor darkening, thirst heat) is absent from the jit
    # -- bit-exact the pre-clock baseline, and NO shape changes anywhere, so no
    # population is invalidated and golden holds. Same convention as `fear_rate=0`.
    day_length: int = 400         # steps in one full day-night cycle; 0 disables
    #                               (compile-time no-op, bit-exact the pre-clock
    #                               kernel). DEFAULT-ON at 400 as of the pred_nocturnal
    #                               landing (docs/day_night.md §5): the shipped world
    #                               now runs a diel clock whose ONLY behavioural lever
    #                               is nocturnal predation (pred_night_amp below) --
    #                               heat/darkness/forage are defaulted OFF (they cull
    #                               or are inert). `--set day_length=0` recovers the
    #                               pre-day-night baseline bit-exact for ablation. Timescale
    #                               (docs/day_night.md): juvenile thirst death
    #                               ~52 steps, fear/trample half-life ~69, memory
    #                               half-life 346, max_age 3000 -- 400 gives adults
    #                               ~7.5 cycles/life and sits well above the 69-step
    #                               fear half-life so the rhythm is not smeared out;
    #                               juveniles see under one cycle, so their thirst
    #                               relief is indirect (via adult commuting).
    heat_water_amp: float = 0.0   # DEFAULT OFF (was a candidate lever): even taxing
    #                               only movement water, midday heat culls via the
    #                               thirst bottleneck (docs/day_night.md §4, +18-21pp
    #                               thirst). Kept for ablation only; set ~1-2 to study.
    #                               --- relative rise in the MOVEMENT water cost at peak
    #                               midday heat: thirst charges move_water_cost*thrust*
    #                               (1 + heat_water_amp*light), so foraging/travelling
    #                               at midday costs more water while RESTING near water
    #                               costs the ordinary base -- the adaptive escape is
    #                               "forage in the cool night, rest by the water at
    #                               midday", a commute rather than a death tax. An
    #                               earlier flat multiplier on base+move was a seed-0
    #                               thirst bomb (+18-21pp, midday culling); see
    #                               docs/day_night.md §4. Read only when day_length>0.
    night_vision_floor: float = 1.0  # DEFAULT OFF (1.0 = no darkening): the darkness
    #                               lever was measured SAFE but spatially inert
    #                               (docs/day_night.md §4), so it is not part of the
    #                               default day/night world; set ~0.4 to study nocturnal
    #                               ambush. --- inter-agent vision multiplier at midnight
    #                               (1.0 = no darkening). Night shrinks how far an
    #                               agent can SEE OTHER AGENTS (prey/pred/peer), the
    #                               nocturnal-ambush lever; the water and food
    #                               channels are deliberately left at full range so
    #                               darkness never worsens water-finding -- the
    #                               standing juvenile-thirst risk (landscape_of_fear
    #                               §3.2). Read only when day_length>0.
    pred_night_amp: float = 1.0   # DEFAULT-ON at 1.0 (docs/day_night.md §5, user
    #                               chose half-strength): 6-seed validated -- hunt_success
    #                               night-day +0.245 (6/6), thirst -13pp (6/6, relieves
    #                               the bottleneck), carn_frac +3.3pp (6/6, the cost).
    #                               amp=2.0 gave a stronger signal but +6pp carn_frac.
    #                               --- nocturnal predation boost: at night a carnivore's
    #                               effective bite reach is scaled by
    #                               (1 + pred_night_amp*(1-light)), strongest at
    #                               midnight (light=0), zero at midday. This is the
    #                               ONE day-night lever the seed-0 sweep found clean
    #                               (docs/day_night.md §4): it makes predation risk
    #                               DIEL (hunt_success splits night>day, monotone in
    #                               amp) with a tiny population swing and thirst that
    #                               FALLS (prey lost to predation, not thirst) --
    #                               unlike the water-taxing levers that cull. Boosted
    #                               reach is clipped to the sense cell (world_size/
    #                               sense_grid) so a bite never reaches prey the
    #                               neighbour table did not gather. Default 0.0 =
    #                               off; read only when day_length>0 (predation gets
    #                               light=None otherwise, a bit-exact no-op).
    forage_heat: float = 0.0      # midday-heat foraging penalty (water-neutral):
    #                               grazing/fruit intake `demand` is scaled by
    #                               (1 - forage_heat*light), so foraging is less
    #                               productive at midday and best in the cool night.
    #                               The seed-0 sweep found this the SAFEST lever
    #                               (thirst ~baseline, tiny population swing) but
    #                               spatially INERT with the reactive 16-unit brain
    #                               (docs/day_night.md §4) -- it is the water-neutral
    #                               SUBSTRATE for the Phase-2 test of whether a larger
    #                               `hidden` evolves a scheduled commute on it (§6).
    #                               Default 0.0 = off; read only when day_length>0.

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
    def attack_index(self) -> int:
        """Column holding the heritable attack-range gene (predator reach). Appended
        after size -- adding it leaves every brain weight and prior trait at the same
        offset, growing `genome_size` by one."""
        return self.brain_params + 3

    @property
    def escape_index(self) -> int:
        """Column holding the prey escape gene, the red-queen counterpart to attack."""
        return self.brain_params + 4

    @property
    def armor_index(self) -> int:
        """Column holding the prey armour gene (bite-damage reduction). Appended after
        escape -- a morphological defence (docs/trait_defense_catalog.md)."""
        return self.brain_params + 5

    @property
    def spike_index(self) -> int:
        """Column holding the prey spike gene (damage reflected onto the attacker)."""
        return self.brain_params + 6

    @property
    def attack_max(self) -> float:
        """Largest evolvable attack range. Must stay under the sense cell size or a
        bite could reach prey the neighbour table never gathered (guarded in
        check_config_invariants)."""
        return self.attack_min + self.attack_span

    @property
    def genome_size(self) -> int:
        return self.brain_params + self.trait_dim
