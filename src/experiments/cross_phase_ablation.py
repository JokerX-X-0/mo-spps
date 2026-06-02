"""Cross-phase ablation: measure incremental contribution of each phase.

Compares:
  1. Phase1: continuous pool + fixed budget
  2. Phase2: + dynamic budget
  3. Phase3: + strategy inheritance
  4. Phase4_Full: + all Phase 4 features
  5. NoPool_Phase1: no pool + fixed budget (pool baseline)
  6. NoPool_Phase4: no pool + all features (pool baseline)
  7. HardCapPool: hard-cap pool (alternative pool mechanism)

Uses unified reference point from optimizer.ref_point for fair HV comparison.
"""

import time
import numpy as np

from ..problems.mo_scsp import generate_moscp_instance
from ..mo_spps import MOSPPSOptimizer
from ..metrics import (
    compute_hypervolume,
    compute_average_jaccard_distance,
    compute_component_entropy,
    compute_pool_occupancy,
    compute_reuse_concentration,
)

VARIANTS = {
    "Phase1": {
        "pool_mode": "continuous",
        "budget_mode": "fixed",
        "base_budget": 3.0,
        "use_strategy_inheritance": False,
        "elimination_interval": 10,
        "gamma": 0.0,
        "kappa": 0.0,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },
    "Phase2": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "base_budget": 2.0,
        "use_strategy_inheritance": False,
        "elimination_interval": 10,
        "gamma": 0.0,
        "kappa": 0.0,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },
    "Phase3": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "base_budget": 2.0,
        "use_strategy_inheritance": True,
        "elimination_interval": 10,
        "gamma": 0.0,
        "kappa": 0.0,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },
    "Phase4_Full": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "base_budget": 2.0,
        "use_strategy_inheritance": True,
        "elimination_interval": 3,
        "gamma": 0.5,
        "kappa": 0.5,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "NoPool_Phase1": {
        "pool_mode": "none",
        "budget_mode": "fixed",
        "base_budget": 3.0,
        "use_strategy_inheritance": False,
        "elimination_interval": 10,
        "gamma": 0.0,
        "kappa": 0.0,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },
    "NoPool_Phase4": {
        "pool_mode": "none",
        "budget_mode": "dynamic",
        "base_budget": 2.0,
        "use_strategy_inheritance": True,
        "elimination_interval": 3,
        "gamma": 0.5,
        "kappa": 0.5,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "HardCapPool": {
        "pool_mode": "hard_cap",
        "budget_mode": "dynamic",
        "base_budget": 2.0,
        "use_strategy_inheritance": True,
        "elimination_interval": 3,
        "gamma": 0.5,
        "kappa": 0.0,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
}


def _make_config(variant, num_components, solution_capacity, max_fe, pop_size, seed):
    return {
        "problem": {
            "num_components": num_components,
            "solution_capacity": solution_capacity,
            "num_objectives": 2,
        },
        "population": {
            "population_size": pop_size,
            "max_function_evaluations": max_fe,
        },
        "shared_pool": {
            "mode": variant["pool_mode"],
            "epsilon": 0.01,
            "tau": 1.0,
            "utility_guidance_kappa": variant["kappa"],
            "capacity_reference": 5,
        },
        "local_search": {
            "shop_size": 5,
            "use_probabilistic_acceptance": variant["use_probabilistic_acceptance"],
            "use_novelty_acceptance": variant["use_novelty_acceptance"],
            "use_release_operation": variant["use_release_operation"],
            "temperature": 1.0,
            "archive_contribution_threshold": 0.0,
            "novelty_threshold": 0.3,
            "quality_loss_threshold": 0.02,
        },
        "budget": {
            "mode": variant["budget_mode"],
            "base_budget": variant["base_budget"],
            "alpha_pareto": 1.0 if variant["budget_mode"] == "dynamic" else 0.0,
            "beta_crowding": 1.0 if variant["budget_mode"] == "dynamic" else 0.0,
            "delta_decision_diversity": 0.4 if variant["budget_mode"] == "dynamic" else 0.0,
            "gamma_exploration": variant["gamma"],
        },
        "rebirth": {
            "use_rebirth": True,
            "use_strategy_inheritance": variant["use_strategy_inheritance"],
            "elimination_interval": variant["elimination_interval"],
            "replacement_rate": 0.2,
            "inheritance_strength": 0.5,
            "inheritance_smoothing": 0.1,
            "preference_learning_rate": 0.01,
            "keep_reference_direction": True,
            "retention_a": 0.5,
            "retention_b": 0.3,
            "retention_d": 0.2,
        },
        "archive": {
            "max_size": 200,
            "prune_method": variant["prune_method"],
            "objective_weight": 0.7,
            "decision_weight": 0.3,
            "remove_duplicates": True,
        },
        "reference_directions": {
            "n_directions": 50,
        },
        "experiment": {"seed": seed},
    }


def _extract_summary(archive, problem, fe_count, agents, elapsed, ref_point):
    arch_objs = archive.get_objectives_array()
    hv = 0.0
    if arch_objs.size > 0 and ref_point is not None:
        hv = compute_hypervolume(arch_objs, ref_point)
    return {
        "fe_count": int(fe_count),
        "runtime_seconds": elapsed,
        "archive_size": len(archive),
        "hypervolume": float(hv),
        "avg_jaccard_distance": float(compute_average_jaccard_distance(archive.solutions)),
        "component_entropy_norm": float(
            compute_component_entropy(
                compute_pool_occupancy(agents), problem.num_components
            )[1]
        ),
        "reuse_concentration": float(compute_reuse_concentration(compute_pool_occupancy(agents))),
    }


def run_cross_phase_ablation(
    problem_type="high_synergy",
    num_components=30,
    solution_capacity=10,
    max_fe=5000,
    population_size=50,
    seed=0,
    n_runs=30,
    verbose=True,
):
    print(f"\n{'#'*90}")
    print(f"#  Cross-Phase Ablation: Incremental Contribution of Each Phase")
    print(f"#  Problem: MOSCSP ({problem_type}), M={num_components}, K={solution_capacity}")
    print(f"#  Budget: {max_fe} FE, {n_runs} runs per variant")
    print(f"{'#'*90}")

    all_results = {name: [] for name in VARIANTS}
    total = len(VARIANTS) * n_runs
    count = 0

    for vname, variant in VARIANTS.items():
        for run in range(n_runs):
            count += 1
            run_seed = seed + run
            if verbose:
                print(f"  [{count}/{total}] {vname} run {run+1}/{n_runs}...", end=" ")

            problem = generate_moscp_instance(num_components, solution_capacity,
                                               problem_type, 1.0, run_seed)
            config = _make_config(variant, num_components, solution_capacity,
                                   max_fe, population_size, run_seed)
            np.random.seed(run_seed)
            start = time.perf_counter()
            opt = MOSPPSOptimizer(problem, config)
            opt.initialize()
            opt.run()
            elapsed = time.perf_counter() - start

            s = _extract_summary(opt.archive, problem, opt.fe_count, opt.agents, elapsed, opt.ref_point)
            all_results[vname].append(s)

            if verbose:
                print(f"|A|={s['archive_size']}, HV={s['hypervolume']:.2f}, JD={s['avg_jaccard_distance']:.4f}")

    # Aggregate
    metrics_keys = ["archive_size", "hypervolume", "avg_jaccard_distance",
                    "component_entropy_norm", "reuse_concentration", "runtime_seconds"]
    aggregated = {}

    print(f"\n{'='*90}")
    print(f"  Cross-Phase Ablation Results (mean over {n_runs} runs)")
    print(f"{'='*90}")
    header = f"{'Variant':<22} {'|A|':>6} {'HV':>10} {'Jaccard':>9} {'Entropy':>9} {'Gini':>9} {'Time(s)':>8}"
    print(header)
    print("-" * 90)

    for vname in VARIANTS:
        runs = all_results[vname]
        agg = {}
        for k in metrics_keys:
            agg[k] = np.mean([r[k] for r in runs])
        aggregated[vname] = agg

        row = f"{vname:<22}"
        for k in metrics_keys:
            val = agg[k]
            if k == "runtime_seconds":
                row += f" {val:>8.2f}"
            elif k in ("avg_jaccard_distance", "component_entropy_norm", "reuse_concentration"):
                row += f" {val:>9.4f}"
            else:
                row += f" {val:>10.3f}"
        print(row)

    print("-" * 90)

    # Cross-phase incremental analysis
    print(f"\n  Incremental contributions:")
    phases = ["Phase1", "Phase2", "Phase3", "Phase4_Full"]
    for i in range(1, len(phases)):
        prev = aggregated[phases[i-1]]
        curr = aggregated[phases[i]]
        print(f"    {phases[i-1]} -> {phases[i]}:")
        print(f"      HV change:       {curr['hypervolume'] - prev['hypervolume']:+.3f}")
        print(f"      Jaccard change:  {curr['avg_jaccard_distance'] - prev['avg_jaccard_distance']:+.4f}")
        print(f"      Entropy change:  {curr['component_entropy_norm'] - prev['component_entropy_norm']:+.4f}")
        print(f"      Gini change:     {curr['reuse_concentration'] - prev['reuse_concentration']:+.4f}")

    # Pool effect
    print(f"\n  Pool contribution (continuous vs none):")
    for label, pool_v, nopool_v in [
        ("Phase1", "Phase1", "NoPool_Phase1"),
        ("Phase4", "Phase4_Full", "NoPool_Phase4"),
    ]:
        pool = aggregated[pool_v]
        nopool = aggregated[nopool_v]
        print(f"    {label}: HV {pool['hypervolume']-nopool['hypervolume']:+.2f}, "
              f"JD {pool['avg_jaccard_distance']-nopool['avg_jaccard_distance']:+.4f}, "
              f"Gini {pool['reuse_concentration']-nopool['reuse_concentration']:+.4f}")

    # HardCap comparison
    p4 = aggregated["Phase4_Full"]
    hc = aggregated["HardCapPool"]
    print(f"\n  Continuous vs HardCap (both Phase4):")
    print(f"    HV change:       {p4['hypervolume'] - hc['hypervolume']:+.3f}")
    print(f"    Jaccard change:  {p4['avg_jaccard_distance'] - hc['avg_jaccard_distance']:+.4f}")
    print(f"    Entropy change:  {p4['component_entropy_norm'] - hc['component_entropy_norm']:+.4f}")
    print(f"    Gini change:     {p4['reuse_concentration'] - hc['reuse_concentration']:+.4f}")

    print(f"{'='*90}\n")

    return {"all_results": all_results, "aggregated": aggregated}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run cross-phase ablation experiments")
    parser.add_argument("--problem", type=str, default="high_synergy",
                        choices=["low_synergy", "high_synergy", "multi_cluster"])
    parser.add_argument("--num_components", type=int, default=30)
    parser.add_argument("--solution_capacity", type=int, default=10)
    parser.add_argument("--max_fe", type=int, default=5000)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_cross_phase_ablation(args.problem, args.num_components, args.solution_capacity,
                             args.max_fe, args.population_size, args.seed, args.n_runs,
                             verbose=not args.quiet)
