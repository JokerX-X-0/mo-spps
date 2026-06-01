"""Phase 4 ablation experiments: validate all Phase 4 features.

Compares:
  1. Phase4_Full (all Phase 4 features on)
  2. Phase3_Baseline (all Phase 4 features off)
  3. NoUtilityGuidance (kappa=0)
  4. NoNoveltyAcceptance
  5. NoProbAcceptance
  6. NoReleaseOp
  7. CrowdingOnlyPrune
  8. HardCapPool

Section 33.4.
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
    "Phase4_Full": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "Phase3_Baseline": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "gamma": 0.0,
        "kappa": 0.0,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },
    "NoUtilityGuidance": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.0,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "NoNoveltyAcceptance": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": False,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "NoProbAcceptance": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "NoReleaseOp": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": False,
        "prune_method": "hybrid_objective_decision",
    },
    "CrowdingOnlyPrune": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "crowding",
    },
    "HardCapPool": {
        "pool_mode": "hard_cap",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.0,
        "use_strategy_inheritance": True,
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
            "base_budget": 2.0,
            "alpha_pareto": 1.0,
            "beta_crowding": 1.0,
            "delta_decision_diversity": 0.4,
            "gamma_exploration": variant["gamma"],
        },
        "rebirth": {
            "use_rebirth": True,
            "use_strategy_inheritance": variant["use_strategy_inheritance"],
            "elimination_interval": 3,
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


def _extract_summary(archive, problem, fe_count, agents, elapsed):
    arch_objs = archive.get_objectives_array()
    hv = 0.0
    if arch_objs.size > 0:
        ref = np.min(arch_objs, axis=0) - 1.0
        hv = compute_hypervolume(arch_objs, ref)
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


def run_phase4_ablation(
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
    print(f"#  Phase 4 Ablation: All Features Validation")
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

            s = _extract_summary(opt.archive, problem, opt.fe_count, opt.agents, elapsed)
            all_results[vname].append(s)

            if verbose:
                print(f"|A|={s['archive_size']}, HV={s['hypervolume']:.2f}, JD={s['avg_jaccard_distance']:.4f}")

    # Aggregate
    metrics_keys = ["archive_size", "hypervolume", "avg_jaccard_distance",
                    "component_entropy_norm", "reuse_concentration", "runtime_seconds"]
    aggregated = {}

    print(f"\n{'='*90}")
    print(f"  Phase 4 Ablation Results (mean over {n_runs} runs)")
    print(f"{'='*90}")
    header = f"{'Variant':<28} {'|A|':>6} {'HV':>10} {'Jaccard':>9} {'Entropy':>9} {'Gini':>9} {'Time(s)':>8}"
    print(header)
    print("-" * 90)

    for vname in VARIANTS:
        runs = all_results[vname]
        agg = {}
        for k in metrics_keys:
            agg[k] = np.mean([r[k] for r in runs])
        aggregated[vname] = agg

        row = f"{vname:<28}"
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

    p4 = aggregated["Phase4_Full"]
    p3 = aggregated["Phase3_Baseline"]
    print(f"\n  Phase 4 vs Phase 3 baseline:")
    print(f"    HV change:       {p4['hypervolume'] - p3['hypervolume']:+.3f}")
    print(f"    Jaccard change:  {p4['avg_jaccard_distance'] - p3['avg_jaccard_distance']:+.4f}")
    print(f"    Entropy change:  {p4['component_entropy_norm'] - p3['component_entropy_norm']:+.4f}")
    print(f"    Gini change:     {p4['reuse_concentration'] - p3['reuse_concentration']:+.4f}")
    print(f"{'='*90}\n")

    return {"all_results": all_results, "aggregated": aggregated}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 4 ablation experiments")
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
    run_phase4_ablation(args.problem, args.num_components, args.solution_capacity,
                         args.max_fe, args.population_size, args.seed, args.n_runs,
                         verbose=not args.quiet)
