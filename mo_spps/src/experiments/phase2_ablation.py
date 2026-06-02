"""Phase 2 ablation experiments: budget scheduling validation.

Compares:
  1. Phase 1 baseline (fixed budget + soft_pressure pool)
  2. Phase 2 dynamic budget (full: alpha, beta, delta)
  3. NoBudget ablation (fixed budget = Phase 1)
  4. NoDecisionDiversity ablation (dynamic, delta=0)
  5. NoPool ablation (mode=none, fixed budget)
  6. NoPool + Dynamic budget (mode=none, dynamic budget)

Section 33.2.
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
    "Phase1_Baseline": {
        "pool_mode": "soft_pressure",
        "budget_mode": "fixed",
        "delta": 0.0,
    },
    "Phase2_Dynamic": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "delta": 1.0,
    },
    "NoBudget_Ablation": {
        "pool_mode": "soft_pressure",
        "budget_mode": "fixed",
        "delta": 0.0,
    },
    "NoDecisionDiversity_Ablation": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "delta": 0.0,
    },
    "NoPool_Fixed": {
        "pool_mode": "none",
        "budget_mode": "fixed",
        "delta": 0.0,
    },
    "NoPool_Dynamic": {
        "pool_mode": "none",
        "budget_mode": "dynamic",
        "delta": 1.0,
    },
}


def _make_config(
    variant: dict,
    num_components: int,
    solution_capacity: int,
    max_fe: int,
    population_size: int,
    seed: int,
) -> dict:
    return {
        "problem": {
            "num_components": num_components,
            "solution_capacity": solution_capacity,
            "num_objectives": 2,
        },
        "population": {
            "population_size": population_size,
            "max_function_evaluations": max_fe,
        },
        "shared_pool": {
            "mode": variant["pool_mode"],
            "epsilon": 0.01,
            "tau": 1.0,
            "capacity_reference": 5,
        },
        "local_search": {
            "shop_size": 5,
            "use_probabilistic_acceptance": False,
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
            "delta_decision_diversity": variant["delta"],
            "gamma_exploration": 0.0,
        },
        "rebirth": {
            "use_rebirth": True,
            "use_strategy_inheritance": False,
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
            "prune_method": "crowding",
            "objective_weight": 0.7,
            "decision_weight": 0.3,
            "remove_duplicates": True,
        },
        "reference_directions": {
            "n_directions": 50,
        },
        "experiment": {
            "seed": seed,
        },
    }


def _extract_summary(
    archive, problem, fe_count, agents, elapsed
) -> dict:
    arch_objs = archive.get_objectives_array()
    hv = 0.0
    if arch_objs.size > 0:
        ref = np.min(arch_objs, axis=0) - 1.0
        hv = compute_hypervolume(arch_objs, ref)

    avg_jd = compute_average_jaccard_distance(archive.solutions)
    occupancy = compute_pool_occupancy(agents)
    entropy, entropy_norm = compute_component_entropy(
        occupancy, problem.num_components
    )
    concentration = compute_reuse_concentration(occupancy)

    return {
        "fe_count": int(fe_count),
        "runtime_seconds": elapsed,
        "archive_size": len(archive),
        "hypervolume": float(hv),
        "avg_jaccard_distance": float(avg_jd),
        "component_entropy": float(entropy),
        "component_entropy_norm": float(entropy_norm),
        "reuse_concentration": float(concentration),
    }


def run_phase2_ablation(
    problem_type: str = "high_synergy",
    num_components: int = 30,
    solution_capacity: int = 10,
    max_fe: int = 5000,
    population_size: int = 50,
    seed: int = 0,
    n_runs: int = 5,
    verbose: bool = True,
) -> dict:
    """Run Phase 2 ablation study comparing budget modes and pool modes.

    Runs each variant `n_runs` times with different seeds and averages results.
    """
    print(f"\n{'#'*90}")
    print(f"#  Phase 2 Ablation: Budget Scheduling Validation")
    print(f"#  Problem: MOSCSP ({problem_type}), M={num_components}, K={solution_capacity}")
    print(f"#  Budget: {max_fe} FE, {n_runs} runs per variant")
    print(f"{'#'*90}")

    all_results: dict[str, list[dict]] = {
        name: [] for name in VARIANTS
    }

    total = len(VARIANTS) * n_runs
    count = 0

    for variant_name, variant in VARIANTS.items():
        for run in range(n_runs):
            count += 1
            run_seed = seed + run
            if verbose:
                print(f"  [{count}/{total}] {variant_name} run {run+1}/{n_runs}...", end=" ")

            problem = generate_moscp_instance(
                num_components=num_components,
                solution_capacity=solution_capacity,
                instance_type=problem_type,
                synergy_strength=1.0,
                seed=run_seed,
            )

            config = _make_config(
                variant, num_components, solution_capacity, max_fe,
                population_size, run_seed,
            )

            np.random.seed(run_seed)
            start = time.perf_counter()
            optimizer = MOSPPSOptimizer(problem, config)
            optimizer.initialize()
            optimizer.run()
            elapsed = time.perf_counter() - start

            summary = _extract_summary(
                optimizer.archive, problem, optimizer.fe_count,
                optimizer.agents, elapsed,
            )

            all_results[variant_name].append(summary)
            if verbose:
                print(f"|A|={summary['archive_size']}, "
                      f"HV={summary['hypervolume']:.3f}, "
                      f"JD={summary['avg_jaccard_distance']:.4f}")

    # Aggregate across runs
    print(f"\n{'='*90}")
    print(f"  Phase 2 Ablation Results (mean over {n_runs} runs)")
    print(f"{'='*90}")
    header = f"{'Variant':<30} {'|A|':>6} {'HV':>10} {'Jaccard':>9} {'Entropy':>9} {'Gini':>9} {'Time(s)':>8}"
    print(header)
    print("-" * 90)

    metrics = [
        "archive_size", "hypervolume", "avg_jaccard_distance",
        "component_entropy_norm", "reuse_concentration", "runtime_seconds",
    ]

    aggregated = {}
    for variant_name in VARIANTS:
        runs = all_results[variant_name]
        agg = {}
        for key in metrics:
            agg[key] = np.mean([r[key] for r in runs])
        aggregated[variant_name] = agg

        row = f"{variant_name:<30}"
        for key in metrics:
            val = agg[key]
            if key == "runtime_seconds":
                row += f" {val:>8.2f}"
            elif key in ("avg_jaccard_distance", "component_entropy_norm", "reuse_concentration"):
                row += f" {val:>9.4f}"
            else:
                row += f" {val:>10.3f}"
        print(row)

    print("-" * 90)

    # Compare key findings
    p1 = aggregated["Phase1_Baseline"]
    p2 = aggregated["Phase2_Dynamic"]

    print(f"\n  Phase 2 vs Phase 1:")
    print(f"    HV change:       {p2['hypervolume'] - p1['hypervolume']:+.3f}")
    print(f"    Jaccard change:  {p2['avg_jaccard_distance'] - p1['avg_jaccard_distance']:+.4f}")
    print(f"    Entropy change:  {p2['component_entropy_norm'] - p1['component_entropy_norm']:+.4f}")
    print(f"    Gini change:     {p2['reuse_concentration'] - p1['reuse_concentration']:+.4f}")
    print(f"{'='*90}\n")

    return {
        "all_results": all_results,
        "aggregated": aggregated,
        "config": {
            "problem_type": problem_type,
            "num_components": num_components,
            "solution_capacity": solution_capacity,
            "max_fe": max_fe,
            "population_size": population_size,
            "n_runs": n_runs,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase 2 ablation experiments"
    )
    parser.add_argument(
        "--problem", type=str, default="high_synergy",
        choices=["low_synergy", "high_synergy", "multi_cluster"],
    )
    parser.add_argument("--num_components", type=int, default=30)
    parser.add_argument("--solution_capacity", type=int, default=10)
    parser.add_argument("--max_fe", type=int, default=5000)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    run_phase2_ablation(
        problem_type=args.problem,
        num_components=args.num_components,
        solution_capacity=args.solution_capacity,
        max_fe=args.max_fe,
        population_size=args.population_size,
        seed=args.seed,
        n_runs=args.n_runs,
        verbose=not args.quiet,
    )
