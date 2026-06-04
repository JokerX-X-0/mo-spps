"""Phase 4 ablation experiments: validate all Phase 4 features.

Compares:
  1. MO_SPPS_Full (all Phase 4 features on, continuous pool, adaptive Q)
  2. MO_SPPS_NoPool (continuous pool disabled, mode=none)
  3. MO_SPPS_OldSoftPool (original truncated soft-pressure formula)
  4. MO_SPPS_ContinuousPool (continuous soft-pressure, fixed Q)
  5. MO_SPPS_AdaptiveQ (archive-driven adaptive capacity)
  6. MO_SPPS_NoUtilityGuidance (kappa=0)
  7. MO_SPPS_NoNoveltyAcceptance
  8. MO_SPPS_NoProbAcceptance
  9. MO_SPPS_NoReleaseOp
 10. MO_SPPS_CrowdingOnlyPrune
 11. MO_SPPS_HardCapPool
 12. MO_SPPS_NoBudget (fixed budget, no dynamic allocation)
 13. MO_SPPS_NoInherit (no strategy preference inheritance)
 14. MO_SPPS_NoDecisionDiversity (delta=0 in budget)

Section 33.4.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "MO_SPPS_Full": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
        "use_adaptive_Q": True,
    },
    "MO_SPPS_NoPool": {
        "pool_mode": "none",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_NoUtilityGuidance": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.0,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_NoNoveltyAcceptance": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": False,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_NoProbAcceptance": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_NoReleaseOp": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": False,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_CrowdingOnlyPrune": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "crowding",
    },
    "MO_SPPS_HardCapPool": {
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
    "MO_SPPS_OldSoftPool": {
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
    "MO_SPPS_ContinuousPool": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
        "use_adaptive_Q": False,
    },
    "MO_SPPS_AdaptiveQ": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
        "use_adaptive_Q": True,
    },
    "MO_SPPS_NoBudget": {
        "pool_mode": "continuous",
        "budget_mode": "fixed",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_NoInherit": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": False,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_NoDecisionDiversity": {
        "pool_mode": "continuous",
        "budget_mode": "dynamic",
        "gamma": 0.5,
        "kappa": 0.5,
        "delta_decision_diversity": 0.0,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
}


def _make_config(variant, num_components, solution_capacity, max_fe, pop_size, seed):
    use_adaptive_q = variant.get("use_adaptive_Q", False)
    delta_decision = variant.get("delta_decision_diversity", 0.4)
    elim_interval = variant.get("elimination_interval", 3)
    base_budget = variant.get("base_budget", 2.0)
    capacity_q0 = variant.get("base_capacity_Q0", 3)

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
            "base_capacity_Q0": capacity_q0,
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
            "base_budget": base_budget,
            "alpha_pareto": 1.0,
            "beta_crowding": 1.0,
            "delta_decision_diversity": delta_decision,
            "gamma_exploration": variant["gamma"],
        },
        "rebirth": {
            "use_rebirth": True,
            "use_strategy_inheritance": variant["use_strategy_inheritance"],
            "elimination_interval": elim_interval,
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
        "adaptive_capacity": {
            "use_adaptive_Q": use_adaptive_q,
            "alpha_Q": 2.0,
            "base_capacity_Q0": capacity_q0,
            "Q_min": 3,
            "Q_max": 30,
            "update_interval": 10,
        },
        "region_novelty": {
            "enabled": False,
            "region_threshold": 0.3,
        },
        "experiment": {"seed": seed},
    }


def _run_one_variant(args):
    """Run a single variant × run combination. Top-level for multiprocessing."""
    vname, variant, run_seed, num_components, solution_capacity, max_fe, pop_size, problem_type = args

    problem = generate_moscp_instance(num_components, solution_capacity,
                                       problem_type, 1.0, run_seed)
    config = _make_config(variant, num_components, solution_capacity,
                           max_fe, pop_size, run_seed)
    np.random.seed(run_seed)
    start = time.perf_counter()
    opt = MOSPPSOptimizer(problem, config)
    opt.initialize()
    opt.run()
    elapsed = time.perf_counter() - start

    s = _extract_summary(opt.archive, problem, opt.fe_count, opt.agents, elapsed, opt.ref_point)
    return vname, run_seed, s


def _extract_summary(archive, problem, fe_count, agents, elapsed, ref_point=None):
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


def run_phase4_ablation(
    problem_type="high_synergy",
    num_components=30,
    solution_capacity=10,
    max_fe=5000,
    population_size=50,
    seed=0,
    n_runs=30,
    verbose=True,
    skip_variants=None,
    workers=1,
):
    print(f"\n{'#'*90}")
    print(f"#  Phase 4 Ablation: All Features Validation")
    print(f"#  Problem: MOSCSP ({problem_type}), M={num_components}, K={solution_capacity}")
    print(f"#  Budget: {max_fe} FE, {n_runs} runs per variant")
    print(f"{'#'*90}")

    all_results = {name: [] for name in VARIANTS}
    skip = set(skip_variants or [])
    active_variants = [(n, v) for n, v in VARIANTS.items() if n not in skip]
    total = len(active_variants) * n_runs

    if workers > 1:
        print(f"#  Parallel workers: {workers}")
    print(f"{'#'*90}")

    # Build list of all jobs
    jobs = []
    for vname, variant in active_variants:
        for run in range(n_runs):
            run_seed = seed + run
            jobs.append((vname, variant, run_seed, num_components, solution_capacity,
                         max_fe, population_size, problem_type))

    count = 0
    if workers <= 1:
        # Sequential execution
        for vname, variant, run_seed, _, _, _, _, _ in jobs:
            count += 1
            if verbose:
                print(f"  [{count}/{total}] {vname} run {run_seed - seed + 1}/{n_runs}...", end=" ")

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
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_variant, j): j for j in jobs}
            for future in as_completed(futures):
                count += 1
                vname, run_seed, s = future.result()
                all_results[vname].append(s)
                if verbose:
                    print(f"  [{count}/{total}] {vname} run {run_seed - seed + 1}/{n_runs}... |A|={s['archive_size']}, HV={s['hypervolume']:.2f}, JD={s['avg_jaccard_distance']:.4f}")

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

    for vname, _ in active_variants:
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
    parser.add_argument("--skip_hardcap", action="store_true",
                        help="Skip MO_SPPS_HardCapPool (slow)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()
    skip = ["MO_SPPS_HardCapPool"] if args.skip_hardcap else None
    run_phase4_ablation(args.problem, args.num_components, args.solution_capacity,
                         args.max_fe, args.population_size, args.seed, args.n_runs,
                         verbose=not args.quiet, skip_variants=skip,
                         workers=args.workers)
