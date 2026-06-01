"""Phase 3 inheritance strength sweep.

Tests multiple inheritance_strength values with paired seeds.
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

STRENGTHS = [0.5, 0.7, 0.9, 1.0]


def _make_config(num_components, solution_capacity, max_fe, pop_size, seed,
                 inheritance_strength):
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
            "mode": "soft_pressure",
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
            "mode": "dynamic",
            "base_budget": 2.0,
            "alpha_pareto": 1.0,
            "beta_crowding": 1.0,
            "delta_decision_diversity": 0.4,
            "gamma_exploration": 0.0,
        },
        "rebirth": {
            "use_rebirth": True,
            "use_strategy_inheritance": True,
            "elimination_interval": 3,
            "replacement_rate": 0.2,
            "inheritance_strength": inheritance_strength,
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


def run_inheritance_sweep(
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
    print(f"#  Phase 3 Inheritance Strength Sweep")
    print(f"#  Strengths: {STRENGTHS}")
    print(f"#  {n_runs} paired runs, MOSCSP ({problem_type}), M={num_components}, K={solution_capacity}")
    print(f"{'#'*90}")

    all_results = {s: [] for s in STRENGTHS}
    total = len(STRENGTHS) * n_runs
    count = 0

    for run in range(n_runs):
        run_seed = seed + run
        # Same problem instance for all strengths (paired design)
        problem = generate_moscp_instance(num_components, solution_capacity,
                                           problem_type, 1.0, run_seed)
        for eta in STRENGTHS:
            count += 1
            config = _make_config(num_components, solution_capacity, max_fe,
                                   population_size, run_seed, eta)
            np.random.seed(run_seed)
            start = time.perf_counter()
            opt = MOSPPSOptimizer(problem, config)
            opt.initialize()
            opt.run()
            elapsed = time.perf_counter() - start

            s = _extract_summary(opt.archive, problem, opt.fe_count, opt.agents, elapsed)
            all_results[eta].append(s)

            if verbose:
                print(f"  [{count}/{total}] eta={eta} run {run+1}/{n_runs}  "
                      f"HV={s['hypervolume']:.2f} JD={s['avg_jaccard_distance']:.4f}")

    # Aggregate
    metrics_keys = ["archive_size", "hypervolume", "avg_jaccard_distance",
                    "component_entropy_norm", "reuse_concentration", "runtime_seconds"]

    print(f"\n{'='*90}")
    print(f"  Inheritance Strength Sweep Results (mean over {n_runs} paired runs)")
    print(f"{'='*90}")
    header = f"{'η':<10} {'|A|':>6} {'HV':>10} {'Jaccard':>9} {'Entropy':>9} {'Gini':>9} {'Time(s)':>8}"
    print(header)
    print("-" * 90)

    aggregated = {}
    for eta in STRENGTHS:
        runs = all_results[eta]
        agg = {}
        for k in metrics_keys:
            agg[k] = np.mean([r[k] for r in runs])
        aggregated[eta] = agg

        row = f"η={eta:<7.1f}"
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

    # Paired comparison: each strength vs 0.5 baseline
    baseline = np.array([r["hypervolume"] for r in all_results[0.5]])
    print(f"\n  Paired differences vs η=0.5 (n={n_runs}):")
    print(f"  {'η':<8} {'HV diff':>10} {'±std':>8} {'95%CI':>18} {'t':>8} {'sig':>8}")
    print(f"  {'-'*60}")
    for eta in STRENGTHS[1:]:
        values = np.array([r["hypervolume"] for r in all_results[eta]])
        diffs = values - baseline
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=1)
        se = std_diff / np.sqrt(n_runs)
        ci_lo = mean_diff - 1.96 * se
        ci_hi = mean_diff + 1.96 * se
        t_stat = mean_diff / se if se > 0 else 0.0
        sig = "***" if abs(t_stat) > 2.58 else ("**" if abs(t_stat) > 1.96 else "n.s.")
        print(f"  η={eta:<6.1f} {mean_diff:>+10.3f} {std_diff:>8.2f}  [{ci_lo:>+7.3f}, {ci_hi:>+7.3f}] {t_stat:>+8.3f} {sig:>6}")

    print(f"{'='*90}\n")

    # Best strength recommendation
    best_eta = max(STRENGTHS, key=lambda e: aggregated[e]["hypervolume"])
    best_hv = aggregated[best_eta]["hypervolume"]
    print(f"  Best HV: η={best_eta} (HV={best_hv:.3f})")
    print(f"  Recommendation: inheritance_strength = {best_eta}")

    return {"all_results": all_results, "aggregated": aggregated}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 inheritance strength sweep")
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
    run_inheritance_sweep(args.problem, args.num_components, args.solution_capacity,
                           args.max_fe, args.population_size, args.seed, args.n_runs,
                           verbose=not args.quiet)
