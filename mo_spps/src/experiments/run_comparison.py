"""Comparison experiment: MO-SPPS vs NSGA-II vs MOEA/D vs MOGWO vs Random vs Greedy.

Runs all algorithms on the same MOSCSP problem instance with equal FE budget
and reports metrics side-by-side. Section 27.4, Section 30.
"""

import json
import time
import numpy as np

from ..problems.mo_scsp import generate_moscp_instance
from ..baselines import (
    run_random_mo_search,
    run_greedy_scalarization,
    run_nsga2,
    run_moead,
    run_mogwo,
)
from ..metrics import (
    compute_hypervolume,
    compute_average_jaccard_distance,
    compute_component_entropy,
    compute_pool_occupancy,
    compute_reuse_concentration,
)


def _run_mo_spps(problem, config: dict, verbose: bool = True) -> dict:
    """Run MO-SPPS and return standard result dict."""
    from ..mo_spps import MOSPPSOptimizer

    seed = config.get("experiment", {}).get("seed", 0)
    np.random.seed(seed)

    if verbose:
        print("  Initializing MO-SPPS...")

    start = time.perf_counter()
    optimizer = MOSPPSOptimizer(problem, config)
    optimizer.initialize()
    history = optimizer.run()
    elapsed = time.perf_counter() - start

    archive = optimizer.archive
    arch_objs = archive.get_objectives_array()
    hv = 0.0
    if arch_objs.size > 0 and optimizer.ref_point is not None:
        hv = compute_hypervolume(arch_objs, optimizer.ref_point)

    return {
        "archive": archive,
        "history": history,
        "summary": _extract_summary(
            archive, arch_objs, hv, problem, optimizer.fe_count, optimizer.agents, elapsed
        ),
    }


def _extract_summary(
    archive, arch_objs, hv, problem, fe_count, agents, elapsed
) -> dict:
    """Extract standard summary metrics."""
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


def _extract_baseline_summary(
    result: dict, problem, elapsed: float
) -> dict:
    """Extract standard summary from baseline result dict."""
    archive = result["archive"]
    arch_objs = archive.get_objectives_array()

    hv = 0.0
    if arch_objs.size > 0:
        ref = np.min(arch_objs, axis=0) - 1.0
        hv = compute_hypervolume(arch_objs, ref)

    avg_jd = compute_average_jaccard_distance(archive.solutions)

    all_sols = list(archive.solutions)
    occupancy: dict[int, int] = {}
    for sol in all_sols:
        for j in sol:
            occupancy[j] = occupancy.get(j, 0) + 1

    entropy, entropy_norm = compute_component_entropy(
        occupancy, problem.num_components
    )
    concentration = compute_reuse_concentration(occupancy)

    return {
        "fe_count": int(result.get("final_fe", 0)),
        "runtime_seconds": elapsed,
        "archive_size": len(archive),
        "hypervolume": float(hv),
        "avg_jaccard_distance": float(avg_jd),
        "component_entropy": float(entropy),
        "component_entropy_norm": float(entropy_norm),
        "reuse_concentration": float(concentration),
    }


def _compare_pareto_fronts(results: dict, title: str, save_path: str | None = None):
    """Plot all Pareto fronts on a single figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    colors = {
        "MO-SPPS": "#2196F3",
        "NSGA-II": "#4CAF50",
        "MOEA/D": "#FF9800",
        "MOGWO": "#9C27B0",
        "Random": "#757575",
        "Greedy": "#F44336",
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, result in results.items():
        objs = result["archive"].get_objectives_array()
        if objs.size == 0:
            continue
        color = colors.get(name, "#000000")
        ax.scatter(
            objs[:, 0], objs[:, 1],
            label=name, c=color, s=25, alpha=0.7, edgecolors="black", linewidth=0.3,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Quality", fontsize=11)
    ax.set_ylabel("-Cost", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    return fig


def _print_comparison_table(
    problem_type: str, summaries: dict[str, dict]
) -> None:
    """Print a formatted comparison table."""
    metrics = [
        ("archive_size", "|Archive|"),
        ("hypervolume", "HV"),
        ("avg_jaccard_distance", "Jaccard Dist"),
        ("component_entropy_norm", "Entropy(norm)"),
        ("reuse_concentration", "Gini Conc."),
        ("runtime_seconds", "Time (s)"),
    ]

    algo_order = ["MO-SPPS", "NSGA-II", "MOEA/D", "MOGWO", "Random", "Greedy"]

    print(f"\n{'='*90}")
    print(f"  Results: {problem_type}")
    print(f"{'='*90}")

    # Header
    header = f"{'Algorithm':<12}"
    for _, short_name in metrics:
        header += f" {short_name:>14}"
    print(header)
    print("-" * 90)

    # Rows
    for name in algo_order:
        if name not in summaries:
            continue
        s = summaries[name]
        row = f"{name:<12}"
        for key, _ in metrics:
            val = s.get(key, 0)
            if key == "runtime_seconds":
                row += f" {val:>14.2f}"
            elif key == "reuse_concentration":
                row += f" {val:>14.4f}"
            elif key in ("component_entropy_norm", "avg_jaccard_distance"):
                row += f" {val:>14.4f}"
            else:
                row += f" {val:>14.3f}"
        print(row)

    print("-" * 90)

    # Best values
    best_row = f"{'Best':<12}"
    for key, _ in metrics:
        if key == "runtime_seconds":
            best_val = min(s.get(key, float("inf")) for s in summaries.values())
            best_row += f" {best_val:>14.2f}"
        elif key == "reuse_concentration":
            best_val = max(s.get(key, float("-inf")) for s in summaries.values())
            best_row += f" {best_val:>14.4f}"
        else:
            best_val = max(s.get(key, float("-inf")) for s in summaries.values())
            if key in ("component_entropy_norm", "avg_jaccard_distance"):
                best_row += f" {best_val:>14.4f}"
            else:
                best_row += f" {best_val:>14.3f}"
    print(best_row)
    print(f"{'='*90}\n")


def run_comparison(
    problem_type: str = "high_synergy",
    num_components: int = 30,
    solution_capacity: int = 10,
    max_fe: int = 10000,
    population_size: int = 100,
    seed: int = 0,
    save_plot: bool = True,
    verbose: bool = True,
) -> dict:
    """Run all algorithms on the same problem instance and compare.

    Args:
        problem_type: MOSCSP instance type (low_synergy, high_synergy, multi_cluster).
        num_components: Number of components (M).
        solution_capacity: Max components per solution (K).
        max_fe: Maximum function evaluations.
        population_size: Population size for population-based algorithms.
        seed: Random seed.
        save_plot: Whether to save the comparison plot.
        verbose: Whether to print progress.

    Returns:
        dict mapping algorithm name -> result dict.
    """
    print(f"\n{'#'*90}")
    print(f"#  Algorithm Comparison: MO-SPPS vs NSGA-II vs MOEA/D vs MOGWO")
    print(f"#  Problem: MOSCSP ({problem_type}), M={num_components}, K={solution_capacity}")
    print(f"#  Budget: {max_fe} FE, Seed: {seed}")
    print(f"{'#'*90}")

    problem = generate_moscp_instance(
        num_components=num_components,
        solution_capacity=solution_capacity,
        instance_type=problem_type,
        synergy_strength=1.0,
        seed=seed,
    )

    results: dict[str, dict] = {}
    summaries: dict[str, dict] = {}

    # =========================================================================
    # 1. MO-SPPS
    # =========================================================================
    if verbose:
        print("\n[1/6] Running MO-SPPS...")

    config = {
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
            "mode": "fixed",
            "base_budget": 3.0,
            "alpha_pareto": 1.0,
            "beta_crowding": 1.0,
            "delta_decision_diversity": 1.0,
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

    result = _run_mo_spps(problem, config, verbose=verbose)
    results["MO-SPPS"] = result
    summaries["MO-SPPS"] = result["summary"]
    if verbose:
        print(f"  MO-SPPS: |A|={result['summary']['archive_size']}, "
              f"HV={result['summary']['hypervolume']:.3f}, "
              f"JD={result['summary']['avg_jaccard_distance']:.4f}")

    # =========================================================================
    # 2. NSGA-II
    # =========================================================================
    if verbose:
        print("\n[2/6] Running NSGA-II...")

    start = time.perf_counter()
    result_nsga2 = run_nsga2(
        problem=problem,
        max_fe=max_fe,
        solution_capacity=solution_capacity,
        archive_size=200,
        population_size=population_size,
        crossover_rate=0.9,
        mutation_rate=0.02,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    results["NSGA-II"] = result_nsga2
    summaries["NSGA-II"] = _extract_baseline_summary(result_nsga2, problem, elapsed)
    if verbose:
        print(f"  NSGA-II: |A|={summaries['NSGA-II']['archive_size']}, "
              f"HV={summaries['NSGA-II']['hypervolume']:.3f}, "
              f"JD={summaries['NSGA-II']['avg_jaccard_distance']:.4f}")

    # =========================================================================
    # 3. MOEA/D
    # =========================================================================
    if verbose:
        print("\n[3/6] Running MOEA/D...")

    start = time.perf_counter()
    result_moead = run_moead(
        problem=problem,
        max_fe=max_fe,
        solution_capacity=solution_capacity,
        archive_size=200,
        population_size=population_size,
        neighborhood_size=20,
        mutation_rate=0.1,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    results["MOEA/D"] = result_moead
    summaries["MOEA/D"] = _extract_baseline_summary(result_moead, problem, elapsed)
    if verbose:
        print(f"  MOEA/D: |A|={summaries['MOEA/D']['archive_size']}, "
              f"HV={summaries['MOEA/D']['hypervolume']:.3f}, "
              f"JD={summaries['MOEA/D']['avg_jaccard_distance']:.4f}")

    # =========================================================================
    # 4. MOGWO
    # =========================================================================
    if verbose:
        print("\n[4/6] Running MOGWO...")

    start = time.perf_counter()
    result_mogwo = run_mogwo(
        problem=problem,
        max_fe=max_fe,
        solution_capacity=solution_capacity,
        archive_size=200,
        population_size=population_size,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    results["MOGWO"] = result_mogwo
    summaries["MOGWO"] = _extract_baseline_summary(result_mogwo, problem, elapsed)
    if verbose:
        print(f"  MOGWO: |A|={summaries['MOGWO']['archive_size']}, "
              f"HV={summaries['MOGWO']['hypervolume']:.3f}, "
              f"JD={summaries['MOGWO']['avg_jaccard_distance']:.4f}")

    # =========================================================================
    # 5. Random Search
    # =========================================================================
    if verbose:
        print("\n[5/6] Running Random Search...")

    start = time.perf_counter()
    result_random = run_random_mo_search(
        problem=problem,
        max_fe=max_fe,
        solution_capacity=solution_capacity,
        archive_size=200,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    results["Random"] = result_random
    summaries["Random"] = _extract_baseline_summary(result_random, problem, elapsed)
    if verbose:
        print(f"  Random: |A|={summaries['Random']['archive_size']}, "
              f"HV={summaries['Random']['hypervolume']:.3f}, "
              f"JD={summaries['Random']['avg_jaccard_distance']:.4f}")

    # =========================================================================
    # 6. Greedy Scalarization
    # =========================================================================
    if verbose:
        print("\n[6/6] Running Greedy Scalarization...")

    start = time.perf_counter()
    result_greedy = run_greedy_scalarization(
        problem=problem,
        max_fe=max_fe,
        solution_capacity=solution_capacity,
        archive_size=200,
        n_directions=50,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    results["Greedy"] = result_greedy
    summaries["Greedy"] = _extract_baseline_summary(result_greedy, problem, elapsed)
    if verbose:
        print(f"  Greedy: |A|={summaries['Greedy']['archive_size']}, "
              f"HV={summaries['Greedy']['hypervolume']:.3f}, "
              f"JD={summaries['Greedy']['avg_jaccard_distance']:.4f}")

    # =========================================================================
    # Print comparison
    # =========================================================================
    _print_comparison_table(f"MOSCSP ({problem_type})", summaries)

    # Plot
    if save_plot:
        plot_title = f"Pareto Front Comparison — MOSCSP ({problem_type}, {max_fe} FE)"
        save_path = f"comparison_pareto_{problem_type}_fe{max_fe}.png"
        _compare_pareto_fronts(results, plot_title, save_path)

    return {
        "results": results,
        "summaries": summaries,
        "problem_type": problem_type,
        "config": {
            "num_components": num_components,
            "solution_capacity": solution_capacity,
            "max_fe": max_fe,
            "population_size": population_size,
            "seed": seed,
        },
    }


def run_all_problem_types(
    num_components: int = 30,
    solution_capacity: int = 10,
    max_fe: int = 10000,
    population_size: int = 100,
    seed: int = 0,
    save_plot: bool = True,
    verbose: bool = True,
) -> dict:
    """Run comparison on all three MOSCSP instance types."""
    all_results = {}
    for ptype in ["low_synergy", "high_synergy", "multi_cluster"]:
        all_results[ptype] = run_comparison(
            problem_type=ptype,
            num_components=num_components,
            solution_capacity=solution_capacity,
            max_fe=max_fe,
            population_size=population_size,
            seed=seed,
            save_plot=save_plot,
            verbose=verbose,
        )

    # Cross-type summary
    print("\n" + "=" * 90)
    print("  Cross-Problem Summary: Avg Jaccard Distance (higher = more diverse)")
    print("=" * 90)
    header = f"{'Algorithm':<12}"
    for ptype in ["low_synergy", "high_synergy", "multi_cluster"]:
        header += f" {ptype:>22}"
    print(header)
    print("-" * 90)

    algo_order = ["MO-SPPS", "NSGA-II", "MOEA/D", "MOGWO", "Random", "Greedy"]
    for name in algo_order:
        row = f"{name:<12}"
        for ptype in ["low_synergy", "high_synergy", "multi_cluster"]:
            s = all_results[ptype]["summaries"].get(name, {})
            jd = s.get("avg_jaccard_distance", 0)
            hv = s.get("hypervolume", 0)
            row += f" JD={jd:.4f} HV={hv:.3f}"
        print(row)

    print("=" * 90)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MO algorithm comparison on MOSCSP"
    )
    parser.add_argument(
        "--problem", type=str, default="high_synergy",
        choices=["low_synergy", "high_synergy", "multi_cluster", "all"],
        help="MOSCSP instance type"
    )
    parser.add_argument("--num_components", type=int, default=30)
    parser.add_argument("--solution_capacity", type=int, default=10)
    parser.add_argument("--max_fe", type=int, default=10000)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_plot", action="store_true", help="Disable plot saving")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    if args.problem == "all":
        run_all_problem_types(
            num_components=args.num_components,
            solution_capacity=args.solution_capacity,
            max_fe=args.max_fe,
            population_size=args.population_size,
            seed=args.seed,
            save_plot=not args.no_plot,
            verbose=not args.quiet,
        )
    else:
        run_comparison(
            problem_type=args.problem,
            num_components=args.num_components,
            solution_capacity=args.solution_capacity,
            max_fe=args.max_fe,
            population_size=args.population_size,
            seed=args.seed,
            save_plot=not args.no_plot,
            verbose=not args.quiet,
        )
