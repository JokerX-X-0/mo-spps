"""Single-run experiment script for MO-SPPS.

Runs one configuration on one problem instance and reports results.
"""

import json
import time
import numpy as np

from ..mo_spps import MOSPPSOptimizer


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: manual config
        return _default_config()


def _default_config() -> dict:
    return {
        "problem": {
            "num_components": 30,
            "solution_capacity": 10,
            "num_objectives": 2,
        },
        "population": {
            "population_size": 50,
            "max_function_evaluations": 5000,
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
            "seed": 0,
            "num_runs": 1,
        },
    }


def run_single_experiment(
    problem,
    config: dict,
    verbose: bool = True,
) -> dict:
    """Run a single MO-SPPS experiment.

    Args:
        problem: MultiObjectiveProblem instance.
        config: Configuration dictionary.
        verbose: Whether to print progress.

    Returns:
        dict with optimizer, history, archive summary, and runtime info.
    """
    seed = config.get("experiment", {}).get("seed", 0)
    np.random.seed(seed)

    if verbose:
        print(f"Initializing MO-SPPS with seed={seed}...")

    optimizer = MOSPPSOptimizer(problem, config)
    optimizer.initialize()

    if verbose:
        print(f"  Initial population: {len(optimizer.agents)} agents")
        print(f"  Initial archive size: {len(optimizer.archive)}")
        print(f"  Max FE: {optimizer.max_fe}")
        print(f"  Pool mode: {optimizer.pool.mode}")
        print(f"  Budget mode: {optimizer.budget_mode}")

    start_time = time.perf_counter()
    history = optimizer.run()
    elapsed = time.perf_counter() - start_time

    final_archive = optimizer.archive

    # Summary
    from ..metrics import (
        compute_hypervolume,
        compute_average_jaccard_distance,
        compute_component_entropy,
        compute_pool_occupancy,
        compute_reuse_concentration,
    )

    archive_objs = final_archive.get_objectives_array()
    hv = 0.0
    if archive_objs.size > 0 and optimizer.ref_point is not None:
        hv = compute_hypervolume(archive_objs, optimizer.ref_point)

    avg_jd = compute_average_jaccard_distance(final_archive.solutions)

    occupancy = compute_pool_occupancy(optimizer.agents)
    entropy, entropy_norm = compute_component_entropy(
        occupancy, problem.num_components
    )
    concentration = compute_reuse_concentration(occupancy)

    if verbose:
        print(f"\n--- Results ---")
        print(f"  Total FE: {optimizer.fe_count}")
        print(f"  Iterations: {optimizer.iteration}")
        print(f"  Runtime: {elapsed:.2f}s")
        print(f"  Final archive size: {len(final_archive)}")
        print(f"  Hypervolume: {hv:.4f}")
        print(f"  Avg Jaccard distance: {avg_jd:.4f}")
        print(f"  Component entropy (norm): {entropy_norm:.4f}")
        print(f"  Reuse concentration: {concentration:.4f}")

    return {
        "optimizer": optimizer,
        "history": history,
        "archive": final_archive,
        "summary": {
            "fe_count": optimizer.fe_count,
            "iterations": optimizer.iteration,
            "runtime_seconds": elapsed,
            "archive_size": len(final_archive),
            "hypervolume": hv,
            "avg_jaccard_distance": avg_jd,
            "component_entropy": entropy,
            "component_entropy_norm": entropy_norm,
            "reuse_concentration": concentration,
        },
    }


def run_with_diagnostic_output(
    problem,
    config: dict,
) -> dict:
    """Run experiment with detailed per-iteration diagnostic output."""
    return run_single_experiment(problem, config, verbose=True)
