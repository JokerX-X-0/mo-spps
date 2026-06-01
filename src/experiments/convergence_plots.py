"""Convergence plots: Phase 1-4 + ablation, metrics vs FE count.

Generates 4 plots (HV, JD, Entropy, Gini) each with 10 curves.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..problems.mo_scsp import generate_moscp_instance
from ..mo_spps import MOSPPSOptimizer


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = {
    # --- Phase 1-3 ---
    "Phase1": {
        "pool_mode": "soft_pressure",
        "budget_mode": "fixed",
        "delta": 0.0,
        "gamma": 0.0,
        "kappa": 0.0,
        "use_strategy_inheritance": False,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },
    "Phase2": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "delta": 1.0,
        "gamma": 0.0,
        "kappa": 0.0,
        "use_strategy_inheritance": False,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },
    "Phase3": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "delta": 0.4,
        "gamma": 0.0,
        "kappa": 0.0,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": False,
        "use_novelty_acceptance": False,
        "use_release_operation": False,
        "prune_method": "crowding",
    },

    # --- Phase 4 Full ---
    "Phase4_Full": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "delta": 0.4,
        "gamma": 0.5,
        "kappa": 0.5,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },

    # --- Phase 4 Ablation ---
    "NoUtilityGuidance": {
        "pool_mode": "soft_pressure",
        "budget_mode": "dynamic",
        "delta": 0.4,
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
        "delta": 0.4,
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
        "delta": 0.4,
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
        "delta": 0.4,
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
        "delta": 0.4,
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
        "delta": 0.4,
        "gamma": 0.5,
        "kappa": 0.0,
        "use_strategy_inheritance": True,
        "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True,
        "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
}

# Plotting: colors + markers
STYLE = {
    "Phase1":              {"color": "#1f77b4", "linestyle": "-",  "linewidth": 2.0},
    "Phase2":              {"color": "#ff7f0e", "linestyle": "-",  "linewidth": 2.0},
    "Phase3":              {"color": "#2ca02c", "linestyle": "-",  "linewidth": 2.0},
    "Phase4_Full":         {"color": "#d62728", "linestyle": "-",  "linewidth": 3.0},
    "NoUtilityGuidance":   {"color": "#9467bd", "linestyle": "--", "linewidth": 1.5},
    "NoNoveltyAcceptance": {"color": "#8c564b", "linestyle": "--", "linewidth": 1.5},
    "NoProbAcceptance":    {"color": "#e377c2", "linestyle": "--", "linewidth": 1.5},
    "NoReleaseOp":         {"color": "#7f7f7f", "linestyle": "--", "linewidth": 1.5},
    "CrowdingOnlyPrune":   {"color": "#bcbd22", "linestyle": "--", "linewidth": 1.5},
    "HardCapPool":         {"color": "#17becf", "linestyle": "--", "linewidth": 1.5},
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
            "delta_decision_diversity": variant["delta"],
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


def _record_current_metrics(opt, elapsed, fe_hist, hv_hist, jd_hist, ent_hist, gini_hist):
    """Record a snapshot of current metrics."""
    archive_objs = opt.archive.get_objectives_array()
    hv = 0.0
    if archive_objs.size > 0 and opt.ref_point is not None:
        hv = opt.metrics_mod.compute_hypervolume(archive_objs, opt.ref_point)
    jd = opt.metrics_mod.compute_average_jaccard_distance(opt.archive.solutions)
    occ = opt.metrics_mod.compute_pool_occupancy(opt.agents)
    _, ent_norm = opt.metrics_mod.compute_component_entropy(occ, opt.num_components)
    gini = opt.metrics_mod.compute_reuse_concentration(occ)

    fe_hist.append(opt.fe_count)
    hv_hist.append(hv)
    jd_hist.append(jd)
    ent_hist.append(ent_norm)
    gini_hist.append(gini)


def run_one_variant(vname, variant, problem, num_components, solution_capacity,
                    max_fe, pop_size, seed, verbose=True):
    """Run one variant, recording metrics every ~50 FE for smooth convergence curves."""
    import time as time_mod
    from .. import metrics as metrics_mod

    if verbose:
        print(f"  Running {vname}...", end=" ", flush=True)
    config = _make_config(variant, num_components, solution_capacity,
                          max_fe, pop_size, seed)
    np.random.seed(seed)
    start = time_mod.perf_counter()
    opt = MOSPPSOptimizer(problem, config)

    # Store metrics_mod reference for use in _record_current_metrics
    opt.metrics_mod = metrics_mod

    opt.initialize()

    # Manual run loop with fine-grained metric recording
    fe_hist = []
    hv_hist = []
    jd_hist = []
    ent_hist = []
    gini_hist = []

    # Record initial state
    _record_current_metrics(opt, 0.0, fe_hist, hv_hist, jd_hist, ent_hist, gini_hist)

    last_recorded_fe = opt.fe_count
    record_interval = max(1, max_fe // 50)  # ~50 data points

    elapsed = 0.0
    while opt.fe_count < opt.max_fe:
        opt.iteration += 1

        # Step sub-methods
        opt.update_archive()
        opt.assign_ranks_and_crowding()
        opt.compute_decision_diversity()
        opt.allocate_budgets()
        if opt.pool.kappa > 0:
            opt.pool.set_component_utilities(opt._compute_component_utilities())

        # Process agents one at a time, recording every ~record_interval FE
        for agent in opt.agents:
            ops_i = int(agent.budget)
            for _ in range(ops_i):
                if opt.fe_count >= opt.max_fe:
                    break
                opt._local_construct_one(agent)
                if opt.fe_count - last_recorded_fe >= record_interval:
                    elapsed = time_mod.perf_counter() - start
                    _record_current_metrics(opt, elapsed, fe_hist, hv_hist, jd_hist,
                                           ent_hist, gini_hist)
                    last_recorded_fe = opt.fe_count

        # Elimination and rebirth
        if (opt.use_rebirth
                and opt.iteration % opt.elimination_interval == 0):
            opt.eliminate_and_rebirth()

    # Final recording
    elapsed = time_mod.perf_counter() - start
    if fe_hist[-1] != opt.fe_count:
        _record_current_metrics(opt, elapsed, fe_hist, hv_hist, jd_hist,
                               ent_hist, gini_hist)

    history = {
        "fe_count": fe_hist,
        "hypervolume": hv_hist,
        "avg_jaccard_distance": jd_hist,
        "component_entropy_norm": ent_hist,
        "reuse_concentration": gini_hist,
    }

    if verbose:
        print(f"done ({elapsed:.1f}s, {len(fe_hist)} data points)")
    return history


def generate_convergence_plots(
    problem_type="high_synergy",
    num_components=30,
    solution_capacity=10,
    max_fe=5000,
    population_size=50,
    seed=0,
    output_dir=".",
    verbose=True,
):
    """Run all variants and generate 4 convergence plots."""
    if verbose:
        print(f"\n{'#'*90}")
        print(f"#  Convergence Plot Data Collection")
        print(f"#  Problem: MOSCSP ({problem_type}), M={num_components}, K={solution_capacity}")
        print(f"#  Budget: {max_fe} FE, seed={seed}")
        print(f"{'#'*90}\n")

    problem = generate_moscp_instance(
        num_components, solution_capacity, problem_type, 1.0, seed
    )

    all_histories = {}
    for vname, variant in VARIANTS.items():
        h = run_one_variant(vname, variant, problem, num_components,
                            solution_capacity, max_fe, population_size, seed, verbose)
        all_histories[vname] = h

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 7,
        "figure.dpi": 150,
    })

    metrics = [
        ("hypervolume", "Hypervolume (HV)"),
        ("avg_jaccard_distance", "Avg Jaccard Distance (JD)"),
        ("component_entropy_norm", "Component Entropy (norm)"),
        ("reuse_concentration", "Reuse Concentration (Gini)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for ax_idx, (key, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for vname in VARIANTS:
            h = all_histories[vname]
            x = np.array(h["fe_count"])
            y = np.array(h[key])
            sty = STYLE.get(vname, {})
            ax.plot(x, y, label=vname, **sty)

        ax.set_xlabel("Function Evaluations")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.7, ncol=2)

    fig.suptitle(
        f"Convergence: MO-SPPS Phase 1–4 + Ablation\n"
        f"MOSCSP ({problem_type}), M={num_components}, K={solution_capacity}, {max_fe} FE",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    import os
    path = os.path.join(output_dir, "convergence_phases_1_4.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved to: {path}")

    return all_histories


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Generate convergence plots for Phase 1-4 + ablation"
    )
    parser.add_argument("--problem", type=str, default="high_synergy",
                        choices=["low_synergy", "high_synergy", "multi_cluster"])
    parser.add_argument("--num_components", type=int, default=30)
    parser.add_argument("--solution_capacity", type=int, default=10)
    parser.add_argument("--max_fe", type=int, default=5000)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    generate_convergence_plots(
        problem_type=args.problem,
        num_components=args.num_components,
        solution_capacity=args.solution_capacity,
        max_fe=args.max_fe,
        population_size=args.population_size,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
