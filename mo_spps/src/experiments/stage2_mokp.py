"""Stage 2 experiment: validate MO-SPPS on MOKP (standard benchmark).

Compares 5 MO-SPPS variants against NSGA-II, SPEA2, MOEA/D, NSGA-II+Div, and PLS
on the Multi-Objective Knapsack Problem.

Section 16, Stage 2.
"""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..problems.mo_kp import generate_mokp_instance
from ..problems.mo_scsp import generate_moscp_instance
from ..mo_spps import MOSPPSOptimizer
from ..baselines import run_nsga2, run_moead
from ..metrics import (
    compute_hypervolume,
    compute_average_jaccard_distance,
    compute_component_entropy,
    compute_pool_occupancy,
    compute_reuse_concentration,
)

# ── MO-SPPS variant configs (from phase4_ablation.py) ──────────────────────────
MO_SPPS_VARIANTS = {
    "MO_SPPS_Full": {
        "pool_mode": "continuous", "budget_mode": "dynamic",
        "gamma": 0.5, "kappa": 0.5,
        "use_strategy_inheritance": True, "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True, "use_release_operation": True,
        "prune_method": "hybrid_objective_decision", "use_adaptive_Q": False,
    },
    "MO_SPPS_NoPool": {
        "pool_mode": "none", "budget_mode": "dynamic",
        "gamma": 0.5, "kappa": 0.5,
        "use_strategy_inheritance": True, "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True, "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_OldSoftPool": {
        "pool_mode": "soft_pressure", "budget_mode": "dynamic",
        "gamma": 0.5, "kappa": 0.5,
        "use_strategy_inheritance": True, "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True, "use_release_operation": True,
        "prune_method": "hybrid_objective_decision",
    },
    "MO_SPPS_ContinuousPool": {
        "pool_mode": "continuous", "budget_mode": "dynamic",
        "gamma": 0.5, "kappa": 0.5,
        "use_strategy_inheritance": True, "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True, "use_release_operation": True,
        "prune_method": "hybrid_objective_decision", "use_adaptive_Q": False,
    },
    "MO_SPPS_AdaptiveQ": {
        "pool_mode": "continuous", "budget_mode": "dynamic",
        "gamma": 0.5, "kappa": 0.5,
        "use_strategy_inheritance": True, "use_probabilistic_acceptance": True,
        "use_novelty_acceptance": True, "use_release_operation": True,
        "prune_method": "hybrid_objective_decision", "use_adaptive_Q": True,
    },
}


# ── Config builder (mirrors phase4_ablation._make_config) ──────────────────────
def _make_config(variant, num_components, max_fe, pop_size, seed, capacity_q0=12):
    use_adaptive_q = variant.get("use_adaptive_Q", False)
    delta_decision = variant.get("delta_decision_diversity", 0.4)
    elim_interval = variant.get("elimination_interval", 3)
    base_budget = variant.get("base_budget", 2.0)

    return {
        "problem": {
            "num_components": num_components,
            "solution_capacity": num_components,
            "num_objectives": 2,
        },
        "population": {
            "population_size": pop_size,
            "max_function_evaluations": max_fe,
        },
        "shared_pool": {
            "mode": variant["pool_mode"],
            "epsilon": 0.01, "tau": 1.0,
            "utility_guidance_kappa": variant["kappa"],
            "base_capacity_Q0": capacity_q0,
        },
        "local_search": {
            "shop_size": 5,
            "use_probabilistic_acceptance": variant["use_probabilistic_acceptance"],
            "use_novelty_acceptance": variant["use_novelty_acceptance"],
            "use_release_operation": variant["use_release_operation"],
            "temperature": 1.0, "archive_contribution_threshold": 0.0,
            "novelty_threshold": 0.3, "quality_loss_threshold": 0.02,
        },
        "budget": {
            "mode": variant["budget_mode"],
            "base_budget": base_budget,
            "alpha_pareto": 1.0, "beta_crowding": 1.0,
            "delta_decision_diversity": delta_decision,
            "gamma_exploration": variant["gamma"],
        },
        "rebirth": {
            "use_rebirth": True,
            "use_strategy_inheritance": variant["use_strategy_inheritance"],
            "elimination_interval": elim_interval,
            "replacement_rate": 0.2, "inheritance_strength": 0.5,
            "inheritance_smoothing": 0.1, "preference_learning_rate": 0.01,
            "keep_reference_direction": True,
            "retention_a": 0.5, "retention_b": 0.3, "retention_d": 0.2,
        },
        "archive": {
            "max_size": 200,
            "prune_method": variant["prune_method"],
            "objective_weight": 0.7, "decision_weight": 0.3,
            "remove_duplicates": True,
        },
        "reference_directions": {"n_directions": 50},
        "adaptive_capacity": {
            "use_adaptive_Q": use_adaptive_q,
            "alpha_Q": 2.0, "base_capacity_Q0": capacity_q0,
            "Q_min": 3, "Q_max": 30, "update_interval": 2,
        },
        "region_novelty": {"enabled": False, "region_threshold": 0.3},
        "experiment": {"seed": seed},
    }


# ── Summary helpers ────────────────────────────────────────────────────────────
def _summary_from_archive_solutions(archive_solutions, archive_objs, problem,
                                     fe_count, elapsed, agents=None,
                                     ref_point=None):
    hv = 0.0
    if archive_objs.size > 0:
        ref = ref_point if ref_point is not None else np.min(archive_objs, axis=0) - 1.0
        hv = compute_hypervolume(archive_objs, ref)
    avg_jd = float(compute_average_jaccard_distance(list(archive_solutions)))

    occupancy = {}
    for sol in archive_solutions:
        for j in sol:
            occupancy[j] = occupancy.get(j, 0) + 1
    if agents is not None:
        pool_occ = compute_pool_occupancy(agents)
    else:
        pool_occ = occupancy

    n = problem.num_components
    entropy = 0.0
    entropy_norm = 0.0
    if occupancy:
        total = sum(occupancy.values())
        freqs = np.array(list(occupancy.values())) / max(total, 1)
        freqs = freqs[freqs > 0]
        entropy = float(-np.sum(freqs * np.log(freqs)))
        if n > 1:
            entropy_norm = entropy / np.log(n)

    gini = 0.0
    if occupancy:
        vals = list(occupancy.values())
        gini = 1.0 - sum(vals) / max(1, len(vals) * max(vals)) if max(vals) > 0 else 0.0

    return {
        "fe_count": int(fe_count), "runtime_seconds": elapsed,
        "archive_size": len(archive_solutions),
        "hypervolume": float(hv),
        "avg_jaccard_distance": avg_jd,
        "component_entropy_norm": float(entropy_norm),
        "reuse_concentration": float(gini),
        "_objs": archive_objs.copy() if isinstance(archive_objs, np.ndarray) and archive_objs.size > 0 else np.zeros((0, 2)),
    }


# ── MO-SPPS runner ─────────────────────────────────────────────────────────────
def _run_mo_spps_variant(args):
    vname, variant, n_items, capacity_ratio, max_fe, pop_size, seed, capacity_q0 = args

    problem = generate_mokp_instance(n_items, capacity_ratio, 2, seed)
    config = _make_config(variant, n_items, max_fe, pop_size, seed, capacity_q0)
    np.random.seed(seed)
    start = time.perf_counter()
    opt = MOSPPSOptimizer(problem, config)
    opt.initialize()
    opt.run()
    elapsed = time.perf_counter() - start

    arch_objs = opt.archive.get_objectives_array()
    s = _summary_from_archive_solutions(
        opt.archive.solutions, arch_objs, problem, opt.fe_count, elapsed, opt.agents
    )
    return vname, s


# ── Pymoo-based baseline runners ────────────────────────────────────────────────
def _mokp_to_pymoo_problem(problem):
    """Convert our MOKP to a pymoo Problem (binary vector representation)."""
    from pymoo.core.problem import Problem as PymooProblem

    class MOKPProblem(PymooProblem):
        def __init__(self):
            super().__init__(
                n_var=problem.num_items,
                n_obj=problem.num_objectives,
                n_ieq_constr=0,
                xl=0, xu=1,
                vtype=int,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            n = x.shape[0]
            f = np.zeros((n, self.n_obj))
            for i in range(n):
                selected = set(np.where(x[i] == 1)[0])
                sol = problem.repair(selected)
                f[i] = problem.evaluate(sol)
            out["F"] = -f

    return MOKPProblem()


def _run_pymoo_baseline(algorithm_name, problem, max_fe, pop_size, seed):
    """Run a pymoo algorithm (NSGA2, SPEA2, MOEAD) on MOKP."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.pntx import SinglePointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling

    pymoo_problem = _mokp_to_pymoo_problem(problem)

    if algorithm_name == "NSGA-II":
        algo = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=1.0 / problem.num_items),
            eliminate_duplicates=True,
        )
    elif algorithm_name == "SPEA2":
        algo = SPEA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=1.0 / problem.num_items),
            eliminate_duplicates=False,
        )
    elif algorithm_name == "MOEA/D":
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1)
        algo = MOEAD(
            ref_dirs,
            n_neighbors=20,
            prob_neighbor_mating=0.9,
            sampling=BinaryRandomSampling(),
            crossover=SinglePointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=1.0 / problem.num_items),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    start = time.perf_counter()
    res = minimize(
        pymoo_problem,
        algo,
        ("n_eval", max_fe),
        seed=seed,
        verbose=False,
    )
    elapsed = time.perf_counter() - start

    # Convert pymoo result to our format
    objs = -res.F  # undo negation (MOKP maximizes)
    if objs.ndim == 1:
        objs = objs.reshape(1, -1)

    # Convert binary vectors to sets
    solutions = []
    for row in res.X:
        if row.ndim == 0:
            row = np.array([row])
        selected = set(int(i) for i, v in enumerate(row) if v > 0.5)
        repaired = problem.repair(selected)
        solutions.append(repaired)

    # Get non-dominated solutions
    from ..pareto import non_dominated_sort
    fronts = non_dominated_sort([-obj for obj in objs])
    nd_idx = fronts[0] if fronts else list(range(len(objs)))
    final_objs = objs[nd_idx]
    final_sols = [solutions[i] for i in nd_idx]

    fe_used = max_fe
    if res.history:
        fe_used = res.history[-1].evaluator.n_eval if hasattr(res.history[-1], 'evaluator') else max_fe
    s = _summary_from_archive_solutions(
        final_sols, final_objs, problem, fe_used, elapsed
    )
    # Override FE count with max_fe since history tracking may differ
    s["fe_count"] = max_fe
    return s


# ── Pareto Local Search ─────────────────────────────────────────────────────────
def _run_pls(problem, max_fe, seed):
    """Simple Pareto Local Search for MOKP.

    Algorithm:
      1. Generate random starting solutions
      2. For each, try all 1-bit and 2-bit flips
      3. Accept improvements (dominating or non-dominated)
      4. Maintain Pareto archive
    """
    rng = np.random.default_rng(seed)
    n = problem.num_items
    start = time.perf_counter()

    archive_sols = []
    archive_objs = []

    def _add_to_archive(sol, obj):
        nonlocal archive_sols, archive_objs
        # Check if dominated by or equal to existing
        for a_obj in archive_objs:
            if np.all(a_obj >= obj) and np.any(a_obj > obj):
                return False  # dominated
        # Remove dominated
        keep_idx = []
        for i, a_obj in enumerate(archive_objs):
            if not (np.all(obj >= a_obj) and np.any(obj > a_obj)):
                keep_idx.append(i)
        archive_sols = [archive_sols[i] for i in keep_idx]
        archive_objs = [archive_objs[i] for i in keep_idx]
        archive_sols.append(sol)
        archive_objs.append(obj)
        return True

    fe_count = 0
    n_restarts = max(1, max_fe // (n * 2))  # estimate

    for _ in range(n_restarts):
        if fe_count >= max_fe:
            break
        # Random starting solution
        k = rng.integers(1, min(n, 50))
        items = rng.choice(n, size=k, replace=False)
        current = set(int(x) for x in items)
        current = problem.repair(current)
        current_obj = problem.evaluate(current)
        fe_count += 1
        _add_to_archive(current, current_obj)

        improved = True
        while improved and fe_count < max_fe:
            improved = False
            # Try 1-bit flips
            candidates = []
            for j in range(n):
                if fe_count >= max_fe:
                    break
                trial = current.copy()
                if j in trial:
                    trial.discard(j)
                else:
                    trial.add(j)
                trial = problem.repair(trial)
                trial_obj = problem.evaluate(trial)
                fe_count += 1
                candidates.append((trial, trial_obj))

            # Accept best non-dominated candidate
            best_added = False
            for trial, trial_obj in candidates:
                if np.all(trial_obj >= current_obj) and np.any(trial_obj > current_obj):
                    if _add_to_archive(trial, trial_obj):
                        current = trial
                        current_obj = trial_obj
                        improved = True
                        best_added = True
                elif not np.all(current_obj >= trial_obj):
                    # Non-dominated
                    if _add_to_archive(trial, trial_obj):
                        if not best_added:
                            current = trial
                            current_obj = trial_obj
                            improved = True
                            best_added = True

            if best_added:
                break  # restart with new current
            break  # no improvement found

    elapsed = time.perf_counter() - start
    if archive_objs:
        archive_objs = np.array(archive_objs)
    else:
        archive_objs = np.zeros((0, 2))

    return _summary_from_archive_solutions(
        archive_sols, archive_objs, problem, fe_count, elapsed
    )


# ── NSGA-II + Decision Diversity ────────────────────────────────────────────────
def _run_nsga2_div(problem, max_fe, pop_size, seed):
    """NSGA-II augmented with decision-space diversity in crowding distance."""
    from .. import pareto

    rng = np.random.default_rng(seed)
    n = problem.num_items
    start = time.perf_counter()

    # Initialize population
    pop = []
    for _ in range(pop_size):
        k = rng.integers(1, min(n, 50))
        sol = frozenset(int(x) for x in rng.choice(n, size=k, replace=False))
        sol = problem.repair(set(sol))
        pop.append(sol)

    eval_count = pop_size
    archive_sols = []
    archive_objs = []

    while eval_count < max_fe:
        # Evaluate
        objs = np.array([problem.evaluate(s) for s in pop])

        # Update archive
        for i, s in enumerate(pop):
            dominated = False
            for a_obj in archive_objs:
                if np.all(a_obj >= objs[i]) and np.any(a_obj > objs[i]):
                    dominated = True
                    break
            if not dominated:
                keep_idx = [j for j, a_obj in enumerate(archive_objs)
                           if not (np.all(objs[i] >= a_obj) and np.any(objs[i] > a_obj))]
                archive_sols = [archive_sols[j] for j in keep_idx]
                archive_objs = [archive_objs[j] for j in keep_idx]
                archive_sols.append(s.copy())
                archive_objs.append(objs[i].copy())

        if eval_count >= max_fe:
            break

        # Non-dominated sorting
        fronts = pareto.non_dominated_sort([o for o in objs])

        # Crowding distance with decision diversity
        all_cd = np.zeros(pop_size)
        for front in fronts:
            f_idx = np.array(front)
            f_obj = objs[f_idx]
            cd = np.zeros(len(f_idx))
            if len(f_idx) > 1:
                for o in range(objs.shape[1]):
                    order = np.argsort(f_obj[:, o])
                    sorted_idx = f_idx[order]
                    if len(f_idx) > 2:
                        cd[order[0]] = np.inf
                        cd[order[-1]] = np.inf
                        obj_range = objs[sorted_idx[-1], o] - objs[sorted_idx[0], o]
                        if obj_range > 1e-12:
                            for k in range(1, len(f_idx) - 1):
                                cd[order[k]] += (objs[sorted_idx[k+1], o] -
                                                  objs[sorted_idx[k-1], o]) / obj_range

                # Decision diversity bonus
                for ki, idx in enumerate(f_idx):
                    sol = pop[int(idx)]
                    others = rng.choice(pop_size, size=min(5, pop_size), replace=False)
                    div_bonus = 0.0
                    n_others = 0
                    for oi in others:
                        if oi == int(idx):
                            continue
                        intersect = len(sol & pop[int(oi)])
                        union = len(sol | pop[int(oi)])
                        if union > 0:
                            div_bonus += 1.0 - intersect / union
                        n_others += 1
                    if n_others > 0:
                        cd[ki] += 0.4 * div_bonus / n_others

            all_cd[f_idx] = cd

        # Precompute ranks for tournament selection
        rank_of = np.full(pop_size, len(fronts), dtype=int)
        for rk, f in enumerate(fronts):
            for idx in f:
                rank_of[idx] = rk

        # Tournament selection
        parents = []
        for _ in range(pop_size):
            i, j = int(rng.integers(0, pop_size)), int(rng.integers(0, pop_size))
            if rank_of[i] < rank_of[j]:
                parents.append(pop[i])
            elif rank_of[j] < rank_of[i]:
                parents.append(pop[j])
            elif all_cd[i] > all_cd[j]:
                parents.append(pop[i])
            else:
                parents.append(pop[j])

        # Crossover + mutation
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[min(i+1, pop_size-1)]
            if rng.random() < 0.9:
                all_items = list(p1 | p2)
                rng.shuffle(all_items)
                split = rng.integers(0, len(all_items))
                c1 = set(all_items[:split])
                c2 = set(all_items[split:])
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            for sol in (c1, c2):
                for j in range(n):
                    if rng.random() < 1.0 / n:
                        if j in sol:
                            sol.discard(j)
                        else:
                            sol.add(j)

            c1 = problem.repair(c1)
            c2 = problem.repair(c2)
            offspring.extend([c1, c2])

        pop = offspring[:pop_size]
        eval_count += pop_size

    elapsed = time.perf_counter() - start
    if archive_objs:
        archive_objs = np.array(archive_objs)
    else:
        archive_objs = np.zeros((0, 2))

    return _summary_from_archive_solutions(
        archive_sols, archive_objs, problem, eval_count, elapsed
    )


# ── Custom NSGA-II runner (reuse existing baselines module) ─────────────────────
def _run_nsga2_custom(problem, max_fe, pop_size, seed):
    """Run our existing NSGA-II baseline."""
    start = time.perf_counter()
    result = run_nsga2(
        problem=problem, max_fe=max_fe, solution_capacity=problem.num_items,
        archive_size=200, population_size=pop_size, crossover_rate=0.9,
        mutation_rate=1.0 / problem.num_items, seed=seed,
    )
    elapsed = time.perf_counter() - start

    archive = result["archive"]
    arch_objs = archive.get_objectives_array()
    return _summary_from_archive_solutions(
        archive.solutions, arch_objs, problem,
        result.get("final_fe", max_fe), elapsed
    )


# ── Job dispatcher (module-level for multiprocessing) ────────────────────────────
def _run_stage2_job(job):
    kind, name, run_seed, variant, n_items, cap_ratio, max_fe, pop_size, cap_q0 = job

    if kind == "mo_spps":
        args = (name, variant, n_items, cap_ratio, max_fe, pop_size, run_seed, cap_q0)
        vname, s = _run_mo_spps_variant(args)
        return vname, s
    else:
        problem = generate_mokp_instance(n_items, cap_ratio, 2, run_seed)
        if name == "NSGA-II":
            s = _run_nsga2_custom(problem, max_fe, pop_size, run_seed)
        elif name in ("SPEA2", "MOEA/D"):
            s = _run_pymoo_baseline(name, problem, max_fe, pop_size, run_seed)
        elif name == "NSGA-II+Div":
            s = _run_nsga2_div(problem, max_fe, pop_size, run_seed)
        elif name == "PLS":
            s = _run_pls(problem, max_fe, run_seed)
        else:
            raise ValueError(f"Unknown baseline: {name}")
        return name, s


# ── Main experiment ─────────────────────────────────────────────────────────────
def run_stage2_mokp(
    n_items=100,
    capacity_ratio=0.5,
    max_fe=20000,
    population_size=100,
    seed=0,
    n_runs=30,
    workers=1,
    verbose=True,
    capacity_q0=12,
):
    print(f"\n{'#'*90}")
    print(f"#  Stage 2: MO-SPPS on MOKP (Standard Benchmark)")
    print(f"#  n={n_items}, W/Σw={capacity_ratio}, FE={max_fe}, runs={n_runs}")
    print(f"{'#'*90}")

    # Build all job tuples: (algorithm_name, run_idx, seed, ...)
    jobs = []

    # MO-SPPS variants
    for vname, variant in MO_SPPS_VARIANTS.items():
        for run in range(n_runs):
            run_seed = seed + run
            jobs.append(("mo_spps", vname, run_seed, variant, n_items, capacity_ratio,
                         max_fe, population_size, capacity_q0))

    # Baselines (each run with different seed)
    for run in range(n_runs):
        run_seed = seed + run
        jobs.append(("baseline", "NSGA-II", run_seed, None, n_items, capacity_ratio,
                     max_fe, population_size, capacity_q0))
        jobs.append(("baseline", "SPEA2", run_seed, None, n_items, capacity_ratio,
                     max_fe, population_size, capacity_q0))
        jobs.append(("baseline", "MOEA/D", run_seed, None, n_items, capacity_ratio,
                     max_fe, population_size, capacity_q0))
        jobs.append(("baseline", "NSGA-II+Div", run_seed, None, n_items, capacity_ratio,
                     max_fe, population_size, capacity_q0))
        jobs.append(("baseline", "PLS", run_seed, None, n_items, capacity_ratio,
                     max_fe, population_size, capacity_q0))

    total = len(jobs)
    print(f"  Total jobs: {total} ({len(MO_SPPS_VARIANTS)} MO-SPPS + 5 baselines) × {n_runs}")
    if workers > 1:
        print(f"  Workers: {workers}")

    all_results = {}
    count = 0

    if workers <= 1:
        for job in jobs:
            count += 1
            name, s = _run_stage2_job(job)
            all_results.setdefault(name, []).append(s)
            if verbose:
                print(f"  [{count}/{total}] {name} HV={s['hypervolume']:.2f} "
                      f"|A|={s['archive_size']} JD={s['avg_jaccard_distance']:.3f}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_stage2_job, j): j for j in jobs}
            for future in as_completed(futures):
                count += 1
                name, s = future.result()
                all_results.setdefault(name, []).append(s)
                if verbose:
                    print(f"  [{count}/{total}] {name} HV={s['hypervolume']:.2f} "
                          f"|A|={s['archive_size']} JD={s['avg_jaccard_distance']:.3f}")

    # Aggregate — first find global reference point for HV
    all_objs_for_ref = []
    for name, runs_data in all_results.items():
        for r in runs_data:
            if "_objs" in r and r["_objs"].size > 0:
                all_objs_for_ref.append(r["_objs"])
    if all_objs_for_ref:
        global_min = np.min(np.vstack(all_objs_for_ref), axis=0)
        global_ref = global_min - 1.0
    else:
        global_ref = np.array([0.0, 0.0])
    print(f"  Global HV reference: {global_ref}")

    # Recompute HV for all results with unified reference
    for name, runs_data in all_results.items():
        for r in runs_data:
            if "_objs" in r and r["_objs"].size > 0:
                r["hypervolume"] = float(compute_hypervolume(r["_objs"], global_ref))

    metrics_keys = ["archive_size", "hypervolume", "avg_jaccard_distance",
                    "component_entropy_norm", "reuse_concentration", "runtime_seconds"]

    algo_order = [
        "MO_SPPS_Full", "MO_SPPS_NoPool", "MO_SPPS_OldSoftPool",
        "MO_SPPS_ContinuousPool", "MO_SPPS_AdaptiveQ",
        "NSGA-II", "SPEA2", "MOEA/D", "NSGA-II+Div", "PLS",
    ]

    print(f"\n{'='*90}")
    print(f"  Stage 2 Results: MOKP (n={n_items}, W/Σw={capacity_ratio}, "
          f"FE={max_fe}, {n_runs} runs)")
    print(f"{'='*90}")
    header = f"{'Algorithm':<24} {'|A|':>6} {'HV':>10} {'Jaccard':>9} {'Entropy':>9} {'Gini':>9} {'Time(s)':>8}"
    print(header)
    print("-" * 90)

    aggregated = {}
    for name in algo_order:
        if name not in all_results:
            continue
        runs_data = all_results[name]
        agg = {k: np.mean([r[k] for r in runs_data]) for k in metrics_keys}
        aggregated[name] = agg

        row = f"{name:<24}"
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

    # Key comparisons
    if "MO_SPPS_Full" in aggregated and "NSGA-II" in aggregated:
        full = aggregated["MO_SPPS_Full"]
        nsga2 = aggregated["NSGA-II"]
        print(f"\n  MO_SPPS_Full vs NSGA-II:")
        print(f"    HV delta:     {full['hypervolume'] - nsga2['hypervolume']:+.3f}")
        print(f"    Jaccard delta: {full['avg_jaccard_distance'] - nsga2['avg_jaccard_distance']:+.4f}")

    if "MO_SPPS_NoPool" in aggregated and "MO_SPPS_Full" in aggregated:
        np_agg = aggregated["MO_SPPS_NoPool"]
        print(f"\n  Pool mechanism contribution (Full vs NoPool):")
        print(f"    HV delta:     {full['hypervolume'] - np_agg['hypervolume']:+.3f}")
        print(f"    Jaccard delta: {full['avg_jaccard_distance'] - np_agg['avg_jaccard_distance']:+.4f}")

    print(f"{'='*90}\n")

    return {"all_results": all_results, "aggregated": aggregated}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: MOKP benchmark comparison")
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--capacity_ratio", type=float, default=0.5)
    parser.add_argument("--max_fe", type=int, default=20000)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--capacity_q0", type=int, default=12)
    args = parser.parse_args()

    run_stage2_mokp(
        n_items=args.n_items,
        capacity_ratio=args.capacity_ratio,
        max_fe=args.max_fe,
        population_size=args.population_size,
        seed=args.seed,
        n_runs=args.n_runs,
        workers=args.workers,
        verbose=not args.quiet,
        capacity_q0=args.capacity_q0,
    )
