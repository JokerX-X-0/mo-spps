"""NSGA-II baseline for set-based multi-objective combinatorial optimization.

Section 27.1. Standard NSGA-II adapted for discrete component selection:
  - Binary vector representation (1 = component selected)
  - Uniform crossover + bit-flip mutation with cardinality repair
  - Binary tournament selection with crowded comparison
  - Non-dominated sorting + crowding distance environmental selection
"""

import numpy as np
from ..archive import ParetoArchive
from ..agent import Agent
from ..pareto import non_dominated_sort, crowding_distance


def _solution_to_set(binary: np.ndarray) -> set[int]:
    return set(int(i) for i, v in enumerate(binary) if v > 0.5)


def _set_to_binary(sol: set[int], M: int) -> np.ndarray:
    b = np.zeros(M, dtype=float)
    for j in sol:
        b[j] = 1.0
    return b


def _repair_binary(binary: np.ndarray, K: int, values: np.ndarray | None = None) -> np.ndarray:
    """Repair binary vector to satisfy |S| <= K.

    If too many 1s, remove lowest-value ones. If values not provided, remove randomly.
    """
    ones = np.where(binary > 0.5)[0]
    if len(ones) <= K:
        return binary
    if values is not None:
        order = np.argsort(values[ones])
    else:
        order = np.random.permutation(len(ones))
    to_remove = ones[order[: len(ones) - K]]
    binary[to_remove] = 0.0
    return binary


def _uniform_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each bit independently from parent A or B."""
    mask = rng.random(len(a)) < 0.5
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1, c2


def _bitflip_mutation(binary: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation: flip each bit with probability `rate`."""
    mutant = binary.copy()
    flips = rng.random(len(binary)) < rate
    mutant[flips] = 1.0 - mutant[flips]
    return mutant


def run_nsga2(
    problem,
    max_fe: int = 15000,
    solution_capacity: int = 10,
    archive_size: int = 200,
    population_size: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.02,
    seed: int = 0,
) -> dict:
    """Run NSGA-II on a set-based multi-objective problem.

    Args:
        problem: MultiObjectiveProblem instance.
        max_fe: Maximum function evaluations.
        solution_capacity: K — max components per solution.
        archive_size: Maximum archive size.
        population_size: Population size N.
        crossover_rate: Probability of crossover.
        mutation_rate: Per-bit mutation probability.
        seed: Random seed.

    Returns:
        dict with archive, history, and metrics.
    """
    rng = np.random.default_rng(seed)
    M = problem.num_components
    K = solution_capacity
    N = population_size

    archive = ParetoArchive(max_size=archive_size)

    # Initialize population: random binary vectors repaired to cardinality
    population_bin: list[np.ndarray] = []
    population_obj: list[np.ndarray] = []
    fe_count = 0

    for _ in range(N):
        # Random solution size between 1 and K
        k = rng.integers(1, K + 1)
        ones = rng.choice(M, size=k, replace=False)
        b = np.zeros(M, dtype=float)
        b[ones] = 1.0
        sol = _solution_to_set(b)
        sol = problem.repair(sol)
        b = _set_to_binary(sol, M)
        obj = problem.evaluate(sol)
        fe_count += 1
        population_bin.append(b)
        population_obj.append(obj)

    # Initial archive
    all_objectives = list(population_obj)

    history = {
        "fe_count": [],
        "archive_size": [],
        "hypervolume": [],
        "avg_jaccard_distance": [],
    }

    from ..metrics import (
        compute_hypervolume,
        compute_average_jaccard_distance,
    )

    generation = 0
    while fe_count < max_fe:
        # Precompute ranks and crowding for tournament selection (once per generation)
        ranks = _get_ranks(population_obj)
        cd_all = crowding_distance(np.array(population_obj))

        # --- Create offspring ---
        offspring_bin: list[np.ndarray] = []
        offspring_obj: list[np.ndarray] = []

        # Tournament selection + crossover + mutation
        for _ in range(N // 2):
            # Binary tournament selection
            idx = rng.choice(N, size=4, replace=False)
            # Select parent 1
            if _crowded_less_precomputed(ranks, cd_all, idx[0], idx[1]):
                p1 = population_bin[idx[0]]
            else:
                p1 = population_bin[idx[1]]
            # Select parent 2
            if _crowded_less_precomputed(ranks, cd_all, idx[2], idx[3]):
                p2 = population_bin[idx[2]]
            else:
                p2 = population_bin[idx[3]]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = _uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = _bitflip_mutation(c1, mutation_rate, rng)
            c2 = _bitflip_mutation(c2, mutation_rate, rng)

            # Repair
            c1 = _repair_binary(c1, K)
            c2 = _repair_binary(c2, K)

            offspring_bin.extend([c1, c2])

        # Evaluate offspring
        for b in offspring_bin:
            if fe_count >= max_fe:
                break
            sol = _solution_to_set(b)
            sol = problem.repair(sol)
            b_final = _set_to_binary(sol, M)
            obj = problem.evaluate(sol)
            fe_count += 1
            offspring_obj.append(obj)
            all_objectives.append(obj)
            # Update the binary in place
            idx = len(offspring_obj) - 1
            offspring_bin[idx] = b_final

        # --- Environmental selection (N from N + N_offspring) ---
        combined_bin = population_bin + offspring_bin[:len(offspring_obj)]
        combined_obj = population_obj + offspring_obj

        # Non-dominated sort
        fronts = non_dominated_sort(combined_obj)

        # Build new population from best fronts
        new_pop_bin = []
        new_pop_obj = []
        for front in fronts:
            if len(new_pop_bin) + len(front) <= N:
                for idx in front:
                    new_pop_bin.append(combined_bin[idx])
                    new_pop_obj.append(combined_obj[idx])
            else:
                # Compute crowding distance for this front
                front_objs = np.array([combined_obj[i] for i in front])
                cd = crowding_distance(front_objs)
                # Take best by crowding distance
                remaining = N - len(new_pop_bin)
                sorted_by_cd = np.argsort(-cd)[:remaining]
                for j in sorted_by_cd:
                    idx = front[j]
                    new_pop_bin.append(combined_bin[idx])
                    new_pop_obj.append(combined_obj[idx])
                break

        population_bin = new_pop_bin
        population_obj = new_pop_obj

        # Batch update archive (once per generation)
        agents = []
        for b, obj in zip(population_bin, population_obj):
            sol = _solution_to_set(b)
            agents.append(Agent(
                solution=sol,
                objectives=obj,
                component_preference=np.ones(1),
                objective_preference=np.ones(1),
            ))
        archive.update(agents)

        # Record metrics
        generation += 1
        if generation % max(1, max_fe // (N * 2 * 50)) == 0 or fe_count >= max_fe:
            problem.update_ideal_nadir(all_objectives)
            arch_objs = archive.get_objectives_array()

            hv = 0.0
            if arch_objs.size > 0:
                nadir = problem.nadir_point
                if nadir is not None:
                    ref = nadir - np.abs(nadir) * 0.1 - 1.0
                else:
                    ref = np.min(arch_objs, axis=0) - 1.0
                hv = compute_hypervolume(arch_objs, ref)

            history["fe_count"].append(fe_count)
            history["archive_size"].append(len(archive))
            history["hypervolume"].append(hv)
            history["avg_jaccard_distance"].append(
                compute_average_jaccard_distance(archive.solutions)
            )

    return {
        "archive": archive,
        "history": history,
        "final_fe": fe_count,
    }


def _crowded_less_precomputed(
    ranks: np.ndarray,
    cd: np.ndarray,
    idx_a: int,
    idx_b: int,
) -> bool:
    """Crowded comparison using precomputed ranks and crowding distances.

    Returns True if a is better than b (lower rank, or same rank + higher crowding).
    """
    rank_a = ranks[idx_a]
    rank_b = ranks[idx_b]
    if rank_a < rank_b:
        return True
    if rank_b < rank_a:
        return False
    return cd[idx_a] > cd[idx_b]


def _get_ranks(objectives: list[np.ndarray]) -> np.ndarray:
    fronts = non_dominated_sort(objectives)
    n = len(objectives)
    ranks = np.ones(n, dtype=int) * len(fronts)
    for l, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = l + 1
    return ranks
