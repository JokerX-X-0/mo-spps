"""MOEA/D baseline for set-based multi-objective combinatorial optimization.

Section 27.1. MOEA/D with Tchebycheff decomposition adapted for discrete
component selection:
  - Set representation (set of component indices)
  - Tchebycheff scalarization: g(x|λ,z*) = max λ_k * (z*_k - f_k(x))
  - Neighborhood-based reproduction (set-intersection crossover + mutation)
  - External Pareto archive
"""

import numpy as np
from ..archive import ParetoArchive
from ..agent import Agent
from ..reference_directions import generate_das_dennis_directions


def _tchebycheff(obj: np.ndarray, weight: np.ndarray, ideal: np.ndarray) -> float:
    """Tchebycheff scalarization for maximization.

    g(x|λ,z*) = max_k λ_k * (z*_k - f_k(x))

    Lower is better. λ_k=0 objectives are ignored.
    """
    diff = ideal - obj
    weighted = weight * diff
    weighted[weight < 1e-12] = 0.0
    if np.all(weight < 1e-12):
        return np.max(diff)
    return float(np.max(weighted))


def _set_crossover(a: set[int], b: set[int], K: int, rng: np.random.Generator) -> set[int]:
    """Set-based crossover: keep intersection, fill from symmetric difference."""
    inter = a & b
    sym_diff = list((a ^ b) - inter)
    rng.shuffle(sym_diff)
    offspring = set(inter)
    needed = K - len(offspring)
    if needed > 0 and sym_diff:
        offspring.update(sym_diff[:needed])
    if len(offspring) == 0:
        offspring = set(rng.choice(list(a | b), size=min(K, len(a | b)), replace=False))
    return offspring


def _set_mutation(
    sol: set[int], M: int, K: int, rate: float, rng: np.random.Generator
) -> set[int]:
    """Mutate by removing one component and adding a new random one."""
    mutant = set(sol)
    if rng.random() < rate and len(mutant) > 0:
        to_remove = rng.choice(list(mutant))
        mutant.remove(to_remove)
    if rng.random() < rate:
        available = list(set(range(M)) - mutant)
        if available:
            mutant.add(int(rng.choice(available)))
    # Repair to capacity
    if len(mutant) > K:
        mutant = set(rng.choice(list(mutant), size=K, replace=False))
    return mutant


def run_moead(
    problem,
    max_fe: int = 15000,
    solution_capacity: int = 10,
    archive_size: int = 200,
    population_size: int = 100,
    neighborhood_size: int = 20,
    mutation_rate: float = 0.1,
    seed: int = 0,
) -> dict:
    """Run MOEA/D on a set-based multi-objective problem.

    Args:
        problem: MultiObjectiveProblem instance.
        max_fe: Maximum function evaluations.
        solution_capacity: K — max components per solution.
        archive_size: Maximum archive size.
        population_size: Population size N.
        neighborhood_size: T — number of neighbors per subproblem.
        mutation_rate: Probability of mutation per offspring.
        seed: Random seed.

    Returns:
        dict with archive, history, and metrics.
    """
    rng = np.random.default_rng(seed)
    M_comp = problem.num_components
    K = solution_capacity
    N = population_size
    T = min(neighborhood_size, N)
    n_obj = problem.num_objectives

    archive = ParetoArchive(max_size=archive_size)

    # Generate weight vectors
    if n_obj == 2:
        weights = np.array([[i / (N - 1), 1.0 - i / (N - 1)] for i in range(N)])
    else:
        weights = generate_das_dennis_directions(n_obj, n_partitions=max(2, int(N ** (1.0 / (n_obj - 1)))))

    if len(weights) < N:
        extra = rng.dirichlet(np.ones(n_obj), size=N - len(weights))
        weights = np.vstack([weights, extra]) if len(weights) > 0 else extra
    weights = weights[:N]

    # Neighborhood: T closest weight vectors by Euclidean distance
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = np.linalg.norm(weights[i] - weights[j])
    neighbors = np.argsort(dist_matrix, axis=1)[:, :T]

    # Initialize population
    population: list[set[int]] = []
    population_obj: list[np.ndarray] = []
    ideal = np.full(n_obj, -np.inf)

    for i in range(N):
        k = rng.integers(1, K + 1)
        sol = set(rng.choice(M_comp, size=k, replace=False))
        sol = problem.repair(sol)
        obj = problem.evaluate(sol)
        population.append(sol)
        population_obj.append(obj)
        ideal = np.maximum(ideal, obj)

    fe_count = N
    all_objectives = list(population_obj)

    history = {
        "fe_count": [],
        "archive_size": [],
        "hypervolume": [],
        "avg_jaccard_distance": [],
    }

    from ..metrics import compute_hypervolume, compute_average_jaccard_distance

    # Main loop
    while fe_count < max_fe:
        gen_agents: list[Agent] = []
        for i in range(N):
            if fe_count >= max_fe:
                break

            # Select two parents from neighborhood
            if T >= 2:
                p1_idx, p2_idx = rng.choice(neighbors[i], size=2, replace=False)
            else:
                p1_idx = neighbors[i][0]
                p2_idx = p1_idx

            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # Crossover
            offspring = _set_crossover(parent1, parent2, K, rng)

            # Mutation
            offspring = _set_mutation(offspring, M_comp, K, mutation_rate, rng)

            # Repair
            offspring = problem.repair(offspring)
            if len(offspring) == 0:
                k = rng.integers(1, K + 1)
                offspring = set(rng.choice(M_comp, size=k, replace=False))

            y_obj = problem.evaluate(offspring)
            fe_count += 1
            all_objectives.append(y_obj)

            # Update ideal point
            ideal = np.maximum(ideal, y_obj)

            # Update neighboring solutions
            for j in neighbors[i]:
                g_y = _tchebycheff(y_obj, weights[j], ideal)
                g_xj = _tchebycheff(population_obj[j], weights[j], ideal)
                if g_y <= g_xj:
                    population[j] = offspring.copy()
                    population_obj[j] = y_obj.copy()

            # Collect for batch archive update
            gen_agents.append(Agent(
                solution=offspring,
                objectives=y_obj,
                component_preference=np.ones(1),
                objective_preference=np.ones(1),
            ))

            if fe_count >= max_fe:
                break

        # Batch update archive (once per generation)
        if gen_agents:
            archive.update(gen_agents)

        # Record metrics periodically
        gen = fe_count // N
        if gen % max(1, max_fe // (N * 50)) == 0 or fe_count >= max_fe:
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
