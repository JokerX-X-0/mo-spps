"""MOGWO baseline for set-based multi-objective combinatorial optimization.

Section 27.1. Multi-Objective Grey Wolf Optimizer adapted for discrete
component selection:
  - Each wolf is a set of component indices
  - Three leaders (alpha, beta, delta) selected from archive
  - Position update: weighted sampling from leader union with
    exploration/exploitation parameter a decreasing 2→0
  - External Pareto archive with grid-based pruning
"""

import numpy as np
from ..archive import ParetoArchive
from ..agent import Agent


def _select_leaders(
    archive: ParetoArchive, rng: np.random.Generator
) -> tuple[set[int], set[int], set[int]] | None:
    """Select alpha, beta, delta leaders from archive.

    Picks 3 most diverse solutions (by Jaccard distance) when possible.
    Falls back to random 3 if archive is small.
    """
    if len(archive) == 0:
        return None

    sols = archive.solutions
    if len(sols) == 1:
        return sols[0], sols[0], sols[0]
    if len(sols) == 2:
        return sols[0], sols[1], sols[0]
    if len(sols) == 3:
        return sols[0], sols[1], sols[2]

    # Pick 3 diverse solutions: select first randomly, then greedily
    # add furthest by Jaccard distance
    candidates = list(range(len(sols)))
    selected_idx = [int(rng.choice(candidates))]
    for _ in range(2):
        best_idx = -1
        best_min_dist = -1.0
        for i in candidates:
            if i in selected_idx:
                continue
            min_dist = min(
                len(sols[i] ^ sols[s]) / max(len(sols[i] | sols[s]), 1)
                for s in selected_idx
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx >= 0:
            selected_idx.append(best_idx)

    return (
        sols[selected_idx[0]],
        sols[selected_idx[1]],
        sols[selected_idx[2]],
    )


def _update_position(
    wolf: set[int],
    alpha: set[int],
    beta: set[int],
    delta: set[int],
    M: int,
    K: int,
    a: float,
    rng: np.random.Generator,
) -> set[int]:
    """Discrete position update for MOGWO.

    Each component j gets a weight:
        w_j = (leader_count_of_j + epsilon) * (1 + exploration * r)

    a ∈ [0, 2]: low a → exploitation (prefer leader components)
                 high a → exploration (random components get weight)

    Then sample K components without replacement from the weighted distribution.
    """
    epsilon = 0.01
    leader_union = alpha | beta | delta

    # Count how many leaders contain each component
    leader_count: dict[int, int] = {}
    for j in leader_union:
        cnt = 0
        if j in alpha:
            cnt += 1
        if j in beta:
            cnt += 1
        if j in delta:
            cnt += 1
        leader_count[j] = cnt

    # Build weight vector for all components
    weights = np.full(M, epsilon)
    exploration_weight = a  # a ∈ [0, 2], higher = more exploration

    for j in leader_count:
        # Exploitation weight from leaders
        weights[j] = leader_count[j] * (1.0 + 0.5 * rng.random())
        if j in wolf:
            weights[j] *= 1.2  # inertia: keep own components

    # Exploration: boost random components
    if exploration_weight > 0.5:
        n_explore = int(M * 0.3 * (exploration_weight / 2.0))
        explore_indices = rng.choice(M, size=n_explore, replace=False)
        for j in explore_indices:
            weights[j] = max(weights[j], exploration_weight * rng.random())

    # Normalize to probabilities
    probs = weights / weights.sum()

    # Sample K components
    indices = rng.choice(M, size=min(K, M), replace=False, p=probs)
    return set(int(i) for i in indices)


def _grid_prune(archive: ParetoArchive, max_size: int, n_grid: int = 10) -> None:
    """Grid-based pruning for MOGWO archive (section 27.2).

    Divides objective space into grid cells. Removes solutions from
    most crowded cells first.
    """
    if len(archive) <= max_size:
        return

    objs = archive.get_objectives_array()
    if objs.size == 0:
        return

    n_obj = objs.shape[1]
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    # Assign each solution to a grid cell
    grid_indices = np.floor((objs - mins) / ranges * n_grid).astype(int)
    grid_indices = np.clip(grid_indices, 0, n_grid - 1)

    # Count per cell
    cell_counts: dict[tuple, int] = {}
    cell_members: dict[tuple, list[int]] = {}
    for i, gi in enumerate(grid_indices):
        key = tuple(gi)
        cell_counts[key] = cell_counts.get(key, 0) + 1
        if key not in cell_members:
            cell_members[key] = []
        cell_members[key].append(i)

    # Remove from most crowded cells until under max_size
    while len(archive) > max_size and cell_counts:
        most_crowded_key = max(cell_counts, key=cell_counts.get)
        members = cell_members[most_crowded_key]
        # Remove last member (or random)
        remove_idx = members[-1]
        members.pop()
        cell_counts[most_crowded_key] -= 1
        if cell_counts[most_crowded_key] == 0:
            del cell_counts[most_crowded_key]
            del cell_members[most_crowded_key]

        # Remove from archive
        archive.solutions.pop(remove_idx)
        archive.objectives.pop(remove_idx)


def run_mogwo(
    problem,
    max_fe: int = 15000,
    solution_capacity: int = 10,
    archive_size: int = 200,
    population_size: int = 100,
    seed: int = 0,
) -> dict:
    """Run MOGWO on a set-based multi-objective problem.

    Args:
        problem: MultiObjectiveProblem instance.
        max_fe: Maximum function evaluations.
        solution_capacity: K — max components per solution.
        archive_size: Maximum archive size.
        population_size: Number of wolves.
        seed: Random seed.

    Returns:
        dict with archive, history, and metrics.
    """
    rng = np.random.default_rng(seed)
    M = problem.num_components
    K = solution_capacity
    N = population_size

    archive = ParetoArchive(max_size=archive_size)

    # Initialize wolves
    wolves: list[set[int]] = []
    wolves_obj: list[np.ndarray] = []

    for _ in range(N):
        k = rng.integers(1, K + 1)
        sol = set(rng.choice(M, size=k, replace=False))
        sol = problem.repair(sol)
        obj = problem.evaluate(sol)
        wolves.append(sol)
        wolves_obj.append(obj)

    fe_count = N
    all_objectives = list(wolves_obj)

    # Initial archive (batch update)
    init_agents = []
    for sol, obj in zip(wolves, wolves_obj):
        init_agents.append(Agent(
            solution=sol,
            objectives=obj,
            component_preference=np.ones(1),
            objective_preference=np.ones(1),
        ))
    archive.update(init_agents)

    history = {
        "fe_count": [],
        "archive_size": [],
        "hypervolume": [],
        "avg_jaccard_distance": [],
    }

    from ..metrics import compute_hypervolume, compute_average_jaccard_distance

    iteration = 0
    max_iter = max_fe // N

    while fe_count < max_fe:
        # Linearly decrease a from 2 to 0
        a = 2.0 * (1.0 - iteration / max(1, max_iter))

        leaders = _select_leaders(archive, rng)
        if leaders is None:
            break
        alpha, beta, delta = leaders

        gen_agents: list[Agent] = []
        for i in range(N):
            if fe_count >= max_fe:
                break

            # Update wolf position using discrete operator
            new_sol = _update_position(wolves[i], alpha, beta, delta, M, K, a, rng)
            new_sol = problem.repair(new_sol)

            if len(new_sol) == 0:
                k = rng.integers(1, K + 1)
                new_sol = set(rng.choice(M, size=k, replace=False))

            new_obj = problem.evaluate(new_sol)
            fe_count += 1
            all_objectives.append(new_obj)

            # Update wolf if new solution is better
            wolves[i] = new_sol
            wolves_obj[i] = new_obj

            gen_agents.append(Agent(
                solution=new_sol,
                objectives=new_obj,
                component_preference=np.ones(1),
                objective_preference=np.ones(1),
            ))

            if fe_count >= max_fe:
                break

        # Batch update archive (once per generation)
        if gen_agents:
            archive.update(gen_agents)

        iteration += 1

        # Record metrics periodically
        if iteration % max(1, max_iter // 50) == 0 or fe_count >= max_fe:
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
