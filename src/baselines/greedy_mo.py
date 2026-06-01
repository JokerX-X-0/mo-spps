"""Greedy Scalarization baseline.

Section 27.1. Runs weighted greedy construction for multiple
scalarization weight vectors and maintains a Pareto archive.
"""

import numpy as np
from ..archive import ParetoArchive
from ..agent import Agent
from ..operators import compute_preference_score


def run_greedy_scalarization(
    problem,
    max_fe: int = 15000,
    solution_capacity: int = 10,
    archive_size: int = 200,
    n_directions: int = 20,
    seed: int = 0,
) -> dict:
    """Run greedy scalarization with multiple weight vectors.

    For each weight vector w_k, greedily construct a solution by adding
    the component that maximizes w_k^T * F_hat(S ∪ {j}) at each step.
    """
    rng = np.random.default_rng(seed)
    archive = ParetoArchive(max_size=archive_size)

    # Generate weight vectors
    alpha = np.ones(problem.num_objectives)
    weights = rng.dirichlet(alpha, size=n_directions)

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

    fe_count = 0
    all_objectives: list[np.ndarray] = []

    for weight in weights:
        if fe_count >= max_fe:
            break

        # Greedy construction
        solution: set[int] = set()
        available = set(range(problem.num_components))

        for _ in range(solution_capacity):
            best_j = -1
            best_score = -np.inf

            for j in available:
                candidate = solution | {j}
                obj = problem.evaluate(candidate)
                fe_count += 1
                all_objectives.append(obj)

                if problem.ideal_point is None:
                    problem.update_ideal_nadir([obj])

                score = compute_preference_score(
                    obj,
                    weight,
                    problem.ideal_point if problem.ideal_point is not None
                    else np.zeros_like(obj),
                    problem.nadir_point if problem.nadir_point is not None
                    else np.ones_like(obj) * -100,
                )

                if score > best_score:
                    best_score = score
                    best_j = j

                if fe_count >= max_fe:
                    break

            if best_j >= 0:
                solution.add(best_j)
                available.remove(best_j)

            if fe_count >= max_fe:
                break

        # Evaluate final solution and add to archive
        if solution:
            obj = problem.evaluate(solution)
            fe_count += 1
            all_objectives.append(obj)

            agent = Agent(
                solution=solution,
                objectives=obj,
                component_preference=np.ones(1),
                objective_preference=weight.copy(),
            )
            archive.update([agent])

    # Build history
    problem.update_ideal_nadir(all_objectives)
    arch_objs = archive.get_objectives_array()
    if arch_objs.size > 0:
        ref = np.min(arch_objs, axis=0) - 1.0
        hv = compute_hypervolume(arch_objs, ref)
    else:
        hv = 0.0

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
