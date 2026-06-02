"""Random Multi-Objective Search baseline.

Section 27.1. Generates random solutions and maintains a Pareto archive.
Serves as the lower-bound baseline for multi-objective performance.
"""

import numpy as np
from ..archive import ParetoArchive
from ..agent import Agent


def run_random_mo_search(
    problem,
    max_fe: int = 15000,
    solution_capacity: int = 10,
    archive_size: int = 200,
    seed: int = 0,
) -> dict:
    """Run random multi-objective search.

    Args:
        problem: MultiObjectiveProblem instance.
        max_fe: Maximum function evaluations.
        solution_capacity: Maximum components per solution (K).
        archive_size: Maximum archive size.
        seed: Random seed.

    Returns:
        dict with archive, history, and metrics.
    """
    rng = np.random.default_rng(seed)
    archive = ParetoArchive(max_size=archive_size)

    history = {
        "fe_count": [],
        "archive_size": [],
        "hypervolume": [],
        "avg_jaccard_distance": [],
    }

    import math
    from ..metrics import (
        compute_hypervolume,
        compute_average_jaccard_distance,
    )

    problem.update_ideal_nadir([])

    record_interval = max(1, max_fe // 50)
    pending_agents: list[Agent] = []

    for fe in range(max_fe):
        # Generate random solution
        k = rng.integers(1, solution_capacity + 1)
        sol = set(
            rng.choice(problem.num_components, size=k, replace=False)
        )
        sol = problem.repair(sol)
        obj = problem.evaluate(sol)

        # Collect agent for batch update
        agent = Agent(
            solution=sol,
            objectives=obj,
            component_preference=np.ones(1),
            objective_preference=np.ones(1),
        )
        pending_agents.append(agent)

        # Record periodically
        if fe % record_interval == 0 or fe == max_fe - 1:
            # Batch update archive
            archive.update(pending_agents)
            pending_agents.clear()
            all_objs = archive.objectives + [obj]
            problem.update_ideal_nadir(all_objs)

            history["fe_count"].append(fe + 1)
            history["archive_size"].append(len(archive))

            arch_objs = archive.get_objectives_array()
            if arch_objs.size > 0:
                ref = np.min(arch_objs, axis=0) - 1.0
                hv = compute_hypervolume(arch_objs, ref)
            else:
                hv = 0.0
            history["hypervolume"].append(hv)

            jd = compute_average_jaccard_distance(archive.solutions)
            history["avg_jaccard_distance"].append(jd)

    return {
        "archive": archive,
        "history": history,
        "final_fe": max_fe,
    }
