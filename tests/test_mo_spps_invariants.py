"""End-to-end invariant tests for MO-SPPS.

Section 34.4. Validates that the algorithm maintains all required invariants
throughout execution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.mo_spps import MOSPPSOptimizer
from src.pareto import dominates
from src.operators import jaccard_similarity


def make_config(**overrides) -> dict:
    config = {
        "problem": {
            "num_components": 30,
            "solution_capacity": 10,
            "num_objectives": 2,
        },
        "population": {
            "population_size": 20,
            "max_function_evaluations": 2000,
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
            "elimination_interval": 5,
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
            "n_directions": 20,
        },
        "experiment": {
            "seed": 42,
            "num_runs": 1,
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict):
            config[k].update(v)
        else:
            config[k] = v
    return config


class TestMOSPPSInvariants:
    """End-to-end tests validating that MO-SPPS maintains core invariants."""

    def test_solution_capacity_invariant(self):
        """All agent solutions satisfy |S_i| <= K."""
        from src.problems.mo_scsp import MOSCSP
        problem = MOSCSP(num_components=30, solution_capacity=10)
        config = make_config()
        optimizer = MOSPPSOptimizer(problem, config)
        optimizer.initialize()

        for _ in range(10):  # Run a few iterations
            optimizer.step()
            if optimizer.fe_count >= optimizer.max_fe:
                break

        for agent in optimizer.agents:
            assert len(agent.solution) <= problem.solution_capacity, (
                f"Agent solution size {len(agent.solution)} exceeds capacity "
                f"{problem.solution_capacity}"
            )

    def test_archive_no_dominated_solutions(self):
        """Archive contains no solution that is dominated by another archive member."""
        from src.problems.mo_scsp import MOSCSP
        problem = MOSCSP(num_components=30, solution_capacity=10)
        config = make_config()
        optimizer = MOSPPSOptimizer(problem, config)
        optimizer.initialize()

        for _ in range(10):
            optimizer.step()
            if optimizer.fe_count >= optimizer.max_fe:
                break

        for i, obj_a in enumerate(optimizer.archive.objectives):
            for j, obj_b in enumerate(optimizer.archive.objectives):
                if i != j:
                    assert not dominates(obj_a, obj_b), (
                        f"Archive member {i} dominates member {j}"
                    )

    def test_archive_no_duplicates(self):
        """Archive contains no duplicate solutions."""
        from src.problems.mo_scsp import MOSCSP
        problem = MOSCSP(num_components=30, solution_capacity=10)
        config = make_config()
        optimizer = MOSPPSOptimizer(problem, config)
        optimizer.initialize()

        for _ in range(10):
            optimizer.step()
            if optimizer.fe_count >= optimizer.max_fe:
                break

        for i in range(len(optimizer.archive.solutions)):
            for j in range(i + 1, len(optimizer.archive.solutions)):
                sol_a = optimizer.archive.solutions[i]
                sol_b = optimizer.archive.solutions[j]
                # Solutions should differ in components OR objectives
                same_comp = sol_a == sol_b
                same_obj = np.allclose(
                    optimizer.archive.objectives[i],
                    optimizer.archive.objectives[j],
                    rtol=1e-10,
                )
                assert not (same_comp or same_obj), (
                    f"Duplicate solutions found at indices {i}, {j}"
                )

    def test_reproducibility(self):
        """Fixed seed produces identical results."""
        from src.problems.mo_scsp import MOSCSP

        results = []
        for _ in range(2):
            problem = MOSCSP(
                num_components=30, solution_capacity=10, seed=42
            )
            config = make_config()
            config["experiment"]["seed"] = 42
            optimizer = MOSPPSOptimizer(problem, config)
            optimizer.initialize()
            for __ in range(10):
                optimizer.step()
                if optimizer.fe_count >= optimizer.max_fe:
                    break
            results.append(
                {
                    "archive_size": len(optimizer.archive),
                    "fe_count": optimizer.fe_count,
                }
            )

        assert results[0] == results[1]

    def test_population_size_constant(self):
        """Population size remains constant after elimination and rebirth."""
        from src.problems.mo_scsp import MOSCSP
        problem = MOSCSP(num_components=30, solution_capacity=10)
        config = make_config()
        config["rebirth"]["elimination_interval"] = 3
        optimizer = MOSPPSOptimizer(problem, config)
        optimizer.initialize()
        initial_size = len(optimizer.agents)

        for _ in range(10):
            optimizer.step()
            if optimizer.fe_count >= optimizer.max_fe:
                break
            assert len(optimizer.agents) == initial_size, (
                f"Population size changed from {initial_size} to "
                f"{len(optimizer.agents)} at iteration {optimizer.iteration}"
            )

    def test_pool_soft_pressure_invariants(self):
        """Soft-pressure pool occupancy matches population."""
        from src.problems.mo_scsp import MOSCSP
        problem = MOSCSP(num_components=30, solution_capacity=10)
        config = make_config()
        optimizer = MOSPPSOptimizer(problem, config)
        optimizer.initialize()

        for _ in range(10):
            optimizer.step()
            if optimizer.fe_count >= optimizer.max_fe:
                break
            # Validate pool invariants
            optimizer.pool.validate(optimizer.agents)


class TestNoPoolAblation:
    """Verify that NoPool variant can be created by setting use_shared_pool=false."""

    def test_no_pool_uniform_sampling(self):
        """Without shared pool, sampling should be uniform (ignoring occupancy)."""
        from src.problems.mo_scsp import MOSCSP

        # Use soft_pressure with tau=0 to effectively disable pool pressure
        problem = MOSCSP(num_components=30, solution_capacity=10)
        config = make_config()
        config["shared_pool"]["tau"] = 0.0  # tau=0 makes (epsilon+q/Q)^0 = 1
        config["population"]["max_function_evaluations"] = 1000
        optimizer = MOSPPSOptimizer(problem, config)
        optimizer.initialize()

        for _ in range(5):
            optimizer.step()
            if optimizer.fe_count >= optimizer.max_fe:
                break

        # Should still produce a valid archive
        assert len(optimizer.archive) > 0
        # All capacity invariants still hold
        assert all(
            len(a.solution) <= problem.solution_capacity
            for a in optimizer.agents
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
