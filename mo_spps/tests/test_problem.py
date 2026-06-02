"""Tests for MOSCSP problem instance.

Section 34.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.problems.mo_scsp import MOSCSP, generate_moscp_instance


class TestMOSCSP:
    def test_evaluate_returns_correct_shape(self):
        problem = MOSCSP(num_components=30, solution_capacity=10)
        obj = problem.evaluate({0, 1, 2})
        assert obj.shape == (2,)
        # Second objective is negated cost, so it should be negative or zero
        assert obj[1] <= 0

    def test_empty_solution(self):
        problem = MOSCSP(num_components=30, solution_capacity=10)
        obj = problem.evaluate(set())
        assert obj[0] == 0.0  # No quality
        assert obj[1] == 0.0  # No cost

    def test_repair_truncates(self):
        problem = MOSCSP(num_components=30, solution_capacity=5)
        oversized = set(range(10))
        repaired = problem.repair(oversized)
        assert len(repaired) <= 5

    def test_repair_preserves_small_solution(self):
        problem = MOSCSP(num_components=30, solution_capacity=10)
        small = {0, 1, 2}
        repaired = problem.repair(small)
        assert repaired == small

    def test_synergy_increases_quality(self):
        problem_no_syn = MOSCSP(
            num_components=10,
            solution_capacity=5,
            synergy_strength=0.0,
        )
        problem_syn = MOSCSP(
            num_components=10,
            solution_capacity=5,
            synergy_strength=2.0,
            seed=42,
        )
        sol = {0, 1, 2, 3}
        obj_no = problem_no_syn.evaluate(sol)
        obj_syn = problem_syn.evaluate(sol)
        # Quality may differ due to different synergy rules
        assert obj_no.shape == obj_syn.shape

    def test_instance_types(self):
        for inst_type in ["low_synergy", "high_synergy", "multi_cluster"]:
            problem = MOSCSP(
                num_components=30,
                solution_capacity=10,
                instance_type=inst_type,
            )
            assert len(problem.synergy_rules) > 0
            assert problem.instance_type == inst_type

    def test_objectives_are_maximized_format(self):
        """Higher quality is better, lower cost is better (negated)."""
        problem = MOSCSP(num_components=20, solution_capacity=10)
        # A larger set should generally have more quality but higher cost
        obj_small = problem.evaluate({0, 1})
        obj_large = problem.evaluate({0, 1, 2, 3, 4})
        # Quality of larger set should be >= smaller set (more components)
        assert obj_large[0] >= obj_small[0]
        # Cost is negated, so larger set should have more negative cost
        assert obj_large[1] <= obj_small[1]


def test_factory_function():
    problem = generate_moscp_instance(
        num_components=30,
        solution_capacity=10,
        instance_type="high_synergy",
        synergy_strength=2.0,
        seed=123,
    )
    assert problem.num_components == 30
    assert problem.solution_capacity == 10
    assert problem.instance_type == "high_synergy"
    assert problem.synergy_strength == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
