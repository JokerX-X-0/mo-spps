"""Tests for Pareto dominance, non-dominated sorting, and crowding distance.

Section 34.2.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.pareto import (
    dominates,
    non_dominated_sort,
    crowding_distance,
    normalize_crowding_score,
    pareto_rank_score,
    is_nondominated_wrt_archive,
    count_dominated_by_candidate,
)


class TestDominates:
    def test_a_dominates_b(self):
        a = np.array([3.0, 2.0])
        b = np.array([1.0, 1.0])
        assert dominates(a, b)

    def test_b_does_not_dominate_a(self):
        a = np.array([3.0, 2.0])
        b = np.array([1.0, 1.0])
        assert not dominates(b, a)

    def test_mutually_non_dominated(self):
        a = np.array([3.0, 1.0])
        b = np.array([1.0, 3.0])
        assert not dominates(a, b)
        assert not dominates(b, a)

    def test_equal_vectors_no_domination(self):
        a = np.array([2.0, 2.0])
        b = np.array([2.0, 2.0])
        assert not dominates(a, b)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            dominates(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


class TestNonDominatedSort:
    def test_single_point(self):
        objs = [np.array([1.0, 1.0])]
        fronts = non_dominated_sort(objs)
        assert len(fronts) == 1
        assert fronts[0] == [0]

    def test_two_nondominated(self):
        objs = [
            np.array([3.0, 1.0]),
            np.array([1.0, 3.0]),
        ]
        fronts = non_dominated_sort(objs)
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1}

    def test_dominated_goes_to_front2(self):
        objs = [
            np.array([3.0, 3.0]),
            np.array([2.0, 2.0]),
        ]
        fronts = non_dominated_sort(objs)
        assert len(fronts) == 2
        assert fronts[0] == [0]
        assert fronts[1] == [1]

    def test_empty_list(self):
        fronts = non_dominated_sort([])
        assert fronts == []


class TestCrowdingDistance:
    def test_two_points_infinite(self):
        objs = np.array([[1.0, 2.0], [3.0, 4.0]])
        cd = crowding_distance(objs)
        assert np.all(np.isinf(cd))

    def test_boundary_infinite(self):
        objs = np.array([[5.0, 1.0], [3.0, 2.0], [1.0, 3.0]])
        cd = crowding_distance(objs)
        assert np.isinf(cd[0])
        assert np.isinf(cd[2])

    def test_crowding_score_normalization(self):
        cd = np.array([1.0, 2.0, np.inf])
        scores = normalize_crowding_score(cd)
        assert 0 <= scores[0] <= 1
        assert scores[2] == 1.0


class TestParetoRankScore:
    def test_all_rank_one(self):
        ranks = np.array([1, 1, 1])
        scores = pareto_rank_score(ranks)
        assert np.allclose(scores, 1.0)

    def test_mixed_ranks(self):
        ranks = np.array([1, 2, 3])
        scores = pareto_rank_score(ranks)
        assert np.isclose(scores[0], 1.0)
        assert np.isclose(scores[1], 0.5)
        assert np.isclose(scores[2], 0.0)


class TestArchiveRelations:
    def test_is_nondominated_wrt_archive(self):
        archive = [np.array([5.0, 5.0]), np.array([4.0, 6.0])]
        candidate = np.array([5.5, 4.5])
        assert is_nondominated_wrt_archive(candidate, archive)

    def test_count_dominated(self):
        archive = [np.array([1.0, 1.0]), np.array([2.0, 0.5])]
        candidate = np.array([3.0, 3.0])
        assert count_dominated_by_candidate(candidate, archive) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
