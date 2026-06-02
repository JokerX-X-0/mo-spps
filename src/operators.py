from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .pool import SharedPool
    from .agent import Agent


def sample_shop(
    pool: SharedPool,
    preference: np.ndarray,
    population: list[Agent],
    shop_size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample candidate components from the shared pool.

    Section 9: L_i^t = {j_1, j_2, ..., j_s} where s = shop_size.
    """
    return pool.sample(preference, population, shop_size, rng)


def generate_add_candidates(
    solution: set[int], shop: list[int], capacity: int
) -> list[set[int]]:
    """Generate candidates by adding shop components not already in solution.

    Section 9.1: S' = S_i ∪ {j}, requires |S_i| < K.
    """
    if len(solution) >= capacity:
        return []
    candidates = []
    for j in shop:
        if j not in solution:
            candidates.append(solution | {j})
    return candidates


def generate_replace_candidates(
    solution: set[int], shop: list[int], rng: np.random.Generator
) -> list[set[int]]:
    """Generate candidates by replacing existing components with shop components.

    Section 9.2: S' = (S_i \\ {r}) ∪ {j}.
    For each shop component j, one random replacement partner r ∈ S_i is
    chosen (rather than all |S_i|, to bound evaluations per operation).
    Excludes identity swaps (j == r) and swaps where j is already in S.
    """
    if not solution:
        return []
    candidates = []
    sol_list = list(solution)
    for j in shop:
        if j in solution:
            continue
        r = int(rng.choice(sol_list))
        new_sol = (solution - {r}) | {j}
        candidates.append(new_sol)
    return candidates


def generate_release_candidates(
    solution: set[int],
) -> list[set[int]]:
    """Generate candidates by removing one component from the solution.

    Section 9.3: S' = S_i \\ {r} for each r in S_i.
    Requires at least 2 components to avoid empty solutions.
    """
    if len(solution) <= 1:
        return []
    return [solution - {r} for r in solution]


def generate_all_candidates(
    solution: set[int], shop: list[int], capacity: int, rng: np.random.Generator,
    include_release: bool = False,
) -> list[set[int]]:
    """Generate all candidates: add (if |S| < K) + replace + optional release.

    Section 9 combined.
    """
    candidates = []
    candidates.extend(generate_add_candidates(solution, shop, capacity))
    candidates.extend(generate_replace_candidates(solution, shop, rng))
    if include_release:
        candidates.extend(generate_release_candidates(solution))
    return candidates


def compute_preference_score(
    objectives: np.ndarray,
    weight_vector: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
) -> float:
    """Compute scalar preference score for an objective vector.

    Section 10.2: g_i(S) = w_i^T * F_hat(S)
    where F_hat is the normalized objective vector.
    """
    denom = ideal - nadir
    denom[denom == 0] = 1.0
    normalized = (objectives - nadir) / (denom + 1e-12)
    return float(np.dot(weight_vector, normalized))


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    """Jaccard similarity between two sets. Section 10.4, Section 13.1.

    sim(S_a, S_b) = |S_a ∩ S_b| / |S_a ∪ S_b|
    """
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def jaccard_distance(a: set[int], b: set[int]) -> float:
    """Jaccard distance = 1 - Jaccard similarity. Section 13.1."""
    return 1.0 - jaccard_similarity(a, b)


def compute_novelty(
    candidate: set[int], archive_solutions: list[set[int]]
) -> float:
    """Decision-space novelty of a candidate relative to the archive.

    Section 10.4:
        Novelty(S') = 1 - max_{S in Archive} sim(S', S)

    Higher values mean the candidate is more structurally different.
    """
    if not archive_solutions:
        return 1.0
    max_sim = max(
        jaccard_similarity(candidate, s) for s in archive_solutions
    )
    return 1.0 - max_sim


def compute_pairwise_jaccard_distances(
    solutions: list[set[int]],
) -> np.ndarray:
    """Compute pairwise Jaccard distances for a list of solutions.

    Returns (N, N) matrix.
    """
    n = len(solutions)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(solutions[i], solutions[j])
            dists[i, j] = d
            dists[j, i] = d
    return dists
