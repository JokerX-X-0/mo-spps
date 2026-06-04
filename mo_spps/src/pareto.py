import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check whether objective vector a dominates b (both maximized).

    a dominates b iff:
      forall k: a_k >= b_k  AND  exists r: a_r > b_r

    Section 4.1.
    """
    # Fast path for 2-objective case (most common, avoids numpy overhead)
    m = a.shape[0]
    if m != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if m == 2:
        a0, a1 = a[0], a[1]
        b0, b1 = b[0], b[1]
        return a0 >= b0 and a1 >= b1 and (a0 > b0 or a1 > b1)
    at_least_as_good = bool(np.all(a >= b))
    strictly_better = bool(np.any(a > b))
    return at_least_as_good and strictly_better


def non_dominated_sort(objectives: list[np.ndarray]) -> list[list[int]]:
    """Partition population into Pareto fronts.

    Returns a list of fronts, where each front is a list of indices
    into the objectives list. Front[0] is the non-dominated set.

    Section 12.1. O(m * N^2) algorithm.
    """
    n = len(objectives)
    if n == 0:
        return []

    dominated_count = np.zeros(n, dtype=int)
    dominates_list = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominates_list[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                dominated_count[i] += 1

    fronts = []
    current_front = [i for i in range(n) if dominated_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distance for each individual.

    objectives: shape (N, m) — population objective vectors (all maximized).

    Boundary points get infinity. Section 12.3.
    """
    n, m = objectives.shape
    cd = np.zeros(n)

    if n <= 2:
        cd[:] = np.inf
        return cd

    for k in range(m):
        obj_k = objectives[:, k]
        sorted_idx = np.argsort(-obj_k)
        f_max = obj_k[sorted_idx[0]]
        f_min = obj_k[sorted_idx[-1]]
        denom = f_max - f_min + 1e-12

        cd[sorted_idx[0]] = np.inf
        cd[sorted_idx[-1]] = np.inf

        for r in range(1, n - 1):
            if np.isinf(cd[sorted_idx[r]]):
                continue
            left_obj = obj_k[sorted_idx[r - 1]]
            right_obj = obj_k[sorted_idx[r + 1]]
            cd[sorted_idx[r]] += (left_obj - right_obj) / denom

    return cd


def normalize_crowding_score(cd: np.ndarray) -> np.ndarray:
    """C_i = CD_i / (max CD_j + epsilon). Section 12.3.

    Boundary points (inf) are set to 1.0. Interior points are normalized
    by the maximum finite crowding distance so that the objective-space
    term in the hybrid pruning score is correctly scaled.
    """
    finite_vals = cd[np.isfinite(cd)]
    if len(finite_vals) == 0:
        return np.ones_like(cd)
    max_cd = float(np.max(finite_vals))
    if max_cd == 0.0:
        return np.where(np.isfinite(cd), 1.0, 1.0)
    result = np.zeros_like(cd, dtype=float)
    result[np.isfinite(cd)] = cd[np.isfinite(cd)] / max_cd
    result[~np.isfinite(cd)] = 1.0
    return result


def pareto_rank_score(ranks: np.ndarray) -> np.ndarray:
    """P_i = 1 - (r_i - 1) / (r_max - 1 + epsilon). Section 12.2."""
    r_max = np.max(ranks)
    if r_max <= 1:
        return np.ones_like(ranks, dtype=float)
    return 1.0 - (ranks.astype(float) - 1.0) / (r_max - 1.0 + 1e-12)


def is_nondominated_wrt_archive(
    candidate: np.ndarray, archive_objectives: list[np.ndarray]
) -> bool:
    """Check if candidate is not dominated by any member of the archive."""
    for arch_obj in archive_objectives:
        if dominates(arch_obj, candidate):
            return False
    return True


def count_dominated_by_candidate(
    candidate: np.ndarray, archive_objectives: list[np.ndarray]
) -> int:
    """Count how many archive solutions are dominated by candidate."""
    count = 0
    for arch_obj in archive_objectives:
        if dominates(candidate, arch_obj):
            count += 1
    return count
