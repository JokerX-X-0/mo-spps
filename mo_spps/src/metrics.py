from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from .operators import jaccard_similarity, jaccard_distance, compute_pairwise_jaccard_distances

if TYPE_CHECKING:
    from .agent import Agent
    from .archive import ParetoArchive


def compute_hypervolume(objectives: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute hypervolume of objective set relative to reference point.

    Section 29.1. Uses exact calculation for 2-objective case,
    Monte Carlo for 3+ objectives.

    All objectives are maximized. Reference point should be worse than
    (i.e., less than or equal to) all objective values.
    """
    if objectives.size == 0:
        return 0.0

    n, m = objectives.shape

    if m == 2:
        return _hv_2d(objectives, ref_point)
    else:
        return _hv_monte_carlo(objectives, ref_point, n_samples=10000)


def _hv_2d(objectives: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact hypervolume for 2-objective case.

    All objectives are maximized. Sorts by f1 ascending, computes
    the area dominated by the Pareto front above the reference point.
    """
    sorted_idx = np.argsort(objectives[:, 0])
    sorted_objs = objectives[sorted_idx]

    hv = 0.0
    prev_f1 = ref_point[0]
    for i in range(len(sorted_objs)):
        f1, f2 = sorted_objs[i]
        if f2 > ref_point[1]:
            width = f1 - prev_f1
            height = f2 - ref_point[1]
            hv += width * height
            prev_f1 = f1

    return hv


def _hv_monte_carlo(
    objectives: np.ndarray, ref_point: np.ndarray, n_samples: int = 10000
) -> float:
    """Monte Carlo hypervolume estimation for 3+ objectives."""
    rng = np.random.default_rng(42)
    mins = np.min(objectives, axis=0)
    maxs = np.max(objectives, axis=0)

    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    samples = rng.uniform(
        low=ref_point,
        high=maxs + ranges * 0.1,
        size=(n_samples, objectives.shape[1]),
    )

    dominated_count = 0
    for sample in samples:
        for obj in objectives:
            if np.all(obj >= sample):
                dominated_count += 1
                break

    volume = np.prod(maxs - ref_point + ranges * 0.1)
    return volume * (dominated_count / n_samples)


def compute_igd(objectives: np.ndarray, ref_front: np.ndarray) -> float:
    """Inverted Generational Distance to reference Pareto front.

    Section 29.1. Lower is better.
    """
    if objectives.size == 0 or ref_front.size == 0:
        return np.inf

    total = 0.0
    for ref_pt in ref_front:
        dists = np.sqrt(np.sum((objectives - ref_pt) ** 2, axis=1))
        total += float(np.min(dists))

    return total / len(ref_front)


def compute_component_entropy(
    occupancy: dict[int, int], num_components: int
) -> tuple[float, float]:
    """Compute component usage entropy. Section 13.3.

    H = -sum(p_j * log(p_j + epsilon))
    H_norm = H / log(M)

    Returns (H, H_norm).
    """
    total = sum(occupancy.values())
    if total == 0:
        return 0.0, 0.0

    entropy = 0.0
    epsilon = 1e-12
    for j in range(num_components):
        u_j = occupancy.get(j, 0)
        if u_j > 0:
            p_j = u_j / total
            entropy -= p_j * np.log(p_j + epsilon)

    log_m = np.log(num_components) if num_components > 1 else 1.0
    return entropy, entropy / log_m


def compute_average_jaccard_distance(
    solutions: list[set[int]],
) -> float:
    """Compute average pairwise Jaccard distance. Section 13.2."""
    n = len(solutions)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += jaccard_distance(solutions[i], solutions[j])
            count += 1

    return total_dist / count if count > 0 else 0.0


def compute_pool_occupancy(
    population: list[Agent],
) -> dict[int, int]:
    """Compute component occupancy in the active population."""
    occupancy: dict[int, int] = {}
    for agent in population:
        for j in agent.solution:
            occupancy[j] = occupancy.get(j, 0) + 1
    return occupancy


def compute_reuse_concentration(
    occupancy: dict[int, int],
) -> float:
    """Compute Gini coefficient of component reuse concentration.

    Higher values mean a few components dominate.
    G = 0 for perfectly uniform usage, G → 1 for extreme concentration.
    """
    values = list(occupancy.values())
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0

    n = len(values)
    sorted_vals = sorted(values)
    # G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    weighted_sum = sum(
        (i + 1) * v for i, v in enumerate(sorted_vals)
    )
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def compute_reference_direction_coverage(
    archive_objectives: list[np.ndarray],
    directions: np.ndarray,
    min_cos_similarity: float = 0.9,
) -> float:
    """Fraction of reference directions with a nearby archive solution.

    Section 29.2.
    """
    if len(archive_objectives) == 0 or len(directions) == 0:
        return 0.0

    covered = 0
    for direction in directions:
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-12:
            covered += 1
            continue
        for obj in archive_objectives:
            obj_norm = np.linalg.norm(obj)
            if obj_norm < 1e-12:
                continue
            cos_sim = float(np.dot(obj, direction) / (obj_norm * dir_norm))
            if cos_sim >= min_cos_similarity:
                covered += 1
                break

    return covered / len(directions)


class MetricsTracker:
    """Records per-iteration metrics during algorithm execution.

    Section 19, step 7. Section 29.
    """

    def __init__(self):
        self.history: dict[str, list] = {
            "iteration": [],
            "fe_count": [],
            "archive_size": [],
            "hypervolume": [],
            "avg_jaccard_distance": [],
            "component_entropy": [],
            "component_entropy_norm": [],
            "reuse_concentration": [],
            "direction_coverage": [],
            "pool_occupancy": [],
            "runtime_seconds": [],
        }

    def record(
        self,
        iteration: int,
        fe_count: int,
        archive_size: int,
        hypervolume: float,
        avg_jaccard_distance: float,
        component_entropy: float,
        component_entropy_norm: float,
        reuse_concentration: float,
        direction_coverage: float = 0.0,
        pool_occupancy: dict[int, int] | None = None,
        runtime_seconds: float = 0.0,
    ) -> None:
        self.history["iteration"].append(iteration)
        self.history["fe_count"].append(fe_count)
        self.history["archive_size"].append(archive_size)
        self.history["hypervolume"].append(hypervolume)
        self.history["avg_jaccard_distance"].append(avg_jaccard_distance)
        self.history["component_entropy"].append(component_entropy)
        self.history["component_entropy_norm"].append(component_entropy_norm)
        self.history["reuse_concentration"].append(reuse_concentration)
        self.history["direction_coverage"].append(direction_coverage)
        self.history["pool_occupancy"].append(
            pool_occupancy.copy() if pool_occupancy else {}
        )
        self.history["runtime_seconds"].append(runtime_seconds)

    def get_history(self) -> dict[str, list]:
        return self.history
