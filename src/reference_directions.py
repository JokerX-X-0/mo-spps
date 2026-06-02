import numpy as np


def generate_das_dennis_directions(n_obj: int, n_partitions: int) -> np.ndarray:
    """Generate systematic reference directions using the Das-Dennis method.

    Section 7.2. Produces C(n_partitions + n_obj - 1, n_obj - 1) directions.
    """
    if n_obj == 2:
        directions = []
        for i in range(n_partitions + 1):
            w = np.array([i / n_partitions, (n_partitions - i) / n_partitions])
            directions.append(w)
        return np.array(directions)

    def _recurse(n_obj_left: int, n_parts_left: int, current: list) -> list:
        if n_obj_left == 1:
            return [np.array(current + [n_parts_left]) / n_partitions]
        results = []
        for k in range(n_parts_left + 1):
            results.extend(
                _recurse(n_obj_left - 1, n_parts_left - k, current + [k])
            )
        return results

    dirs = _recurse(n_obj, n_partitions, [])
    if len(dirs) == 0:
        return np.empty((0, n_obj))
    return np.array(dirs)


def generate_random_directions(
    n_obj: int, n_directions: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate random reference directions via Dirichlet sampling.

    Each direction is drawn from Dirichlet(alpha=[1,...,1]).
    """
    alpha = np.ones(n_obj)
    return rng.dirichlet(alpha, size=n_directions)


def assign_directions(
    n_agents: int, directions: np.ndarray, mode: str = "round_robin"
) -> list[np.ndarray]:
    """Assign reference directions to agents.

    mode="round_robin": cycle through directions, repeating if needed.
    mode="random_sample": sample with replacement if n_agents > len(directions).
    """
    n_dirs = len(directions)
    if n_dirs == 0:
        raise ValueError("No directions provided")

    result = []
    if mode == "round_robin":
        for i in range(n_agents):
            result.append(directions[i % n_dirs].copy())
    elif mode == "random_sample":
        idx = np.random.default_rng().integers(0, n_dirs, size=n_agents)
        for i in idx:
            result.append(directions[i].copy())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return result
