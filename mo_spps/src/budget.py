import numpy as np


def allocate_fixed_budget(n_agents: int, base_budget: float) -> np.ndarray:
    """Phase 1: all agents receive the same fixed budget.

    Section 15.1: G_i = G_0.
    """
    return np.full(n_agents, base_budget, dtype=float)


def allocate_dynamic_budget(
    rank_scores: np.ndarray,
    crowding_scores: np.ndarray,
    diversity_scores: np.ndarray,
    base_budget: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 1.0,
    gamma: float = 0.0,
    rank_threshold: float = 0.5,
) -> np.ndarray:
    """Allocate budgets based on Pareto rank, crowding, and decision diversity.

    Section 15.5 (main algorithm):
        G_i = G_0 + alpha * P_i + beta * C_i + delta * D_i

    Optional exploration compensation (Section 15.4, Phase 4):
        + gamma * I(r_i > r_threshold) * D_i

    Args:
        rank_scores: P_i for each agent.
        crowding_scores: C_i for each agent.
        diversity_scores: D_i for each agent.
        base_budget: G_0.
        alpha: Pareto rank weight.
        beta: Crowding distance weight.
        delta: Decision diversity weight.
        gamma: Exploration compensation weight (Phase 4).
        rank_threshold: Rank threshold for exploration compensation.

    Returns:
        Budget array of shape (N,).
    """
    n = len(rank_scores)
    budget = np.full(n, base_budget, dtype=float)
    budget += alpha * rank_scores
    budget += beta * crowding_scores
    budget += delta * diversity_scores

    if gamma > 0:
        exploration = np.where(
            rank_scores < rank_threshold, diversity_scores, 0.0
        )
        budget += gamma * exploration

    return budget


def budget_to_operations(budget: np.ndarray) -> np.ndarray:
    """Convert continuous budget to discrete operation counts.

    Section 15.1: ops_i = max(1, round(G_i))
    """
    return np.maximum(1, np.round(budget)).astype(int)
