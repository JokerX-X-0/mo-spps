"""MOSCSP: Multi-Objective Synergistic Component Selection Problem.

Section 26.1. The primary mechanism-validation problem for MO-SPPS.

Candidate solutions are subsets S ⊆ V with |S| ≤ K.

Objectives (2-objective default):
  F(S) = (Quality(S), -Cost(S))

  Quality(S) = sum(v_j for j in S) + lambda * sum(B_r for satisfied synergy rules r)

Three instance types:
  - low_synergy: Low synergy, low conflict (check pool has no negative side-effects)
  - high_synergy: High synergy, popular component concentration (demonstrate pool suppression)
  - multi_cluster: Multiple synergy clusters (verify discovery of structurally diverse Pareto routes)
"""

import numpy as np
from ..problem_base import MultiObjectiveProblem


class MOSCSP(MultiObjectiveProblem):
    """Multi-Objective Synergistic Component Selection Problem.

    Generates synthetic components with individual values v_j, costs c_j,
    and synergy rules R that reward co-selection of certain component pairs.
    """

    def __init__(
        self,
        num_components: int = 30,
        solution_capacity: int = 10,
        synergy_strength: float = 1.0,
        instance_type: str = "low_synergy",
        seed: int = 0,
    ):
        super().__init__(
            num_components=num_components,
            solution_capacity=solution_capacity,
            num_objectives=2,
        )
        self.synergy_strength = synergy_strength
        self.instance_type = instance_type
        rng = np.random.default_rng(seed)

        # Generate component values and costs
        self.values = rng.uniform(0.5, 2.0, size=num_components)
        self.costs = rng.uniform(0.3, 1.5, size=num_components)

        # Generate synergy rules
        self.synergy_rules: list[tuple[set[int], float]] = []
        self._generate_synergy_rules(rng)

    def _generate_synergy_rules(self, rng: np.random.Generator) -> None:
        """Generate synergy rules based on instance type.

        Each rule is (component_set, bonus_weight).
        Bonus = bonus_weight * I(all components in rule are selected).
        """
        n = self.num_components

        if self.instance_type == "low_synergy":
            # Few weak pairwise synergies
            n_rules = max(1, n // 10)
            for _ in range(n_rules):
                pair = tuple(rng.choice(n, size=2, replace=False))
                bonus = rng.uniform(0.1, 0.3) * self.synergy_strength
                self.synergy_rules.append((set(pair), bonus))

        elif self.instance_type == "high_synergy":
            # Many strong synergies concentrated on popular components
            popular = set(rng.choice(n, size=max(2, n // 5), replace=False))
            n_rules = n // 3
            for _ in range(n_rules):
                j1 = int(rng.choice(list(popular)))
                j2 = int(rng.choice(n))
                if j1 == j2:
                    j2 = (j2 + 1) % n
                bonus = rng.uniform(0.5, 2.0) * self.synergy_strength
                self.synergy_rules.append(({j1, j2}, bonus))

        elif self.instance_type == "multi_cluster":
            # Multiple independent clusters, each with internal synergies
            n_clusters = 3
            cluster_size = n // n_clusters
            for c in range(n_clusters):
                cluster = set(
                    range(c * cluster_size, min((c + 1) * cluster_size, n))
                )
                n_rules = len(cluster) // 2
                cluster_list = list(cluster)
                for _ in range(n_rules):
                    if len(cluster_list) >= 2:
                        pair = tuple(
                            rng.choice(cluster_list, size=2, replace=False)
                        )
                        bonus = rng.uniform(0.3, 1.0) * self.synergy_strength
                        self.synergy_rules.append((set(pair), bonus))

        else:
            raise ValueError(f"Unknown instance type: {self.instance_type}")

    def evaluate(self, solution: set[int]) -> np.ndarray:
        """Evaluate a candidate solution.

        Returns: [Quality(S), -Cost(S)]
        """
        # Base quality: sum of individual component values
        quality = sum(self.values[j] for j in solution)

        # Synergy bonus
        for rule_set, bonus in self.synergy_rules:
            if rule_set.issubset(solution):
                quality += bonus

        # Cost
        cost = sum(self.costs[j] for j in solution)

        return np.array([quality, -cost])

    def repair(self, solution: set[int]) -> set[int]:
        """Truncate to capacity, keeping components with best quality/cost ratio."""
        if len(solution) <= self.solution_capacity:
            return solution
        ratios = {
            j: self.values[j] / max(self.costs[j], 1e-12)
            for j in solution
        }
        sorted_components = sorted(ratios, key=ratios.get, reverse=True)
        return set(sorted_components[: self.solution_capacity])

    def get_true_pareto_front(self, n_points: int = 200) -> np.ndarray | None:
        """Approximate the true Pareto front via exhaustive enumeration
        of dominant solutions from random search. Only feasible for small M, K.
        """
        # For small instances, we can approximate via extensive sampling
        rng = np.random.default_rng(42)
        best_quality = -np.inf
        best_front: dict[float, float] = {}

        for _ in range(100000):
            k = rng.integers(1, self.solution_capacity + 1)
            sol = set(rng.choice(self.num_components, size=k, replace=False))
            obj = self.evaluate(sol)
            quality, neg_cost = float(obj[0]), float(obj[1])
            cost = -neg_cost

            # Keep non-dominated
            dominated = False
            to_remove = []
            for existing_cost, existing_quality in best_front.items():
                if quality >= existing_quality and cost <= existing_cost:
                    to_remove.append(existing_cost)
                    if quality > existing_quality or cost < existing_cost:
                        pass  # strictly dominates
                elif existing_quality >= quality and existing_cost <= cost:
                    dominated = True
                    break

            if not dominated:
                for c in to_remove:
                    del best_front[c]
                best_front[cost] = quality

        if not best_front:
            return None

        front = np.array(
            [[q, -c] for c, q in sorted(best_front.items())]
        )
        return front


def generate_moscp_instance(
    num_components: int = 30,
    solution_capacity: int = 10,
    instance_type: str = "low_synergy",
    synergy_strength: float = 1.0,
    seed: int = 0,
) -> MOSCSP:
    """Factory function for creating MOSCSP instances."""
    return MOSCSP(
        num_components=num_components,
        solution_capacity=solution_capacity,
        synergy_strength=synergy_strength,
        instance_type=instance_type,
        seed=seed,
    )
