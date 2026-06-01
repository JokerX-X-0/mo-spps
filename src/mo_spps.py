from __future__ import annotations

import time
from typing import TYPE_CHECKING
import numpy as np

from .components import Component
from .pool import SharedPool
from .agent import Agent, AgentFactory, reset_id_counter
from .archive import ParetoArchive
from . import pareto
from . import reference_directions as refdir
from . import operators
from . import budget as budget_mod
from . import metrics as metrics_mod

if TYPE_CHECKING:
    from .problem_base import MultiObjectiveProblem


class MOSPPSOptimizer:
    """Main MO-SPPS optimizer. Section 18-19, Section 25.2.

    Orchestrates the full algorithm loop:
      Initialize -> Iterate (evaluate, archive, rank, diversity, budget,
      local construct, pool update, preferences, elimination, metrics)
      -> Output Pareto archive.
    """

    def __init__(self, problem: MultiObjectiveProblem, config: dict):
        self.problem = problem
        self.config = config

        # Seed
        seed = config.get("experiment", {}).get("seed", 0)
        self.rng = np.random.default_rng(seed)

        # Build component objects
        pop_cfg = config.get("population", {})
        pool_cfg = config.get("shared_pool", {})
        self.num_components = problem.num_components
        self.solution_capacity = problem.solution_capacity
        self.num_objectives = problem.num_objectives

        capacity_ref = pool_cfg.get("capacity_reference", 5)
        self.components: list[Component] = []
        for j in range(self.num_components):
            self.components.append(
                Component(
                    id=j,
                    attributes={},
                    base_weight=1.0,
                    pool_capacity=capacity_ref,
                )
            )

        # Shared pool
        capacities = {
            j: c.pool_capacity for j, c in enumerate(self.components)
        }
        base_weights = {
            j: c.base_weight for j, c in enumerate(self.components)
        }
        self.pool = SharedPool(
            capacities=capacities,
            base_weights=base_weights,
            mode=pool_cfg.get("mode", "soft_pressure"),
            epsilon=pool_cfg.get("epsilon", 0.01),
            tau=pool_cfg.get("tau", 1.0),
        )

        # Population
        self.population_size = pop_cfg.get("population_size", 100)

        # Archive
        archive_cfg = config.get("archive", {})
        self.archive = ParetoArchive(
            max_size=archive_cfg.get("max_size", 200)
        )

        # Reference directions
        ref_cfg = config.get("reference_directions", {})
        n_dirs = ref_cfg.get("n_directions", self.population_size)
        self.directions = refdir.generate_random_directions(
            self.num_objectives, n_dirs, self.rng
        )

        # Agents
        self.agents: list[Agent] = []

        # Budget
        budget_cfg = config.get("budget", {})
        self.base_budget = budget_cfg.get("base_budget", 3.0)
        self.budget_mode = budget_cfg.get("mode", "fixed")
        self.alpha = budget_cfg.get("alpha_pareto", 1.0)
        self.beta = budget_cfg.get("beta_crowding", 1.0)
        self.delta = budget_cfg.get("delta_decision_diversity", 1.0)
        self.gamma = budget_cfg.get("gamma_exploration", 0.0)

        # Rebirth
        rebirth_cfg = config.get("rebirth", {})
        self.use_rebirth = rebirth_cfg.get("use_rebirth", True)
        self.elimination_interval = rebirth_cfg.get("elimination_interval", 10)
        self.replacement_rate = rebirth_cfg.get("replacement_rate", 0.2)
        self.keep_reference_direction = rebirth_cfg.get(
            "keep_reference_direction", True
        )
        # Weights for retention score R_i = a*P_i + b*C_i + d*D_i
        self.retention_a = rebirth_cfg.get("retention_a", 0.5)
        self.retention_b = rebirth_cfg.get("retention_b", 0.3)
        self.retention_d = rebirth_cfg.get("retention_d", 0.2)

        # Strategy inheritance (Phase 3)
        self.inheritance_strength = rebirth_cfg.get("inheritance_strength", 0.5)
        self.inheritance_smoothing = rebirth_cfg.get(
            "inheritance_smoothing", 0.1
        )
        self.preference_learning_rate = rebirth_cfg.get(
            "preference_learning_rate", 0.01
        )
        self.use_strategy_inheritance = rebirth_cfg.get(
            "use_strategy_inheritance", False
        )

        # Local search
        local_cfg = config.get("local_search", {})
        self.shop_size = local_cfg.get("shop_size", 5)
        self.use_probabilistic_acceptance = local_cfg.get(
            "use_probabilistic_acceptance", False
        )
        self.temperature = local_cfg.get("temperature", 1.0)
        self.archive_contribution_threshold = local_cfg.get(
            "archive_contribution_threshold", 0.0
        )
        self.novelty_threshold = local_cfg.get("novelty_threshold", 0.3)
        self.quality_loss_threshold = local_cfg.get(
            "quality_loss_threshold", 0.02
        )

        # Termination
        self.max_fe = config.get("population", {}).get(
            "max_function_evaluations", 150000
        )
        self.fe_count = 0
        self.iteration = 0

        # Metrics
        self.tracker = metrics_mod.MetricsTracker()

        # Reference point for HV (set after initialization)
        self.ref_point: np.ndarray | None = None

    # ------------------------------------------------------------------
    #  MAIN INTERFACE
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Algorithm initialization. Section 18.1, Section 19 (Initialize block)."""
        reset_id_counter()

        dir_assignments = refdir.assign_directions(
            self.population_size, self.directions, mode="round_robin"
        )

        self.agents = []
        for i in range(self.population_size):
            w_i = dir_assignments[i]
            # Uniform initial component preference
            pi_i = np.ones(self.num_components) / self.num_components

            # Construct initial solution by pool sampling (no population yet:
            # pool.sample sees no occupancy, so distribution is uniform
            # weighted by base_weight and pi_i)
            initial_components = self.pool.sample(
                pi_i, self.agents, self.solution_capacity, self.rng
            )
            S_i = set(initial_components[: self.solution_capacity])
            S_i = self.problem.repair(S_i)

            F_i = self.problem.evaluate(S_i)
            self.fe_count += 1

            agent = Agent(
                solution=S_i,
                objectives=F_i,
                component_preference=pi_i.copy(),
                objective_preference=w_i.copy(),
                budget=self.base_budget,
            )
            self.agents.append(agent)

        # Initial archive
        self.archive.update(self.agents)

        # Estimate ideal/nadir
        all_objs = [a.objectives for a in self.agents] + self.archive.objectives
        self.problem.update_ideal_nadir(all_objs)

        # Reference point for HV (nadir - margin)
        if self.problem.nadir_point is not None:
            margin = np.abs(self.problem.nadir_point) * 0.1 + 1.0
            self.ref_point = self.problem.nadir_point - margin
        else:
            self.ref_point = np.full(self.num_objectives, -100.0)

        # Initial metrics
        self._record_metrics(runtime_seconds=0.0)

    def run(self) -> dict:
        """Run the main optimization loop. Section 19.

        Returns metrics history dict.
        """
        if not self.agents:
            self.initialize()

        start_time = time.perf_counter()

        while self.fe_count < self.max_fe:
            self.step()
            elapsed = time.perf_counter() - start_time
            self._record_metrics(runtime_seconds=elapsed)

        return self.tracker.get_history()

    def step(self) -> None:
        """Execute one full iteration. Section 19, steps 1-7."""
        self.iteration += 1

        # 1. Evaluate population (already evaluated via local_construct)
        # 2. Update archive
        self.update_archive()

        # 3. Multi-objective ranking
        self.assign_ranks_and_crowding()

        # 4. Compute decision-space diversity
        self.compute_decision_diversity()

        # 5. Allocate budgets
        self.allocate_budgets()

        # 6. Local construction for each agent
        self._local_construct_all()

        # 7. Elimination and rebirth
        if (
            self.use_rebirth
            and self.iteration % self.elimination_interval == 0
        ):
            self.eliminate_and_rebirth()

    # ------------------------------------------------------------------
    #  STEP SUB-METHODS
    # ------------------------------------------------------------------

    def evaluate_population(self) -> None:
        """Evaluate any agents that need evaluation.

        Normally, evaluation happens during local_construct.
        """
        pass

    def update_archive(self) -> None:
        """Merge population into external Pareto archive.

        A <- non_dominated_update(A ∪ Pop)
        """
        self.archive.update(self.agents)

        # Update ideal/nadir with current archive
        all_objs = [a.objectives for a in self.agents] + self.archive.objectives
        self.problem.update_ideal_nadir(all_objs)

    def assign_ranks_and_crowding(self) -> None:
        """Non-dominated sorting and crowding distance assignment.

        Section 12: Assigns r_i, P_i, CD_i, C_i to each agent.
        """
        objectives_list = [a.objectives for a in self.agents]
        if not objectives_list:
            return

        # Non-dominated sort
        fronts = pareto.non_dominated_sort(objectives_list)

        # Assign ranks
        ranks = np.ones(len(self.agents), dtype=int) * len(fronts)
        for l, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = l + 1

        for i, agent in enumerate(self.agents):
            agent.pareto_rank = int(ranks[i])

        # Rank scores
        rank_scores = pareto.pareto_rank_score(ranks)
        for i, agent in enumerate(self.agents):
            agent.rank_score = float(rank_scores[i])

        # Crowding distance per front
        objs_array = np.stack(objectives_list)
        all_cd = np.zeros(len(self.agents))

        for front in fronts:
            if len(front) <= 1:
                for idx in front:
                    all_cd[idx] = np.inf
            else:
                front_objs = objs_array[front]
                front_cd = pareto.crowding_distance(front_objs)
                for j, idx in enumerate(front):
                    all_cd[idx] = front_cd[j]

        crowding_scores = pareto.normalize_crowding_score(all_cd)

        for i, agent in enumerate(self.agents):
            agent.crowding_distance = float(all_cd[i])
            agent.crowding_score = float(crowding_scores[i])

    def compute_decision_diversity(self) -> None:
        """Compute decision-space diversity for each agent.

        Section 13.2:
            AvgSim_i = mean_{j != i} sim(S_i, S_j)
            D_i = 1 - AvgSim_i
        """
        solutions = [a.solution for a in self.agents]
        n = len(solutions)
        if n <= 1:
            for agent in self.agents:
                agent.decision_diversity = 0.0
            return

        for i in range(n):
            total_sim = 0.0
            for j in range(n):
                if i != j:
                    total_sim += operators.jaccard_similarity(
                        solutions[i], solutions[j]
                    )
            avg_sim = total_sim / (n - 1)
            self.agents[i].decision_diversity = 1.0 - avg_sim

    def allocate_budgets(self) -> None:
        """Allocate search budget to each agent.

        Phase 1: fixed budget G_i = G_0.
        Section 15.1.
        """
        if self.budget_mode == "fixed":
            budgets = budget_mod.allocate_fixed_budget(
                len(self.agents), self.base_budget
            )
        elif self.budget_mode in (
            "pareto_crowding_decision",
            "dynamic",
        ):
            rank_scores = np.array([a.rank_score for a in self.agents])
            crowding_scores = np.array([a.crowding_score for a in self.agents])
            diversity_scores = np.array(
                [a.decision_diversity for a in self.agents]
            )
            budgets = budget_mod.allocate_dynamic_budget(
                rank_scores=rank_scores,
                crowding_scores=crowding_scores,
                diversity_scores=diversity_scores,
                base_budget=self.base_budget,
                alpha=self.alpha,
                beta=self.beta,
                delta=self.delta,
                gamma=self.gamma,
            )
        else:
            budgets = budget_mod.allocate_fixed_budget(
                len(self.agents), self.base_budget
            )

        ops = budget_mod.budget_to_operations(budgets)
        for i, agent in enumerate(self.agents):
            agent.budget = int(ops[i])

    # ------------------------------------------------------------------
    #  LOCAL CONSTRUCTION
    # ------------------------------------------------------------------

    def _local_construct_all(self) -> None:
        """Apply local construction to each agent.

        Section 19, step 5. Each agent executes ops_i construction attempts.
        """
        for agent in self.agents:
            ops_i = int(agent.budget)
            for _ in range(ops_i):
                if self.fe_count >= self.max_fe:
                    return
                self._local_construct_one(agent)

    def _local_construct_one(self, agent: Agent) -> None:
        """Execute one local construction step for a single agent.

        Section 9-10:
        1. Sample shop L from pool.
        2. Generate add/replace candidates.
        3. Evaluate and select best.
        4. Accept or reject.
        """
        # Sample shop
        shop = operators.sample_shop(
            self.pool,
            agent.component_preference,
            self.agents,
            self.shop_size,
            self.rng,
        )

        # Generate candidates
        candidates = operators.generate_all_candidates(
            agent.solution, shop, self.solution_capacity
        )

        if not candidates:
            return

        # Filter infeasible candidates
        feasible: list[tuple[set[int], np.ndarray]] = []
        for cand_sol in candidates:
            repaired = self.problem.repair(cand_sol)
            if len(repaired) > self.solution_capacity:
                continue
            cand_obj = self.problem.evaluate(repaired)
            self.fe_count += 1
            feasible.append((repaired, cand_obj))

        if not feasible:
            return

        best_candidate = None
        best_score = -np.inf

        for cand_sol, cand_obj in feasible:
            if self._accept_candidate(cand_obj, cand_sol, agent):
                score = operators.compute_preference_score(
                    cand_obj,
                    agent.objective_preference,
                    self.problem.ideal_point,
                    self.problem.nadir_point,
                )
                if score > best_score:
                    best_score = score
                    best_candidate = (cand_sol, cand_obj)

        if best_candidate is not None:
            new_sol, new_obj = best_candidate
            old_sol = agent.solution
            old_obj = agent.objectives.copy()

            # Update pool state
            self.pool.update_transition(old_sol, new_sol)

            # Update preferences (pass both old and new objectives)
            self._update_preferences(agent, old_sol, new_sol, old_obj, new_obj)

            # Update agent
            agent.solution = new_sol
            agent.objectives = new_obj

            # Update archive
            agent.no_archive_contribution_steps = 0
            self.archive.update([agent])
        else:
            agent.no_archive_contribution_steps += 1

    def _accept_candidate(
        self,
        candidate_obj: np.ndarray,
        candidate_sol: set[int],
        current: Agent,
    ) -> bool:
        """Acceptance criteria. Section 10.5, Phase 1 simplified.

        1. If S' dominates S_i, accept.
        2. If S' is non-dominated and contributes to archive, accept.
        3. If mutually non-dominated and g_i(S') > g_i(S_i), accept.
        4. Otherwise reject.
        """
        current_obj = current.objectives

        # Rule 1: S' dominates S_i
        if pareto.dominates(candidate_obj, current_obj):
            return True

        # Rule 2: S' is non-dominated (vs S_i) and contributes to archive
        # Section 10.5: "若 S' 非支配，且对 Archive 具有正贡献，接受"
        if not pareto.dominates(current_obj, candidate_obj):
            dominated_count = pareto.count_dominated_by_candidate(
                candidate_obj, self.archive.objectives
            )
            if dominated_count > self.archive_contribution_threshold:
                return True

        # Rule 3: mutually non-dominated and preference score is better
        if not pareto.dominates(current_obj, candidate_obj):
            cand_score = operators.compute_preference_score(
                candidate_obj,
                current.objective_preference,
                self.problem.ideal_point,
                self.problem.nadir_point,
            )
            curr_score = operators.compute_preference_score(
                current_obj,
                current.objective_preference,
                self.problem.ideal_point,
                self.problem.nadir_point,
            )
            if cand_score > curr_score:
                return True

        # Rule 4 (deferred): novelty-based acceptance
        # Rule 5: probabilistic acceptance (deferred)
        return False

    def _update_preferences(
        self,
        agent: Agent,
        old_sol: set[int],
        new_sol: set[int],
        old_obj: np.ndarray,
        new_obj: np.ndarray,
    ) -> None:
        """Update component preferences after solution change.

        Section 16:
          Delta F = F(S') - F(S_i)
          Delta_F_hat = normalize(Delta F)
          Delta g_i = w_i^T * Delta_F_hat

          If Delta g_i > 0:
            pi_{i,j} += mu * Delta g_i  for j in J_add
            pi_{i,j} *= (1 - mu)         for j in J_remove
          Then normalize pi_i.

        Phase 1: stub — only applies if use_strategy_inheritance is True.
        """
        if not self.use_strategy_inheritance:
            return

        j_add = new_sol - old_sol
        j_remove = old_sol - new_sol

        # delta_g = w_i^T * (F_hat(S') - F_hat(S_i))
        old_hat = self.problem.normalize_objectives(old_obj)
        new_hat = self.problem.normalize_objectives(new_obj)
        delta_f_hat = new_hat - old_hat
        delta_g = float(np.dot(agent.objective_preference, delta_f_hat))

        mu = self.preference_learning_rate

        if delta_g > 0:
            # Reinforce added components (Section 16: pi_{i,j} += mu * delta_g)
            for j in j_add:
                if 0 <= j < len(agent.component_preference):
                    agent.component_preference[j] += mu * delta_g
            # Decay removed components (Section 16: pi_{i,j} *= (1 - mu))
            for j in j_remove:
                if 0 <= j < len(agent.component_preference):
                    agent.component_preference[j] *= (1.0 - mu)

        # Normalize (Section 16: pi_i <- pi_i / sum(pi_{i,j}))
        total = agent.component_preference.sum()
        if total > 0:
            agent.component_preference /= total
        else:
            agent.component_preference[:] = (
                1.0 / self.problem.num_components
            )

    # ------------------------------------------------------------------
    #  ELIMINATION AND REBIRTH
    # ------------------------------------------------------------------

    def eliminate_and_rebirth(self) -> None:
        """Periodic elimination and rebirth of low-performing agents.

        Section 17. Phase 1: simplified (no strategy inheritance).
        Phase 3: adds smooth strategy preference inheritance.
        """
        n = len(self.agents)
        n_replace = max(1, int(n * self.replacement_rate))

        # Compute retention scores
        scores = self._compute_retention_scores()

        # Sort by score (ascending) — worst first
        sorted_indices = np.argsort(scores)
        remove_indices = set(sorted_indices[:n_replace].tolist())

        # Save objective preferences of removed agents BEFORE removing them
        removed_preferences: list[np.ndarray] = []
        for i in sorted_indices[:n_replace]:
            removed_preferences.append(
                self.agents[i].objective_preference.copy()
            )

        # Remove agents and release their components
        new_agents = []
        for i, agent in enumerate(self.agents):
            if i in remove_indices:
                # Release components (hard-cap pool)
                if self.pool.mode == "hard_cap":
                    for j in agent.solution:
                        if j in self.pool._remaining:
                            self.pool._remaining[j] += 1
            else:
                new_agents.append(agent)

        self.agents = new_agents

        # Reborn: create replacement agents
        for pref_idx, w_old in enumerate(removed_preferences):
            if self.keep_reference_direction:
                w_new = w_old.copy()
            else:
                w_new = self.rng.dirichlet(
                    np.ones(self.num_objectives)
                )

            # Component preference
            if self.use_strategy_inheritance:
                # Phase 3: smooth inheritance from archive elite matching w_new
                pi_new = self._inherit_preference(w_new)
            else:
                # Phase 1: random
                pi_new = self.rng.dirichlet(
                    np.ones(self.num_components)
                )

            # Construct initial solution via pool sampling
            initial_components = self.pool.sample(
                pi_new, self.agents, self.solution_capacity, self.rng
            )
            S_new = set(initial_components[: self.solution_capacity])
            S_new = self.problem.repair(S_new)

            F_new = self.problem.evaluate(S_new)
            self.fe_count += 1

            new_agent = Agent(
                solution=S_new,
                objectives=F_new,
                component_preference=pi_new,
                objective_preference=w_new,
                budget=self.base_budget,
            )
            self.agents.append(new_agent)

    def _compute_retention_scores(self) -> np.ndarray:
        """Compute retention scores for elimination.

        Section 17.1:
            R_i = a * P_i + b * C_i + d * D_i
        """
        scores = np.zeros(len(self.agents))
        for i, agent in enumerate(self.agents):
            scores[i] = (
                self.retention_a * agent.rank_score
                + self.retention_b * agent.crowding_score
                + self.retention_d * agent.decision_diversity
            )
        return scores

    def _inherit_preference(self, w_new: np.ndarray) -> np.ndarray:
        """Smooth strategy preference inheritance. Section 17.5.

        Phase 3: pi_new = (1 - eta) * pi_random + eta * pi_elite
        where pi_elite is derived from the archive elite matching w_new.
        """
        # Find archive elite matching the target objective preference
        elite_result = self.archive.select_elite_by_direction(w_new)

        if elite_result is None:
            return self.rng.dirichlet(np.ones(self.num_components))

        elite_sol, _ = elite_result
        M = self.num_components
        lambda_s = self.inheritance_smoothing
        eta = self.inheritance_strength

        # Derive pi_elite from elite solution (Section 17.5 formula)
        pi_elite = np.full(M, lambda_s / M)
        elite_size = len(elite_sol)
        if elite_size > 0:
            elite_weight = (1.0 - lambda_s) / elite_size
            for j in elite_sol:
                if 0 <= j < M:
                    pi_elite[j] += elite_weight

        # Smooth blend
        pi_random = self.rng.dirichlet(np.ones(M))
        pi_new = (1.0 - eta) * pi_random + eta * pi_elite
        pi_new /= pi_new.sum()

        return pi_new

    # ------------------------------------------------------------------
    #  METRICS
    # ------------------------------------------------------------------

    def _record_metrics(self, runtime_seconds: float = 0.0) -> None:
        """Record current iteration metrics. Section 19, step 7."""
        archive_objs = self.archive.get_objectives_array()

        # Hypervolume
        hv = 0.0
        if archive_objs.size > 0 and self.ref_point is not None:
            hv = metrics_mod.compute_hypervolume(archive_objs, self.ref_point)

        # Jaccard distance
        avg_jd = metrics_mod.compute_average_jaccard_distance(
            self.archive.solutions
        )

        # Component entropy
        occupancy = metrics_mod.compute_pool_occupancy(self.agents)
        entropy, entropy_norm = metrics_mod.compute_component_entropy(
            occupancy, self.num_components
        )

        # Reuse concentration
        concentration = metrics_mod.compute_reuse_concentration(occupancy)

        # Reference direction coverage
        dir_coverage = 0.0
        if self.archive.objectives and len(self.directions) > 0:
            dir_coverage = metrics_mod.compute_reference_direction_coverage(
                self.archive.objectives, self.directions
            )

        self.tracker.record(
            iteration=self.iteration,
            fe_count=self.fe_count,
            archive_size=len(self.archive),
            hypervolume=hv,
            avg_jaccard_distance=avg_jd,
            component_entropy=entropy,
            component_entropy_norm=entropy_norm,
            reuse_concentration=concentration,
            direction_coverage=dir_coverage,
            pool_occupancy=occupancy,
            runtime_seconds=runtime_seconds,
        )
