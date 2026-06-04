# MO-SPPS 一区标准实验测试方案

本文档用于指导 **MO-SPPS（Multi-Objective Shared-Pool Population Search）** 后续实验测试。目标是按照较高水平期刊的实验要求，系统验证算法在多目标组合优化中的有效性、泛化性、稳定性和可解释性。

实验主线包括五类问题：

1. MOSCSP 多目标协同组件选择；
2. MOKP / MOMKP 多目标背包与多目标多维背包；
3. Multi-objective Maximum Coverage 多目标最大覆盖；
4. 多目标特征选择；
5. Multi-objective QUBO / Max-Cut，作为强交互变量问题扩展。

所有实验均应围绕一个核心命题展开：

> MO-SPPS 是否能够在保持 Pareto 目标质量不显著下降的前提下，通过共享池机制获得更高的决策空间结构多样性，并且优于仅在选择阶段加入决策多样性补偿的传统 MOEA 变体。

---

## 1. 总体实验结构

### 1.1 实验问题总表

| 编号 | 问题 | 是否必做 | 主要作用 | 推荐优先级 |
|---|---|---|---|---|
| P1 | MOSCSP 多目标协同组件选择 | 必做 | 机制验证，展示共享池对热门组件复用和结构同质化的抑制作用 | 最高 |
| P2 | MOKP / MOMKP 多目标背包 | 必做 | 经典公开组合优化 benchmark，验证算法泛化能力 | 最高 |
| P3 | Multi-objective Maximum Coverage | 必做 | 子集选择类公开问题，验证共享池在覆盖类问题中的效果 | 高 |
| P4 | 多目标特征选择 | 必做 | 实际应用验证，体现算法应用价值 | 高 |
| P5 | Multi-objective QUBO / Max-Cut | 推荐 | 强交互变量验证，提高论文实验强度 | 中高 |

若计算资源有限，最低实验组合为：

```text
MOSCSP + MOKP/MOMKP + Maximum Coverage + Feature Selection
```

若目标为一区标准，建议完整执行五类问题。

---

## 2. 统一实验原则

### 2.1 评估预算公平

所有算法必须使用相同的目标函数评估次数：

\[
FE_{max}
\]

不同算法可具有不同迭代次数，但最终比较必须基于相同函数评估预算。

同时报告：

| 项目 | 含义 |
|---|---|
| Function Evaluations | 目标函数评估次数 |
| Runtime | 实际运行时间 |
| Iterations | 算法迭代次数 |
| Time-to-HV | 达到指定 HV 阈值所需时间 |

### 2.2 随机种子

每组实验至少运行 30 次独立重复实验。

推荐种子：

```yaml
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
```

### 2.3 参数调优公平

所有算法应遵循相同调参原则：

1. 在小规模验证集上进行预调参；
2. 主实验固定参数；
3. 报告所有关键参数；
4. 不在测试结果上反复调参；
5. 对 MO-SPPS 和对比算法使用相同评估预算。

### 2.4 统计检验

建议使用以下统计检验：

| 检验方法 | 用途 |
|---|---|
| Wilcoxon signed-rank test | 两算法配对比较 |
| Mann–Whitney U test | 两算法非配对比较 |
| Friedman test | 多算法整体比较 |
| Holm post-hoc test | 多重比较校正 |

显著性水平：

\[
p<0.05
\]

报告格式建议：

```text
mean ± std
rank
p-value
win/tie/loss
```

---

## 3. MO-SPPS 算法版本

### 3.1 主算法版本

| 版本 | 说明 | 是否必跑 |
|---|---|---|
| MO_SPPS_Full | 完整算法 | 必跑 |
| MO_SPPS_NoPool | 移除共享池压力（pool_mode=none） | 必跑 |
| MO_SPPS_OldSoftPool | 原始截断软压力公式（pool_mode=soft_pressure） | 必跑 |
| MO_SPPS_ContinuousPool | 连续型软压力公式 + 固定 \(Q_j\)（pool_mode=continuous） | 必跑 |
| MO_SPPS_AdaptiveQ | 自适应共享池容量 \(Q_j\) | 必跑 |
| MO_SPPS_NoBudget | 移除动态预算，使用固定预算 | 必跑 |
| MO_SPPS_NoInherit | 移除策略偏好继承 | 必跑 |
| MO_SPPS_NoDecisionDiversity | 移除决策空间多样性项（delta=0） | 必跑 |
| MO_SPPS_HardCapPool | 使用硬容量共享池 | 推荐 |

### 3.2 消融实验分层

为避免实验量过大，建议按层次执行消融。

#### 核心消融，所有问题都做

| 版本 | 目的 |
|---|---|
| MO_SPPS_NoPool | 验证共享池是否有效 |
| MO_SPPS_ContinuousPool | 验证连续软压力公式 |
| MO_SPPS_AdaptiveQ | 验证自适应 \(Q_j\) 是否有效 |
| MO_SPPS_Full | 完整算法表现 |

#### 完整消融，至少在 MOSCSP 与 MOKP/MOMKP 上做

| 版本 | 目的 |
|---|---|
| MO_SPPS_OldSoftPool | 对比截断压力与连续压力 |
| MO_SPPS_NoBudget | 验证预算调度贡献 |
| MO_SPPS_NoInherit | 验证策略继承贡献 |
| MO_SPPS_NoDecisionDiversity | 验证决策空间多样性项贡献 |
| MO_SPPS_HardCapPool | 检查硬容量池风险 |

---

## 4. MO-SPPS 默认参数

### 4.1 共享池参数

```yaml
shared_pool:
  mode: continuous
  epsilon: 0.01
  tau: 1.0
  utility_guidance_kappa: 0.5
  base_capacity_Q0: 12
```

连续型软压力项：

\[
\phi_j^t=
\left(
\epsilon+
\frac{1}{1+u_j^t/Q_j^t}
\right)^\tau
\]

采样概率：

\[
p_{i,j}^t=
\frac{
\phi_j^t\rho_j\pi_{i,j}^t
}{
\sum_{l=1}^{M}\phi_l^t\rho_l\pi_{i,l}^t
}
\]

参数定义：

| 符号 | 含义 | 默认值 |
|---|---|---|
| \(u_j^t\) | 第 \(t\) 轮组件 \(j\) 的活跃种群占用次数 | 动态统计 |
| \(Q_j^t\) | 组件 \(j\) 的参考容量或压力尺度 | 12 或自适应 |
| \(\epsilon\) | 最小采样保底项 | 0.01 |
| \(\tau\) | 共享池压力强度 | 1.0 |
| \(\kappa\) | 效用引导强度 | 0.5 |
| \(\pi_{i,j}^t\) | Agent \(i\) 对组件 \(j\) 的偏好 | 动态学习 |

### 4.2 自适应 \(Q_j\)

```yaml
adaptive_capacity:
  use_adaptive_Q: false
  base_capacity_Q0: 12
  alpha_Q: 2.0
  Q_min: 3
  Q_max: 30
  update_interval: 10
```

自适应容量：

\[
Q_j^t=
clip
\left(
Q_0(1+\alpha_Q C_j^t),
Q_{min},
Q_{max}
\right)
\]

贡献得分：

\[
C_j^t=
\frac{
\sum_{S\in \mathcal A^t}I(j\in S)q(S)
}{
\sum_{S\in \mathcal A^t}q(S)+\epsilon
}
\]

质量权重推荐：

\[
q(S)=1+CD_{obj}(S)
\]

参数定义：

| 符号 | 含义 | 默认值 |
|---|---|---|
| \(Q_0\) | 基础容量 | 5 |
| \(\alpha_Q\) | Archive 贡献放大系数 | 2.0 |
| \(C_j^t\) | 组件历史贡献得分 | 动态计算 |
| \(Q_{min}\) | 最小容量 | 1 |
| \(Q_{max}\) | 最大容量 | 20 |
| \(q(S)\) | Archive 解质量权重 | \(1+CD_{obj}(S)\) |

### 4.3 种群与搜索预算参数

```yaml
population:
  population_size: 50
  max_function_evaluations: 5000

local_search:
  shop_size: 5
  use_probabilistic_acceptance: true
  use_novelty_acceptance: true
  use_release_operation: true
  max_release_candidates: 3
  release_quality_loss_threshold: 0.01
  temperature: 0.5
  archive_contribution_threshold: 0.0
  novelty_threshold: 0.3
  quality_loss_threshold: 0.02

budget:
  mode: dynamic
  base_budget: 2.0
  alpha_pareto: 1.0
  beta_crowding: 1.0
  delta_decision_diversity: 0.4
  gamma_exploration: 0.5

rebirth:
  use_rebirth: true
  use_strategy_inheritance: true
  elimination_interval: 3
  replacement_rate: 0.2
  inheritance_strength: 0.5
  inheritance_smoothing: 0.1
  preference_learning_rate: 0.01
  keep_reference_direction: true
  retention_a: 0.5
  retention_b: 0.3
  retention_d: 0.2

archive:
  max_size: 200
  prune_method: hybrid_objective_decision
  objective_weight: 0.7
  decision_weight: 0.3
  remove_duplicates: true
```

---

## 5. 统一评价指标

### 5.1 目标空间指标

| 指标 | 含义 | 趋势 |
|---|---|---|
| Hypervolume, HV | Pareto 解集覆盖体积 | 越大越好 |
| IGD | 到参考前沿的平均距离 | 越小越好 |
| GD | 解集到参考前沿的距离 | 越小越好 |
| Coverage Metric | 一个算法支配另一个算法解集的比例 | 越大越好 |
| Epsilon Indicator | 逼近前沿所需最小偏移 | 越小越好 |
| Spread | 前沿分布范围 | 越大或越均匀越好 |
| Spacing | 相邻解距离一致性 | 越小越均匀 |
| Archive Size | 非支配解数量 | 结合质量判断 |

### 5.2 决策空间结构指标

| 指标 | 含义 | 趋势 |
|---|---|---|
| Average Jaccard Distance | Pareto 解结构平均差异 | 越大越好 |
| Component Entropy | 组件占用熵 | 越大越分散 |
| Unique Pattern Count | 不同组件模式数量 | 越大越好 |
| Pareto Set Structural Diversity | Pareto 解集结构多样性 | 越大越好 |
| Near-equivalent Structural Diversity | 目标相近解之间的结构差异 | 越大越好 |

### 5.3 共享池机制指标

| 指标 | 含义 | 用途 |
|---|---|---|
| Pool Occupancy Curve | 组件占用曲线 | 观察共享池压力是否生效 |
| Scarcity Index | 热门组件稀缺程度 | 衡量组件复用压力 |
| Reuse Concentration | 组件复用集中度 | 判断是否存在热门组件垄断 |
| Route Transition Rate | Agent 组合路线转移频率 | 判断搜索是否能跳出同质结构 |
| Pool Pressure Sensitivity | 不同 \(\tau\)、\(Q_j\) 下的结果变化 | 参数敏感性分析 |

### 5.4 效率指标

| 指标 | 含义 |
|---|---|
| Runtime | 实际运行时间 |
| Function Evaluations | 目标函数评估次数 |
| Time-to-HV | 达到指定 HV 所需评估次数或时间 |
| Archive Update Cost | Archive 维护成本 |
| Memory Cost | 内存占用 |

---

## 6. P1：MOSCSP 多目标协同组件选择

### 6.1 问题作用

MOSCSP 是 MO-SPPS 的主机制验证问题，用于展示共享池机制在热门组件复用、多簇协同结构和决策空间同质化场景下的作用。

### 6.2 问题定义

候选解：

\[
S\subseteq V,\quad |S|\le K
\]

目标：

\[
F(S)=(Quality(S),-Cost(S))
\]

质量函数：

\[
Quality(S)=\sum_{j\in S}v_j+\lambda\sum_{r\in R}B_rI(S\ satisfies\ r)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(v_j\) | 组件 \(j\) 的基础价值 |
| \(\lambda\) | 协同强度 |
| \(R\) | 协同规则集合 |
| \(B_r\) | 协同规则 \(r\) 的奖励 |
| \(I(\cdot)\) | 指示函数 |

### 6.3 实例设置

| 实例类型 | 目的 | 推荐规模 |
|---|---|---|
| MOSCSP-LowSynergy | 检查共享池是否有副作用 | \(M=50,100\), \(K=5,10\) |
| MOSCSP-HotComponents | 测试热门组件复用抑制 | \(M=50,100,200\), \(K=10,15\) |
| MOSCSP-MultiCluster | 测试多条结构路线发现能力 | \(M=100,200\), \(K=10,20\) |
| MOSCSP-Large | 测试扩展性 | \(M=500\), \(K=20\) |

推荐参数：

```yaml
moscsp:
  M: [50, 100, 200, 500]
  K: [5, 10, 15, 20]
  objectives: [2, 3]
  synergy_lambda: [0.5, 1.0, 2.0, 5.0]
  cluster_count: [2, 4, 8]
```

### 6.4 对比算法

| 类别 | 算法 | 是否必跑 |
|---|---|---|
| 基础算法 | Random MO Search | 必跑 |
| 基础算法 | Greedy Scalarization | 必跑 |
| 经典 MOEA | NSGA-II | 必跑 |
| 经典 MOEA | SPEA2 | 必跑 |
| 经典 MOEA | MOEA/D | 必跑 |
| 经典 MOEA | IBEA | 必跑 |
| 决策多样性对照 | NSGA-II + Decision Diversity | 必跑 |
| 组合优化强对照 | Pareto Local Search | 必跑 |
| 组合优化强对照 | MOEA/D + Local Search | 必跑 |
| 本文算法 | MO_SPPS_Full, MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_NoBudget, MO_SPPS_NoInherit, MO_SPPS_NoDecisionDiversity, MO_SPPS_HardCapPool | 必跑 |

### 6.5 重点指标

| 指标 | 目的 |
|---|---|
| HV / IGD | 判断目标空间质量 |
| Average Jaccard Distance | 判断 Pareto 解结构差异 |
| Component Entropy | 判断组件使用是否更分散 |
| Reuse Concentration | 判断热门组件是否被抑制 |
| Route Transition Rate | 判断 Agent 是否出现路线切换 |
| Pool Occupancy Curve | 展示共享池压力的动态作用 |

### 6.6 预期结论

MO-SPPS 应在 HotComponents 和 MultiCluster 实例中表现出更强优势：

1. HV / IGD 不弱于主要基线；
2. Jaccard Distance 和 Component Entropy 显著高于 MO_SPPS_NoPool 与 NSGA-II；
3. Reuse Concentration 明显下降；
4. Archive 中出现更多结构不同但目标质量接近的 Pareto 解。

---

## 7. P2：MOKP / MOMKP 多目标背包

### 7.1 问题作用

MOKP / MOMKP 用于验证 MO-SPPS 在经典公开多目标组合优化问题上的泛化能力。

### 7.2 问题定义

候选解：

\[
S\subseteq V
\]

多目标收益：

\[
\max F(S)=
\left(
\sum_{j\in S}p_{1j},
\sum_{j\in S}p_{2j},
\dots,
\sum_{j\in S}p_{mj}
\right)
\]

容量约束：

\[
\sum_{j\in S}w_{rj}\le C_r,\quad r=1,\dots,R
\]

其中：

| 符号 | 含义 |
|---|---|
| \(p_{kj}\) | 物品 \(j\) 在目标 \(k\) 上的收益 |
| \(w_{rj}\) | 物品 \(j\) 在约束维度 \(r\) 上的重量 |
| \(C_r\) | 第 \(r\) 个背包容量 |
| \(m\) | 目标数量 |
| \(R\) | 约束维度数量 |

### 7.3 实例设置

| 实例类型 | 目标数 | 物品数 | 约束维度 | 目的 |
|---|---:|---:|---:|---|
| MOKP-2obj | 2 | 250, 500, 750, 1000 | 1 | 基础公开 benchmark |
| MOKP-3obj | 3 | 250, 500, 750 | 1 | 多目标扩展测试 |
| MOKP-4obj | 4 | 250, 500 | 1 | 高目标数压力测试 |
| MOMKP-2obj | 2 | 250, 500 | 2, 5, 10 | 多维约束测试 |
| MOMKP-3obj | 3 | 250, 500 | 2, 5 | 多目标多约束测试 |
| Multi-cluster MOKP | 2, 3 | 250, 500 | 1, 5 | 多簇替代结构测试 |
| Hot-item MOKP | 2, 3 | 250, 500 | 1, 5 | 热门物品复用测试 |

### 7.4 MO-SPPS 适配

MOKP 不应强制使用 \(|S|\le K\) 作为主约束，应使用容量约束与 repair 操作。

局部操作：

| 操作 | 说明 |
|---|---|
| Add | 若加入物品后满足容量约束，则加入 |
| Replace | 用新物品替换旧物品后检查容量 |
| Remove | 主动释放低贡献或高重量物品 |
| Repair | 若超容量，按低贡献/高重量比移除物品直到可行 |

建议 repair 规则：

\[
score_j=\frac{\sum_{k=1}^{m}\hat p_{kj}}{\sum_{r=1}^{R}\hat w_{rj}+\epsilon}
\]

若解不可行，则优先移除 \(score_j\) 最低的物品。

### 7.5 对比算法

| 类别 | 算法 | 是否必跑 |
|---|---|---|
| 经典 MOEA | NSGA-II | 必跑 |
| 经典 MOEA | SPEA2 | 必跑 |
| 经典 MOEA | MOEA/D | 必跑 |
| 经典 MOEA | IBEA | 必跑 |
| 指标驱动 MOEA | SMS-EMOA | 必跑 |
| 组合优化强对照 | MOGLS | 必跑 |
| 组合优化强对照 | Pareto Local Search | 必跑 |
| 组合优化强对照 | 2PPLS | 必跑 |
| 组合优化强对照 | MOEA/D + Local Search | 必跑 |
| 决策多样性对照 | NSGA-II + Decision Diversity | 必跑 |
| 本文算法 | MO_SPPS_NoPool | 必跑 |
| 本文算法 | MO_SPPS_OldSoftPool | 必跑 |
| 本文算法 | MO_SPPS_ContinuousPool | 必跑 |
| 本文算法 | MO_SPPS_AdaptiveQ | 必跑 |
| 本文算法 | MO_SPPS_Full | 必跑 |

### 7.6 重点指标

| 指标 | 目的 |
|---|---|
| HV / IGD / GD | 目标质量与收敛性 |
| Coverage Metric | 与强基线的支配关系 |
| Jaccard Distance | 背包方案结构差异 |
| Component Entropy | 物品选择是否更均衡 |
| Feasibility Rate | 候选解可行率 |
| Runtime | 检查复杂度 |

### 7.7 预期结论

MO-SPPS 不一定必须在所有标准随机 MOKP 上明显优于 2PPLS 或 MOGLS，但应至少满足：

1. HV / IGD 接近强基线；
2. 决策空间结构多样性明显更高；
3. 在 Hot-item 和 Multi-cluster MOKP 上 Full 显著优于 MO_SPPS_NoPool；
4. MO_SPPS_AdaptiveQ 能缓解关键物品被过度抑制的问题。

---

## 8. P3：Multi-objective Maximum Coverage

### 8.1 问题作用

Maximum Coverage 用于验证 MO-SPPS 在覆盖类子集选择问题上的泛化能力。该问题天然存在集合重叠和热门集合复用现象，适合检验共享池机制。

### 8.2 问题定义

设元素全集为 \(U\)，候选集合为：

\[
V=\{A_1,A_2,\dots,A_M\}
\]

候选解：

\[
S\subseteq V,\quad |S|\le K
\]

覆盖目标：

\[
f_1(S)=\left|\bigcup_{A_j\in S}A_j\right|
\]

成本目标：

\[
f_2(S)=-\sum_{A_j\in S}cost_j
\]

多目标形式：

\[
F(S)=(Coverage(S),-Cost(S))
\]

### 8.3 实例设置

| 实例类型 | 设置 | 目的 |
|---|---|---|
| Normal Coverage | 中等重叠率 | 基础泛化测试 |
| High-overlap Coverage | 少数集合覆盖大量元素 | 热门集合复用测试 |
| Multi-cluster Coverage | 多组集合覆盖相似元素 | 多结构路线测试 |
| Large Coverage | 元素数 1000–5000，集合数 500–2000 | 扩展性测试 |

推荐参数：

```yaml
maximum_coverage:
  num_elements: [500, 1000, 5000]
  num_sets: [200, 500, 1000, 2000]
  K: [10, 20, 50]
  overlap_rate: [0.2, 0.5, 0.8]
  cluster_count: [2, 4, 8]
```

### 8.4 对比算法

| 类别 | 算法 | 是否必跑 |
|---|---|---|
| 基础算法 | Random MO Search | 必跑 |
| 基础算法 | Greedy Scalarization | 必跑 |
| 贪心算法 | Pareto Greedy | 必跑 |
| 经典 MOEA | NSGA-II | 必跑 |
| 经典 MOEA | SPEA2 | 必跑 |
| 经典 MOEA | MOEA/D | 必跑 |
| 经典 MOEA | IBEA | 必跑 |
| 组合优化强对照 | Pareto Local Search | 必跑 |
| 组合优化强对照 | Greedy + Local Search | 必跑 |
| 组合优化强对照 | MOEA/D + Local Search | 推荐 |
| 决策多样性对照 | NSGA-II + Decision Diversity | 必跑 |
| 本文算法 | MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full | 必跑 |

核心消融版本包括：

```text
MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full
```

### 8.5 重点指标

| 指标 | 目的 |
|---|---|
| HV / IGD | 目标质量 |
| Coverage Metric | 与其他算法支配关系 |
| Component Entropy | 集合选择是否集中 |
| Average Jaccard Distance | Pareto 解结构差异 |
| Reuse Concentration | 热门集合是否被抑制 |
| Runtime | 覆盖计算成本 |

### 8.6 预期结论

MO-SPPS 应在 High-overlap 和 Multi-cluster Coverage 中表现出明显优势：

1. 高重叠实例中组件熵更高；
2. 多簇覆盖实例中结构路线更多；
3. HV / IGD 不明显弱于 Greedy-LS 和 PLS；
4. Full 显著优于 MO_SPPS_NoPool 与 NSGA-II + Decision Diversity。

---

## 9. P4：多目标特征选择

### 9.1 问题作用

多目标特征选择用于实际应用验证。该问题中每个特征可视为一个组件，MO-SPPS 的共享池机制可用于避免 Pareto 解集中重复依赖少数热门特征。

### 9.2 问题定义

候选解：

\[
S\subseteq V
\]

二目标形式：

\[
F(S)=
\left(
Accuracy(S),-rac{|S|}{M}
\right)
\]

可选三目标形式：

\[
F(S)=
\left(
Accuracy(S),-rac{|S|}{M},-InferenceCost(S)
\right)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(S\) | 被选择的特征子集 |
| \(M\) | 总特征数 |
| \(Accuracy(S)\) | 使用特征子集 \(S\) 的分类准确率 |
| \(InferenceCost(S)\) | 推理代价，可由特征采集成本或模型推理时间近似 |

### 9.3 数据集设置

| 数据类型 | 数量 | 说明 |
|---|---:|---|
| UCI 小中型分类数据集 | 8–10 | 验证一般分类任务 |
| 高维基因表达数据集 | 5–8 | 验证高维小样本场景 |
| 高维稀疏数据集 | 2–3 | 验证稀疏特征选择能力 |

推荐记录：

| 数据集信息 | 必须记录 |
|---|---|
| 样本数 | 是 |
| 特征数 | 是 |
| 类别数 | 是 |
| 缺失值处理 | 是 |
| 标准化方式 | 是 |
| 训练/验证/测试划分 | 是 |

### 9.4 分类器设置

建议至少使用一种主分类器，最好使用两种验证稳定性。

| 分类器 | 是否推荐 | 说明 |
|---|---|---|
| KNN | 必做 | 特征选择常用，计算简单 |
| SVM | 推荐 | 检查泛化能力 |
| Random Forest | 可选 | 非线性分类器对照 |

推荐验证方式：

```yaml
feature_selection:
  validation: stratified_5_fold_cv
  classifier_primary: knn
  classifier_secondary: svm
  test_split: 0.2
```

### 9.5 对比算法

| 类别 | 算法 | 是否必跑 |
|---|---|---|
| 基础方法 | Filter Top-K | 必跑 |
| 基础方法 | Random Feature Subset | 必跑 |
| 经典 MOEA | Binary NSGA-II | 必跑 |
| 经典 MOEA | Binary SPEA2 | 必跑 |
| 经典 MOEA | Binary MOEA/D | 必跑 |
| 特征选择专用算法 | MOPSO-ASFS | 必跑 |
| 特征选择专用算法 | MOEA/D-FS | 必跑 |
| 特征选择专用算法 | NSGAIII-FS | 推荐 |
| 特征选择专用算法 | CNSGA-II | 推荐 |
| 决策多样性对照 | Binary NSGA-II + Decision Diversity | 必跑 |
| 本文算法 | MO_SPPS_NoPool | 必跑 |
| 本文算法 | MO_SPPS_ContinuousPool | 必跑 |
| 本文算法 | MO_SPPS_AdaptiveQ | 必跑 |
| 本文算法 | MO_SPPS_Full | 必跑 |

### 9.6 重点指标

| 指标 | 目的 |
|---|---|
| Accuracy | 分类性能 |
| Feature Count | 特征压缩能力 |
| HV / IGD | 多目标质量 |
| Stability | 多次运行选择稳定性 |
| Feature Entropy | 特征使用是否过度集中 |
| Jaccard Distance | Pareto 特征子集结构差异 |
| Runtime | 特征选择耗时 |

特征选择稳定性可定义为多次运行得到的 Pareto 解之间的平均 Jaccard 相似度或 Kuncheva 指数。

### 9.7 预期结论

MO-SPPS 在特征选择中应重点证明：

1. Accuracy 不弱于专用特征选择算法；
2. Feature Count 有竞争力；
3. 在相似 Accuracy 下能输出更多结构不同的特征子集；
4. MO_SPPS_AdaptiveQ 能避免真正强特征被过度抑制；
5. Full 优于 MO_SPPS_NoPool 和 Binary NSGA-II + Decision Diversity。

---

## 10. P5：Multi-objective QUBO / Max-Cut

### 10.1 问题作用

QUBO / Max-Cut 用于验证 MO-SPPS 在强变量交互问题上的能力。该问题可防止论文被认为只适用于简单加和型目标。

### 10.2 问题定义

二进制变量：

\[
x\in\{0,1\}^M
\]

多目标 QUBO：

\[
f_k(x)=x^TQ_kx,\quad k=1,\dots,m
\]

多目标形式：

\[
F(x)=\left(f_1(x),f_2(x),\dots,f_m(x)\right)
\]

Max-Cut 变体：

\[
F(x)=
\left(
CutWeight(x),-BalancePenalty(x)
\right)
\]

### 10.3 实例设置

| 实例类型 | 规模 | 目的 |
|---|---|---|
| Random QUBO | \(M=100,200,500\) | 基础强交互测试 |
| Correlated QUBO | \(M=100,200,500\) | 目标相关性测试 |
| Multi-cluster QUBO | \(M=200,500\) | 多结构路线测试 |
| Multi-objective Max-Cut | 节点数 100, 250, 500 | 图划分问题验证 |

推荐目标数：

```yaml
qubo:
  objectives: [2, 3]
  variables: [100, 200, 500]
  density: [0.05, 0.1, 0.2]
```

### 10.4 MO-SPPS 适配

在 QUBO / Max-Cut 中，组件可以理解为变量取 1 的选择，候选解为：

\[
S=\{j\mid x_j=1\}
\]

局部操作：

| 操作 | 含义 |
|---|---|
| Add | 将变量从 0 翻转为 1 |
| Remove | 将变量从 1 翻转为 0 |
| Replace | 一个 1 变 0，同时一个 0 变 1 |
| Flip | 单变量翻转，可作为额外局部操作 |

### 10.5 对比算法

| 类别 | 算法 | 是否必跑 |
|---|---|---|
| 经典 MOEA | NSGA-II | 必跑 |
| 经典 MOEA | SPEA2 | 必跑 |
| 经典 MOEA | MOEA/D | 必跑 |
| 经典 MOEA | IBEA | 必跑 |
| 组合优化强对照 | Pareto Local Search | 必跑 |
| 组合优化强对照 | Multi-objective Tabu Search | 必跑 |
| 组合优化强对照 | Memetic NSGA-II | 必跑 |
| 决策多样性对照 | NSGA-II + Decision Diversity | 必跑 |
| 本文算法 | MO_SPPS_NoPool | 必跑 |
| 本文算法 | MO_SPPS_ContinuousPool | 必跑 |
| 本文算法 | MO_SPPS_AdaptiveQ | 必跑 |
| 本文算法 | MO_SPPS_Full | 必跑 |

### 10.6 重点指标

| 指标 | 目的 |
|---|---|
| HV / IGD | 强交互目标下的目标质量 |
| Hamming Distance | 二进制解结构差异 |
| Component Entropy | 变量选择分散程度 |
| Runtime | 强交互计算成本 |
| Route Transition Rate | 搜索路线变化 |

### 10.7 预期结论

QUBO / Max-Cut 是扩展验证问题，不要求 MO-SPPS 在所有实例上压倒强局部搜索算法，但应证明：

1. 在 Multi-cluster QUBO 中结构多样性优势明显；
2. Full 优于 MO_SPPS_NoPool；
3. MO_SPPS_ContinuousPool 和 MO_SPPS_AdaptiveQ 对强交互场景仍然稳定；
4. 目标质量与强基线接近或在部分实例上占优。

---

## 11. 总体对比算法清单

### 11.1 通用基线

| 算法 | 适用问题 | 是否必跑 |
|---|---|---|
| Random MO Search | MOSCSP, Coverage, Feature Selection | 必跑 |
| Greedy Scalarization | MOSCSP, Coverage, MOKP | 必跑 |
| NSGA-II | 全部问题 | 必跑 |
| SPEA2 | 全部问题 | 必跑 |
| MOEA/D | 全部问题 | 必跑 |
| IBEA | 全部问题 | 必跑 |
| SMS-EMOA | MOKP/MOMKP，其他可选 | 推荐 |

### 11.2 组合优化强基线

| 算法 | 适用问题 | 是否必跑 |
|---|---|---|
| Pareto Local Search | MOSCSP, MOKP, Coverage, QUBO | 必跑 |
| 2PPLS | MOKP/MOMKP | 必跑 |
| MOGLS | MOKP/MOMKP | 必跑 |
| MOEA/D + Local Search | MOSCSP, MOKP, Coverage | 必跑 |
| Greedy + Local Search | Coverage | 必跑 |
| Multi-objective Tabu Search | QUBO/Max-Cut | 必跑 |
| Memetic NSGA-II | QUBO/Max-Cut，MOKP 可选 | 推荐 |

### 11.3 决策空间多样性对照

| 算法 | 适用问题 | 是否必跑 |
|---|---|---|
| NSGA-II + Decision Diversity | 全部问题 | 必跑 |
| SPEA2 + Decision Diversity | MOSCSP, MOKP 可选 | 可选 |
| MOEA/D + Decision Diversity | MOSCSP, MOKP 可选 | 可选 |

其中，NSGA-II + Decision Diversity 是关键对照，用于证明生成阶段共享池机制优于选择阶段事后多样性补偿。

### 11.4 特征选择专用算法

| 算法 | 是否必跑 |
|---|---|
| Filter Top-K | 必跑 |
| Random Feature Subset | 必跑 |
| Binary NSGA-II | 必跑 |
| Binary SPEA2 | 必跑 |
| Binary MOEA/D | 必跑 |
| MOPSO-ASFS | 必跑 |
| MOEA/D-FS | 必跑 |
| NSGAIII-FS | 推荐 |
| CNSGA-II | 推荐 |
| Binary NSGA-II + Decision Diversity | 必跑 |

---

## 12. 最终实验矩阵

### 12.1 完整实验矩阵

| 问题 | 对比算法 | MO-SPPS 版本 |
|---|---|---|
| MOSCSP | Random, Greedy, NSGA-II, SPEA2, MOEA/D, IBEA, PLS, MOEA/D-LS, NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_NoBudget, MO_SPPS_NoInherit, MO_SPPS_NoDecisionDiversity, MO_SPPS_HardCapPool, MO_SPPS_Full |
| MOKP / MOMKP | NSGA-II, SPEA2, MOEA/D, IBEA, SMS-EMOA, MOGLS, PLS, 2PPLS, MOEA/D-LS, NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_NoBudget, MO_SPPS_NoInherit, MO_SPPS_NoDecisionDiversity, MO_SPPS_HardCapPool, MO_SPPS_Full |
| Maximum Coverage | Random, Greedy, Pareto Greedy, NSGA-II, SPEA2, MOEA/D, IBEA, PLS, Greedy-LS, MOEA/D-LS, NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |
| Feature Selection | Filter Top-K, Random Subset, Binary NSGA-II, Binary SPEA2, Binary MOEA/D, MOPSO-ASFS, MOEA/D-FS, NSGAIII-FS, CNSGA-II, Binary NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |
| QUBO / Max-Cut | NSGA-II, SPEA2, MOEA/D, IBEA, PLS, MOTS, Memetic NSGA-II, NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |

### 12.2 精简但仍较强的矩阵

如果资源不足，可采用下表：

| 问题 | 对比算法 | MO-SPPS 版本 |
|---|---|---|
| MOSCSP | Random, Greedy, NSGA-II, SPEA2, MOEA/D, IBEA, NSGA-II+Div, PLS | MO_SPPS_NoPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |
| MOKP / MOMKP | NSGA-II, SPEA2, MOEA/D, IBEA, MOGLS, PLS, 2PPLS, NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |
| Maximum Coverage | Greedy, Pareto Greedy, NSGA-II, SPEA2, MOEA/D, IBEA, PLS, NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |
| Feature Selection | Filter Top-K, Binary NSGA-II, Binary MOEA/D, MOPSO-ASFS, MOEA/D-FS, Binary NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |
| QUBO / Max-Cut | NSGA-II, MOEA/D, IBEA, PLS, MOTS, NSGA-II+Div | MO_SPPS_NoPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full |

---

## 13. 参数敏感性分析

### 13.1 必做参数

| 参数 | 取值 | 目的 |
|---|---|---|
| \(\tau\) | 0, 0.5, 1.0, 2.0, 4.0 | 共享池压力强度 |
| \(Q_0\) | 1, 3, 5, 10, 20 | 基础容量尺度 |
| \(\alpha_Q\) | 0, 0.5, 1.0, 2.0, 5.0 | 自适应容量强度 |
| \(A_{max}\) | 100, 200, 500 | Archive 大小影响 |
| \(\delta\) | 0, 0.5, 1.0, 2.0 | 决策多样性预算权重 |
| \(\eta\) | 0, 0.25, 0.5, 0.75 | 策略继承强度 |

### 13.2 建议测试问题

参数敏感性不必在所有问题上完整展开，建议选择：

| 问题 | 原因 |
|---|---|
| MOSCSP-HotComponents | 最能体现共享池压力 |
| MOSCSP-MultiCluster | 最能体现结构路线多样性 |
| MOKP-3obj-500 | 公开 benchmark 代表 |
| Maximum Coverage-HighOverlap | 覆盖类热门组件代表 |

---

## 14. 结果图表清单

### 14.1 主结果表

每个问题至少给出：

| 表格 | 内容 |
|---|---|
| Table 1 | HV mean ± std + rank |
| Table 2 | IGD mean ± std + rank |
| Table 3 | Average Jaccard Distance mean ± std |
| Table 4 | Component Entropy mean ± std |
| Table 5 | Runtime mean ± std |
| Table 6 | Win/Tie/Loss 统计 |

### 14.2 消融表

| 表格 | 内容 |
|---|---|
| Ablation-HV | 各 MO-SPPS 版本 HV 对比 |
| Ablation-Diversity | 各版本决策多样性对比 |
| Ablation-Runtime | 各版本运行时间对比 |
| Ablation-Mechanism | MO_SPPS_NoPool、MO_SPPS_ContinuousPool、MO_SPPS_AdaptiveQ 的机制指标对比 |

### 14.3 推荐图

| 图 | 内容 |
|---|---|
| Figure 1 | HV 收敛曲线 |
| Figure 2 | IGD 收敛曲线 |
| Figure 3 | Component Entropy 曲线 |
| Figure 4 | Average Jaccard Distance 曲线 |
| Figure 5 | Pool Occupancy 热力图 |
| Figure 6 | Reuse Concentration 曲线 |
| Figure 7 | Pareto 前沿散点图 |
| Figure 8 | Pareto 解结构聚类图 |
| Figure 9 | 参数敏感性热力图 |
| Figure 10 | 目标质量—结构多样性 trade-off 图 |

---

## 15. 结果判断标准

### 15.1 认为 MO-SPPS 有效的条件

至少应满足：

1. MO_SPPS_Full 的 HV 高于或不弱于 MO_SPPS_NoPool；
2. MO_SPPS_Full 的决策空间多样性显著高于 MO_SPPS_NoPool；
3. MO_SPPS_Full 的决策空间多样性高于 NSGA-II、SPEA2 和 MOEA/D；
4. MO_SPPS_Full 的 HV / IGD 不弱于主要经典 MOEA；
5. 在 HotComponents、HighOverlap、MultiCluster 等结构性实例中优势更明显；
6. MO_SPPS_Full 优于或接近 NSGA-II + Decision Diversity；
7. MO_SPPS_ContinuousPool 优于 MO_SPPS_OldSoftPool 或更稳定；
8. MO_SPPS_AdaptiveQ 能减少关键组件被过度抑制导致的目标质量下降。

### 15.2 可接受结果

以下结果可以接受：

1. MO-SPPS 不总是击败 2PPLS、MOGLS 或 PLS；
2. 在随机 MOKP 上优势不明显；
3. HV 与 NSGA-II 接近，但决策空间多样性显著更好；
4. 在低协同或低重叠问题上优势有限；
5. HV 略低但结构多样性显著更高，可作为质量—结构多样性折中。

### 15.3 危险结果

以下结果说明算法需要调整：

1. Full 与 MO_SPPS_NoPool 无显著差异；
2. 决策空间多样性提高，但 HV / IGD 明显恶化；
3. NSGA-II + Decision Diversity 明显优于 MO-SPPS；
4. MO_SPPS_AdaptiveQ 不能保护关键组件，甚至降低性能；
5. 共享池参数极端敏感，缺少稳定有效区间；
6. 只在 MOSCSP 有效，在公开问题上无优势；
7. Runtime 明显高于强基线且目标质量没有补偿收益。

---

## 16. 推荐实验执行顺序

### 阶段一：最小机制验证

问题：

```text
MOSCSP-HotComponents
MOSCSP-MultiCluster
```

算法：

```text
MO_SPPS_NoPool
MO_SPPS_OldSoftPool
MO_SPPS_ContinuousPool
MO_SPPS_AdaptiveQ
MO_SPPS_Full
NSGA-II
NSGA-II + Decision Diversity
```

目标：

```text
确认共享池机制是否真的提高结构多样性，并且不明显损害 HV / IGD。
```

### 阶段二：公开组合优化验证

问题：

```text
MOKP / MOMKP
Maximum Coverage
```

算法：

```text
NSGA-II
SPEA2
MOEA/D
IBEA
PLS
MOGLS / 2PPLS
NSGA-II + Decision Diversity
MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full
```

目标：

```text
确认 MO-SPPS 不只适用于自定义 MOSCSP。
```

### 阶段三：应用验证

问题：

```text
Feature Selection
```

算法：

```text
Filter Top-K
Binary NSGA-II
Binary MOEA/D
MOPSO-ASFS
MOEA/D-FS
Binary NSGA-II + Decision Diversity
MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full
```

目标：

```text
证明 MO-SPPS 在实际组件选择问题中有应用价值。
```

### 阶段四：强交互扩展验证

问题：

```text
QUBO / Max-Cut
```

算法：

```text
NSGA-II
MOEA/D
IBEA
PLS
MOTS
Memetic NSGA-II
NSGA-II + Decision Diversity
MO_SPPS_NoPool, MO_SPPS_OldSoftPool, MO_SPPS_ContinuousPool, MO_SPPS_AdaptiveQ, MO_SPPS_Full
```

目标：

```text
证明 MO-SPPS 不只适用于加和型目标，也能处理强交互变量问题。
```

### 阶段五：统计检验与论文图表整理

任务：

1. 汇总 30 次独立运行结果；
2. 计算 mean、std、rank、p-value；
3. 执行 Friedman + Holm post-hoc；
4. 绘制收敛曲线、熵曲线、Jaccard 曲线；
5. 整理消融实验表；
6. 形成最终论文结果表。

---

## 17. 推荐目录结构

```text
experiments/
├── configs/
│   ├── moscsp/
│   ├── mokp/
│   ├── maximum_coverage/
│   ├── feature_selection/
│   └── qubo_maxcut/
├── data/
│   ├── raw/
│   ├── generated/
│   └── processed/
├── results/
│   ├── raw/
│   ├── metrics/
│   ├── statistics/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── run_single.py
│   ├── run_batch.py
│   ├── run_ablation.py
│   ├── run_sensitivity.py
│   ├── summarize_metrics.py
│   ├── statistical_tests.py
│   └── plot_figures.py
└── logs/
    ├── moscsp/
    ├── mokp/
    ├── maximum_coverage/
    ├── feature_selection/
    └── qubo_maxcut/
```

---

## 18. 每次运行必须保存的信息

每次实验运行必须保存：

```yaml
run_record:
  problem_name: string
  instance_id: string
  algorithm_name: string
  algorithm_version: string
  seed: int
  FE_max: int
  population_size: int
  archive_size: int
  start_time: timestamp
  end_time: timestamp
  runtime_seconds: float
  final_archive_size: int
  final_HV: float
  final_IGD: float
  final_GD: float
  final_jaccard_distance: float
  final_component_entropy: float
  final_reuse_concentration: float
  parameter_config_path: string
  result_archive_path: string
```

同时保存完整 Pareto Archive：

```text
solution_id
selected_components
objective_vector
rank
crowding_distance
decision_diversity
component_count
```

---

## 19. 最终投稿级结论模板

若实验结果理想，论文结论可围绕以下主张展开：

1. MO-SPPS 在候选解生成阶段引入共享池压力，能够有效缓解多目标组合优化中的结构同质化；
2. 连续型软压力公式比原始截断式软压力更稳定；
3. 自适应 \(Q_j\) 能保护高贡献组件，减少共享池对关键组件的误伤；
4. 相比 NSGA-II + Decision Diversity，MO-SPPS 的生成阶段调控能获得更高的结构多样性和更稳定的 Pareto 质量；
5. 在 MOSCSP、MOKP/MOMKP、Maximum Coverage、Feature Selection 和 QUBO/Max-Cut 上，MO-SPPS 展现出良好的泛化能力；
6. MO-SPPS 尤其适合存在热门组件复用、多簇替代结构和目标—结构双多样性需求的多目标组合优化问题。

---

## 20. 最终建议

按照一区标准，实验不应只证明 MO-SPPS 比 NSGA-II 好，而应证明以下三点：

```text
1. 比普通 MOEA 更能保持决策空间结构多样性；
2. 相比选择阶段决策多样性补偿，生成阶段共享池调控更有效；
3. 在公开组合优化问题和实际应用问题上具有泛化能力。
```

因此，最关键的实验组合是：

```text
MOSCSP + MOKP/MOMKP + Maximum Coverage + Feature Selection + QUBO/Max-Cut
```

最关键的对比算法是：

```text
NSGA-II, SPEA2, MOEA/D, IBEA,
PLS, MOGLS, 2PPLS, MOEA/D-LS,
NSGA-II + Decision Diversity,
Feature Selection 专用算法,
MO_SPPS_Full + 全消融变体
```

最关键的判断指标是：

```text
HV / IGD + Jaccard Distance + Component Entropy + Reuse Concentration + Runtime
```

如果上述实验结果成立，MO-SPPS 的论文说服力将明显接近一区期刊标准。
