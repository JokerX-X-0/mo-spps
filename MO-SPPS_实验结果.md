# MO-SPPS 实验结果

本文档与《MO-SPPS 实验测试方案》一一对应，记录每一项实验的详细结果与总结。实验未进行的部分留空，后续补充。

---

## 6. P1：MOSCSP 多目标协同组件选择

### 6.1 实验配置

| 参数 | 值 |
|---|---|
| 问题类型 | MOSCSP (high_synergy) |
| 组件数 M | 50 |
| 解容量 K | 10 |
| 目标数 | 2 |
| 最大函数评估 FE | 50000 |
| 种群大小 | 50 |
| 随机种子 | 0 |
| 独立运行次数 | 30 |

### 6.2 参测变体

| 变体 | pool_mode | use_adaptive_Q | 说明 |
|---|---|---|---|
| MO_SPPS_Full | continuous | False | 完整算法（连续池 + 固定 Q） |
| MO_SPPS_NoPool | none | — | 移除共享池 |
| MO_SPPS_OldSoftPool | soft_pressure | — | 原始截断软压力公式 |
| MO_SPPS_ContinuousPool | continuous | False | 连续软压力 + 固定 Q |
| MO_SPPS_AdaptiveQ | continuous | True | 自适应 Q_j |
| MO_SPPS_NoUtilityGuidance | continuous | False | κ = 0 |
| MO_SPPS_NoNoveltyAcceptance | continuous | False | 移除新颖性接受 |
| MO_SPPS_NoProbAcceptance | continuous | False | 移除概率接受 |
| MO_SPPS_NoReleaseOp | continuous | False | 移除 Release 操作 |
| MO_SPPS_CrowdingOnlyPrune | continuous | False | 纯拥挤距离剪枝 |
| MO_SPPS_HardCapPool | hard_cap | — | 硬容量共享池 |
| MO_SPPS_NoBudget | continuous | False | 固定预算 |
| MO_SPPS_NoInherit | continuous | False | 移除策略继承 |
| MO_SPPS_NoDecisionDiversity | continuous | False | δ = 0 |

### 6.3 结果

| 变体 | \|A\|| HV | Jaccard | Entropy | Gini | Time(s) |
|---|---|---|---|---|---|---|
| *(待填入)* | | | | | |

### 6.4 关键发现

*(待填入)*

---

## 7. P2：MOKP / MOMKP 多目标背包

### 7.1 实验配置

| 参数 | 值 |
|---|---|
| 物品数 n | 100 |
| 容量比例 W/Σw | 0.5 |
| 目标数 | 2 |
| 最大函数评估 FE | 20000 |
| 种群大小 | 100 |
| Q0 | 12 |
| 独立运行次数 | 30 |
| 并行 worker | 8 |

### 7.2 参测算法

**MO-SPPS 变体：**

| 变体 | pool_mode | use_adaptive_Q | 说明 |
|---|---|---|---|
| MO_SPPS_Full | continuous | False | 完整算法（连续池 + 固定 Q，默认策略） |
| MO_SPPS_NoPool | none | — | 移除共享池 |
| MO_SPPS_OldSoftPool | soft_pressure | — | 原始截断软压力公式 |
| MO_SPPS_ContinuousPool | continuous | False | 连续软压力 + 固定 Q（≡ Full） |
| MO_SPPS_AdaptiveQ | continuous | True | 自适应 Q_j（可选策略） |

**对比算法：**

| 算法 | 实现 |
|---|---|
| NSGA-II | 自定义实现（pareto.py + baselines/nsga2.py），二进制编码 |
| SPEA2 | pymoo 实现 |
| MOEA/D | pymoo 实现 |
| NSGA-II+Div | 自定义 NSGA-II + 决策空间多样性（Jaccard）补偿 |
| PLS | 自定义 Pareto Local Search（1-bit flip 邻域） |

### 7.3 结果

| Algorithm | \|A\|| HV | Jaccard | Entropy | Gini | Time(s) |
|---|---|---|---|---|---|---|
| MO_SPPS_Full | 20.133 | 3694756.800 | 0.0988 | 0.9246 | 0.1779 | 12.22 |
| MO_SPPS_NoPool | 18.300 | 3669636.700 | 0.0998 | 0.9243 | 0.1737 | 8.32 |
| MO_SPPS_OldSoftPool | 18.200 | 3665285.400 | 0.0905 | 0.9239 | 0.1788 | 11.09 |
| MO_SPPS_ContinuousPool | 20.133 | 3694756.800 | 0.0988 | 0.9246 | 0.1779 | 11.78 |
| MO_SPPS_AdaptiveQ | 18.767 | 3679290.800 | 0.0938 | 0.9240 | 0.1743 | 12.36 |
| NSGA-II | 39.967 | 3795607.767 | 0.0876 | 0.9230 | 0.1271 | 40.61 |
| SPEA2 | 100.000 | 3432508.233 | 0.0526 | 0.9130 | 0.0747 | 6.44 |
| MOEA/D | 99.967 | 2792870.933 | 0.0243 | 0.9006 | 0.0292 | 16.08 |
| NSGA-II+Div | 6.100 | 3386662.300 | 0.0925 | 0.9168 | 0.0704 | 7.62 |
| PLS | 9.200 | 451137.233 | 0.3995 | 0.9271 | 0.3632 | 0.49 |

> HV 使用全局统一参考点 [1915, 2144]（30 runs × 10 algorithms 的全局最小值 −1）。

### 7.4 核心对比

| 对比 | HV Δ | Jaccard Δ | 解读 |
|---|---|---|---|
| Full vs NSGA-II | −100851 (−2.7%) | +0.0112 (+12.8%) | HV 略低但决策多样性显著更高，快 3.3× |
| Full vs NoPool | +25120 (+0.68%) | −0.0011 | 池贡献正向但幅度不大 |
| Full vs OldSoftPool | +29471 (+0.80%) | +0.0083 | 连续压力优于截断压力 |
| Full vs AdaptiveQ | +15466 (+0.42%) | +0.0050 | 固定 Q 在 MOKP 上略优于自适应 Q |
| Full vs SPEA2 | +268249 (+7.6%) | +0.0462 | 全面优于 SPEA2 |
| Full vs MOEA/D | +903886 (+32.3%) | +0.0745 | 大幅优于 MOEA/D |
| Full vs NSGA-II+Div | +308095 (+9.1%) | +0.0063 | 生成阶段池调控优于事后多样性补偿 |
| Full vs PLS | +3243620 | −0.3007 | HV 碾压，PLS 仅多样性高但目标质量极差 |

### 7.5 关键发现

1. **MO-SPPS Full 以 −2.7% HV 的代价换取了 +12.8% 的决策空间多样性**，同时速度快 3.3×（12s vs 41s）。这符合实验方案 15.2 节"可接受结果"第 3 条：HV 接近但多样性显著更好。

2. **池机制贡献**：Full vs NoPool = +0.68% HV，正向但幅度不大。在 MOKP 上池的作用不如 MOSCSP 明显，可能因为 MOKP 的物品筛选主要受重量约束驱动，池压力起辅助作用。

3. **连续压力优于截断压力**：Full (+0.80% HV, +0.0083 Jaccard vs OldSoftPool)，验证了连续型软压力公式的设计优势。

4. **自适应 Q 在 MOKP 上略逊于固定 Q**（−0.42% HV, −0.005 Jaccard）。自适应 Q 放大热门组件的容量→削弱惩罚→加剧收敛，与池的分散功能相悖。将其作为可选策略而非默认是正确的。

5. **MO-SPPS 生成阶段调控 > 事后补偿**：Full vs NSGA-II+Div = +9.1% HV, +0.0063 Jaccard。在生成阶段通过池压力分散 agent，优于在环境选择阶段用 Jaccard 补偿拥挤距离。

6. **MOEA/D 在 MOKP 上表现很差**（HV 仅 2,792,871），标量化方法不适合离散背包问题的目标空间。

7. **性能优化**：`dominates()` 针对 2 目标问题的纯 Python 快速路径将 NSGA-II 从 300s 加速至 41s（7.4×），算法逻辑不变。

### 7.6 与预期结论对照（§7.7）

| 预期 | 实际 | 状态 |
|---|---|---|
| HV/IGD 接近强基线 | HV −2.7% vs NSGA-II，但优于 SPEA2/MOEA/D | 满足 |
| 决策空间结构多样性明显更高 | Jaccard +12.8% vs NSGA-II | 满足 |
| Full 优于 NoPool | +0.68% HV | 满足 |
| AdaptiveQ 缓解关键物品被过度抑制 | 未验证（MOKP 上无此场景） | 待验证 |

---

## 8. P3：Multi-objective Maximum Coverage

### 8.1 实验配置

*(待进行)*

### 8.2 结果

*(待填入)*

### 8.3 关键发现

*(待填入)*

---

## 9. P4：多目标特征选择

### 9.1 实验配置

*(待进行)*

### 9.2 结果

*(待填入)*

### 9.3 关键发现

*(待填入)*

---

## 10. P5：Multi-objective QUBO / Max-Cut

### 10.1 实验配置

*(待进行)*

### 10.2 结果

*(待填入)*

### 10.3 关键发现

*(待填入)*

---

## 13. 参数敏感性分析

*(待进行)*

---

## 15. 结果判断标准对照

### 15.1 MO-SPPS 有效性条件核查

| # | 条件 | P1 MOSCSP | P2 MOKP | 状态 |
|---|---|---|---|---|
| 1 | Full HV 高于或不弱于 NoPool | *(待填入)* | +0.68% | P2 满足 |
| 2 | Full 决策多样性显著高于 NoPool | *(待填入)* | −0.0011 | P2 未显著满足 |
| 3 | Full 决策多样性高于 NSGA-II/SPEA2/MOEA/D | *(待填入)* | +12.8%/+88%/+306% | P2 满足 |
| 4 | Full HV/IGD 不弱于主要经典 MOEA | *(待填入)* | −2.7% vs NSGA-II, 优于 SPEA2/MOEA/D | P2 部分满足 |
| 5 | 结构性实例中优势更明显 | *(待填入)* | — | 待验证 |
| 6 | Full 优于或接近 NSGA-II+Div | *(待填入)* | +9.1% HV, +0.0063 JD | P2 满足 |
| 7 | ContinuousPool 优于 OldSoftPool 或更稳定 | *(待填入)* | +0.80% HV, +0.0083 JD | P2 满足 |
| 8 | AdaptiveQ 能减少关键组件被过度抑制 | *(待填入)* | 未验证 | 待验证 |

### 15.3 危险信号核

| # | 危险信号 | 评估 |
|---|---|---|
| 1 | Full 与 NoPool 无显著差异 | P2: 差异仅 +0.68%，在 MOKP 上池作用不显著 |
| 2 | 多样性提高但 HV 明显恶化 | P2: HV −2.7% 换 JD +12.8%，属于可接受 trade-off |
| 4 | AdaptiveQ 降低性能 | P2: −0.42%，但已不作为默认策略 |

---

*最后更新：2026-06-04*
