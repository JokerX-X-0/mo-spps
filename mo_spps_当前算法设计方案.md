# MO-SPPS 多目标共享池种群搜索算法设计方案

**Multi-Objective Shared-Pool Population Search**  
面向多目标组合优化中决策空间多样性保持的共享池构造式进化算法

---

## 0. 当前工作

本文档描述 **MO-SPPS（Multi-Objective Shared-Pool Population Search，多目标共享池种群搜索算法）** 的当前算法设计方案。

MO-SPPS 面向多目标组合优化问题，尤其关注如下场景：

1. 候选解由离散组件构成；
2. 不同组件之间存在协同、冗余或冲突；
3. 多个目标之间存在明显权衡；
4. 传统多目标算法容易在目标空间保持一定分布，但在决策空间出现结构同质化；
5. 需要获得目标质量较好、分布较均匀、且组件结构差异明显的 Pareto 解集。

MO-SPPS 的核心思想是：

> 在候选解生成阶段引入共享组件池，通过组件可获得性或组件稀缺压力影响采样分布，使热门组件的过度复用受到动态抑制，从而促进决策空间多样性；同时结合 Pareto Archive、非支配排序、拥挤距离、目标偏好搜索和策略偏好继承机制，形成面向多目标组合优化的构造式种群搜索框架。

---

## 1. 算法名称与定位

### 1.1 算法名称

中文名称：

```text
多目标共享池种群搜索算法
```

英文名称：

```text
Multi-Objective Shared-Pool Population Search
```

缩写：

```text
MO-SPPS
```

### 1.2 一句话定义

MO-SPPS 是一种面向多目标组合优化问题的构造式种群搜索算法。它通过共享组件池在候选解生成阶段调节组件采样分布，并结合 Pareto 档案维护、目标空间拥挤距离、决策空间结构多样性、搜索预算分配和策略偏好继承机制，获得目标空间与决策空间双多样性的非支配解集。

### 1.3 算法定位

MO-SPPS 当前定位为：

```text
面向多目标组合优化问题的共享池构造式进化搜索框架
```

它不定位为连续优化算法，也不定位为通用替代 NSGA-II、SPEA2 或 MOEA/D 的算法，而是聚焦于以下问题类型：

- 候选解由离散组件、元素、模块、变量子集或结构片段构成；
- 少数组件容易被高频复用，导致解结构同质化；
- 存在多个结构不同但目标表现接近的可行解；
- 需要同时关注 Pareto 前沿质量和 Pareto 解集结构多样性。

---

## 2. 研究问题与适用范围

### 2.1 多目标组合优化中的双多样性问题

多目标优化通常关注目标空间多样性，即希望非支配解在目标空间中分布均匀。但是在组合优化中，仅保持目标空间多样性并不充分。

可能出现如下现象：

```text
Pareto 解在目标空间中看似分散，但在决策空间中高度相似。
```

例如：

- 多个特征选择解的准确率和特征数量不同，但都依赖同一批核心特征；
- 多个覆盖问题解的覆盖率和成本不同，但反复选择少数高覆盖集合；
- 多个神经结构搜索结果性能相近，但使用高度类似的模块结构。

因此，MO-SPPS 同时关注两类多样性。

| 多样性类型 | 含义 | 典型风险 |
|---|---|---|
| 目标空间多样性 | Pareto 解在目标空间中的分布 | 前沿集中、覆盖不足 |
| 决策空间多样性 | 解的组件构成差异 | 结构同质化、热门组件过度复用 |

### 2.2 核心研究问题

MO-SPPS 关注的核心研究问题是：

> 在多目标组合优化中，是否可以通过候选解生成阶段的共享组件池机制，主动维持决策空间多样性，并与 Pareto 档案机制协同，提高非支配解集的质量、分布和结构多样性？

### 2.3 适用问题类型

MO-SPPS 适合以下问题：

1. 多目标特征选择；
2. 多目标最大覆盖；
3. 多目标协同组件选择；
4. 多目标模块组合；
5. 多目标神经结构搜索；
6. 多目标子集选择；
7. 具有组件协同关系的多目标组合优化问题。

### 2.4 不适合直接使用的场景

MO-SPPS 不适合直接用于以下场景：

1. 连续变量优化问题；
2. 解不具备明确组件结构的问题；
3. 决策空间结构多样性并不重要的问题；
4. 所有高质量解必须高度依赖同一小组关键组件的问题；
5. 对运行时间极其敏感、无法承担种群多样性计算成本的问题。

---

## 3. 多目标组合优化问题定义

### 3.1 组件集合

设组件集合为：

\[
V=\{1,2,\dots,M\}
\]

其中：

- \(M\)：组件数量；
- 每个组件是构成候选解的基本单元。

组件 \(j\) 定义为：

\[
component_j=(id_j, attr_j, \rho_j, Q_j)
\]

| 符号 | 含义 |
|---|---|
| \(id_j\) | 组件编号 |
| \(attr_j\) | 组件属性，例如类别、成本、覆盖元素、特征标签等 |
| \(\rho_j\) | 基础采样权重 |
| \(Q_j\) | 共享池容量或共享池压力参数 |

### 3.2 候选解

候选解为组件子集：

\[
S\subseteq V
\]

通常满足容量约束：

\[
|S|\le K
\]

其中：

- \(S\)：候选解；
- \(K\)：最大组件数量。

不同问题中，候选解可以表示为：

- 固定大小组件集合；
- 不定长组件集合；
- 二进制选择向量；
- 模块组合；
- 资源分配组合；
- 变量取值组合。

### 3.3 多目标函数

设有 \(m\) 个优化目标：

\[
F(S)=\left(f_1(S),f_2(S),\dots,f_m(S)\right)
\]

默认所有目标统一转化为最大化形式：

\[
\max F(S)
\]

若某个目标原本是最小化，例如成本 \(Cost(S)\)，则写为：

\[
f_k(S)=-Cost(S)
\]

示例一：多目标特征选择

\[
F(S)=\left(Accuracy(S),-\frac{|S|}{M}\right)
\]

示例二：多目标最大覆盖

\[
F(S)=\left(Coverage(S),-Cost(S)\right)
\]

示例三：多目标神经结构搜索

\[
F(S)=\left(Performance(S),-Params(S),-Latency(S)\right)
\]

---

## 4. Pareto 支配关系与外部档案

### 4.1 Pareto 支配关系

对于两个候选解 \(S_a\) 和 \(S_b\)，若满足：

\[
\forall k\in\{1,\dots,m\},\quad f_k(S_a)\ge f_k(S_b)
\]

且至少存在一个目标 \(r\)：

\[
f_r(S_a)>f_r(S_b)
\]

则称 \(S_a\) 支配 \(S_b\)，记作：

\[
S_a \prec S_b
\]

若两个解互不支配，则它们代表不同的目标权衡。

### 4.2 外部 Pareto 档案

MO-SPPS 维护外部非支配解档案：

\[
\mathcal{A}
\]

用于保存历史搜索过程中发现的非支配解。

档案更新为：

\[
\mathcal{A}^{t+1}=ND\left(\mathcal{A}^{t}\cup Pop^{t}\cup C^{t}\right)
\]

其中：

- \(ND(\cdot)\)：提取非支配解；
- \(Pop^t\)：第 \(t\) 轮活跃种群；
- \(C^t\)：第 \(t\) 轮生成的候选解集合。

若档案大小超过上限 \(A_{max}\)，则执行裁剪：

\[
|\mathcal{A}^{t+1}|\le A_{max}
\]

### 4.3 档案不占用共享池

外部 Pareto 档案只保存历史搜索结果，不占用共享池资源。

原因是：

```text
Archive 是历史记录，不应长期锁定组件资源；共享池只约束或影响活跃种群的当前搜索过程。
```

---

## 5. 共享池机制语义

共享池是 MO-SPPS 的核心机制。它用于调节组件在活跃种群中的复用程度，使热门组件在被大量占用后产生稀缺压力，从而引导搜索转向替代组件组合。

当前设计中，共享池具有三种语义模式。

### 5.1 Hard-Cap Pool：硬容量共享池

硬容量共享池将 \(Q_j\) 解释为组件 \(j\) 在活跃种群中最多可被同时占用的次数。

第 \(t\) 轮共享池状态为：

\[
P^t=\{q_1^t,q_2^t,\dots,q_M^t\}
\]

其中：

\[
0\le q_j^t\le Q_j
\]

| 符号 | 含义 |
|---|---|
| \(Q_j\) | 组件 \(j\) 的最大可占用容量 |
| \(q_j^t\) | 第 \(t\) 轮组件 \(j\) 的剩余容量 |
| \(Q_j-q_j^t\) | 当前活跃种群对组件 \(j\) 的占用数量 |

硬容量模式下，若：

\[
q_j^t=0
\]

则组件 \(j\) 不可被采样。

该模式多样性压力强，但可能导致关键组件无法充分传播。

### 5.2 Soft-Pressure Pool：软压力共享池

软压力共享池将 \(Q_j\) 解释为组件 \(j\) 的参考容量或压力尺度，而非严格容量上限。

当某组件被高频占用时，其采样概率降低，但不一定完全禁止再次采样。

该模式适合作为 MO-SPPS 的默认模式，因为它在保持多样性压力的同时，允许关键组件继续以较低概率传播。

### 5.3 Hybrid Pool：混合共享池

混合共享池结合硬容量和软压力：

- 在一般情况下使用软压力采样；
- 当组件占用超过预设上限时禁止继续采样；
- 或者对超过容量的组件施加强惩罚。

该模式适合资源约束较强的问题，但实现复杂度高于软压力池。

### 5.4 当前默认模式

当前主算法采用：

```yaml
shared_pool_mode: soft_pressure
```

硬容量池作为消融版本或约束更强问题的变体。

---

## 6. 共享池可行性条件

### 6.1 硬容量池的可行性条件

若使用硬容量共享池，并且每个活跃 Agent 初始解都要求固定大小 \(K\)，则必须满足：

\[
\sum_{j=1}^{M}Q_j\ge NK
\]

其中：

- \(N\)：活跃种群规模；
- \(K\)：每个解的组件数量上限。

若该条件不满足，则无法保证所有 Agent 都能构造长度为 \(K\) 的初始解。

### 6.2 不定长解情形

若候选解只要求：

\[
|S_i|\le K
\]

则初始化时允许部分 Agent 的初始解小于 \(K\)。此时需要在局部构造阶段通过加入操作继续补全或扩展解。

### 6.3 软压力池情形

软压力池中 \(Q_j\) 不是严格容量上限，而是控制稀缺压力的尺度参数，因此不要求：

\[
\sum_{j=1}^{M}Q_j\ge NK
\]

但仍建议设置合理的 \(Q_j\)，使组件压力与种群规模、解容量之间保持可解释关系。

---

## 7. Agent 状态定义

第 \(i\) 个 Agent 在第 \(t\) 轮的状态定义为：

\[
A_i^t=\{S_i^t,F_i^t,G_i^t,\pi_i^t,w_i,r_i^t,c_i^t,d_i^t,h_i^t\}
\]

| 符号 | 含义 |
|---|---|
| \(S_i^t\) | 当前候选解 |
| \(F_i^t\) | 当前目标向量 |
| \(G_i^t\) | 当前搜索预算 |
| \(\pi_i^t\) | 组件选择偏好 |
| \(w_i\) | 目标偏好向量 |
| \(r_i^t\) | Pareto rank |
| \(c_i^t\) | 目标空间拥挤距离 |
| \(d_i^t\) | 决策空间多样性得分 |
| \(h_i^t\) | 历史状态，例如未改进次数、历史贡献等 |

### 7.1 组件选择偏好

组件偏好为：

\[
\pi_i^t=(\pi_{i,1}^t,\pi_{i,2}^t,\dots,\pi_{i,M}^t)
\]

满足：

\[
\pi_{i,j}^t\ge 0,\quad \sum_{j=1}^{M}\pi_{i,j}^t=1
\]

它表示 Agent \(i\) 对组件 \(j\) 的选择倾向。

### 7.2 目标偏好向量

目标偏好向量为：

\[
w_i=(w_{i,1},w_{i,2},\dots,w_{i,m})
\]

满足：

\[
w_{i,k}\ge 0,\quad \sum_{k=1}^{m}w_{i,k}=1
\]

目标偏好向量用于引导不同 Agent 搜索不同 Pareto 区域。

初始化方式包括：

1. 预设参考方向；
2. Dirichlet 随机采样；
3. Das-Dennis 方法生成参考方向；
4. 根据 Archive 稀疏区域动态生成。

当前第一版采用：

```text
预设参考方向
```

---

## 8. 共享池采样机制

### 8.1 硬容量采样

硬容量模式下，组件 \(j\) 的采样概率为：

\[
p_{i,j}^t
=
\frac{
q_j^t\cdot \rho_j\cdot \pi_{i,j}^t
}{
\sum_{l=1}^{M}q_l^t\cdot \rho_l\cdot \pi_{i,l}^t
}
\]

若：

\[
q_j^t=0
\]

则：

\[
p_{i,j}^t=0
\]

硬容量采样适合显式资源受限问题。

### 8.2 软压力采样

软压力模式下，组件 \(j\) 的采样概率为：

\[
p_{i,j}^t
=
\frac{
\left(\epsilon+\frac{\tilde q_j^t}{Q_j}\right)^{\tau}
\cdot \rho_j
\cdot \pi_{i,j}^t
\cdot \left(1+\kappa U_{i,j}^t\right)
}{
\sum_{l=1}^{M}
\left[
\left(\epsilon+\frac{\tilde q_l^t}{Q_l}\right)^{\tau}
\cdot \rho_l
\cdot \pi_{i,l}^t
\cdot \left(1+\kappa U_{i,l}^t\right)
\right]
}
\]

其中：

\[
\tilde q_j^t=\max(Q_j-u_j^t,0)
\]

\(u_j^t\) 表示组件 \(j\) 在当前活跃种群中的占用次数。

| 符号 | 含义 |
|---|---|
| \(\epsilon\) | 最小采样保底项 |
| \(\tau\) | 共享池压力强度 |
| \(\rho_j\) | 组件基础采样权重 |
| \(\pi_{i,j}^t\) | Agent 对组件的选择偏好 |
| \(U_{i,j}^t\) | 组件对当前目标偏好的预估效用 |
| \(\kappa\) | 目标偏好引导强度 |
| \(u_j^t\) | 当前活跃种群对组件 \(j\) 的占用次数 |

当某组件被频繁占用时，\(\tilde q_j^t/Q_j\) 降低，其采样概率随之降低。

### 8.3 当前第一版设置

当前第一版建议使用：

\[
U_{i,j}^t=0
\]

即采样只依赖：

1. 共享池压力；
2. 基础权重；
3. Agent 组件偏好。

此时采样公式简化为：

\[
p_{i,j}^t
=
\frac{
\left(\epsilon+\frac{\tilde q_j^t}{Q_j}\right)^{\tau}
\cdot \rho_j
\cdot \pi_{i,j}^t
}{
\sum_{l=1}^{M}
\left[
\left(\epsilon+\frac{\tilde q_l^t}{Q_l}\right)^{\tau}
\cdot \rho_l
\cdot \pi_{i,l}^t
\right]
}
\]

这样可以降低实现复杂度，并更清楚地验证共享池机制本身的作用。

---

## 9. 多目标候选操作

每个 Agent 从共享池中采样一个候选组件列表：

\[
L_i^t=\{j_1,j_2,\dots,j_s\}
\]

其中 \(s\) 为 `shop_size`。

### 9.1 加入操作

若：

\[
|S_i|<K
\]

则可以加入组件 \(j\)：

\[
S'=S_i\cup\{j\}
\]

### 9.2 替换操作

若：

\[
|S_i|=K
\]

则可以使用候选组件 \(j\) 替换已有组件 \(r\)：

\[
S'=(S_i\setminus \{r\})\cup\{j\}
\]

### 9.3 释放操作

可主动释放低贡献组件：

\[
S'=S_i\setminus \{r\}
\]

释放操作用于：

1. 解除低质量结构；
2. 增加后续构造空间；
3. 释放共享池资源；
4. 支持搜索路线转型。

当前第一版可以不单独实现释放操作，仅通过替换操作完成组件释放。

---

## 10. 候选解选择与接受准则

### 10.1 目标归一化

候选解选择中使用归一化目标向量：

\[
\widehat{f}_k(S)=
\frac{f_k(S)-f_k^{min}}
{f_k^{max}-f_k^{min}+\epsilon}
\]

其中 \(f_k^{min}\) 和 \(f_k^{max}\) 可由当前种群与 Archive 共同估计。

### 10.2 偏好标量得分

Agent \(i\) 对候选解 \(S\) 的偏好得分为：

\[
g_i(S)=w_i^\top \widehat{F}(S)
\]

该得分只用于局部构造选择，不替代 Pareto Archive。

### 10.3 Archive 贡献

候选解 \(S'\) 对 Archive 的贡献可以定义为：

\[
AC(S')=CD(S',\mathcal{A}\cup\{S'\})
\]

其中 \(CD\) 表示目标空间拥挤距离贡献。

也可以使用超体积贡献：

\[
AC(S')=HV(\mathcal{A}\cup\{S'\})-HV(\mathcal{A})
\]

当前第一版采用拥挤距离贡献，避免高维 HV 计算成本。

### 10.4 决策空间新颖性

候选解 \(S'\) 相对于 Archive 的决策空间新颖性定义为：

\[
Novelty(S')=1-\max_{S\in\mathcal{A}}sim(S',S)
\]

其中 Jaccard 相似度为：

\[
sim(S_a,S_b)=
\frac{|S_a\cap S_b|}
{|S_a\cup S_b|}
\]

\(Novelty(S')\) 越大，说明该候选解与已有 Archive 解结构差异越大。

### 10.5 接受准则

候选解 \(S'\) 相对于当前解 \(S_i\) 的接受规则如下：

1. 若 \(S'\) 支配 \(S_i\)，接受；
2. 若 \(S'\) 非支配，且对 Archive 具有正贡献，接受；
3. 若 \(S'\) 与 \(S_i\) 互不支配，且 \(g_i(S')>g_i(S_i)\)，接受；
4. 若 \(S'\) 具有显著决策空间新颖性，且目标质量损失不超过阈值，接受；
5. 若 \(S'\) 被 \(S_i\) 支配，通常拒绝。

其中，Archive 正贡献可以定义为：

\[
AC(S')>\theta_{AC}
\]

决策新颖性接受条件可以定义为：

\[
Novelty(S')>\theta_{novel}
\]

且：

\[
g_i(S')\ge g_i(S_i)-\theta_{loss}
\]

该规则使算法不仅接受目标改善解，也允许保留结构新颖且目标质量损失可控的候选解。

### 10.6 概率接受

可选地，以小概率接受劣解：

\[
P(accept)=
\exp\left(\frac{g_i(S')-g_i(S_i)}{T_t}\right)
\]

仅在：

\[
g_i(S')<g_i(S_i)
\]

时使用。

当前第一版关闭概率接受：

```yaml
use_probabilistic_acceptance: false
```

---

## 11. 共享池状态更新

### 11.1 组件变更集合

若 Agent 从旧解 \(S\) 更新为新解 \(S'\)，则：

加入组件集合为：

\[
J_{add}=S'\setminus S
\]

释放组件集合为：

\[
J_{remove}=S\setminus S'
\]

### 11.2 硬容量池更新

硬容量模式下：

\[
q_j\leftarrow q_j-1,\quad j\in J_{add}
\]

\[
q_j\leftarrow q_j+1,\quad j\in J_{remove}
\]

必须满足：

\[
0\le q_j\le Q_j
\]

并保持守恒：

\[
q_j^t+u_j^t=Q_j
\]

其中 \(u_j^t\) 为活跃种群中组件 \(j\) 的占用数量。

### 11.3 软压力池更新

软压力模式下，不直接通过 \(q_j\) 限制容量，而是根据活跃种群实时统计：

\[
u_j^t=\sum_{i=1}^{N}I(j\in S_i^t)
\]

并计算压力余量：

\[
\tilde q_j^t=\max(Q_j-u_j^t,0)
\]

该模式允许组件占用超过 \(Q_j\)，但超过后采样概率受到强惩罚。

### 11.4 共享池不变量

实现中需要检查以下不变量：

1. Agent 解中不存在重复组件；
2. 所有解满足 \(|S_i|\le K\)；
3. 硬容量模式下 \(0\le q_j\le Q_j\)；
4. 硬容量模式下 \(q_j+u_j=Q_j\)；
5. 软压力模式下 \(u_j\) 与当前种群一致；
6. Archive 不影响共享池占用。

---

## 12. 非支配排序与拥挤距离

### 12.1 非支配排序

将活跃种群划分为若干 Pareto 前沿：

\[
Front_1,Front_2,\dots,Front_L
\]

其中：

- \(Front_1\)：当前种群中的非支配解；
- \(Front_2\)：去掉 \(Front_1\) 后的非支配解；
- 依此类推。

Agent \(i\) 的 Pareto rank 为：

\[
r_i=l,\quad A_i\in Front_l
\]

### 12.2 Pareto rank 得分

定义：

\[
P_i=1-\frac{r_i-1}{r_{max}-1+\epsilon}
\]

其中 \(r_{max}\) 是当前种群最大 rank。

\(P_i\) 越大，表示该 Agent 的目标空间质量越好。

### 12.3 拥挤距离

对于每个目标 \(k\)，按照目标值排序。边界点拥挤距离设为较大值。

中间点 \(i\) 在目标 \(k\) 上的拥挤贡献为：

\[
CD_{i,k}=
\frac{
f_k(S_{i+1})-f_k(S_{i-1})
}{
f_k^{max}-f_k^{min}+\epsilon
}
\]

总拥挤距离为：

\[
CD_i=\sum_{k=1}^{m}CD_{i,k}
\]

归一化拥挤得分为：

\[
C_i=\frac{CD_i}{\max_j CD_j+\epsilon}
\]

---

## 13. 决策空间多样性

### 13.1 Jaccard 相似度

两个候选解的 Jaccard 相似度为：

\[
sim(S_i,S_j)=
\frac{|S_i\cap S_j|}
{|S_i\cup S_j|}
\]

Jaccard 距离为：

\[
dist(S_i,S_j)=1-sim(S_i,S_j)
\]

### 13.2 Agent 决策多样性得分

Agent \(i\) 与其他个体的平均相似度为：

\[
AvgSim_i=
\frac{1}{N-1}
\sum_{j\ne i}sim(S_i,S_j)
\]

决策空间多样性得分为：

\[
D_i=1-AvgSim_i
\]

\(D_i\) 越大，说明该 Agent 的组件结构越独特。

### 13.3 组件占用熵

组件 \(j\) 在活跃种群中的占用数量为：

\[
u_j=\sum_{i=1}^{N}I(j\in S_i)
\]

占用频率为：

\[
p_j=\frac{u_j}{\sum_{l=1}^{M}u_l}
\]

组件占用熵为：

\[
H=-\sum_{j=1}^{M}p_j\log(p_j+\epsilon)
\]

归一化熵为：

\[
H_{norm}=\frac{H}{\log M}
\]

组件占用熵越高，表示组件使用越分散。

---

## 14. Archive 裁剪机制

### 14.1 目标空间裁剪

基础版本使用目标空间拥挤距离裁剪 Archive。

当：

\[
|\mathcal{A}|>A_{max}
\]

优先移除拥挤距离较低的解。

### 14.2 目标—决策混合裁剪

为了保持最终 Archive 的结构多样性，可定义混合裁剪得分：

\[
Score(S)=\omega_o CD_{obj}(S)+\omega_d D_{archive}(S)
\]

其中：

\[
D_{archive}(S)=
1-
\frac{1}{|\mathcal{A}|-1}
\sum_{S'\in\mathcal{A},S'\ne S}sim(S,S')
\]

| 符号 | 含义 |
|---|---|
| \(CD_{obj}(S)\) | 目标空间拥挤距离 |
| \(D_{archive}(S)\) | Archive 内部决策空间多样性 |
| \(\omega_o\) | 目标空间权重 |
| \(\omega_d\) | 决策空间权重 |

当前主实验可设置：

\[
\omega_o=0.7,\quad \omega_d=0.3
\]

### 14.3 重复解处理

Archive 更新时需要移除重复解。

重复解包括：

1. 组件集合完全相同；
2. 目标向量完全相同且结构完全相同；
3. 在浮点误差范围内目标值相同且结构相同。

---

## 15. 搜索预算分配

MO-SPPS 中，Agent 的搜索预算决定该 Agent 本轮执行多少次局部构造操作。

### 15.1 固定预算版本

第一版使用固定预算：

\[
G_i=G_0
\]

局部操作次数为：

\[
ops_i=\max(1,round(G_i))
\]

固定预算版本用于验证共享池机制本身。

### 15.2 质量—拥挤预算版本

第二版预算为：

\[
G_i=G_0+\alpha P_i+\beta C_i
\]

该版本强调目标空间质量和稀疏区域开发。

### 15.3 双多样性预算版本

第三版预算为：

\[
G_i=G_0+\alpha P_i+\beta C_i+\delta D_i
\]

该版本同时奖励目标空间稀疏性和决策空间独特性。

### 15.4 探索补偿版本

探索补偿定义为：

\[
E_i=I(r_i>r_{threshold})\cdot D_i
\]

预算为：

\[
G_i=G_0+\alpha P_i+\beta C_i+\delta D_i+\gamma E_i
\]

该机制用于保留目标 rank 暂时较差但结构独特的搜索路线。

### 15.5 当前主算法预算

当前主算法采用：

\[
G_i=G_0+\alpha P_i+\beta C_i+\delta D_i
\]

探索补偿作为扩展版本或消融分析内容。

---

## 16. 策略偏好更新

当 Agent 从 \(S_i\) 更新为 \(S'\) 时，目标变化为：

\[
\Delta F=F(S')-F(S_i)
\]

归一化后得到：

\[
\widehat{\Delta F}
\]

Agent 偏好方向上的收益为：

\[
\Delta g_i=w_i^\top \widehat{\Delta F}
\]

若：

\[
\Delta g_i>0
\]

则强化新增组件偏好。

新增组件：

\[
J_{add}=S'\setminus S_i
\]

移除组件：

\[
J_{remove}=S_i\setminus S'
\]

对新增组件：

\[
\pi_{i,j}\leftarrow \pi_{i,j}+\mu \Delta g_i,\quad j\in J_{add}
\]

对移除组件：

\[
\pi_{i,j}\leftarrow (1-\mu)\pi_{i,j},\quad j\in J_{remove}
\]

最后归一化：

\[
\pi_i\leftarrow
\frac{\pi_i}{\sum_{j=1}^{M}\pi_{i,j}}
\]

其中 \(\mu\) 为组件偏好学习率。

当前建议：

```yaml
preference_learning_rate: 0.01
```

---

## 17. 淘汰与重生机制

### 17.1 淘汰分数

多目标搜索中不能只按单一适应度淘汰个体。定义保留分数：

\[
R_i=aP_i+bC_i+dD_i
\]

| 符号 | 含义 |
|---|---|
| \(P_i\) | Pareto rank 得分 |
| \(C_i\) | 目标空间拥挤得分 |
| \(D_i\) | 决策空间多样性得分 |
| \(a,b,d\) | 权重参数 |

分数越低，Agent 越容易被淘汰。

建议默认：

\[
a=0.5,\quad b=0.3,\quad d=0.2
\]

### 17.2 淘汰比例

每隔若干轮执行一次淘汰：

\[
t\mod T_{elim}=0
\]

淘汰比例为：

\[
replacement\_rate\in[0,1]
\]

例如：

```yaml
replacement_rate: 0.2
```

表示替换保留分数最低的 20% Agent。

### 17.3 目标偏好保持

重生时，新 Agent 保持被淘汰 Agent 的目标偏好：

\[
w_{new}=w_{old}
\]

这样可以维持参考方向覆盖，不让所有 Agent 集中到少数目标区域。

### 17.4 Archive 精英选择

选择与 \(w_{new}\) 最接近的 Archive 精英：

\[
elite=\arg\max_{S\in\mathcal{A}}cos(\widehat{F}(S),w_{new})
\]

该精英表示对应目标偏好区域内的历史优质搜索经验。

### 17.5 平滑策略继承

若 Archive 中保存了精英对应的组件偏好 \(\pi_{elite}\)，则：

\[
\pi_{new}=(1-\eta)\pi_{random}+\eta\pi_{elite}
\]

若 Archive 只保存解，不保存偏好，则从精英解反推平滑偏好：

\[
\pi_{elite,j}=
\begin{cases}
\frac{1-\lambda_s}{|S_{elite}|}+\frac{\lambda_s}{M}, & j\in S_{elite}\\
\frac{\lambda_s}{M}, & j\notin S_{elite}
\end{cases}
\]

然后归一化。

其中：

- \(\eta\)：继承强度；
- \(\lambda_s\)：平滑系数。

建议默认：

```yaml
inheritance_strength: 0.5
inheritance_smoothing: 0.1
```

平滑继承可以保留精英经验，同时避免新 Agent 被锁定在少数组件附近。

---

## 18. MO-SPPS 完整流程

### 18.1 初始化阶段

1. 输入多目标组合优化问题；
2. 初始化组件集合 \(V\)；
3. 设置共享池模式与参数；
4. 生成参考方向集合 \(W\)；
5. 初始化 \(N\) 个 Agent；
6. 为每个 Agent 分配目标偏好 \(w_i\)；
7. 初始化组件偏好 \(\pi_i\)；
8. 基于共享池采样构造初始解 \(S_i\)；
9. 计算目标向量 \(F(S_i)\)；
10. 初始化外部 Pareto 档案 \(\mathcal{A}\)。

### 18.2 迭代阶段

每轮迭代执行：

1. 评估所有 Agent 的目标向量；
2. 更新外部 Pareto 档案；
3. 对活跃种群进行非支配排序；
4. 计算拥挤距离；
5. 计算决策空间多样性；
6. 分配搜索预算；
7. 每个 Agent 执行多目标局部构造；
8. 更新共享池状态或共享池压力；
9. 更新组件偏好；
10. 周期性执行淘汰与重生；
11. 记录目标空间、决策空间和共享池指标。

### 18.3 输出

最终输出：

\[
\mathcal{A}_{Pareto}
\]

即外部 Pareto 档案。

附加输出包括：

- Archive 历史变化；
- Hypervolume 曲线；
- IGD 曲线；
- 组件占用熵曲线；
- 平均 Jaccard 距离曲线；
- 共享池占用曲线；
- 各参考方向覆盖情况；
- 组件复用集中度；
- 结构相近 Pareto 解簇。

---

## 19. MO-SPPS 伪代码

```text
Input:
  Component set V
  Multi-objective function F(S) = (f1(S), ..., fm(S))
  Pool capacities or pressure parameters Q
  Population size N
  Solution capacity K
  Archive size A_max
  Reference directions W
  Maximum function evaluations FE_max

Initialize:
  PoolState ← initialize_pool(Q, mode)
  Pop ← ∅
  Archive A ← ∅

  For i = 1 to N:
      w_i ← assign_reference_direction(W)
      π_i ← initialize_component_preference(M)
      S_i ← construct_solution_by_pool_sampling(PoolState, π_i, K)
      F_i ← evaluate(S_i)
      A_i ← Agent(S_i, F_i, π_i, w_i)
      Pop ← Pop ∪ {A_i}

  A ← update_archive(A, Pop)

While FE < FE_max:

  1. Evaluate population if needed

  2. Update Archive:
      A ← nondominated_update(A ∪ Pop)
      remove duplicated solutions
      If |A| > A_max:
          A ← prune_archive_by_hybrid_score(A, A_max)

  3. Multi-objective ranking:
      fronts ← non_dominated_sort(Pop)
      For each Agent A_i:
          r_i ← pareto_rank(A_i)
          c_i ← crowding_distance(A_i)
          d_i ← decision_diversity(A_i, Pop)

  4. Allocate budget:
      For each Agent A_i:
          P_i ← pareto_rank_score(r_i)
          C_i ← normalized_crowding_score(c_i)
          D_i ← decision_diversity_score(d_i)
          G_i ← G0 + αP_i + βC_i + δD_i
          ops_i ← max(1, round(G_i))

  5. Local construction:
      For each Agent A_i:
          Repeat ops_i times:
              L ← sample_shop(PoolState, π_i, w_i)
              Candidates ← generate_add_replace_candidates(S_i, L)
              Evaluate Candidates
              S_new ← select_candidate(
                          Candidates,
                          current=S_i,
                          archive=A,
                          preference=w_i
                      )

              If accept(S_new, S_i, A, w_i):
                  update_pool_state(PoolState, S_i, S_new)
                  π_i ← update_preference(π_i, S_i, S_new, w_i)
                  S_i ← S_new
                  F_i ← evaluate(S_i)
                  A ← update_archive(A ∪ {S_new})

  6. Elimination and rebirth:
      If iteration mod elimination_interval == 0:
          For each Agent A_i:
              R_i ← aP_i + bC_i + dD_i

          Remove worst replacement_rate Agents
          Release their occupied components if using hard-cap pool

          For each removed Agent:
              w_new ← w_old
              elite ← select_archive_elite(A, w_new)
              π_new ← inherit_smoothed_preference(elite, η, λ_s)
              S_new ← construct_solution_by_pool_sampling(PoolState, π_new, K)
              F_new ← evaluate(S_new)
              Insert new Agent

  7. Record metrics:
      HV, IGD, ArchiveSize
      ComponentEntropy
      AverageJaccardDistance
      PoolOccupancy
      ReuseConcentration
      ReferenceDirectionCoverage

Output:
  Pareto Archive A
```

---

## 20. 推荐默认配置

```yaml
problem:
  num_objectives: 2
  solution_capacity: 10
  archive_size: 200

population:
  population_size: 100
  max_function_evaluations: 150000

shared_pool:
  mode: soft_pressure
  epsilon: 0.01
  tau: 1.0
  utility_guidance_kappa: 0.0
  capacity_reference: 5

local_search:
  shop_size: 5
  use_probabilistic_acceptance: false
  temperature: 1.0
  archive_contribution_threshold: 0.0
  novelty_threshold: 0.3
  quality_loss_threshold: 0.02

budget:
  mode: pareto_crowding_decision
  base_budget: 3.0
  alpha_pareto: 1.0
  beta_crowding: 1.0
  delta_decision_diversity: 1.0
  gamma_exploration: 0.0

rebirth:
  use_rebirth: true
  elimination_interval: 10
  replacement_rate: 0.2
  inheritance_strength: 0.5
  inheritance_smoothing: 0.1
  preference_learning_rate: 0.01
  keep_reference_direction: true

archive:
  max_size: 200
  prune_method: hybrid_objective_decision
  objective_weight: 0.7
  decision_weight: 0.3
  remove_duplicates: true

experiment:
  num_runs: 30
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
```

---

## 21. 关键参数说明

### 21.1 问题参数

| 参数 | 含义 | 建议值 |
|---|---|---|
| \(M\) | 组件数量 | 30, 50, 100, 500 |
| \(K\) | 解容量 | 5, 10, 20 或按问题设定 |
| \(m\) | 目标数量 | 2 或 3 起步 |
| \(Q_j\) | 容量或压力尺度 | 1, 3, 5, 10 |
| \(A_{max}\) | Archive 最大容量 | 100, 200, 500 |

### 21.2 种群参数

| 参数 | 含义 | 建议值 |
|---|---|---|
| \(N\) | 活跃种群规模 | 50, 100 |
| \(FE_{max}\) | 最大目标函数评估次数 | 按问题规模设定 |
| `shop_size` | 每次采样候选组件数量 | 5, 10 |
| `initial_solution_mode` | 初始解生成方式 | random / greedy-random / pool-sampling |

### 21.3 共享池参数

| 参数 | 含义 | 建议值 |
|---|---|---|
| \(\epsilon\) | 软压力保底项 | 0.01 |
| \(\tau\) | 共享池压力强度 | 0.5, 1, 2 |
| \(\rho_j\) | 组件基础权重 | 默认 1 |
| \(\kappa\) | 效用引导强度 | 0 或 0.5 |

### 21.4 预算参数

| 参数 | 含义 | 建议值 |
|---|---|---|
| \(G_0\) | 基础预算 | 2 或 3 |
| \(\alpha\) | Pareto rank 权重 | 1.0 |
| \(\beta\) | 拥挤距离权重 | 1.0 |
| \(\delta\) | 决策多样性权重 | 1.0 |
| \(\gamma\) | 探索补偿权重 | 0 或 0.5 |

### 21.5 淘汰与继承参数

| 参数 | 含义 | 建议值 |
|---|---|---|
| \(T_{elim}\) | 淘汰间隔 | 10 |
| `replacement_rate` | 淘汰比例 | 0.1 或 0.2 |
| \(\eta\) | 策略继承强度 | 0.5 |
| \(\lambda_s\) | 继承平滑系数 | 0.1 |
| \(\mu\) | 偏好学习率 | 0.01 |
| \(a,b,d\) | 淘汰分数权重 | 0.5, 0.3, 0.2 |

---

## 22. 算法复杂度分析

### 22.1 非支配排序复杂度

对于种群规模 \(N\)、目标数量 \(m\)，普通非支配排序复杂度约为：

\[
O(mN^2)
\]

### 22.2 Archive 更新复杂度

若 Archive 大小为 \(A\)，候选解数量为 \(C\)，则 Archive 非支配更新成本约为：

\[
O(m(A+C)^2)
\]

### 22.3 决策空间多样性复杂度

若每个解最多包含 \(K\) 个组件，计算种群两两 Jaccard 相似度的复杂度约为：

\[
O(N^2K)
\]

### 22.4 局部构造复杂度

若每个 Agent 平均预算为 \(\bar G\)，每次采样候选组件数量为 \(s\)，则每轮候选评估数量约为：

\[
O(N\bar G s)
\]

若每个候选都需要目标函数评估，则运行成本主要由该项决定。

### 22.5 实现建议

1. 主实验优先使用 2 目标问题；
2. Archive 裁剪优先使用拥挤距离或混合距离，不默认使用 HV 贡献；
3. 决策空间多样性每轮计算一次，不在每次候选评估时重复计算；
4. 大规模问题中可用采样估计平均 Jaccard；
5. 所有算法对比使用相同函数评估次数，而不是相同迭代次数。

---

## 23. 创新点表述

MO-SPPS 的核心贡献集中在以下三点。

### 23.1 候选解生成阶段的共享池多样性调控

MO-SPPS 在候选解生成阶段引入共享组件池，通过组件可获得性或稀缺压力改变采样分布，使热门组件的过度复用受到动态抑制。

该机制不同于传统多目标算法主要在选择阶段通过拥挤距离或密度估计维持多样性。

### 23.2 目标空间—决策空间双多样性协同搜索

MO-SPPS 同时维护：

1. 目标空间多样性；
2. 决策空间结构多样性。

其中：

- 目标空间由 Pareto rank、crowding distance 和 Archive 维护；
- 决策空间由共享池压力、Jaccard 距离、组件占用熵和混合 Archive 裁剪维护。

### 23.3 非复制式策略偏好继承

MO-SPPS 在淘汰重生阶段不直接复制精英解，而是继承其组件选择偏好。

该机制实现了搜索经验迁移，同时降低直接复制解造成的结构同质化和共享池资源冲突。

---

## 24. 与典型多目标算法的区别

| 机制 | NSGA-II | SPEA2 | MOEA/D | MO-SPPS |
|---|---|---|---|---|
| 多目标选择 | 非支配排序 + 拥挤距离 | 支配强度 + Archive | 分解子问题 | Pareto rank + Archive |
| 解生成 | 交叉 + 变异 | 交叉 + 变异 | 邻域交叉变异 | 共享池压力采样 + 局部构造 |
| 目标空间多样性 | 有 | 有 | 有 | 有 |
| 决策空间多样性 | 间接 | 间接 | 间接 | 显式维护 |
| 组件复用调控 | 无 | 无 | 无 | 有 |
| 个体状态 | 解 | 解 | 解 | 解 + 组件偏好 + 目标偏好 + 预算 |
| 知识迁移 | 基因继承 | 基因继承 | 邻域传播 | 策略偏好继承 |
| 输出目标 | Pareto 解集 | Pareto 解集 | Pareto 解集 | 结构多样的 Pareto 解集 |

---

## 25. 代码架构设计

### 25.1 推荐目录结构

```text
mo_spps/
├── README.md
├── configs/
│   ├── default.yaml
│   ├── moscsp.yaml
│   ├── maximum_coverage.yaml
│   ├── feature_selection.yaml
│   ├── ablation_no_pool.yaml
│   ├── ablation_no_budget.yaml
│   ├── ablation_no_inherit.yaml
│   ├── ablation_no_decision_diversity.yaml
│   └── ablation_nsga2_decision_diversity.yaml
├── data/
│   ├── raw/
│   └── generated/
├── results/
│   ├── raw/
│   ├── summary/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── components.py
│   ├── problem_base.py
│   ├── problems/
│   │   ├── mo_scsp.py
│   │   ├── maximum_coverage.py
│   │   ├── feature_selection.py
│   │   └── mo_qubo.py
│   ├── pool.py
│   ├── agent.py
│   ├── archive.py
│   ├── pareto.py
│   ├── reference_directions.py
│   ├── operators.py
│   ├── budget.py
│   ├── metrics.py
│   ├── mo_spps.py
│   ├── baselines/
│   │   ├── nsga2.py
│   │   ├── nsga2_decision_diversity.py
│   │   ├── spea2.py
│   │   ├── moead.py
│   │   ├── random_mo.py
│   │   └── greedy_mo.py
│   ├── experiments/
│   │   ├── run_single.py
│   │   ├── run_grid.py
│   │   ├── run_ablation.py
│   │   └── run_baselines.py
│   └── visualization.py
└── tests/
    ├── test_pool.py
    ├── test_pareto.py
    ├── test_archive.py
    ├── test_metrics.py
    ├── test_problem.py
    └── test_mo_spps_invariants.py
```

### 25.2 核心类设计

#### Component

```python
@dataclass(frozen=True)
class Component:
    id: int
    attributes: dict
    base_weight: float = 1.0
    pool_capacity: int = 1
```

#### MultiObjectiveProblem

```python
class MultiObjectiveProblem:
    def evaluate(self, solution: set[int]) -> np.ndarray:
        """Return objective vector F(S). All objectives are maximized."""
        raise NotImplementedError

    def repair(self, solution: set[int]) -> set[int]:
        raise NotImplementedError
```

#### SharedPool

```python
@dataclass
class SharedPool:
    capacities: dict[int, int]
    mode: str = "soft_pressure"
    epsilon: float = 0.01
    tau: float = 1.0

    def compute_occupancy(self, population: list["Agent"]) -> dict[int, int]:
        ...

    def sample(
        self,
        preference: np.ndarray,
        population: list["Agent"],
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        ...

    def update_transition(
        self,
        old_solution: set[int],
        new_solution: set[int],
    ) -> None:
        ...

    def validate(self, population: list["Agent"]) -> None:
        ...
```

#### Agent

```python
@dataclass
class Agent:
    solution: set[int]
    objectives: np.ndarray
    component_preference: np.ndarray
    objective_preference: np.ndarray
    budget: float = 0.0
    pareto_rank: int = 1
    crowding_distance: float = 0.0
    decision_diversity: float = 0.0
    no_archive_contribution_steps: int = 0
```

#### ParetoArchive

```python
class ParetoArchive:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.solutions = []
        self.objectives = []

    def update(self, candidates: list[Agent]) -> None:
        ...

    def remove_duplicates(self) -> None:
        ...

    def prune(self) -> None:
        ...

    def select_elite_by_direction(self, direction: np.ndarray) -> Agent:
        ...
```

#### MOSPPSOptimizer

```python
class MOSPPSOptimizer:
    def __init__(self, problem, config):
        ...

    def initialize(self) -> None:
        ...

    def run(self) -> dict:
        ...

    def step(self) -> None:
        ...

    def evaluate_population(self) -> None:
        ...

    def update_archive(self) -> None:
        ...

    def assign_ranks_and_crowding(self) -> None:
        ...

    def compute_decision_diversity(self) -> None:
        ...

    def allocate_budgets(self) -> None:
        ...

    def local_construct(self, agent: Agent) -> None:
        ...

    def eliminate_and_rebirth(self) -> None:
        ...

    def record_metrics(self) -> None:
        ...
```

---

## 26. 实验问题设计

### 26.1 MOSCSP 多目标协同组件选择问题

MOSCSP 是主机制验证问题。

候选解为：

\[
S\subseteq V,\quad |S|\le K
\]

推荐目标：

\[
F(S)=(Quality(S),-Cost(S))
\]

其中：

\[
Quality(S)=\sum_{j\in S}v_j+\lambda\sum_{r\in R}B_rI(S\ satisfies\ r)
\]

参数建议：

| 参数 | 建议值 |
|---|---|
| \(M\) | 30, 50, 100 |
| \(K\) | 5, 10, 15 |
| 目标数 | 2 或 3 |
| 协同强度 \(\lambda\) | 0.5, 1, 2, 5 |
| 共享池容量 \(Q_j\) | 1, 3, 5, 10 |
| 种群规模 | 50, 100 |

MOSCSP 应包含三类实例：

| 实例类型 | 用途 |
|---|---|
| 低协同、低冲突 | 检查共享池是否无明显副作用 |
| 高协同、热门组件集中 | 展示共享池对热门组件过度复用的抑制效果 |
| 多簇协同结构 | 验证算法是否能发现多条结构不同的 Pareto 路线 |

### 26.2 多目标 Maximum Coverage

目标：

\[
f_1(S)=Coverage(S)
\]

\[
f_2(S)=-Cost(S)
\]

约束：

\[
|S|\le K
\]

该问题用于验证 MO-SPPS 在公开组合优化问题上的泛化能力。

### 26.3 多目标特征选择

目标：

\[
f_1(S)=Accuracy(S)
\]

\[
f_2(S)=-\frac{|S|}{M}
\]

可选第三目标：

\[
f_3(S)=-InferenceCost(S)
\]

该问题用于验证算法在实际应用中的价值。

### 26.4 多目标 QUBO / Max-Cut 扩展

目标可以设计为：

\[
f_1(x)=x^TQ_1x
\]

\[
f_2(x)=x^TQ_2x
\]

或者：

\[
F(x)=(CutWeight(x),-BalancePenalty(x))
\]

该问题适合验证强交互变量场景。

---

## 27. 对比算法

### 27.1 必做对比算法

| 算法 | 作用 |
|---|---|
| Random MO Search | 随机多目标下限 |
| Greedy Scalarization | 简单加权贪心 |
| NSGA-II | 标准多目标进化算法对照 |
| SPEA2 | Archive 与密度估计对照 |
| MOEA/D | 目标分解与参考方向对照 |
| NSGA-II + Decision Diversity | 验证生成阶段共享池是否优于选择阶段多样性补偿 |
| MO-SPPS-NoPool | 验证共享池贡献 |
| MO-SPPS-NoBudget | 验证预算贡献 |
| MO-SPPS-NoInherit | 验证策略继承贡献 |
| MO-SPPS-NoDecisionDiversity | 验证决策多样性贡献 |
| MO-SPPS-Full | 主算法 |

### 27.2 可选强对照

| 算法 | 适用场景 |
|---|---|
| MOEA/D + Local Search | 强分解式对照 |
| Memetic NSGA-II | 强混合算法对照 |
| Multi-objective Tabu Search | 组合优化强对照 |

---

## 28. 消融实验设计

### 28.1 MO-SPPS-Full

完整版本：

```yaml
use_shared_pool: true
shared_pool_mode: soft_pressure
use_budget: true
use_strategy_inheritance: true
use_decision_diversity: true
use_archive: true
archive_prune_method: hybrid_objective_decision
```

### 28.2 MO-SPPS-NoPool

移除共享池压力：

```yaml
use_shared_pool: false
```

采样时忽略组件占用状态。

目的：

```text
验证共享池机制是否有效。
```

### 28.3 MO-SPPS-NoBudget

固定预算：

\[
G_i=G_0
\]

目的：

```text
验证预算调度是否有效。
```

### 28.4 MO-SPPS-NoInherit

随机重生：

\[
\pi_{new}=\pi_{random}
\]

目的：

```text
验证策略继承是否有效。
```

### 28.5 MO-SPPS-NoDecisionDiversity

预算、淘汰和 Archive 裁剪中移除决策空间多样性项。

目的：

```text
验证决策空间多样性指标是否有贡献。
```

### 28.6 MO-SPPS-HardPool vs SoftPool

比较硬容量共享池与软压力共享池。

目的：

```text
分析共享池压力强度和容量约束对 Pareto 质量与结构多样性的影响。
```

### 28.7 NSGA-II + Decision Diversity

在 NSGA-II 环境选择阶段加入决策空间多样性奖励。

目的：

```text
验证 MO-SPPS 的生成阶段共享池机制是否优于选择阶段的事后多样性补偿。
```

---

## 29. 评价指标

### 29.1 目标空间指标

| 指标 | 含义 |
|---|---|
| Hypervolume, HV | Pareto 解集覆盖体积，越大越好 |
| IGD | 到参考前沿的平均距离，越小越好 |
| GD | 解集到参考前沿的距离，越小越好 |
| Spread | 前沿分布均匀性 |
| Spacing | 相邻解距离一致性 |
| Archive Size | 非支配解数量 |
| Coverage Metric | 一个算法解集支配另一个算法解集的比例 |

### 29.2 决策空间指标

| 指标 | 含义 |
|---|---|
| Average Jaccard Distance | 解结构平均差异 |
| Component Entropy | 组件占用熵 |
| Unique Pattern Count | 不同组件模式数量 |
| Decision-space Spread | 决策空间分布范围 |
| Pareto Set Structural Diversity | Pareto 解集结构多样性 |

### 29.3 共享池机制指标

| 指标 | 含义 |
|---|---|
| Pool Occupancy Curve | 组件占用量随迭代变化 |
| Scarcity Index | 热门组件稀缺程度 |
| Reuse Concentration | 组件复用集中度 |
| Route Transition Rate | Agent 组合路线转移频率 |
| Pool Pressure Sensitivity | 不同 \(\tau\)、\(Q_j\) 下的结果变化 |

### 29.4 效率指标

| 指标 | 含义 |
|---|---|
| Runtime | 运行时间 |
| Function Evaluations | 目标函数评估次数 |
| Time-to-HV | 达到指定 HV 的时间 |
| Archive Update Cost | 档案维护成本 |

---

## 30. 实验安排

### 30.1 阶段一：机制验证实验

问题：

```text
MOSCSP
```

目的：

```text
验证共享池机制、预算调度和策略继承是否有效。
```

实验变量：

| 参数 | 取值 |
|---|---|
| \(\lambda\) | 0.5, 1, 2, 5 |
| \(Q_j\) | 1, 3, 5, 10 |
| \(\tau\) | 0, 0.5, 1, 2 |
| \(N\) | 50, 100 |

算法：

- Random MO Search；
- Greedy Scalarization；
- NSGA-II；
- SPEA2；
- MOEA/D；
- NSGA-II + Decision Diversity；
- MO-SPPS-NoPool；
- MO-SPPS-NoBudget；
- MO-SPPS-NoInherit；
- MO-SPPS-NoDecisionDiversity；
- MO-SPPS-Full。

主要图表：

- HV 曲线；
- IGD 曲线；
- Archive size 曲线；
- Component entropy 曲线；
- Average Jaccard distance 曲线；
- 共享池占用曲线；
- 热门组件复用集中度曲线；
- 消融实验柱状图；
- Pareto 解结构可视化图。

### 30.2 阶段二：公开组合优化验证

问题：

```text
Multi-objective Maximum Coverage
```

目的：

```text
验证 MO-SPPS 不只适用于自定义问题。
```

目标：

\[
F(S)=(Coverage(S),-Cost(S))
\]

算法：

- Greedy Scalarization；
- NSGA-II；
- SPEA2；
- MOEA/D；
- NSGA-II + Decision Diversity；
- MO-SPPS-NoPool；
- MO-SPPS-Full。

指标：

- HV；
- IGD；
- Coverage metric；
- Pareto 解集数量；
- 组件熵；
- Jaccard 距离。

### 30.3 阶段三：应用验证

问题：

```text
多目标特征选择
```

目标：

\[
F(S)=\left(Accuracy(S),-\frac{|S|}{M}\right)
\]

数据集：

- 小规模分类数据集；
- 中等维度数据集；
- 高维稀疏数据集。

对比：

- Filter Top-K；
- NSGA-II；
- SPEA2；
- Binary MOEA/D；
- NSGA-II + Decision Diversity；
- MO-SPPS-NoPool；
- MO-SPPS-Full。

指标：

- Accuracy；
- Feature count；
- HV；
- Pareto 解集；
- 特征选择稳定性；
- 特征占用熵。

### 30.4 阶段四：参数敏感性分析

分析参数：

1. 共享池容量或压力尺度 \(Q_j\)；
2. 共享池压力 \(\tau\)；
3. Archive 大小 \(A_{max}\)；
4. 决策多样性权重 \(\delta\)；
5. 继承强度 \(\eta\)；
6. 继承平滑系数 \(\lambda_s\)；
7. 预算参数 \(\alpha,\beta,\gamma\)。

---

## 31. 公平性控制

### 31.1 评估次数公平

所有算法使用相同目标函数评估次数：

\[
FE_{max}
\]

不同算法结构不同，因此同时报告：

- 迭代次数；
- 函数评估次数；
- 实际运行时间。

### 31.2 随机种子

每组实验至少 30 次独立运行。

统一随机种子列表：

```text
seeds = [0, 1, 2, ..., 29]
```

### 31.3 参数调优公平

参数调优原则：

1. 对所有算法进行小规模预调参；
2. 固定主实验参数；
3. 报告所有参数；
4. 不在测试结果上反复调参；
5. 对主算法和对比算法使用相同评估预算。

### 31.4 统计检验

建议使用：

- Wilcoxon signed-rank test；
- Mann–Whitney U test；
- Friedman test；
- Holm post-hoc test。

显著性水平：

\[
p<0.05
\]

---

## 32. 结果判断标准

### 32.1 认为 MO-SPPS 有效的条件

至少应满足：

1. MO-SPPS-Full 的 HV 高于或不弱于 MO-SPPS-NoPool；
2. MO-SPPS-Full 的决策空间多样性显著高于 MO-SPPS-NoPool；
3. MO-SPPS-Full 的决策空间多样性高于 NSGA-II 和 SPEA2；
4. MO-SPPS-Full 的 HV / IGD 不弱于 NSGA-II 和 SPEA2；
5. 在 MOSCSP 中，优势随协同强度 \(\lambda\) 增大而增强；
6. 在公开问题中，MO-SPPS-Full 仍优于 MO-SPPS-NoPool；
7. 组件占用曲线能显示热门组件被动态抑制；
8. Archive 中存在更多结构不同但目标质量接近的 Pareto 解。

### 32.2 可接受结果

以下结果可以接受：

1. MO-SPPS 不总是击败 MOEA/D；
2. 在低协同强度问题上优势不明显；
3. HV 与 NSGA-II 接近，但决策空间多样性更好；
4. 共享池容量或压力尺度存在最佳区间；
5. 软压力池优于硬容量池；
6. HV 略低但结构多样性显著更高。

对于第 6 类情况，可表述为：

```text
MO-SPPS 在目标质量与结构多样性之间提供了更优折中。
```

### 32.3 危险结果

以下结果说明算法需要进一步调整：

1. Full 与 NoPool 无显著差异；
2. 决策空间多样性提高，但 HV 明显下降；
3. Archive 解数量多，但质量差；
4. 所有 Agent 集中到少数参考方向；
5. 共享池导致关键组件无法传播；
6. NSGA-II + Decision Diversity 明显优于 MO-SPPS；
7. 共享池参数对结果极端敏感，缺少稳定有效区间。

---

## 33. 分阶段实现计划

### 33.1 第一阶段：最小可运行版本

保留：

- 软压力共享池采样；
- Pareto Archive；
- 非支配排序；
- 拥挤距离；
- 固定预算；
- 加入 / 替换操作；
- NoPool 消融。

暂时关闭：

- 策略继承；
- 动态预算；
- 概率接受；
- 组件效用引导；
- 复杂 Archive 贡献判断。

该阶段目标：

```text
验证共享池是否能提高决策空间多样性，并保持 Pareto 质量不明显下降。
```

### 33.2 第二阶段：预算调度版本

加入：

- Pareto rank 预算；
- 拥挤距离预算；
- 决策多样性预算；
- NoBudget 消融；
- NoDecisionDiversity 消融。

该阶段目标：

```text
验证搜索预算调度是否能够提升收敛效率和多样性保持能力。
```

### 33.3 第三阶段：策略继承版本

加入：

- 淘汰重生；
- Archive 精英选择；
- 平滑策略继承；
- NoInherit 消融。

该阶段目标：

```text
验证策略继承是否能够提升搜索经验迁移效率，并避免直接复制导致的结构同质化。
```

### 33.4 第四阶段：完整实验版本

加入：

- 混合 Archive 裁剪；
- NSGA-II + Decision Diversity 强对照；
- 多问题验证；
- 参数敏感性分析；
- 统计检验。

该阶段目标：

```text
形成可用于论文实验的完整算法与实验体系。
```

---

## 34. 测试与不变量检查

实现中需要建立单元测试与集成测试。

### 34.1 共享池测试

检查：

1. 硬容量池中 \(q_j\) 不越界；
2. 硬容量池中 \(q_j+u_j=Q_j\)；
3. 软压力池中占用统计与种群一致；
4. \(q_j=0\) 时硬容量池不采样该组件；
5. 软压力池中高占用组件采样概率下降。

### 34.2 Pareto 测试

检查：

1. 支配关系判断正确；
2. 非支配排序正确；
3. Archive 中不存在被支配解；
4. Archive 中不存在重复解；
5. 所有目标均按最大化处理。

### 34.3 决策多样性测试

检查：

1. Jaccard 相似度计算正确；
2. 平均 Jaccard 距离计算正确；
3. 组件占用熵计算正确；
4. Archive 决策多样性得分计算正确。

### 34.4 算法整体测试

检查：

1. 所有 Agent 解满足 \(|S_i|\le K\)；
2. 局部构造不会产生重复组件；
3. 淘汰重生后种群规模保持不变；
4. NoPool 与 Full 只在共享池机制上存在差异；
5. 固定随机种子下结果可复现。

---

## 35. 后续扩展方向

### 35.1 多模态多目标优化

MO-SPPS 可以进一步扩展到多模态多目标组合优化，强调：

```text
同一目标区域附近存在多个结构不同的 Pareto 解。
```

### 35.2 动态共享池

让 \(Q_j\) 随迭代变化：

\[
Q_j^{t+1}=Q_j^t+\Delta Q_j^t
\]

用于模拟动态资源环境或自适应组件压力。

### 35.3 层级组件池

组件可以分层：

- 基础组件；
- 模块组件；
- 子结构组件；
- 解片段组件。

该方向适合神经结构搜索、路径规划和调度问题。

### 35.4 多区域共享池

不同目标区域使用不同子池：

\[
P_r^t,\quad r=1,\dots,R
\]

也可以采用：

```text
全局池 + 区域池
```

联合机制。

### 35.5 理论分析

可分析：

1. 共享池压力对组件占用熵的影响；
2. 共享池压力对种群同质化速度的影响；
3. 软压力参数 \(\tau\) 与采样分布熵之间的关系；
4. 决策空间多样性与 Hypervolume 的相关性；
5. 共享池机制在多簇协同结构问题中的搜索路径分化能力。

---

## 36. 论文贡献建议写法

论文贡献可写为：

1. 提出一种面向多目标组合优化的共享池构造式种群搜索算法 MO-SPPS，在候选解生成阶段通过共享池压力调节组件复用行为。
2. 设计目标空间—决策空间双多样性协同机制，将 Pareto rank、拥挤距离、组件占用状态和结构差异共同用于搜索过程控制。
3. 提出非复制式策略偏好继承机制，在淘汰重生阶段迁移有效构造经验，同时减少直接复制解带来的结构同质化。
4. 通过多目标协同组件选择、公开组合优化和多目标特征选择任务验证算法在 Pareto 质量、目标分布和结构多样性方面的有效性。

---

## 37. 当前版本总结

MO-SPPS 当前形成的完整算法框架为：

```text
多偏好 Agent
+ 软压力共享组件池
+ 多目标局部构造
+ Pareto Archive
+ 目标空间拥挤距离
+ 决策空间结构多样性
+ 搜索预算调度
+ 平滑策略偏好继承
+ 混合 Archive 裁剪
```

其最关键的研究命题是：

> 在多目标组合优化中，共享池压力是否能够在候选解生成阶段有效抑制热门组件过度复用，从而发现更多结构不同且目标质量不差的 Pareto 解？

当前主算法应优先证明：

1. MO-SPPS-Full 相比 MO-SPPS-NoPool 能显著提升决策空间多样性；
2. MO-SPPS-Full 的 HV / IGD 不明显弱于主流多目标算法；
3. MO-SPPS 能在高协同、多簇结构或热门组件集中问题中产生更丰富的 Pareto 结构路线；
4. 软压力共享池存在稳定有效的参数区间；
5. 生成阶段共享池机制优于单纯在选择阶段加入决策多样性补偿。

如果上述命题在实验中成立，MO-SPPS 可以形成明确的研究价值：

```text
它不是简单追求更多 Pareto 点，而是追求目标质量、目标分布和解结构多样性之间的协同平衡。
```

