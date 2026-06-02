# MO-SPPS 三项关键完善机制

本文档整理 MO-SPPS（Multi-Objective Shared-Pool Population Search）中建议加入的三项关键完善机制，用于增强算法的稳定性、可解释性和实验说服力。

三项机制分别为：

1. 连续型软压力共享池；
2. 质量约束的决策空间多样性；
3. Archive 贡献驱动的自适应共享池容量 \(Q_j\)。

这三项机制对应 MO-SPPS 当前最关键的三个问题：

| 完善机制 | 解决的问题 | 主要作用 | 定位 |
|---|---|---|---|
| 连续型软压力共享池 | 原软压力在 \(u_j \ge Q_j\) 后惩罚饱和 | 使组件复用压力连续变化 | **默认机制** |
| 偏好区域约束的 Novelty 参考集 | 限制 Agent 只在其偏好区域内衡量结构新颖性 | 控制探索范围，聚焦偏好相关多样性 | **可选扩展** |
| 自适应 \(Q_j\) | 避免高贡献组件被过度抑制 | 保护关键组件，同时抑制低贡献热门组件 | **默认增强项** |

---

## 1. 连续型软压力共享池

### 1.1 改进动机

原始软压力池中，组件 \(j\) 的压力余量定义为：

\[
\tilde q_j^t=\max(Q_j-u_j^t,0)
\]

其中 \(u_j^t\) 表示第 \(t\) 轮组件 \(j\) 在活跃种群中的占用次数，\(Q_j\) 表示组件 \(j\) 的参考容量或压力尺度。

该设计存在一个问题：当 \(u_j^t\ge Q_j\) 后，\(\tilde q_j^t\) 被截断为 0。此时轻微超额占用和严重超额占用会被压缩到同一低采样区间，难以区分组件复用程度的差异。

因此，将原始截断式软压力改为连续衰减式软压力。

---

### 1.2 连续型压力项

定义组件 \(j\) 在第 \(t\) 轮的连续软压力权重为：

\[
\phi_j^t=
\left(
\epsilon+
\frac{1}{1+u_j^t/Q_j^t}
\right)^\tau
\]

其中：

| 符号 | 含义 |
|---|---|
| \(u_j^t\) | 第 \(t\) 轮组件 \(j\) 在活跃种群中的占用次数 |
| \(Q_j^t\) | 第 \(t\) 轮组件 \(j\) 的参考容量或压力尺度 |
| \(\epsilon\) | 最小采样保底项，防止采样概率变为 0 |
| \(\tau\) | 共享池压力强度 |
| \(\phi_j^t\) | 组件 \(j\) 的连续软压力权重 |

当 \(u_j^t\) 增大时，\(\frac{1}{1+u_j^t/Q_j^t}\) 连续下降，因此 \(\phi_j^t\) 也连续下降。该形式可以避免原始 \(\max(Q_j-u_j^t,0)\) 带来的截断饱和问题。

---

### 1.3 连续型软压力采样概率

在连续型软压力池下，Agent \(i\) 在第 \(t\) 轮选择组件 \(j\) 的概率定义为：

\[
p_{i,j}^t=
\frac{
\phi_j^t \rho_j \pi_{i,j}^t
}{
\sum_{l=1}^{M}
\phi_l^t \rho_l \pi_{i,l}^t
}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(p_{i,j}^t\) | Agent \(i\) 在第 \(t\) 轮采样组件 \(j\) 的概率 |
| \(\phi_j^t\) | 组件 \(j\) 的连续软压力权重 |
| \(\rho_j\) | 组件 \(j\) 的基础采样权重 |
| \(\pi_{i,j}^t\) | Agent \(i\) 对组件 \(j\) 的选择偏好 |
| \(M\) | 组件总数 |

该采样公式仍保留 MO-SPPS 的核心思想：高占用组件会被降低采样概率，但不会被完全禁止继续传播。

---

### 1.4 默认参数

```yaml
soft_pressure:
  mode: continuous
  epsilon: 0.01
  tau: 1.0
  base_capacity_Q0: 5
  rho_j: 1.0
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epsilon` | 0.01 | 最小采样保底项 |
| `tau` | 1.0 | 共享池压力强度 |
| `base_capacity_Q0` | 5 | 默认参考容量 |
| `rho_j` | 1.0 | 默认基础采样权重 |

---

## 2. 偏好区域约束的 Novelty 参考集（可选机制）

### 2.1 定位

**可选扩展机制，不作为默认主机制。**

在默认 MO-SPPS 中，Pareto Archive 本身就是非支配解集——所有存档解已是高质量解。同时，Novelty 接受规则（Rule 4）已包含质量损失阈值 \(\theta_{loss}\)，候选解必须满足 \(g_i(S') \ge g_i(S_i) - \theta_{loss}\) 才能通过新颖性接受。因此**无需额外对 Archive 进行质量过滤**——质量约束已内嵌在算法结构中。

本节定义的偏好区域约束 Novelty 是一个扩展选项，用于希望**进一步限制 Agent 只在与其偏好方向高度相关的区域**内衡量结构新颖性的场景。

---

### 2.2 默认 Novelty

默认 Novelty 定义为候选解相对于全部 Archive 的最大结构相似度：

\[
Novelty(S')=1-\max_{S\in\mathcal A} sim(S',S)
\]

其中：

\[
sim(S_a,S_b)=\frac{|S_a\cap S_b|}{|S_a\cup S_b|}
\]

---

### 2.3 偏好区域参考集（扩展）

当希望限制 Agent 只在其目标偏好区域内衡量结构新颖性时，定义偏好相关参考集：

\[
\mathcal A_i^{rel}
=
\{S\in\mathcal A \mid \cos(\widehat F(S),w_i)\ge \theta_{rel}\}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\mathcal A_i^{rel}\) | Agent \(i\) 的偏好相关 Archive 子集 |
| \(\widehat F(S)\) | 解 \(S\) 的归一化目标向量 |
| \(w_i\) | Agent \(i\) 的目标偏好权重向量 |
| \(\theta_{rel}\) | 偏好相关性的余弦相似度阈值 |

---

### 2.4 区域 Novelty（扩展）

在偏好区域参考集上定义区域 Novelty：

\[
Novelty_i^{rel}(S')
=
1-\max_{S\in\mathcal A_i^{rel}}sim(S',S)
\]

若 \(\mathcal A_i^{rel}\) 为空，则退化为全 Archive Novelty：

\[
Novelty_i^{rel}(S')=Novelty(S')
\]

---

### 2.5 默认参数

```yaml
region_novelty:
  enabled: false
  region_threshold: 0.3
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `enabled` | false | 是否启用偏好区域 Novelty（默认关闭） |
| `region_threshold` | 0.3 | 余弦相似度阈值，控制偏好区域宽度 |

---

## 3. Archive 贡献驱动的自适应 \(Q_j\)

### 3.1 改进动机

固定 \(Q_j\) 默认所有组件具有相同参考容量，但在实际组合优化问题中，不同组件的价值可能不同。

某些组件虽然被频繁复用，但确实对高质量 Pareto 解有重要贡献。如果对这些组件施加过强共享池压力，可能会损害算法的收敛性能。

因此，引入基于 Archive 贡献的自适应容量 \(Q_j^t\)，使高贡献组件获得更大的参考容量，低贡献热门组件仍然受到共享池抑制。

---

### 3.2 自适应参考容量

定义组件 \(j\) 在第 \(t\) 轮的自适应参考容量为：

\[
Q_j^t=
clip
\left(
Q_0(1+\alpha_Q C_j^t),
Q_{min},
Q_{max}
\right)
\]

其中：

| 符号 | 含义 |
|---|---|
| \(Q_j^t\) | 第 \(t\) 轮组件 \(j\) 的自适应参考容量 |
| \(Q_0\) | 基础参考容量 |
| \(\alpha_Q\) | Archive 贡献对容量的放大系数 |
| \(C_j^t\) | 组件 \(j\) 的历史贡献得分 |
| \(Q_{min}\) | 最小参考容量 |
| \(Q_{max}\) | 最大参考容量 |
| \(clip(\cdot)\) | 将容量限制在 \([Q_{min},Q_{max}]\) 范围内 |

当 \(C_j^t\) 较大时，组件 \(j\) 的参考容量 \(Q_j^t\) 增大，从而降低其因高频复用而受到的共享池惩罚。

---

### 3.3 组件历史贡献得分

组件 \(j\) 的历史贡献得分定义为：

\[
C_j^t=
\frac{
\sum_{S\in \mathcal{A}^t}
I(j\in S)\cdot q(S)
}{
\sum_{S\in \mathcal{A}^t} q(S)+\epsilon
}
\]

其中：

| 符号 | 含义 |
|---|---|
| \(C_j^t\) | 第 \(t\) 轮组件 \(j\) 的 Archive 历史贡献得分 |
| \(I(j\in S)\) | 若组件 \(j\) 属于解 \(S\)，则为 1，否则为 0 |
| \(q(S)\) | Archive 解 \(S\) 的质量权重 |
| \(\epsilon\) | 防止分母为 0 的小常数 |

质量权重 \(q(S)\) 可以定义为：

\[
q(S)=1+CD_{obj}(S)
\]

其中 \(CD_{obj}(S)\) 为目标空间拥挤距离。

也可以使用归一化目标得分或超体积贡献近似值作为 \(q(S)\)。第一版建议使用拥挤距离形式，避免高维超体积贡献带来的计算成本。

---

### 3.4 与连续软压力池的结合

自适应 \(Q_j^t\) 直接进入连续软压力项：

\[
\phi_j^t=
\left(
\epsilon+
\frac{1}{1+u_j^t/Q_j^t}
\right)^\tau
\]

因此，高贡献组件会因为 \(Q_j^t\) 较大而受到较弱惩罚；低贡献且高频复用的组件仍会受到较强共享池压力。

该机制可以形成如下效果：

```text
高频但低贡献组件：继续受到抑制
高频且高贡献组件：获得更高 Q_j，避免被过度压制
低频但潜在有效组件：仍有探索机会
```

---

### 3.5 默认参数

```yaml
adaptive_capacity:
  use_adaptive_Q: true
  base_capacity_Q0: 5
  alpha_Q: 2.0
  Q_min: 1
  Q_max: 20
  update_interval: 10
  contribution_metric: archive_frequency_weighted_crowding
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `use_adaptive_Q` | true | 是否启用自适应容量 |
| `base_capacity_Q0` | 5 | 基础参考容量 |
| `alpha_Q` | 2.0 | Archive 贡献放大系数 |
| `Q_min` | 1 | 最小参考容量 |
| `Q_max` | 20 | 最大参考容量 |
| `update_interval` | 10 | 每隔多少轮更新一次 \(Q_j^t\) |
| `contribution_metric` | archive_frequency_weighted_crowding | 组件贡献计算方式 |

---

## 4. 推荐整合方式

三项机制在 MO-SPPS 原算法中的插入位置如下：

| 机制 | 插入位置 | 说明 |
|---|---|---|
| 连续型软压力共享池 | 替换原软压力采样公式 | 作为主算法默认采样机制 |
| 偏好区域 Novelty（可选） | Novelty 接受规则中的参考集定义 | 可选扩展，默认关闭 |
| 自适应 \(Q_j\) | 补充到共享池状态更新和参数敏感性分析 | 保护高贡献关键组件 |

推荐主算法默认配置如下：

```yaml
shared_pool:
  mode: soft_pressure_continuous
  epsilon: 0.01
  tau: 1.0
  base_capacity_Q0: 5

region_novelty:
  enabled: false
  region_threshold: 0.3

adaptive_capacity:
  use_adaptive_Q: true
  alpha_Q: 2.0
  Q_min: 1
  Q_max: 20
  update_interval: 10
  contribution_metric: archive_frequency_weighted_crowding
```

---

## 5. 消融实验建议

为验证三项完善机制的独立贡献，建议增加以下消融版本：

| 版本 | 设置 | 目的 |
|---|---|---|
| MO-SPPS-Base | 原始软压力池 + 固定 \(Q_j\) | 基础对照 |
| MO-SPPS-ContinuousPool | 连续型软压力池 | 验证连续压力是否优于截断压力 |
| MO-SPPS-AdaptiveQ | 连续池 + 自适应 \(Q_j\) | 验证高贡献组件保护机制 |
| MO-SPPS-RegionNovelty | 连续池 + 偏好区域 Novelty | 验证偏好区域约束对探索的影响（可选扩展） |
| MO-SPPS-Full | 加入预算、继承、混合 Archive 裁剪等完整机制 | 最终主算法 |

建议所有版本使用相同函数评估次数、相同随机种子和相同对比算法，确保实验公平。

---

## 6. 简要总结

这三项完善机制分别定位于 MO-SPPS 的不同层次：

1. **连续型软压力共享池**——默认采样机制，解决共享池压力不连续和惩罚饱和问题；
2. **偏好区域约束的 Novelty 参考集**——可选扩展，在需要控制探索范围时启用。默认 MO-SPPS 已通过 Pareto Archive 的非支配性和 Novelty 接受规则的质量阈值内嵌了质量约束，此机制作为额外的区域控制选项；
3. **自适应 \(Q_j\)**——默认增强项，解决高贡献关键组件可能被过度抑制的问题，通过消融实验单独验证。
