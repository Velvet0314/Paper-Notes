1. FRAGFM 的框架
	1. 片段图的表示
		- 细粒度图（原子级别的图）
			- 将分子在原子层面表示为一个图 $G = (V, E)$，其中 $V$ 是原子集合，$E$ 代表原子间的化学键
			- 每个节点 $v_k \in V$ 对应一个独立的原子，而边 $e_{kl} \in E$ 表示原子 $v_k$ 和 $v_l$ 之间的键（包括非键相互作用）
		- 粗粒度图（片段级别的图）
			- 分子的粗粒度表示定义为一个图 $G = (\mathcal{X}, \mathcal{E})$。此处，$x_i \in \mathcal{F}$ 对应一个片段，而每条边 $\varepsilon_{ij} \in \mathcal{E}$ 对应片段间的连通性。**每个片段被解释为一个原子级别的图**
			- $\{x_i\}_{i} = \{(V_i, E_i)\}_{i}$ 是 $G = (V, E)$ 的不相交子图，其中 $V_i \subseteq V$ 且 $E_i \subseteq E$，并且对于不同的片段索引 $i, j$，有 $V_i \cap V_j = \emptyset$
			- 片段级图 $\mathcal{E}$ 中的边是从 $E$ 中诱导而来的，这意味着如果两个片段 $F_i, F_j \in \mathcal{X}$ 对应的原子之间**至少存在一个化学键，则这两个片段是连通的**，即
			$$ε_{ij} \in \mathcal{E} \quad \text{if} \quad \exists e_{kl} \in E \quad \text{such that} \quad v_k \in V_i, v_l \in V_j \tag{1}$$
			* **变量定义**：
			    * $ε_{ij}$：片段图 $\mathcal{G}$ 中连接片段 $x_i$ 和 $x_j$ 的边
			    * $e_{kl}$：原子图 $G$ 中连接原子 $v_k$ 和 $v_l$ 的键
			    * $V_i, V_j$：分别是片段 $x_i$ 和 $x_j$ 包含的原子集合
			- **解释**：
				- 只要在原始分子中，**存在至少一个化学键，其连接的两个原子分属于不同的片段**（一个在 $V_i$ 中，一个在 $V_j$ 中），那么在片段图上，我们就认为这**两个片段是相连的**
	2. 基于粗粒度到细粒度的自编码器实现分子图压缩
		- 从原子图到片段图的转换（Coarsening）是一个**有损压缩**过程。片段图 $\mathcal{G}$ 只告诉我们片段A和片段B相连，但**丢失了关键的细节信息**：连接的具体位置未知（究竟是A中的哪个原子与B中的哪个原子相连）
		- 通过**粗-细粒度自编码器 (Coarse-to-Fine Autoencoder)** 来解决（具体内容补充到附录中）
	3. 用于粗粒度图的离散流匹配
		- **目标**：学习一个生成模型 $p(\mathcal{G}, z)$，使其能够生成片段级图 $\mathcal{G}$ 及其对应的连接信息隐变量 $z$
		- **分类**：
			1. 由于连接性变量 $\varepsilon \in E$ 是二元的 —— 采用离散流匹配
			2. 潜在变量 $z$ 是低维实值向量（连续的） ——  采用标准的连续流匹配
		- **定义时序边缘分布** (Temporal Marginal Distribution)
			$$p_{t|1}(x_t | x_1, \mathcal{B}) = t\delta^\mathcal{B}(x_t, x_1) + (1-t)p_0(x_t|\mathcal{B}) \tag{2}$$

			* **变量定义**：
			    * $x_1$：目标数据点，即一个真实的、“干净”的片段类型 (来自目标粗粒度图)
			    * $x_t$：在时间 $t$ 时的随机变量，代表一个片段类型。$t=0$ 对应纯噪声，$t=1$ 对应真实数据
			    * $\mathcal{B}$：一个**片段袋 (Fragment Bag)**，它是一个包含了当前样本可能用到的所有片段类型的集合。这是一个关键的上下文信息
			    * $p_{t|1}(x_t | x_1, \mathcal{B})$：给定目标片段 $x_1$ 和片段袋 $B$ 的条件下，在时间 $t$ 观察到片段 $x_t$ 的概率。这定义了从 $x_1$ “加噪”到时间 $t$ 的边缘分布
			    * $p_0(x_t|\mathcal{B})$：**先验分布 (prior distribution)**，即 $t=0$ 时的噪声分布。在这里，它被定义为在片段袋 $B$ 内的**均匀分布**
			    * $\delta^\mathcal{B}(x_t, x_1)$：一个经过修改的**克罗内克函数 (Kronecker delta)**。它等于 $1_{\mathcal{B}}(x_t) \cdot \delta(x_t, x_1)$，当且仅当 $x_t = x_1$ 且 $x_t$ 属于片段袋 $B$ 时为 1，否则为 0
			- **解释**：
				- 当 $t=1$ 时，公式变为 $p_{1|1}(x_t|x_1, \mathcal{B}) = \delta^\mathcal{B}(x_t, x_1)$，意味着概率质量完全集中在目标数据 $x_1$ 上
				- 当 $t=0$ 时，公式变为 $p_{0|1}(x_t|x_1, \mathcal{B}) = p_0(x_t|\mathcal{B})$，即一个在袋 $\mathcal{B}$ 内的均匀噪声分布
			- **问题**：分子片段的种类非常繁多，如果要在所有可能的片段上定义一个连续时间马尔可夫链 (CTMC) 的完整的转移矩阵 $R_t$，其维度将会巨大（$|\mathcal{F}| \times |\mathcal{F}|$）
			- **解决**：引入了一种随机包选择策略
				- 对于一个**完整的训练分子** $x_{1:D}$（它由 $D$ 个片段构成），为这个分子构建的片段袋 $\mathcal{B}$，**必须包含**这个分子本身所拥有的那 $D$ 种片段
					- $D$ 就是指一个分子被分解后，所包含的**片段的总数量**。例如，如果一个分子被分解成了3个片段，那么 $D=3$，数据点就是 $x_{1:3} = \{x_1, x_2, x_3\}$
				- 片段袋 $\mathcal{B}$ 会再加上从数据集中**随机采样的其他一些分子的片段**
				- **训练采样**
					- 从原本的广阔的片段库缩减到了特定分子的片段袋 $\mathcal{B}$，即 $\mathcal{B} \sim \mathbb{Q}(\cdot | x_{1:D})$
				- **推理采样**
					- 开始并不知道目标 $x_1$。因此，模型通过对整个数据集的期望来构建一个**无条件的片段袋 $Q = \mathbb{E}_{x_1 \sim p_{\text{data}}}[Q(\cdot|x_1)]$**。在实践中，这意味着随机采样一批（例如256个）训练分子，把它们所有的片段收集起来形成一个推理时使用的片段袋 $\mathcal{B}$。然后从这个 $B$ 中采样初始噪声 $x_0$，并开始生成过程
		- **定义生成动力学** (Kolmogorov Forward Equation)
			$$p_{t+dt|t}(y|x_t, \mathcal{B}) = \delta^\mathcal{B}(x_t, y) + R_t(x_t, y|\mathcal{B})dt \tag{3}$$

			* **变量定义:**
			    * $p_{t+dt|t}(y|x_t, \mathcal{B})$：从时间 $t$ 的状态 $x_t$ 到时间 $t+dt$ 的状态 $y$ 的转移概率。
			    * $R_t(x_t, y|\mathcal{B})$：**转移速率矩阵 (transition rate matrix)** $R_t(i, j|\mathcal{B})\ \text{for}\ i \neq j$ 表示在时间 $t$ 从片段 $i$ 跳转到片段 $j$ 的瞬时速率
			- 解释：这个公式是描述连续时间马尔可夫链 (CTMC) 演化的**基本微分方程**。在极小的一段时间 $dt$ 内：
			    * 系统有 $1 - \sum_{y \neq x_t} R_t(x_t, y|\mathcal{B})dt$ 的概率保持在原状态 $x_t$ 不变（对应 $\delta^\mathcal{B}(x_t, y)$ 项）
			    * 系统有 $R_t(x_t, y|\mathcal{B})dt$ 的概率从状态 $x_t$ 跳转到另一个状态 $y$
2. Method 细节
	1. 粗-细粒度自编码器细节
		- 编码与解码过程的形式化定义
			$$\begin{align*}\mathcal{G} &= \text{Fragmentation}(G) \\z &\sim q_{\theta}(z|G) = \mathcal{N}(\text{Encoder}(G; \theta), \sigma) \tag{4} \\ \hat{E} &= \text{Decoder}(\mathcal{G}, z; \theta)\end{align*}$$
			* **变量定义**：
			    * $G, \mathcal{G}$：分别是原子级图和片段级图
			    * $z$：连续隐变量
			    * $q_{\theta}(z|G)$：编码器（Encoder）定义的后验分布，用于从 $G$ 推断 $z$。它被建模为一个均值由神经网络 $\text{Encoder}(G; \theta)$ 预测、方差 $\sigma$ 通常固定的高斯分布
			    * $\hat{E}$：解码器（Decoder）重构出的跨片段原子键的集合（或其概率）
			* **解释:**
			    1. **分解 (Fragmentation)**：将输入的原子图 $G$ 按照预设规则（如BRICS）分解为粗糙的片段图 $\mathcal{G}$。这是一个确定性的、无学习参数的过程
			    2. **编码 (Encoding)**：一个参数为 $\theta$ 的神经网络（Encoder）读取整个原子图 $G$，并**输出一个高斯分布的参数**（主要是均值）。然后我们从这个分布中采样一个隐变量 $z$。这个 $z$ 捕获了分解过程中丢失的精细连接信息
			    3. **解码 (Decoding)**：另一个神经网络（Decoder）接收粗糙图 $\mathcal{G}$ 和隐变量 $z$ 作为输入，它的任务是**预测**那些连接不同片段的原子之间应该存在哪些化学键，即输出 $\hat{E}$（==仅重建与粗粒度表示中碎片连通性对应的那些原子级别边 $\hat{E}$==）
		- 自编码器的损失函数 (VAE Loss)
			$$\mathcal{L}_{\text{VAE}}(\theta) = \mathbb{E}_{G \sim p_{\text{data}}} \left[ \mathcal{L}_{\text{CE}}(E, \hat{E}(\theta)) + \beta D_{\text{KL}}(q_{\theta}(z|G) || p(z)) \right] \tag{5}$$

			* **变量定义**：
			    * $\mathcal{L}_{\text{CE}}$：**交叉熵损失 (Cross-Entropy Loss)**，也称为重构损失
			    * $D_{\text{KL}}$：**KL散度 (Kullback-Leibler Divergence)**，也称为正则化项
			    * $p(z)$：隐变量 $z$ 的先验分布，通常设为标准正态分布 $\mathcal{N}(0, I)$
			    * $\beta$：一个超参数，用于平衡重构损失和KL散度项的权重
			- **解释**：
				 1. **重构项 $\mathcal{L}_{\text{CE}}(E, \hat{E}(\theta))$**：这部分的目标是让解码器**尽可能完美地重构出原始的分子**。它度量了预测的键连接 $\hat{E}$ 与真实的键连接 $E$ 之间的差异。如果解码器完美重构，这个损失为 0
				 2. **正则化项 $\beta D_{\text{KL}}(...)$**：它的目标是让编码器产生的后验分布 $q_{\theta}(z|G)$ **尽可能地接近**一个简单的、预设的先验分布 $p(z)$ —— 防止学到一个非常复杂、支离破碎的隐空间，每个分子都对应一个孤立的点，无法采样
	2. 片段上的去噪流匹配
		- 定义转移速率矩阵
			$$R_t^*(x_t, y | x_1, \mathcal{B}) = \frac{\text{ReLU}\left[ \partial_t p_{t|1}(y|x_1, \mathcal{B}) - \partial_t p_{t|1}(x_t|x_1, \mathcal{B}) \right]}{Z_t^{>0} p_{t|1}(x_t|x_1, \mathcal{B})} \quad \text{for } x_t \neq y \tag{6}$$
			* **变量定义**：
			    * $R_t^*(x_t, y | x_1, \mathcal{B})$：从当前噪声状态 $x_t$ 跳转到另一个状态 $y$ 的**目标转移速率**。星号 `*` 代表这是我们希望模型学习的“标准”
			    * $p_{t|1}(...)$：我们在公式(2)中定义的**预设概率路径**。这是连接噪声和数据的“轨道”
			    * $\partial_t p_{t|1}(...)$：**$p_{t|1}$ 对时间 $t$ 的偏导数**
				    * 代入公式(2)，计算得：
					    - $\partial_t p_{t|1}(x_t | x_1, \mathcal{B}) = \frac{\partial}{\partial t} \left[ t\delta_{\mathcal{B}}(x_t, x_1) + (1-t)p_0(x_t|\mathcal{B}) \right] =  \delta_{\mathcal{B}}(x_t, x_1) - p_0(x_t|\mathcal{B})$
			    * $Z_t^{>0}$：一个归一化常数
			- **解释**：
				- 偏导数项的意义：在时间 $t$，概率在 $x_t$ 这个点的**“瞬时流速”或“变化速率”**。
				    1. 如果 $x_t$ 是**目标数据** ($x_t = x_1$)，那么 $\delta_{\mathcal{B}}(x_t, x_1)=1$。流速为 $1 - p_0(x_1|\mathcal{B})$，这是一个**正值**（因为 $p_0$ 是概率，小于1）。这意味着随着时间推移，概率正在**流入**并**汇集**到目标数据点上
				    2. 如果 $x_t$ 是**任何其他点** ($x_t \neq x_1$)，那么 $\delta_{\mathcal{B}}(x_t, x_1)=0$。流速为 $-p_0(x_t|\mathcal{B})$，这是一个**负值**。这意味着随着时间推移，概率正在从这些非目标点**流出**
		- 离散时间步的采样近似
			$$\tilde{p}_{t+\Delta t | t}(x_{t+\Delta t}^{1:D} | x_t^{1:D}, x_1^{1:D}, \mathcal{B}) = \prod_{d=1}^D  \left(\delta^{\mathcal{B}}(x_t^{(d)}, x_{t+\Delta t}^{(d)}) + \mathbb{E}_{p_{t|1}^{(d)}(x_1^{(d)}|x_1^{1:D}, \mathcal{B})} \left[ R_t^{(d)}(x_t^{(d)}, x_{t+\Delta t}^{(d)}) | x_1^{(d)}, \mathcal{B}) \Delta t \right]\right) \tag{7}$$
			* **解释**：
			    - 这是一个**欧拉法 (Euler method)** 的应用。CTMC的演化是一个微分方程，难以精确求解。因此，我们把它离散化成许多小的时间步 $\Delta t$。即在一个很小的时间段 $\Delta t$ 内：
				    * 一个片段有很大的概率保持不变（$\delta_{\mathcal{B}}$ 项）
				    * 有 $R_t(...) \Delta t$ 的小概率从当前状态 $x_t^{(d)}$ 跳转到另一个状态 $y^{(d)}$
				    * 连乘符号 $\prod_{d=1}^D$ 表明，在做这个近似时，我们**假设分子中的 $D$ 个片段是独立演化的**。这是一种简化，但在小时间步下是合理的
		- 片段袋 $\mathcal{B}$ 的构建 —— **蒙特卡洛近似**
			- **训练阶段**：构建片段袋是很容易的，因为我们有一个明确的目标分子 $x_1$。我们可以围绕这个 $x_1$ 来构建一个量身定制的袋子，这个过程我们记为从条件分布 $\mathbb{Q}(\cdot|x_1)$ 中采样
				1. 从数据分布中选择固定数量的分子，并将目标分子 $x_1$ 包含在其中
				2. 这些分子中的所有片段以形成片段袋 $\mathcal{B}$
			- **采样阶段**：最终要生成的分子 $x_1$ 非先验可用，所以不能以任何特定的 $x_1$ 为条件，必须构建一个**无条件的 (unconditional)**、具有普适性的片段袋，定义该无条件分布 $\mathbb{Q}$：
				$$\mathbb{Q} = \mathbb{E}_{x_1 \sim p_{\text{data}}} [\mathbb{Q}(\cdot|x_1)] \tag{8}$$
				- 类似于训练阶段，但是不用包含特定的分子
	3. 神经网络的参数化
	4. 