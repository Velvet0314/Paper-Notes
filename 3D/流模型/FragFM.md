1. FragFM 的框架
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
					- 开始并不知道目标 $x_1$。因此，模型通过对整个数据集的期望来构建一个**无条件的片段袋 $Q = \mathbb{E}_{x_1 \sim p_{\text{data}}}[Q(\cdot|x_1)]$**。在实践中，这意味着随机采样一批（例如256个）训练分子，把它们所有的片段收集起来形成一个推理时使用的片段袋 $\mathcal{B}$。然后从这个$\mathcal{B}$ 中采样初始噪声 $x_0$，并开始生成过程
		- **定义生成动力学** (==**Kolmogorov**== Forward Equation)
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
		<div style="text-align: center;"> <img src="denosingmodule.png" width="500"> </div>

		1. 片段袋嵌入（Fragment Embedder）
			- 由上面步骤构建的片段袋 $\mathcal{B}$，$\mathcal{B}$ 由一个 MPNN 会**逐一处理**袋中的每一个片段。它将每个片段的化学结构（一个小的原子图）转换成一个高维向量
			$$h_i = \text{FragmentEncoder}(x_i; \phi), \quad \text{for } x_i \in \mathcal{F} \tag{9}$$
			- 这个公式定义了如何将一个化学片段（一个小的原子图 $x_i$）转换成一个数学向量 $h_i$。$\text{FragmentEncoder}$ 是一个消息传递神经网络（MPNN），它在片段的原子图上运行，最终聚合信息得到一个代表整个片段的向量。参数 $\phi$ 是可学习的
		2. 片段级图表示和上下文信息提取
			- 从一个带噪声的片段图 $\mathcal{G}_t$ 开始，图中的每个节点都需要一个初始的特征向量，从第一步的片段袋中找
			- 再通过一个图 Transformer 得到一个**富含上下文的新的节点嵌入（Node Embedding）**
		3. 做出预测
			- Node Embedding 和 初始的 Fragment Embedder 做内积算相似度，经 softmax 被转换成一个概率分布，得到最终的**干净片段**的预测
			$$\hat{p}_i = \text{Softmax}\left( \{h_i^{(l)} \cdot h_k^{(0)} \}_{k \in \mathcal{B}}\right) \tag{10}$$
			- **变量定义**：
			    * $\hat{p}_i$：模型对第 $i$ 个节点的**最终预测**。这是一个概率分布向量，其维度等于片段袋 $\mathcal{B}$ 的大小。向量的第 $k$ 个元素表示第 $i$ 个节点是片段 $x_k$ 的概率
			    * $h_i^{(l)}$：这是第 $i$ 个节点的**最终嵌入向量**。上标 `(l)` 至关重要，它代表这个向量是**经过了 $l$ 层图Transformer处理之后**的结果。因此，$h_i^{(l)}$ 是一个**富含上下文信息 (context-aware)** 的表示。它不仅知道自己最初是什么，更重要的是，它“理解”了自己在整个图结构中的角色
			    * $h_k^{(0)}$：这是片段袋 $\mathcal{B}$ 中第 $k$ 个候选片段 $x_k$ 的**初始嵌入向量**。上标 `(0)` 同样至关重要，它代表这是由 `Fragment Embedder` 直接产生的、**未经任何图Transformer处理**的向量。因此，$h_k^{(0)}$ 是一个**上下文无关 (context-free)** 的表示，可以被看作是片段 $x_k$ 纯粹的**“身份ID”或“数学指纹”**
			    * $h_i^{(l)} \cdot h_k^{(0)}$：计算这两个向量的**点积 (dot product)**。点积是衡量两个向量相似度的常用方法。
			    * $\{ \cdot \}_{x_k \in \mathcal{B}}$：这个花括号表示，点积操作要对片段袋 $\mathcal{B}$ 中的**每一个**候选片段 $x_k$ 都执行一次。最终会得到一个与袋大小相同的分数列表（logits）
		4. CTMC 驱动流匹配
			- 由第三步的**干净片段**得到**理论转移速率矩阵** $R_t^*$，然后经 CTMC 从 $t=0$ 到 $t=1$ 进行完整的去噪流程
	4. 从片段图进行原子级别图重构
		- 目标：解决片段拼接的歧义性 —— 片段和片段之间到底是哪两个原子相连？
		- 方法：将“选择最佳连接点”的问题，转换成了一个经典的图论问题——**最大权匹配 (Maximum Weighted Matching)**
			1. 标记接口原子 $V_m$，用 `*` 标记 —— **基于化学反应规则的分解 (BRICS)**
			2. 定义连接边的候选集及其配对分数
				$$e_{kl} \in E_m \quad \text{if} \quad v_k \in \hat{V_i}, \quad v_l \in \hat{V_j}, \quad \text{and} \quad ε_{ij} \tag{11}$$
				- 如果原子 $v_k$ 和 $v_l$ 分别属于不同的片段（$V_i$ 和 $V_j$），并且这两个片段在粗糙图上是相连的（$ε_{ij} \in \mathcal{E}$），那么 $(v_k, v_l)$ 这条边就属于我们考虑的候选边集合 $E_m$
				- 解码器接收片段图 $\mathcal{G}$ 和隐变量 $z$，然后对每一个可能的配对 $(v_i, v_j)$，都预测一个分数 $w_{ij}$
				$$M^* = \text{argmax}_{M \subseteq E_m} \sum_{(i,j) \in M} w_{ij} \tag{12}$$
				- 注：**一个接口原子最多只能参与一次配对** —— 在图论中，满足这个“一对一”规则的边的集合，就叫做一个**匹配 (Matching)**
			3. 利用**Blossom算法**求解最大权匹配
				- 注：Blossom算法的复杂度是 $O(N^3)$，其中 $N$ 是接口原子的数量，看似很高。但在实践中**完全可以忽略不计 (negligible)**。因为**一个分子中的接口原子数量 $N$ 通常非常少**，远远小于分子的总原子数
	5. 采样技巧
		1. 目标引导（Target Guidance）
			- **动机**：神经网络在时刻 $t$ 其实已经对最终的“干净”数据 $x_1$ 做出一个相当不错的**预测**了（称之为 $\hat{x}_1$）。既然已经有了一个关于“终点”在哪里的猜测（$\hat{x}_1$），就可以在理论速率的基础上，额外增加一个**直接指向我们预测的终点 $\hat{x}_1$ 的“引力”
			- 新的速率矩阵
				$$\begin{align*}R_t(x_t, y | x_1) &= R_t^*(x_t, y | x_1) + R_t^\omega(x_t, y | x_1) \tag{13} \\[7pt] R_t^\omega(x_t, y | x_1) &= \omega \cdot \frac{\delta(y, x_1)}{Z_t^{\gt 0} p_{t|1}(x_t|x_1)} \tag{14}\end{align*}$$
				- **变量定义**：
					* $R_t^\omega$：**引导速率 (Guidance Rate)**。这是一个新增的、起引导作用的“引力项”
					* $\omega$：**引导强度 (guidance strength)**。这是一个超参数，控制“引力”的大小
					* $\delta(y, \hat{x}_1)$：**克罗内克函数**。当且仅当目标状态 $y$ **恰好**是我们预测的干净状态 $\hat{x}_1$ 时，它才为 1，否则为 0
					- 注：此修改会引入 Kolmogorov 方程轻微的 $O(ω)$ 违反。然而，经验结果表明，小的 $ω$ 值在不显著扭曲学习到的分布的情况下可以改善样本质量 —— 论文取 $\omega = 0.002$
				- 问题：$\delta(y, \hat{x}_1)$ 克罗内克函数的具体定义有待商榷 —— 因为如果是严格的二元0/1，那么 $t$ 在开始时其实就没有引导作用，这里推测是用 softmax 的概率输出做一个加权来取一个平滑的引导
		2. 细致平衡（Detailed Balance）
			- **动机**：在 CTMC 中，一个标准的速率矩阵 $R_t^*$ 可以添加额外的随机性，并且整个系统 **仍然满足Kolmogorov方程**
			- 新的速率矩阵
				$$\begin{align*}p_{t|1}(x_t | x_1) &R_t^{DB}(x_t, y | x_1) = p_{t|1}(y | x_1) R_t^{DB}(y, x_t | x_1) \tag{15} \\[7pt]  &R_t^\eta = R_t^* + \eta R_t^{DB}, \quad \eta \in \mathbb{R}^+ \tag{16}\end{align*}$$
				- **变量定义**：
					* $R_t^{DB}$：任何一个满足此条件的速率矩阵。DB 代表 Detailed Balance
				- **解释**：添加额外的随机性必须满足**细致平衡条件**
					- 正向流动和反向流动必须平衡
						- 左边：在状态 $x_t$​ 的概率 × 从 $x_t$ 转移到 $y$ 的速率
						- 右边：在状态 $y$ 的概率 × 从 $y$ 转移回 $x_t$ ​的速率
					- 状态 $x_t$ 随时间的变化率 = 所有流入 $x_t$ 的流量之和 - 所有流出 $x_t$ 的流量之和
						$$\begin{align*}\frac{d}{dt} p_{t|1}(x_t) &= \sum_{y \neq x_t} \left[ \text{Flux}(y \to x_t) - \text{Flux}(x_t \to y) \right] \\[7pt] &= \sum_{y \neq x_t} \left[ p_{t|1}(y) R_t^{DB}(y, x_t) - p_{t|1}(x_t) R_t^{DB}(x_t, y)  \right]  \xlongequal{\text{under eq.15}} 0\end{align*}$$
				- 优点：
					1. **稳定的边缘分布**：$p_{t|1}(x_t)$ 保持不变
					2. **可控的生成过程**：添加满足详细平衡的 $R_t^{DB}$ 不会改变目标分布
					3. **数学一致性**：修改后的速率矩阵 $R_t^\eta = R_t^* + \eta R_t^{DB}$​ 仍然满足 Kolmogorov 方程
	6. 无分类器引导
		- 方法：通过在**条件模型和无条件模型之间进行插值**，无需**显式属性分类器**即可实现可控的分子生成
		- 具体实现：训练时，有90%的数据会提供属性标签 $c$，模型学习条件速率 $R_t(x_t, y|c)$；另外10%的数据则用一个特殊的“空”标签 $\phi$ 来代替，模型学习无条件速率 $R_t(x_t, y| φ)$
		- 引导采样
			$$R_t^{\theta, \gamma}(x_t, y | c) = R_t^\theta(x_t, y | c)^\gamma \cdot R_t^\theta(x_t, y | φ)^{1-\gamma} \tag{17}$$

			- **解释**：在采样时，我们想生成一个满足条件 $c$ 的分子，于是同时计算出模型预测的**条件速率** $R_t(\cdot|c)$ 和**无条件速率** $R_t(\cdot|φ)$。然后，通过一个**引导强度参数 $\gamma$** 来将它们进行**插值或外推**（在**对数空间**中）
				$$\log R_t^{\theta, \gamma}(x_t, y | c) = \gamma \cdot R_t^\theta(x_t, y | c) + (1-\gamma) \cdot \log R_t^\theta(x_t, y | φ)$$
			    * **$\gamma = 0$**: 最终预测就是基准点（无条件生成）
			    * **$\gamma = 1$**: 最终预测就是有条件预测点
			    * **$0 < \gamma < 1$**: 在基准点和有条件点之间进行**插值**
			    * **$\gamma > 1$**: 沿着从基准点到有条件点的方向，**前进超过100%的距离**，进行**外推**。这相当于**放大了**条件 $c$ 的影响，使得生成的样本更强烈地体现出该属性