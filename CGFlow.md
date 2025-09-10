1. GFlowNets 生成流网络的基础知识
	- **生成流网络的定义**
		- 生成流网络是一类概率模型，用于学习随机策略来构造组合对象 $x \in \mathcal{X}$，**该策略与终端状态 $R(x)$ 的奖励成比例**，即 $p(x) \propto R(x)$。每个对象 $x$ 通过轨迹 $\tau = (s_0 \rightarrow \ldots \rightarrow s_n = x) \in \mathcal{T}$ 从初始状态 $s_0$ 和一系列状态转移 $s \rightarrow s'$ 构造而成，其中终端状态 $s_n \in \mathcal{X}$ 对应 $x \in \mathcal{X}$
		- GFlowNet 将流 $F$ 建模为沿有向无环图（DAG）$\mathcal{G} = (\mathcal{S}, \mathcal{A})$ 的非归一化密度函数，其中 $\mathcal{S}$ 表示状态空间，$\mathcal{A}$ 表示转移
			- $\mathcal{S}$ 是**图的节点** —— 包含一个初始状态 $s_0$ (e.g. 空分子)、所有中间状态 (e.g. 构建了一半的分子)、所有终端状态 $x \in \mathcal{X}$ (e.g. 完整的分子)
			- $\mathcal{A}$ 是**图的边** —— 代表了从一个状态到另一个状态的合法转换
			- **轨迹流** $F(\tau)$ 定义为通过轨迹 $\tau$ 的流 ——  一条从 $s_0$ 到某个终端 $x$ 的完整路径 $\tau = (s_0 \rightarrow \ldots \rightarrow s_n = x) \in \mathcal{T}$。这代表了构建对象 $x$ 的一个完整步骤序列
			- **节点流** $F(s)$ 定义为通过节点 $s$ 的流，即 $F(s) = \sum_{\tau \in \mathcal{T}: s \in \tau} F(\tau)$
			- **边流** $F(s \rightarrow s')$ 定义为沿边 $s \rightarrow s'$ 的总流，即 $F(s \rightarrow s') = \sum_{\tau \in \mathcal{T}: (s \rightarrow s') \in \tau} F(\tau)$ 
		- 简单来说：我们希望从空间 $\mathcal{X}$ 中采样，但不是均匀采样，而是倾向于**采样那些“好”的对象** —— 通过一个**奖励函数** $R(x)$ 来量化，$x$ 是一个完整的对象。$R(x)$ **越高，对象越好**
	- **策略分布**
		- 从流网络中，我们定义两个策略分布
			1. **前向策略** $P_F(s'\mid s)$ 执行状态转移 $s \rightarrow s'$，来自流分布，即 $P_F(s'|s) = F(s \rightarrow s')/F(s)$
				- 解释：当处于状态 $s$ 时，它决定了将流向下一个可能状态 $s'$ 的流量比例。这也就是模型在状态 $s$ 时，选择动作 $s \rightarrow s'$ 的概率
			2. **后向策略** $P_B(s\mid s')$ 将节点流 $F(s)$ 分配给反向转移 $s \leftarrow s'$，即 $P_B(s|s') = F(s' \leftarrow s)/F(s')$
				- 解释：当已经到达状态 $s'$ 时，它回头看，估计从上一步状态 $s$ 过来的流量占 $s'$ 总流入流量的比例。它用于在训练中形成一个“闭环”约束
	- **边界条件**
		- 为了匹配生成 $x \in \mathcal{X}$ 的可能性与奖励函数 $R$，需要实现两个边界条件
			1. 首先，每个终端状态 $x$ 的节点流（表示采样对象 $x$ 的非归一化概率）必须等于其奖励，即 $F(x) = R(x)$
			2. 其次，初始节点流 $s_0$（表示**划分函数** ）必须等于所有奖励的总和，即 $Z = \sum_{x \in \mathcal{X}} R(x)$
			- 满足两个边界条件后才能保证 $p(x) \propto R(x)$
	- **训练目标**
		- 实现这些条件的一个目标是**轨迹平衡（TB,trajectory balance）**，定义如下：$$\mathcal{L}_{TB}(\tau) = \left( \log \frac{Z_0 \prod_{t=1}^T P_F(s_t|s_{t-1}; \theta)}{R(x) \prod_{t=1}^T P_B(s_{t-1}|s_t; \theta)} \right)^2 \tag{1}$$
			其中 $P_F$、$P_B$ 和 $Z$ 直接参数化以最小化 TB 目标
2. CGFlow
	1. **数据表示**
		- 将对象 $x$ 表示为元组 $(\mathcal{C}, \mathcal{S})$，其中 $\mathcal{C}$ 表示组合结构，$\mathcal{S}$ 表示与状态相关的配置状态
			- $\mathcal{C}$ : 对象的 **离散组合结构**。在本文的应用中，定义为一个有序的序列 $\mathcal{C} = (\mathcal{C}^{(i)})_{i=1}^n = (\mathcal{C}^{(1)}, \mathcal{C}^{(2)}, ..., \mathcal{C}^{(n)})$，代表了**分子的合成路径**
				- 其中 $n$ 表示其组合组件的数量（e.g. 分子构建块）。第 $i$ 个组件 $C^{(i)}$ 被添加到轨迹 $\tau$ 的第 $i$ 个生成步骤（e.g. 在合成途径中）
			- $\mathcal{S}$：$\mathcal{S}^{(i)}$ 表示与组件 $i$ 相关的状态，大小为 $(m_i, d)$。连续状态 $\mathcal{S}$ 定义为来自每个组件的所有状态的有序元组 $\mathcal{S} = (\mathcal{S}^{(i)})_{i=1}^n$
				- 每个组合组件 $\mathcal{C}^{(i)}$ 包含 $m_i$ 个点（例如，原子），并且 $m_i$ 在不同组件之间变化。每个点都有一个维度为 $d$ 的关联连续状态（e.g. 原子位置）
			- 标准的流匹配方法**仅对状态变量进行建模**，而忽略了对象的组成结构和生成顺序。在 CGFlow 中，对象的组**成结构和状态变量被联合建模**，从而确保了生成对象的组合有效性
	2. **联合条件流过程（Joint Conditional Flow Process）**
		- 联合条件流的初始变量定义
			- $x_0 = (\mathcal{C}_0, \mathcal{S}_0)$：初始状态。$\mathcal{C}_0 = \phi$ 是一个空结构（空图），$\mathcal{S}_0 = \big[\ \big]$ 是空的坐标集合
			- $x_1 = (\mathcal{C}_1, \mathcal{S}_1)$：最终状态。$\mathcal{C}_1$ 是完整的目标分子合成路径，$\mathcal{S}_1$ 是其最终的、稳定的 3D 构象。代表我们希望生成的真实数据
			- $\mathcal{P}_{t|1}(\cdot \mid x_1)$：这是一个条件概率路径，描述了在给定最终目标 $x_1$ 的情况下，中间状态 $x_t$ 的分布。生成模型的目标就是学习这个过程的逆过程
				- 条件概率路径必须满足下面的边界条件
					$$\mathcal{P}_{t|1}(x_t|x_1) = \begin{cases} \delta(x_t = x_0), & t = 0, \\ \delta(x_t = x_1), & t = 1. \end{cases} \tag{2}$$
					这表示在 $t=0$ 时，初始状态一定是 $x_0$；同时在 $t=1$ 时，最终状态一定是真实数据 $x_1$
		1. 组合流（Compositional Flow）
			- 组合流定义了一个在组合结构 $\mathcal{C}$ 上的条件概率流，逐步将其从空图 $\mathcal{C}_0$ 过渡到完整的结构 $\mathcal{C}_1$
			- 函数 $k(t)$ 返回在时间 t 时，已经加入了多少个组件
				$$k(t) = \begin{cases} 0, & t = 0, \\ \text{min}(\lfloor t/\lambda \rfloor +1,n = x_1), & t \gt 0. \end{cases} \tag{2}$$
				- 其中：
					- $\lambda$ 定义了添加每个组合组件之间的时间间隔
					- 在时间 $t$，组合结构 $\mathcal{C}_t$ 包括来自该顺序的前 $k(t)$ 个组件
					- $k(t)$ 随着 $t$ 的推进以离散的步长增大，从 $k(0) = 0$ 开始，并确保 $k(1) = 1$，使得所有 $n$ 个组件在 $t = 1$ 时被添加
					- 每个组件 $\mathcal{C}^{(i)}$ 在时间 $t_{add}^{(i)} = \lambda \cdot (i - 1)$ 时生成
					- 为了确保所有组件都在有效时间范围内生成，我们要求对所有数据点满足 $\lambda \leq 1/n$，满足 $t_{gen} \leq 1 - \lambda$。时间 $t$ 的组合结构由下式给出：$$\mathcal{C}_t = (\mathcal{C}^{(i)})_{i=1}^{k(t)} \tag{4}$$
						这种表述保证了从空状态 $C_0$ 到完全构建结构 $C_1$ 的逐步和顺序构造，在固定间隔处进行
		2. 状态流（State Flow）
			- 状态流定义了连续状态空间 $\mathcal{S}=(\mathbf{S}^{(i)})_{i=1}^n$ 上的条件概率路径
			- 每个连续状态 $\mathbf{S}^{(i)}$ 在其对应的组件分量 $\mathbf{C}^{(i)}$ 生成时被初始化，且各组件在不同时间生成
				- 直观地说，相比于那些较早生成的组件，最近添加的组件的连续状态具有更大的不确定性
			- 解决方式：引入一个时间偏置，全局时间 $t$ 被重新参数化为组件级的局部时间 $t_{local}^{(i)}$，定义为：
				$$t_{local}^{(i)} = \text{clip}\left(\frac{t - t_{gen}^{(i)}}{t_{window}}\right) \tag{5}$$
				- 其中：
					- $t_{gen}^{(i)}$ 是组件 $C^{(i)}$ 的生成时刻的**全局时间点**
					- $t_{window}$ 是插值时间窗口，定义了任何一个构件从“诞生”（$t_{local}=0$）到“成熟”（$t_{local}=1$）所需要花费的**时间窗口长度**，也就是该构件的生命周期
					- $\text{clip}(x)$ 确保 $t_{local}^{(i)} \in [0,1]$
						- 如果当前时间为 $t = t_{gen}^{(i)}$ ，$t_{local}^{(i)} = 0$
						- 如果当前时间超过 $t_{gen}^{(i)} + t_{window}$ ，$t_{local}^{(i)} = 1$
					![[localtime.png]]
			- 状态流被建模为基于 $t_{local}^{(i)}$ 的线性插值，并结合贯穿整个过程的高斯噪声
				$$\mathbf{S}_t^{(i)} = \begin{cases} \mathcal{N}\left(t_{local}^{(i)}\mathbf{S}_1^{(i)} + (1 - t_{local}^{(i)})\mathbf{S}_0^{(i)}, \sigma^2\right), & \text{if } t \gt t_{gen}^{(i)}, \\ \bigl[\ \bigr], & \text{else}. \end{cases} \tag{6}$$
			- 其中：
				- $\mathbf{S}_0^{(i)}$ 表示第 $i$ 个组件完全由噪声组成的初始状态
				- $\mathbf{S}_1^{(i)}$ 表示第 $i$ 个组件的最终真实状态
				- 连续状态 $\mathbf{S}_t^{(i)}$ 仅在对应的组件 $C_t^{(i)}$ 已被生成时才存在，即当 $t \gt t_{gen}^{(i)}$ 时
			- 状态流使得当相关组件按顺序生成时，能够对连续状态进行插值
	3. 采样
		- 两个交错循环 —— 执行离散的构件选择和连续的状态变化
			- 状态流模型 $p_1^θ|t$ 的的积分
				- 控制第 $i$ 个组件连续状态的向量场定义为 $\hat{S}_1^{(i)} - S_t^{(i)}$，其中 $\hat{S}_1^{(i)}$ 是由 $p_{1\mid t}^\theta$ 预测的去噪后的理想状态，在这个向量场中前进的速率 $κ^{(i)}$ 由状态 $S^{(i)}$ 插值过程中剩余的时间决定：
					$$\kappa^{(i)} = \frac{\min(t_{end}^{(i)} - t, \Delta t)}{t_{window}} \tag{7}$$
					- 其中：
						- $t_{end}^{(i)}$ ：第 $i$ 个构件应该完成其演化（即 $t_{local} = 1$）的全局时间点
						- $\Delta t$：ODE 求解器的步长
						- $t_{window}$：归一化因子。如果一个构件的演化窗口很长，那么它的每一步演化速率 $\kappa^{(i)}$ 就会相应地变小，确保整个过程更加平滑和稳定
						- 如果 $t \gt t_{end}^{(i)}$，状态 $S_t^{(i)}$ 直接设置为预测的去噪后的理想状态 $\hat{S}_1^{(i)}$，确保连续状态的一致性
				- 状态值 $S^{(i)}$ 使用欧拉法更新
					$$S_{t+\Delta t}^{(i)} = S_t^{(i)} + (\hat{S}_1^{(i)} - S_t^{(i)}) \cdot \kappa^{(i)} \Delta t \tag{8}$$
			- 组合流策略 $π^θ$ 的采样动作
				- 
	4. 训练目标