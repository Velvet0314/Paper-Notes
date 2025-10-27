1. 简要概述：
	- 将分子编码为两部分潜在表示 $z = [z_\mathcal{T}, z_G]$，其中 $z_\mathcal{T}$ 编码了树结构以及树中的簇是什么，但并未完全捕捉簇之间确切的相互连接方式。$z_G$ 编码了图以捕捉精细的连接性。这两个部分由树和图编码器 $q(z_\mathcal{T} \mid \mathcal{T})$ 和 $q(z_G \mid G)$ 创建。然后，潜在表示分两个阶段解码回分子图
	- 首先基于 $z_\mathcal{T}$ 中的信息，使用树解码器 $p(\mathcal{T} \mid z_\mathcal{T})$ 重构连接树
	- 其次，我们使用图解码器 $p(G \mid \mathcal{T}, z_G)$ 预测连接树中簇之间的精细连接性，以生成完整的分子图
2. 定义
	- 分子图被定义为 $G = (V, E)$，其中 $V$ 是原子（顶点）集合，$E$ 是化学键（边）集合
	- 令 $N(x$) 为 $x$ 的邻居
	-  sigmoid 函数表示为 $\sigma(\cdot)$，ReLU 函数表示为 $\tau(\cdot)$
	- 使用 $i, j, k$ 表示树中的结点，使用 $u, v, w$ 表示图中的结点
3. 连接树（Junction Tree）
	- 树分解通过将某些顶点收缩为单个结点将图 $G$ 映射到连接树中，使 $G$ 变得**无循环**
	- 形式上，给定图 $G$，连接树 $\mathcal{T}_G=(V,E,X)$ 一个连接标记树，其结点集为 $V=\{C_1,\cdots,C_n\}$，边集为 $E$。每个结点或簇 $C_i=(V_i, E_i)$ 都是 $G$ 的诱导子图，满足以下约束：
		1. 所有簇的并集等于 $G$，即 $\bigcup_i V_i = V$ 和 $\bigcup_i E_i = E$
		2. 运行交集：对于所有簇 $C_i$, $C_j$ 和 $C_k$,，如果 $C_k$ 从 $C_i$ 到 $C_j$ 的路径上，那么有 $V_i \cap V_j \subseteq V_k$
4. 图编码器（Graph Encoder）
	- 目标：将一个完整的分子图  编码成一个低维、连续的隐向量 $z_G$。这个向量需要捕捉分子的精细连接信息
	- 通过图消息传递网络对 $G$ 的潜在表示进行编码。每个顶点 $v$ 有一个特征向量 $\mathbf{x}_v$，表示原子类型、化合价和其他性质
	- 每条边 $(u,v) \in E$ 有一个特征向量 $\mathbf{x}_{uv}$，表示其键类型，以及两个隐藏向量 $\mathbf{ν}_{uv}$ 和 $\mathbf{ν}_{vu}$，分别表示从 $u$ 到 $v$ 以及反之的消息。由于图的循环结构，消息以循环置信传播的方式进行交换：$$\boldsymbol{\nu}_{uv}^{(t)} = \tau(\mathbf{W}_1^g\mathbf{x}_u + \mathbf{W}_2^g\mathbf{x}_{uv} + \mathbf{W}_3^g \sum_{w \in N(u) \backslash v} \boldsymbol{\nu}_{wu}^{(t-1)}) \tag{1}$$
	- 变量定义：
	    * $\boldsymbol{\nu}_{uv}^{(t)}$：在第 $t$ 轮迭代中，从原子（图结点）$u$ 传递给邻居原子 $v$ 的消息向量
	    * $\mathbf{x}_u$：原子 $u$ 的初始特征向量（如原子类型、化合价等）
	    * $\mathbf{x}_{uv}$：连接原子 $u$ 和 $v$ 的化学键的特征向量（如键的类型是单键、双键等）
	    * $\mathcal{N}(u) \setminus v$：原子 $u$ 的所有邻居中，除了 $v$ 之外的集合
	    * $\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3$：可学习的权重矩阵
	    * $\tau(\cdot)$：非线性激活函数，如 ReLU
	$$\begin{align}\mathbf{h}_u &= \tau(\mathbf{U}_1^g \mathbf{x}_u + \mathbf{U}_2^g \sum_{v \in \mathcal{N}(u)} \boldsymbol{\nu}_{vu}^{(T)}) \tag{2} \\[8pt] \mathbf{h}_G &= \frac{\sum_i \mathbf{h}_i}{|V|}\end{align}$$
	- 变量定义：
	    * $\mathbf{h}_u$：经过 $T$ 轮消息传递后，原子 $u$ 的最终表示向量
	    * $\mathbf{h}_G$：整个分子图 $G$ 的最终表示向量
	    * $\mathbf{U}_1^g, \mathbf{U}_2^g$：可学习的权重矩阵
	    * $|V|$：分子中的原子总数
	* 得到的 $\mathbf{h}_G$ 向量被送入两个独立的线性层，分别预测出变分后验分布 $q(z_G|G) = \mathcal{N}(z_G; \mu_G, \sigma_G^2)$ 的均值 $\mu_G$ 和对数方差 $\log \sigma_G^2$
    * 在训练时，使用**重参数技巧**从该分布中采样得到 $z_G = \mu_G + \sigma_G \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。这使得采样过程可微，从而能够通过梯度下降进行端到端训练
6. 树编码器（Tree Encoder）
	* 目标：将分子的连接树 $T_G$ 编码成一个低维、连续的隐向量 $z_T$。这个向量需要捕捉分子的粗粒度拓扑结构和构件信息
	$$\mathbf{m}_{ij} = \text{GRU}(\mathbf{x}_i, \{\mathbf{m}_{ki}\}_{k \in \mathcal{N}(i) \setminus j}) \tag{3}$$
	- 变量定义：
		* $i, j, k$：连接树中的结点（即化学簇）
	    * $\mathbf{m}_{ij}$：从树结点 $i$ 传递给邻居结点 $j$ 的消息向量
	    * $\mathbf{x}_i$：树结点 $i$ 的特征向量（代表该簇的类型，如“苯环”或“C-C单键”，通常是一个独热编码或可学习的嵌入）
	    * $\mathbf{m}_{ki}$：一组来自邻居的输入消息 $\mathbf{m}_{ki}$
	    * $\mathcal{N}(i) \setminus j$：结点 $i$ 在树上的邻居，除了 $j$
	- GRU 的定义
		$$\begin{align*}s_{ij} &= \sum_{k \in \mathcal{N}(i)\setminus j} \mathbf{m}_{ki} \tag{4} & \text{// 输入消息聚合} \\[8pt] z_{ij} &= \sigma(\mathbf{W}^z x_i + \mathbf{U}^z \mathbf{s_{ij}} + \mathbf{b}^z) \tag{5} & \text{// 更新门} \\[8pt] r_{ki} &= \sigma(\mathbf{W}^r x_i + \mathbf{U}^r \mathbf{m}_{ki} + \mathbf{b}^r) \tag{6} & \text{// 重置门 (每个输入一个)} \\[8pt] \tilde{m}_{ij} &= \tanh(\mathbf{W} \mathbf{x}_i + \mathbf{U} \sum_{k \in \mathcal{N}(i)\setminus j} (\mathbf{r}_{ki} \odot \mathbf{m}_{ki})) \tag{7} & \text{// 候选消息} \\[8pt] m_{ij} &= (1 - \mathbf{z}_{ij}) \odot \mathbf{s}_{ij} + \mathbf{z}_{ij} \odot \tilde{\mathbf{m}}_{ij} \tag{8} & \text{// 最终消息}\end{align*}$$
	- 对所有来自邻居的消息进行聚合：（$\mathcal{N}(i)$ 表示结点 $i$ 在树上所有邻居结点的集合）
		$$\begin{align}\mathbf{h}_i &= \tau(\mathbf{W}^o \mathbf{x}_i + \sum_{k \in \mathcal{N}(i)} \mathbf{U}^o \mathbf{m}_{ki}) \tag{9} \end{align}$$
	- 最终的树： $\mathbf{h}_{\mathcal{T}_G} = \mathbf{h}_{\text{root}}$
	- 在**图编码器**中，模型通过对所有原子（结点）的表示求平均来得到图的表示 ($h_G = \frac{\sum h_u}{|V|}$)。这种**平均池化 (average pooling)** 操作会丢失所有结点的顺序和位置信息，得到一个对结点排列不变的表示。对于分子图来说，这是一个合理的做法
	- 但在**树编码器**中，作者明确指出**不能**使用平均池化。原因是：这样会让树解码器不知道应该先生成哪个结点
	- 得到的 $\mathbf{h}_\mathcal{T}$ 被用来预测变分后验分布 $q(z_\mathcal{T}|\mathcal{T}_G) = \mathcal{N}(z_\mathcal{T}; \mu_\mathcal{T}, \sigma_\mathcal{T}^2)$ 的参数，并从中采样得到 $z_\mathcal{T}$
7. 树解码器（Tree Decoder）
	- 目标：接收一个从隐空间采样得到的向量 $z_\mathcal{T}$，并将其解码 (decode) 成一棵完整的、结构和标签都正确的连接树 $\mathcal{T}$
	- 方法：自回归 (autoregressive) 策略：在树上进行**深度优先遍历（DFS）** 的过程来逐步构建这棵树
		- **拓扑问题 (Topological Question):** 在当前这个结点，还需要向下探索、生成新的子结点吗？还是说这个方向已经到头了，应该回溯 (backtrack)？
		- **标签问题 (Label Question):** 如果决定要生成一个新的子结点，那么这个子结点应该是什么类型的化学簇（例如，苯环、乙炔键等）？
	- 解码过程中的消息传递
		$$\mathbf{h}_{i_t, j_t} = \text{GRU}(\mathbf{x}_{i_t}, \{\mathbf{h}_{k, i_t}\}_{(k, i_t) \in \tilde{\mathcal{E}}_t, k \neq j_t}) \tag{10}$$
		* **变量定义**：
		    * $t$：表示生成过程的当前时间步
		    * $i_t$：在时间步 $t$ 访问的结点
		    * $j_t$：将要移动到的下一个结点（可能是新的子结点，也可能是父结点用于回溯）
		    * $\mathbf{h}_{i_t, j_t}$：从结点 $i_t$ 发往结点 $j_t$ 的消息向量。这个消息是**动态计算**的，代表了到目前为止的生成历史和上下文
		    * $\mathbf{x}_{i_t}$：结点 $i_t$ 的标签（化学簇）对应的特征向量
		    * $\tilde{\mathcal{E}}_t$：在时间步 $t$ 之前，已经被遍历过的边的集合
		    * GRU：与树编码器中使用的门控循环单元是**同一个**
		- 含义：聚合了结点 $i_t$ 自身的信息 ($\mathbf{x}_{i_t}$) 和从它刚刚走过的路径上传来的消息（除了它将要去的方向 $j_t$）
	- 拓扑预测 (Topological Prediction)
		$$p_t = \sigma(\mathbf{u}^d \cdot \tau(\mathbf{W}_1^d \mathbf{x}_{i_t} + \mathbf{W}_2^d z_\mathcal{T} + \mathbf{W}_3^d \sum_{(k, i_t) \in \tilde{\mathcal{E}}_t} \mathbf{h}_{k, i_t})) \tag{11}$$
		- 变量定义：
			- $p_t$：一个概率值 (0 到 1之间)，代表在当前结点 $i_t$ **继续生成新子结点的概率**。如果 $p_t$ 很高，模型就倾向于“扩展”；如果很低，就倾向于“回溯”
			- $\mathbf{h}_{k, i_t}$：从结点 $k$ 移动到结点 $i_t$ 时，留下的“上下文”或“路径记忆”
			- $\sum_{(k, i_t) \in \tilde{\mathcal{E}}_t} \mathbf{h}_{k, i_t}$：所有**已经遍历过的、指向当前结点 $i_t$ 的路径**所携带的上下文信息的**聚合**
			- $\tau$：sigmoid 激活函数
	- 标签预测 (Label Prediction) ：当子结点 $j$ 从其父结点 $i$ 生成时，我们预测其结点标签
		$$\mathbf{q}_j = \text{softmax}(\mathbf{U}^l \tau(\mathbf{W}_1^l z_\mathcal{T} + \mathbf{W}_2^l \mathbf{h}_{ij})) \tag{12}$$
		- $\mathbf{q}_j$ 是在标签词汇表 $\mathcal{X}$ 上的一个分布
		- 当 $j$ 是根结点时，其父结点 $i$ 是一个虚拟结点，并且 $\mathbf{h}_{ij} = 0$	
	- 学习目标 (Learning Loss)：树解码器旨在最大化似然 $p(\mathcal{T} |z_{\mathcal{T}})$。令 $p_t \in \{0, 1\}$ 和 $\mathbf{q}_j$ 为真实拓扑和标签值，解码器最小化以下交叉熵损失
		$$\mathcal{L}_c(\mathcal{T}) = \sum_t \mathcal{L}^d(p_t, \hat{p}_t) + \sum_j \mathcal{L}^l(\mathbf{q}_j, \hat{\mathbf{q}}_j) \tag{13}$$
		- 与序列生成类似，在训练过程中进行**教师强制（teacher forcing）**：在每一步进行拓扑和标签预测后，我们将它们替换为真实值，以便模型根据正确的历史进行预测 —— 即使模型在第 $t$ 步做出了错误的预测（比如，它预测要回溯，但实际上应该扩展），在第 $t+1$ 步的输入中，我们仍然会**强制使用真实的数据**。这能极大地稳定训练过程，**防止误差累积**
	- **Algorithm 1** 将上述所有公式串联成一个实际的生成算法。其中最关键的一步是**可行性检查 (Feasibility Check)**。在预测新结点标签时，模型首先会生成一个概率分布 $q_j$。然而，并非所有的化学簇都可以连接到当前结点上。因此，算法会：
		1.  确定一个化学上**允许的**候选簇集合 $\mathcal{X}_i$
		2.  将 $q_j$ 中不属于 $\mathcal{X}_i$ 的簇的概率**屏蔽掉 (mask out)**
		3.  在剩余的有效候选中，根据它们的概率进行采样
	- 这个步骤是**注入化学领域知识**的关键，也是 JT-VAE 能够生成 100% 化学有效分子的重要保障之一。它确保了模型在每一步构建的都是一个“可实现”的局部结构
8. 图解码器（Graph Decoder）
	- 目标：重构预测的连接树 $\hat{\mathcal{T}} = (\hat{\mathcal{V}}, \hat{\mathcal{E}})$ 所属的分子图 $G$。注意，这一步并非确定性的，因为可能有多个分子对应于同一个连接树 —— 将子图（树中的结点）组合成正确的分子图
	- 定义：
		$$\hat{G} = \arg\max_{G^\prime \in \mathcal{G}(\hat{\mathcal{T}})} f^a(G^\prime)$$
		* **变量定义**：
		    *   $\hat{G}$：最终预测出的分子图
		    *   $G'$：一个候选分子图
		    *   $\mathcal{G}(\hat{\mathcal{T}})$：所有与给定的连接树 $\hat{T}$ 的拓扑结构相兼容的分子图的集合
		    *   $f^a(G')$：学习到的评分函数，输出一个标量分数
	- 评分部分
		- 定义 $G_i$
		    * $G_i$ 是一个**候选子图 (candidate subgraph)**
		    * 它由树结点 $C_i$ 和它的所有邻居结点 $\{C_j\}_{j \in \mathcal{N}_{\hat{T}}(i)}$ 组装而成
		- 评分的核心公式：$f^a(G_i) = \mathbf{h}_{G_i} \cdot \mathbf{z}_G$
		    * **变量定义**：
			    * $f^a(G_i)$：候选子图 $G_i$ 的最终标量分数
			    * $\mathbf{h}_{G_i}$：候选子图 $G_i$ 向量表示 (vector representation)。这个向量需要捕捉到 $G_i$ 的所有结构信息
			    * $\mathbf{z}_G$：从编码器传来的、代表目标分子整体精细结构的隐向量
		- 计算 $\mathbf{h}_{G_i}$：注入树上下文的消息传递网络
			$$\begin{align}\boldsymbol{\mu}_{uv}^{(t)} &= \tau(\mathbf{W}_1^a \mathbf{x}_u + \mathbf{W}_2^a \mathbf{x}_{uv} + \mathbf{W}_3^a \boldsymbol{\tilde{\mu}}_{uv}^{(t-1)}) \tag{15} \\[10pt] \boldsymbol{\tilde{\mu}}_{uv}^{(t-1)} &= \begin{cases} \sum_{w \in N(u) \backslash v} \boldsymbol{\mu}_{wu}^{(t-1)} & \alpha_u = \alpha_v \\ \tilde{\mathbf{m}}_{\alpha_u, \alpha_v} + \sum_{w \in N(u) \backslash v} \boldsymbol{\mu}_{wu}^{(t-1)} & \alpha_u \neq \alpha_v \end{cases}\end{align}$$
			- **变量定义**：
				- 情况 1：$u$ 和 $v$ 在同一个化学簇中 ($\alpha_u = \alpha_v$)
				    *   这意味着化学键 $(u, v)$ 是簇内部的连接
				    *   此时，历史信息 $\tilde{\mu}_{uv}^{(t-1)} = \sum_{w \in \mathcal{N}(u)\setminus v} \mu_{wu}^{(t-1)}$
				    *   这与标准图编码器公式(1)完全一样。它聚合了来自 $u$ 的其他邻居 $w$ 的消息
				* 情况 2：$u$ 和 $v$ 在不同的化学簇中 ($\alpha_u \neq \alpha_v$)
				    * 这意味着化学键 $(u, v)$ 是连接两个不同簇 $C_{\alpha_u}$ 和 $C_{\alpha_v}$ 的桥梁
				    * 此时，历史信息 $\tilde{\mu}_{uv}^{(t-1)} = \hat{m}_{\alpha_u, \alpha_v} + \sum_{w \in \mathcal{N}(u)\setminus v} \mu_{wu}^{(t-1)}$
				    * 除了聚合来自 $u$ 的其他邻居的消息外，这里还 **额外注入了来自树的消息** $\hat{m}_{\alpha_u, \alpha_v}$
				- $\hat{m}_{\alpha_u, \alpha_v}$ 是在**连接树 $\hat{\mathcal{T}}$ 上**，从结点 $\alpha_u$ 发往结点 $\alpha_v$ 的消息。这个消息是通过在解码器生成的树 $\hat{\mathcal{T}}$ 上运行树的消息传递算法（类似Tree Encoder的过程）得到的 —— **其总结了以结点 $\alpha_u$ 为根、沿着 $(\alpha_u, \alpha_v)$ 方向看过去的整个子树的信息
	- 学习目标
		$$\mathcal{L}_g(G) = \sum_i \left( f^a(G_i) - \log \sum_{G'_i \in \mathcal{G}_i} \exp(f^a(G'_i)) \right)$$
		- $G_i$: 在树结点 $i$ 的局部邻域内，真实（正确）的子图组装方式
	    * $\mathcal{G}_i$: 在结点 $i$ 邻域内，所有可能的候选组装方式的集合