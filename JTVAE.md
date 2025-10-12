1. 简要概述：
	- 将分子编码为两部分潜在表示 $z = [z_\mathcal{T}, z_G]$，其中 $z_\mathcal{T}$ 编码了树结构以及树中的簇是什么，但并未完全捕捉簇之间确切的相互连接方式。$z_G$ 编码了图以捕捉精细的连接性。这两个部分由树和图编码器 $q(z_\mathcal{T} \mid \mathcal{T})$ 和 $q(z_G \mid G)$ 创建。然后，潜在表示分两个阶段解码回分子图
	- 首先基于 $z_\mathcal{T}$ 中的信息，使用树解码器 $p(\mathcal{T} \mid z_\mathcal{T})$ 重构连接树
	- 其次，我们使用图解码器 $p(G \mid \mathcal{T}, z_G)$ 预测连接树中簇之间的精细连接性，以生成完整的分子图
2. 定义
	- 分子图被定义为 $G = (V, E)$，其中 $V$ 是原子（顶点）集合，$E$ 是化学键（边）集合
	- 令 $N(x$) 为 $x$ 的邻居
	-  sigmoid 函数表示为 $\sigma(\cdot)$，ReLU 函数表示为 $\tau(\cdot)$
	- 使用 $i, j, k$ 表示树中的节点，使用 $u, v, w$ 表示图中的节点
3. 连接树（Junction Tree）
	- 树分解通过将某些顶点收缩为单个节点将图 $G$ 映射到连接树中，使 $G$ 变得无循环
	- 形式上，给定图 $G$，连接树 $\mathcal{T}_G=(V,E,X)$ 一个连接标记树，其节点集为 $V=\{C_1,\cdots,C_n\}$，边集为 $E$。每个节点或簇 $C_i=(V_i, E_i)$ 都是 $G$ 的诱导子图，满足以下约束：
		1. 所有簇的并集等于 $G$，即 $\bigcup_i V_i = V$ 和 $\bigcup_i E_i = E$
		2. 运行交集：对于所有簇 $C_i$, $C_j$ 和 $C_k$,，如果 $C_k$ 从 $C_i$ 到 $C_j$ 的路径上，那么有 $V_i \cap V_j \subseteq V_k$
4. 图编码器（Graph Encoder）
	- 通过图消息传递网络对 $G$ 的潜在表示进行编码。每个顶点 $v$ 有一个特征向量 $\mathbf{x}_v$，表示原子类型、化合价和其他性质
	- 每条边 $(u,v) \in E$ 有一个特征向量 $\mathbf{x}_{uv}$，表示其键类型，以及两个隐藏向量 $\mathbf{ν}_{uv}$ 和 $\mathbf{ν}_{vu}$，分别表示从 $u$ 到 $v$ 以及反之的消息。由于图的循环结构，消息以循环置信传播的方式进行交换：$$\boldsymbol{\nu}_{uv}^{(t)} = \tau(\mathbf{W}_1^g\mathbf{x}_u + \mathbf{W}_2^g\mathbf{x}_{uv} + \mathbf{W}_3^g \sum_{w \in N(u) \backslash v} \boldsymbol{\nu}_{wu}^{(t-1)}) \tag{1}$$
	- 变量定义：
		- 1
		- 2
		- 3
		- 4