1. 定义
	* $G = (V, E)$：输入的图，$V$ 是结点集合，$E$ 是边集合
	* $A$：图的邻接矩阵 ($N \times N$)。论文中假设 $A$ 是对称的，并且对角线元素为1（即每个结点有自环）
	* $X$：结点的特征矩阵 ($N \times D$)，每一行代表一个结点的 $D$ 维特征向量
	* $Z$：结点的潜在表示矩阵 ($N \times F$)，$F$是潜在空间的维度，通常 $F \ll D$。VGAE模型的核心就是学习 $Z$
	* $z_i$：矩阵 $Z$ 的第 $i$ 行，代表结点 $i$ 的 $F$ 维潜在表示向量
2. 推理模型
	$$q(\mathbf{Z} | \mathbf{X}, \mathbf{A}) = \prod_{i=1}^{N} q(\mathbf{z}_i | \mathbf{X}, \mathbf{A}), \quad \text{with} \quad q(\mathbf{z}_i | \mathbf{X}, \mathbf{A}) = \mathcal{N}(\mathbf{z}_i | \boldsymbol{\mu}_i, \text{diag}(\boldsymbol{\sigma}_i^2))$$
	* **变量定义：**
	    * $q(Z | X, A)$：这是**变分后验分布（Variational Posterior）**。它是一个近似分布，用来模拟“真实”但无法计算的后验分布 $p(Z | X, A)$
	    * $q(z_i | X, A)$：结点 $i$ 的潜在表示 $z_i$ 的后验分布
	    * $N(z_i | μ_i, diag(σ_i^2))$：一个高斯分布，其均值为向量 $μ_i$，协方差矩阵为对角矩阵 $diag(σ_i^2)$。这意味着 $z_i$ 的每个维度都是相互独立的
	    * $μ = GCN_μ(X, A)$：所有结点均值向量 $μ_i$ 组成的矩阵 ($N \times F$)
	    * $log σ = GCN_σ(X, A)$：所有结点对数标准差向量 $log σ_i$ 组成的矩阵 ($N \times F$)
3. 生成模型
	$$p(\mathbf{A} | \mathbf{Z}) = \prod_{i=1}^{N} \prod_{j=1}^{N} p(A_{ij} | \mathbf{z}_i, \mathbf{z}_j), \quad \text{with} \quad p(A_{ij}=1 | \mathbf{z}_i, \mathbf{z}_j) = \sigma(\mathbf{z}_i^\top \mathbf{z}_j)$$
	* **变量定义：**
	    * $p(A | Z)$：**生成模型**或**解码器**。它描述了在给定所有结点的潜在表示 $Z$ 的条件下，生成邻接矩阵 $A$ 的概率
	    * $A_{ij}$：邻接矩阵 $A$ 中第 $i$ 行第 $j$ 列的元素，$A_{ij}=1$ 表示结点 $i$ 和 $j$ 之间有边
	    * $σ(·)$：Logistic Sigmoid 函数 $σ(x) = 1 / (1 + exp(-x))$
	    * $z_i^\top z_j$：结点 $i$ 和结点 $j$ 潜在表示向量的内积 (Inner Product) —— 本质上衡量的是 **结点 $i$ 和 $j$ 在潜在空间里的相似程度**
		    * 内积的含义：如果两个向量方向相近，值就大；方向相反，值就小甚至为负 —— 所以它本质上衡量的是 **节点 $i$ 和 $j$ 在潜在空间里的相似程度**
4. 学习目标
	$$\mathcal{L} = \mathbb{E}_{q(\mathbf{Z}|\mathbf{X},\mathbf{A})}[\log p(\mathbf{A}|\mathbf{Z})] - \text{KL}[q(\mathbf{Z}|\mathbf{X},\mathbf{A}) || p(\mathbf{Z})]$$
	* **变量定义：**
	    * $L$：优化的目标函数，即 **证据下界 (Evidence Lower Bound, ELBO)**。我们的目标是最大化 $L$
	    * $E_q[...]$：**期望**。表示在由编码器 $q$ 产生的 $Z$ 的分布下，$[...]$ 内部项的平均值
	    * $log p(A|Z)$：重建项 (Reconstruction Term)。表示在给定$Z$的情况下，观测到真实邻接矩阵 $A$ 的对数似然
	    * $KL[...]$：KL散度 (Kullback-Leibler Divergence)
	    * $p(Z)$：**先验分布 (Prior Distribution)**。我们对潜在表示 $Z$ 的一个先验假设。论文中选择了一个简单的标准正态分布 $p(Z) = \prod_i N(z_i \mid 0, I)$