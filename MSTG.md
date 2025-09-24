1. Molecular substructure tree
2. Encoder
	- 子结构树（Substructure Tree）通过多通道子结构图门控循环单元（MSGG，Multichannel Substructure-Graph GRU）映射到潜在空间
2. Decoder
	- Root generation 根结点生成
		$$V_r = \sigma(f_r(z_r)) \tag{3}$$
		- $z_r$：Encoder 的 MSGG 将原始子结构树映射到潜在空间 $z$ 中。从潜在空间中采样初始向量 $z_r$
		- $f_r$：一个多层感知机 MLP
		- $\sigma(\cdot)$：一个 softmax 激活层，将预测的类别概率归一化到 $[0,1]$ 中且所有概率的总和为 1
		- $V_r$：预测的根结点
	- Dynamic features of generated substructure tree 生成的子结构树的动态特征
		$$ I_i^t = [X_{V_i}, \sum_{V_j \in N^t(V_i)} X_{V_j}] \tag{4}$$
		- $I_i^t$： 在时间步 $t$，为图中第 $i$ 个节点 $V_i$ 准备的**输入特征**
	    * $X_{V_i}$：节点 $V_i$ 自身的静态特征向量（代表了这是哪种子结构）
	    * $N^t(V_i)$：在时间步 $t$，节点 $V_i$ 的邻居节点集合
	    * $\sum_{V_j \in N^t(V_i)} X_{V_j}$：将 $V_i$ 所有邻居节点的特征向量求和（一种简单的聚合操作）
	    $$h_i^t, Y_i^t = \text{GRUcell}=(I_i^t, Y_{i-1}^t)$$
		- 1
		- 2
		- 3
	- Topological connection prediction
	- Edge generation
	- Node generation
	- Stop prediction