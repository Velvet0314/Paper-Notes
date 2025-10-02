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
	    $$h_i^t, Y_i^t = \text{GRUcell}=(I_i^t, Y_{i-1}^t) \tag{5}$$
		- $I_i^t$：第 $i$ 个 GRUcell 的当前输入，即第 $i$ 个ji.点的特征
		- $Y_{i-1}^t$：第 $i$个 GRUcell 的上一个状态输入，即前一个 GRUcell（即第 $i-1$个）的输出
		- $h_i^t$：第 $i$ 个 GRUcell 计算出的新隐藏状态 (hidden state)
		$$ H^t, Y^t = \text{GRU}(T^t) = \text{GRU}(I_1^t, I_2^t, \dots,I_i^t, \dots, I_t^t) \tag{6}$$
		$$H^t = \begin{bmatrix} \; h_1^t \; \\ \; \vdots \; \\ \; h_t^t \; \end{bmatrix} \tag{7}$$
		$$ Y^t = y_t^t \tag{8}$$
		- GRU单元的数量动态增加。在步骤 $t$ 时，作用于生成的子结构树 $T_t$ 上的整个 GRU 模型简化为 $(6),(7),(8)$
	- Topological connection prediction 拓扑连接预测 
		$$ C_{t+1} = \sigma(f_c([Y^t, z_r])) \tag{9}$$
	    * $C_{t+1}$：一个概率分布，表示新节点应该连接到前面 $t$ 个节点中的哪一个
	    * $Y^t$：当前图的全局状态摘要（来自GRU）
	    * $z_r$：从 VAE 潜在空间采样的向量
	    * $f_c$：一个多层感知机 MLP
	    * $\sigma(\cdot)$: softmax 激活函数
	- Edge generation 边生成
		$$ E_{t+1,p} = \sigma(f_e([X_{V_p}, \sum_{V_j \in N^t(V_p)} X_{E_{j, p}}])) \tag{10}$$
		*  $E_{t+1,p}$：新结点 $V_{t+1}$ 与其父结点 $V_p$ 之间的边类型的概率分布
	    *  $V_p$：上一步预测出的父结点（核心）
	    *  $X_{V_p}$：父结点的特征
	    *  $X_{E_{j, p}}$：父结点 $V_p$ 已经连接的其他边的特征
	    *  $f_e$：第二个多层感知机 MLP
	- Node generation 结点生成
		$$ V_{t+1} = \sigma(f_n([X_{V_p}, X_{E_{p,t+1}}, \sum_{V_j \in N^t(V_p)} X_{V_j}, Y^t, z_r])) \tag{11}$$
		* $V_{t+1}$：新结点的类型（即是哪个子结构）的概率分布
	    * $X_{V_p}, X_{E_{p,t+1}}$：父结点和连接边的特征
	    * $\sum X_{V_j}$：父结点邻居的特征
	    * $f_n$：第三个多层感知机 MLP
	- Stop prediction 停止预测
		$$S_{t+1} = \sigma(f_s([Y^{t+1}, z_r])) \tag{12}$$
		- $S_{t+1}$：是终止的二元标签
		- $Y^{t+1}$：是第 $t+1$ 步子结构树的 GRU 模型输出
		- $f_s$：第三个多层感知机 MLP
		- $\sigma$：sigmoid 激活函数，它保证如果函数的输入是一个非常大的负数或一个非常大的正数，输出始终在 0 和 1 之间