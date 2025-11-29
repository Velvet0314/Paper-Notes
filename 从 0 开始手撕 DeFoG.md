1. 超参数与各类配置
	- 通过 Hydra 和 OmegaConf 进行配置
2. 数据集加载 —— 以 QM9_no_H 为例
	- 暂时空着
3. 额外的图特征提取初始化 —— `extra_features`
	- 目的：为图神经网络补充结构特征 —— 代码中给出了七种方式，默认采用的是 `rrwp`，针对不同的分子集的特征应该选用不同的方式
		1. `cycles`：仅使用循环特征 (k-cycles)
		2. `eigenvalues`：使用图拉普拉斯矩阵的特征值
		3. `rrwp`：随机游走核心特征（Regularized Random Walk Kernels）
		4. `rrwp_double`：RRWP 特征 + 非归一化版本
		5. `rrwp_only`：仅 RRWP（不含循环特征）
		6. `rrwp_comp`：RRWP 补集版本（考虑图的补图）
		7. `all`：所有特征的组合（循环 + 特征值 + 特征向量）
	- 额外的特征
		1. 图的最大节点数 `self.max_n_nodes = dataset.info.max_nodes = 9`
		2. 循环特征计算 `self.ncycles = NodeCycleFeatures()`
			- 概念：
				- k-cycles：**k-圈（k-cycle）** 是指长度为 k 的环，即由 k 个节点和 k 条边组成的闭合路径
		3. 提取特征的方式 `features_type = extra_features_type = rrwp` 
		4. `rrwp` 的步数 `self.rrwp_steps = rrwp_steps = 12`
		5. 归一化的随机游走核 `self.RRWP = RRWPFeatures()`
		6. 非归一化的随机游走核 `self.RWP = RRWPFeatures(normalize=False)`
4. 分子的特征提取初始化 —— `domain_features`
	- 目的：从数据集中提取分子的化学特征
	- 分子的特征
		1. 电荷 `self.charge = ChargeFeature()`
		2. 化合价 `self.valency = ValencyFeature()`
		3. 分子量 `self.weight = WeightFeature()`
5. 计算模型的输入/输出的维度
	- PyG 稀疏图结构
		1. 节点特征矩阵 `x` —— `x.shape = [num_nodes, num_node_features]`
			- 作用：保存分子图的原子类型
			- 例子：一个分子有 5 个原子，每个原子用 $\text{one-hot}$ 编码（C, N, O, H）
				```python
				x = [[1, 0, 0, 0],   # 节点 0: 碳
					 [1, 0, 0, 0],   # 节点 1: 碳  
					 [0, 1, 0, 0],   # 节点 2: 氮
					 [0, 0, 1, 0],   # 节点 3: 氧
					 [0, 0, 0, 1]]   # 节点 4: 氢
				# x.shape:[5, 4]
				```
		2. 边索引矩阵 `edge_index` —— `edge_index.shape = [2, nums_edges]`
			- 作用：保存图中节点和节点的连接关系
			- 例子：第一行是源节点，第二行是目标节点
			```python
				边列表（无向图需要双向存储）：
				0 → 1, 1 → 0, 1 → 2, 2 → 1, 1 → 3, 3 → 1
				
				edge_index = [[0, 1, 1, 2, 1, 3],   # 源节点
				              [1, 0, 2, 1, 3, 1]]   # 目标节点
				# edge_index.shape = [2, 6]
			```
		3. 边特征矩阵 `edge_attr` —— `edge_attr.shape = [num_edges, num_edge_features]`
			- 作用：保存每条边的特征，与 `edge_index` 一一对应
			- 例子：化学键类型（$\text{one-hot}$：单键、双键、三键等）
			```python
			# 特征长度+1是因为预留了 [1,0,0,0] 表示无边，在稀疏矩阵里不会存储无边
			edge_attr = [[0,1,0,0],  # 边 0→1: 单键
			             [0,1,0,0],  # 边 1→0: 单键
			             [0,0,1,0],  # 边 1→2: 双键
			             [0,0,1,0],  # 边 2→1: 双键
			             [0,1,0,0],  # 边 1→3: 单键
			             [0,1,0,0]]  # 边 3→1: 单键
			# edge_attr.shape:[6, 4]
			```
		4.  批次索引 `batch` —— `batch.shape = [total_num_nodes]`
			- 作用：当多个图被打包成一个 batch 时，`batch` 标记每个节点属于哪个图
			- 例子：3 个图打包在一起
			```python
			图 0: 2 个节点 (节点 0, 1)
			图 1: 3 个节点 (节点 2, 3, 4)
			图 2: 2 个节点 (节点 5, 6)
			batch = [0, 0, 1, 1, 1, 2, 2]
			         ↑  ↑  ↑  ↑  ↑  ↑  ↑
					图0    图1      图2
			```
	- 稀疏图转稠密图
		1. 节点特征矩阵转稠密矩阵 —— 调用 PyG 的函数 `to_dense_batch()`
			- 返回稠密图和一个 `node_mask`，`node_mask` 表示每个图哪些节点是真实的（因为要 padding 到 `max_num_nodes`，需要一个 mask 无效掉 padding 的部分）
			- 例子：
			```python
			In：
			# 3个图：图0有2节点，图1有3节点，图2有2节点
			x = [[1,0],    # 节点0 (图0)
				 [0,1],    # 节点1 (图0)
				 [1,0],    # 节点2 (图1)
				 [1,1],    # 节点3 (图1)
				 [0,1],    # 节点4 (图1)
				 [1,0],    # 节点5 (图2)
				 [0,0]]    # 节点6 (图2)
			# x.shape = [7, 2]
			batch = [0, 0, 1, 1, 1, 2, 2]
			
			Out:
			# max_num_nodes = 3（图1最大）
			X = [[[1,0], [0,1], [0,0]],    # 图0: 2节点 + 1个padding
				 [[1,0], [1,1], [0,1]],    # 图1: 3节点
				 [[1,0], [0,0], [0,0]]]    # 图2: 2节点 + 1个padding
			# X.shape: [3, 3, 2]  → [batch_size, max_nodes, features]
			
			node_mask = [[True,  True,  False],   # 图0: 前2个是真实节点
						 [True,  True,  True ],   # 图1: 全是真实节点
						 [True,  True,  False]]   # 图2: 前2个是真实节点
			# node_mask.shape:[3, 3]
			```
		2. 去掉边索引和边特征（`edge_index, edge_attr`）矩阵中的自环 —— 调用 PyG `torch_geometric.utils.remove_self_loops()`
		3. 边列表转邻接矩阵 —— 调用 PyG `to_dense_adj()`
			- 例子：
			```python
			In：
			# 图0: 0-1 单键
			# 图1: 0-1 双键, 1-2 单键
			edge_index = [[0, 1, 2, 3, 3, 4],   # 源节点(全局索引)
						  [1, 0, 3, 2, 4, 3]]   # 目标节点
			
			edge_attr = [[0, 1, 0, 0],  # 0→1 单键
						 [0, 1, 0, 0],  # 1→0 单键
						 [0, 0, 1, 0],  # 2→3 双键(只看图1的索引是节点0→1)
						 [0, 0, 1, 0],  # 3→2 双键
						 [0, 1, 0, 0],  # 3→4 单键(只看图1的索引是节点1→2)
						 [0, 1, 0, 0]]  # 4→3 单键
			
			batch = [0, 0, 1, 1, 1]
			
			Out：
			E.shape = [batch_size, max_nodes, max_nodes, edge_features]
					= [2, 3, 3, 4]
			
			# 图0 的邻接矩阵 (3×3×4)
			E[0] = [
				#    节点0          节点1     节点2(padding)
				[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],  # 从节点0出发
				[[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # 从节点1出发
				[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]   # padding 行
			]
			
			# 图1 的邻接矩阵 (3×3×4)
			E[1] = [
				#    节点0          节点1         节点2
				[[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],  # 只看图1的索引是节点0→1: 双键
				[[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]],  # 只看图1的索引是节点1→0: 双键, 1→2: 单键
				[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]   # 只看图1的索引是节点2→1: 单键
			]
			```
		4. 编码无边类型 —— 调用 `encode_no_edge()`
			- 作用：因为稀疏图只存储存在的边，所以不需要表示"无边"，转换成稠密图后，由于无边的位置被padding 成了 0，导致没有有效的 $\text{one-hot}$ 与之对应
			1. 判断是否有边特征
				```python
				if E.shape[-1] == 0: # 因为 E 的最后一个维度就是 edge_features
					return E
				```
			2. 对最后一个维度求和 —— `no_edge = torch.sum(E, dim=3) == 0`
				- 作用：找到哪些是“无边”
				```python
				E[0] = [
					#    节点0          节点1     节点2(padding)
					[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],  # 从节点0出发
					[[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # 从节点1出发
					[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]   # padding 行
				]
				
				E[0]_sum = torch.sum(E[0], dim=3) # 求和后消去了最后一个维度
				
				# no_edge.shape:[2, 3, 3]
				
				# 相当于对每个代表边特征的 one-hot 向量内部元素求和
				# 如果是全 0 的无边类型，那么一定是 0
				E[0]_sum = [
					[0, 1, 0],  
					[1, 0, 0],
					[0, 0, 0]
				] → 这个就是 no_edge 里的第一个维度的第一个元素：图 0
				no_edge[0] = [
					[False, True, False],  
					[True, False, False],
					[False, False, False]
				]
				```
			3. 填充无边 padding 为预留的 $\text{one-hot}$ `[1,0,...,0]`
				- 作用：将无边也能用 $\text{one-hot}$ 表示
				```python
				first_elt = E[:, :, :, 0] # 取最后一个维度的第一个元素
				# first_elt.shape:[2, 3, 3] = no_edge.shape:[2 ,3, 3]
				
				# Pytorch 布尔索引语法：tensor[布尔掩码] = 值，需要 shape 匹配才能用
				first_elt[no_edge] = 1    # 把所有的无边的第一个元素置为 1
				E[:, :, :, 0] = first_elt # 显式写回 E
				```
			4. 修正对角线的问题
				- 作用：因为转成稠密矩阵的时候没有自环，所以对角线默认也是无边的 padding 是相同的，但是第 3 步操作将对角线也标记成无边的 $\text{one-hot}$ `[1,0,...,0]`了，所以需要修正回来
				```python
				diag = (
					torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
				)
				# torch.eye(E.shape[1], dtype=torch.bool) shape:[3, 3]
				# unsqueeze(0): 在最前面(位置 0)添加一个 batch 维度(1) shape:[1, 3, 3]
				# expand(E.shape[0], -1, -1): 扩展到 batch_size 份(不复制内存，只是广播) shape:[2, 3, 3]
				
				E[diag] = 0
				# 通过广播机制，让第四个维度不变，只改变了对角线的 one-hot，是直接将整个对角线向量改成 0 向量，而不是像之前是选出第四个维度的第一个元素去修改
				```
	- 计算输入特征
		1. 从稠密图提取 `X, E`
			- 节点维度
			- 边维度
			- `node_mask` 的维度
		2. 数据集本身传递的一些图的全局特征 `y`
			- 再 `+1`，用于传递时间步的信息（原注释：# + 1 due to time conditioning）
		3. 额外的图特征 —— `extra_features: __call__ function`
			1. 计算图的大小比例
				```python
				# dim = 1 按行求和(对第二个维度里面的内容求和，消去了第二个维度)
				# node_mask = [[True, True, False], 图0
				#			[True, True, True],     图1
				#			[True, True, False]]    图2
				# node_mask.sum(dim=1) = [2, 3, 2]
				# node_mask.sum(dim=1).unsqueeze(1) = [[2], [3], [2]] 在位置 1 添加一个 batch 维度(1)
				# 思考：node_mask.sum(dim=1).unsqueeze(1) 其实应为 [[2, 3, 2]]
				n = noisy_data["node_mask"].sum(dim=1).unsqueeze(1) / self.max_n_nodes # 除以最大节点数，归一化到 [0, 1]
				```
			2. 计算 k-循环特征 —— 加入到图的全局特征 `y` 中
			3. 计算选定的特征类型 —— 标准 RRWP
				- RRWP 详解
					1. RRWP 的概念：Random Walk with Restart Positional Encoding（随机游走位置编码）
						- 每一步，从当前节点随机选择一条边走到邻居节点
						- RRWP 计算的是：从节点 $i$ 出发，走 $k$ 步后到达节点 $j$ 的概率
					2. 数学原理
						- 概率转移矩阵
							$$P = D^{-1} A$$
							其中：
							- $A$：邻接矩阵（$A[i, j] = 1$ 表示 $i$ 和 $j$ 之间有边）
							- $D$：度矩阵（对角矩阵，$D[i, i]=$ 节点 $i$ 的度数）
							- $P$：概率转移矩阵（$P[i ,j] =$ 从 $i$ 一步走到 $j$ 的概率）
						- $k$ 步随机游走
							- $P^k[i,j] = \text{从节点 i 出发，恰好走 k 步到达节点 j 的概率}$
					3. 代码解析
						1. 排除无边类别 —— rrwp 只关心有没有边，所以可以不用最后一维
							```python
							# 排除最后一维的第 0 个元素，之前编码的无边类型就失效了
							# 再对最后一个维度求和，只有单一的 0/1 代表有无边
							E = noisy_data["E_t"].float()[..., 1:].sum(-1)
							# E.shape:[batch_size, n ,n]
							```
						2. 计算随机游走核特征：由 `extra_features.py —— RRWPFeatures()` 实现
							```python
							rrwp_edge_attr = self.RRWP(E, k=self.rrwp_steps)
							self.RRWP = RRWPFeatures()
							
							class RRWPFeatures:
								def __call__(self, E, k=None):
									k = k or self.k
									bs, n, _ = E.shape
									# 如果需要归一化，计算转移矩阵 D^{-1}A
									if self.normalize:
										# 初始化度矩阵
										degree = torch.zeros(bs, n, n, device=E.device)
										
										# 计算出度（每个节点的度数）
										# 对于无向图：出度 = 入度 = 度数
										# 由于 D 是对角矩阵，所以 D 的逆矩阵 D^{-1} 就是对角线元素取倒 
										to_fill = 1 / (E.sum(dim=-1).float())
										
										# 孤立节点度数为 0，1/0 = inf
										# 找到原本为 0 的地方，全部将 inf 替换为 0
										to_fill[E.sum(dim=-1).float() == 0] = 0
										# to_fill.shape:[非孤立节点数, 1]
										
										# 构造对角度矩阵
										# degree 是三维向量，写回到第2、3维度里
										degree = torch.diagonal_scatter(degree, to_fill, dim1=1, dim2=2)
										
										# 转移矩阵 = D^{-1} @ A
										E = degree @ E
							```
						3. 开始游走，迭代计算
							```python
							# 初始化：第 0 步即是单位矩阵 I
							# [n, n] 单位矩阵，unsqueeze(0) 增加 batch 维度到 [1, n, n]，repeat(bs, 1, 1) 复制 bs 份 [bs, n, n] 
							id = torch.eye(n, device=E.device).unsqueeze(0).repeat(bs, 1, 1)
							# 列表用来存储所有步数的矩阵，目前只有 P^0
							rrwp_list = [id]
							
							# 迭代计算 P^1, P^2, ..., P^{k-1}
							for i in range(k - 1):
								cur_rrwp = rrwp_list[-1] @ E  #  P^{i+1} = P^i @ E
								rrwp_list.append(cur_rrwp)
							
							# 堆栈成 [bs, n, n, k]
							return torch.stack(rrwp_list, -1)
							```
						4. 提取节点特征
							```python
							# 从对角线上提取节点特征（A^k[i,i] 对角元素）
							# 生成对角线索引
							diag_index = torch.arange(rrwp_edge_attr.shape[1])
							
							# 对于 diag_index = [0, 1, 2]
							# rrwp_edge_attr[:, [0,1,2], [0,1,2], :]
							# 展开就是：
							# rrwp_edge_attr[:, 0, 0, :]  → 节点0的对角特征
							# rrwp_edge_attr[:, 1, 1, :]  → 节点1的对角特征
							# rrwp_edge_attr[:, 2, 2, :]  → 节点2的对角特征
							# 由于取 n 个对角线元素，维度减少了一个
							rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]  # (bs, n, k)
							```
					4. 总结
						- `rrwp_edge_attr` 告诉模型：节点 $i$ 和节点 $j$ 之间的"图上距离"是多少，它们通过多少步可以互相到达，以及它们之间有多少条路径连接
						- `rrwp_node_attr` 告诉模型：这个节点在图中处于什么样的"位置"—— 是在**密集的中心（高返回概率，因为邻居多容易走回来）**，还是在**稀疏的边缘（低返回概率，不易走回来）**，以及它周围的局部连接模式是什么样的
		4. 分子的特征
			1. 1
			2. 2
			3. 3