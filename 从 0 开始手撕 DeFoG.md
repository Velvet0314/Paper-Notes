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
			1. 填充无边 padding 为预留的 $\text{one-hot}$ `[1,0,...,0]`
				- 作用：将无边也能用 $\text{one-hot}$ 表示
				```python
				first_elt = E[:, :, :, 0] # 取最后一个维度的第一个元素
				# first_elt.shape:[2, 3, 3] = no_edge.shape:[2 ,3, 3]
				
				# Pytorch 布尔索引语法：tensor[布尔掩码] = 值，需要 shape 匹配才能用
				first_elt[no_edge] = 1    # 把所有的无边的第一个元素置为 1
				E[:, :, :, 0] = first_elt # 显式写回 E
				```
			2. 修正对角线的问题
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
			2. 计算 k-循环特征
			3. 计算选定的特征类型 —— 标准 RRWP
		4. 分子的特征