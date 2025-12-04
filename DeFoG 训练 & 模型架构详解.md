1. `graph_discrete_flow_model.py` —— `training_step`
	1.  将`batch_data`稀疏图转换为`dense_data`稠密图 
		- `utils.to_dense()` 详细过程在 [[从 0 开始的 DeFoG#^todense]]
	2. 对稠密图进行掩码`mask`
		- 将 padding 的 （无效的）节点和边全部 mask 掉
			```python
			def mask(self, node_mask, collapse=False):
				# mask 节点(原子)的特征
				x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
				# mask 边(键)特征 E 的源节点
				e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
				# mask 边(键)特征 E 的目标节点
				e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
				
	        if collapse: # 生成阶段
				# 将 one-hot 转换成类别索引
				self.X = torch.argmax(self.X, dim=-1)
				self.E = torch.argmax(self.E, dim=-1)
				
				# 索引转换后原本第一个类别的 one-hot 被映射成 0 了，所以需要把 padding 的节点标记为 -1(原本是 0)
				self.X[node_mask == 0] = -1
				self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
			else: # 训练阶段
				# self.X → PlaceHolder(X, E, y=None)，X 是转成稠密图的节点特征
				# X.shape: [batch_size, max_nodes, dx] → dx 是原子类型数量
				❓原本 padding 的原子类型就是 0，为什么还要再处理？
				# 确保无论经过什么处理，padding 位置都保持为 0
				self.X = self.X * x_mask # 广播，将 padding 的原子变为 0
				
				# self.E → PlaceHolder(X, E, y=None)，E 是转成稠密图的边特征
				# E.shape: [bs, n, n, de] → de 是边类型数量
				# 将 padding 的节点的边的对应行、列全部清零，因为之前转稠密处理的时候全部标记成无边 [1, ..., 0, 0]
				# 真实节点 - 真实节点（无边）需要保留
				# 真实节点 - padding、padding - padding 都是无意义的
				self.E = self.E * e_mask1 * e_mask2
				# 检查边矩阵的对称性
				assert torch.allclose(self.E, torch.transpose(self.E, 1, 2)) # # 交换第 1, 2 维（行列互换）
			return self
			```
	3. 采样噪声，并对数据进行加噪
		1. 采样一个时间步
			```python
			 def apply_noise(self, X, E, y, node_mask, t=None):
				"""Sample noise and apply it to the data."""
				# Sample a timestep t.
				
				# 获取 batch_size，是 X 的第一个维度的长度
				bs = X.size(0)
				if t is None:
					t_float = self.time_distorter.train_ft(bs, self.device) # [bs, 1]
				else:
					t_float = t
			```
			- 时间变形器 `time_distorter`
				```python
				def train_ft(self, batch_size, device):
					# 从均匀分布 U(0, 1) 采样生成随机时间步
					t_uniform = torch.rand((batch_size, 1), device=device)
					# 应用时间变形函数
					t_distort = self.apply_distortion(t_uniform, self.train_distortion)
					
					return t_distort
				```
			- 应用时间变形函数 `apply_distortion()`
				```python
				def apply_distortion(self, t, distortion_type):
				# 断言检查采样的随机时间步是否全在 [0, 1] 区间内
				assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"
				
				# 均匀分布 —— 不变
				if distortion_type == "identity":
					ft = t
				# 余弦变形
				elif distortion_type == "cos":
					ft = (1 - torch.cos(t * torch.pi)) / 2
				# 反余弦变形
				elif distortion_type == "revcos":
					ft = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
				# 二次递增
				elif distortion_type == "polyinc":
					ft = t**2
				# 二次递减 
				elif distortion_type == "polydec":
					ft = 2 * t - t**2
				elif distortion_type == "beta":
					raise ValueError(f"Unsupported for now: {distortion_type}")
				elif distortion_type == "logitnormal":
					raise ValueError(f"Unsupported for now: {distortion_type}")
				else:
					raise ValueError(f"Unknown distortion type: {distortion_type}")
					
				return ft
				```
				<div style="text-align: center;"> <img src="time_step.png" width="500"> </div>
		1. 计算概率路径
			- 1
				```python
				# sample random step
				
				# X 的最后一维，就是节点类型的特征
				X_1_label = torch.argmax(X, dim=-1) # [bs, n]
				# E 的最后一维，就是边类型的特征
				E_1_label = torch.argmax(E, dim=-1) # [bs, n, n]
				
				prob_X_t, prob_E_t = p_xt_g_x1(
					X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
				)
			```
			- DFM 的概率路径定义
				$$p_{t|1}(z_t|z_1) = t \delta(z_t, z_1) + (1-t) p_0(z_t) \tag{paper Eq. 1}$$
				```python
				def p_xt_g_x1(X1, E1, t, limit_dist):
					# x1 (B, D)
					# t float
					# returns (B, D, S) for varying x_t value
					
					device = X1.device
					# X 节点的噪声分布
					limit_dist.X = limit_dist.X.to(device) # [dx] 原子类型数量
					# E 边的噪声分布
					limit_dist.E = limit_dist.E.to(device) # [de] 边类型数量
					
					# 先去掉最后一个长度为 1 的维度，再添加两个长度为 1 的维度
					t_time = t.squeeze(-1)[:, None, None] # [bs, 1, 1]
					
					# 之前把最后一维映射合并了，现在需要根据 limit_dist 来重新展开构建正确的 one-hot
					# 例如，如果是 absorbing 增加了虚拟类，这样原本的 one-hot 就不匹配了
					X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float() # [bs, n, dx]
					E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float() # [bs, n, n, de]
					
					# 核心公式
					# one_hot 对应了克罗内克 δ 函数;limit_dist 对应了先验噪声分布
					# limit_dist.X 扩展为 [1, 1, dx]
					Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
					Et = (
						t_time[:, None] * E1_onehot
						+ (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
					)
					
					assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
						(Et.sum(-1) - 1).abs() < 1e-4
					).all()
					
					return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)
				```
	1. 