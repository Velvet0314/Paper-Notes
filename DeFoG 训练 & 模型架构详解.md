1. `graph_discrete_flow_model.py` —— `training_step`
	1.  将`batch_data`稀疏图转换为`dense_data`稠密图 —— `dense_data, node_mask = utils.to_dense(...)`
		- `utils.to_dense()` 详细过程在 [[从 0 开始的 DeFoG#^todense]]
	2. 对稠密图进行掩码`mask` —— `dense_data = dense_data.mask(node_mask)`
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
	3. 采样噪声，并对数据进行加噪 ——  `noisy_data = self.apply_noise(X, E, data.y, node_mask)`
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
		2. 计算该时间步的概率分布
			- 采样的是整个过程的随机一步
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
			- DFM 的概率路径定义：已知标签，计算随机的中间过程的一步的概率分布
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
					# limit_dist.X → [1, 1, dx]
					Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
					# t_time → [bs, 1, 1, 1];limit_dist.E → [1, 1, 1, de]
					Et = (
						t_time[:, None] * E1_onehot
						+ (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
					)
					
					# 概率分布验证
					# Xt.sum(-1) 对最后一维求和，得到每个原子类型的概率，其和应该为 1
					# 1e-4 允许浮点运算误差;.all() 只有 batch 内所有都满足 True 才返回 True
					assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
						(Et.sum(-1) - 1).abs() < 1e-4
					).all()
					
					# clamp 确保所有概率都在 [0, 1] 范围内
					return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)
				```
		3. 采样加噪数据
			- 从上一步计算的中间过程的概率分布中采样，得到加噪数据
				```python
				# step 4 - sample noised data
				
				sampled_t = flow_matching_utils.sample_discrete_features(
					probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
				)
				```
			- 采样流程
				```python
				def sample_discrete_features(probX, probE, node_mask, mask=False):
					"""Sample features from multinomial distribution with given probabilities (probX, probE, proby)
					:param probX: bs, n, dx_out        node features
					:param probE: bs, n, n, de_out     edge features
					:param proby: bs, dy_out           global features.
					"""
					
					bs, n, _ = probX.shape # [bs, n, dx]
					# Noise X
					# The masked rows should define probability distributions as well
					# 为了满足 multinomial 采样，要求每一行的概率和必须为 1（严格的概率分布）
					❓padding 节点在之前不都是设为全 0 了吗，为什么还能有正常的概率?
					# 因为计算随机时间步的概率分布时 F.one_hot 将其转换成了 [1, ..., 0, 0]，所以这里需要再处理
					probX[~node_mask] = 1 / probX.shape[-1] # 反转 mask，并将 padding 节点的每个原子类型概率设为均匀分布
					
					# Flatten the probability tensor to sample with multinomial
					# 每行代表一个节点的原子类型概率分布
					probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)
					
					# Sample X
					# 从多项分布中采样 1 个样本，采样结果是类别的索引
					X_t = probX.multinomial(1, replacement=True)  # (bs * n, 1)
					# X_t = Categorical(probs=probX).sample()  # (bs * n, 1)
					X_t = X_t.reshape(bs, n)  # (bs, n)
					
					# Noise E
					# The masked rows should define probability distributions as well
					# node_mask 相乘后 [i, j] 表示 i 和 j 都是真实节点
					# 反转 mask 后， [i, j] 表示表示至少有一个节点是 padding (无效边)
					inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2)) # [bs, 1, n] * [bs, n, 1] → [bs, n, n]
					# 和节点的 padding 类似，之前也设为全 0 了，但是经 F.ont_hot 转换了
					diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1) # [n, n] → [1, n, n] → [bs, n, n]
					
					# 将 padding 节点的相关边类型概率设为均匀分布，对角线同理
					probE[inverse_edge_mask] = 1 / probE.shape[-1]
					probE[diag_mask.bool()] = 1 / probE.shape[-1]
					
					# 每一行代表一条边的概率分布
					probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)
					
					# Sample E
					E_t = probE.multinomial(1, replacement=True).reshape(bs, n, n)  # (bs, 1) → (bs, n, n)
					# E_t = Categorical(probs=probE).sample().reshape(bs, n, n)  # (bs, n, n)
					
					# 取上三角 + 对称扩展，保证无向图的对称性
					E_t = torch.triu(E_t, diagonal=1)
					E_t = E_t + torch.transpose(E_t, 1, 2)
					
					# 是否立即清零 mask 对应的值
					if mask:
						X_t = X_t * node_mask
						E_t = E_t * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
						
					return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))
				```
			- one-hot 映射
				```python
				noise_dims = self.noise_dist.get_noise_dims()
				# 将索引转回 one-hot
				X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
				E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])
				```
		4. 打包加噪好的数据 + mask padding 数据
			```python
			# step 5 - create the PlaceHolder
			
			# 将之前没有处理的均匀分布采样的 padding mask 掉
			z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
				
			noisy_data = {
				"t": t_float,
				"X_t": z_t.X,
				"E_t": z_t.E,
				"y_t": z_t.y,
				"node_mask": node_mask,
			}
				
			return noisy_data
			```
	4. 计算特征 —— `extra_data = self.compute_extra_data(noisy_data)`
		1. 图的额外特征
			- `rrwp` 详细过程在 [[从 0 开始的 DeFoG#^extragraphfeatures]]
		2. 分子的额外特征
			- `domain_features` 详细过程在 [[从 0 开始的 DeFoG#^extramolfeatures]]
		3. 时间特征
			- `t = noisy_data["t"]`，并将时间作为全局特征 `y` 加入：`extra_y = torch.cat((extra_y, t), dim=1)` 
	5. 模型训练 —— `pred = self.forward(noisy_data, extra_data, node_mask)`
		1. 前向传播函数
			```python
			def forward(self, noisy_data, extra_data, node_mask):
				# 节点特征拼接 [bs, n, dx] + [bs, n, k] 沿维度二拼接 → [bs, n, dx+k]
				X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
				# 边特征拼接 [bs, n, n, de] + [bs, n, n, k] 沿维度三拼接 → [bs, n, n, de+k]
				E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
				# 全局特征拼接 [bs, dy] + [bs, n_cycles] 沿最后一个维度拼接 → [bs, dy+n_cycles]
				y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
				
				return self.model(X, E, y, node_mask) # 传入模型进行预测
			```
		1. 单层 Transformer 架构
			- 初始化
				```python
				class XEyTransformerLayer(nn.Module):
					"""Transformer that updates node, edge and global features
					d_x: node features
					d_e: edge features
					dz : global features
					n_head: the number of heads in the multi_head_attention
					dim_feedforward: the dimension of the feedforward network model after self-attention
					dropout: dropout probablility. 0 to disable
					layer_norm_eps: eps value in layer normalizations.
					"""
					def __init__(
						self,
						dx: int, # 节点特征维度
						de: int, # 边特征维度
						dy: int, # 全局特征维度
						n_head: int, # 注意力头数
						dim_ffX: int = 2048, # 节点 FFN 隐藏层维度
						dim_ffE: int = 128,  # 边 FFN 隐藏层维度
						dim_ffy: int = 2048, # # 全局 FFN 隐藏层维度
						dropout: float = 0.1,
						layer_norm_eps: float = 1e-5,
						device=None,
						dtype=None,
					) -> None:
						kw = {"device": device, "dtype": dtype}
						super().__init__()
						
						self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)
						
						# 节点特征的 FFN
						self.linX1 = Linear(dx, dim_ffX, **kw) # dx -> 2048
						self.linX2 = Linear(dim_ffX, dx, **kw) # 2048 -> dx
						self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw) # 两层 LayerNorm
						self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
						self.dropoutX1 = Dropout(dropout) # 三层 DropOut
						self.dropoutX2 = Dropout(dropout)
						self.dropoutX3 = Dropout(dropout)
						
						# 边特征的 FFN
						self.linE1 = Linear(de, dim_ffE, **kw) # de -> 128
						self.linE2 = Linear(dim_ffE, de, **kw) # 128 -> de
						self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw) # 两层 LayerNorm
						self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
						self.dropoutE1 = Dropout(dropout) # 三层 DropOut
						self.dropoutE2 = Dropout(dropout)
						self.dropoutE3 = Dropout(dropout)
						
						self.lin_y1 = Linear(dy, dim_ffy, **kw) # dy -> 2048
						self.lin_y2 = Linear(dim_ffy, dy, **kw) # 2048 -> dy
						self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw) # 两层 LayerNorm
						self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
						self.dropout_y1 = Dropout(dropout) # 三层 DropOut
						self.dropout_y2 = Dropout(dropout)
						self.dropout_y3 = Dropout(dropout)
						
						self.activation = F.relu
				```
			- 前向传播
				```python
				def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
					"""Pass the input through the encoder layer.
					X: (bs, n, d)
					E: (bs, n, n, d)
					y: (bs, dy)
					node_mask: (bs, n) Mask for the src keys per batch (optional)
					Output: newX, newE, new_y with the same shape.
					"""
					# 标准的 Transformer 步骤
					# step 1 self-attention
					newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)
					
					# step 2 Res + LayerNorm
					newX_d = self.dropoutX1(newX)
					X = self.normX1(X + newX_d)
					# X = self.normX1(newX_d + nn.functional.sigmoid(X))
					
					newE_d = self.dropoutE1(newE)
					E = self.normE1(E + newE_d)
					# newE_d = self.normE1(newE_d + nn.functional.sigmoid(E))
					
					new_y_d = self.dropout_y1(new_y)
					y = self.norm_y1(y + new_y_d)
					# new_y_d = self.norm_y1(new_y_d + nn.functional.sigmoid(y))
					
					# step 3 FFN + Res + LayerNorm
					ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
					ff_outputX = self.dropoutX3(ff_outputX)
					X = self.normX2(X + ff_outputX)
					
					ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
					ff_outputE = self.dropoutE3(ff_outputE)
					E = self.normE2(E + ff_outputE)
					
					ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
					ff_output_y = self.dropout_y3(ff_output_y)
					y = self.norm_y2(y + ff_output_y)
					
					return X, E, y
				```
		2. 单层中的核心：self-attention
			- 初始化
				```python
				class NodeEdgeBlock(nn.Module):
					"""Self attention layer that also updates the representations on the edges."""
					def __init__(self, dx, de, dy, n_head, **kwargs):
						super().__init__()
						assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}" # dx 必须能被 n_head 整除
						
						self.dropout_attn = Dropout(0.1)
						self.dropout_X = Dropout(0.1)
						self.dropout_E = Dropout(0.1)
						
						self.dx = dx 
						self.de = de 
						self.dy = dy 
						self.df = int(dx / n_head) # 每个注意力头的特征维度
						self.n_head = n_head
						
						# Attention
						self.q = Linear(dx, dx) # Query
						self.k = Linear(dx, dx) # Key
						self.v = Linear(dx, dx) # Value
						
						# Feature-wise Linear Modulation 特征线性调制
						# output = γ ⋅input + β
						
						# FiLM E to X
						# 用边特征调制注意力分数
						self.e_add = Linear(de, dx) # 偏置 β
						self.e_mul = Linear(de, dx) # 缩放 γ
						
						# FiLM y to E
						# 用全局特征调制边
						self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
						self.y_e_add = Linear(dy, dx)
						
						# FiLM y to X
						# 用全局特征调制节点
						self.y_x_mul = Linear(dy, dx)
						self.y_x_add = Linear(dy, dx)
						
						# Process y
						self.y_y = Linear(dy, dy)
						self.x_y = Xtoy(dx, dy)
						self.e_y = Etoy(de, dy)
						
						# Output layers
						self.x_out = Linear(dx, dx)
						self.e_out = Linear(dx, de)
						self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))
				```
			- 前向传播
				```python
				def forward(self, X, E, y, node_mask):
						"""
						:param X: bs, n, d        node features
						:param E: bs, n, n, d     edge features
						:param y: bs, dz           global features
						:param node_mask: bs, n
						:return: newX, newE, new_y with the same shape.
						"""
						
						bs, n, _ = X.shape
						x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
						e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
						e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
						
						# 1. Map X to keys and queries
						Q = self.q(X) * x_mask  # (bs, n, dx)
						K = self.k(X) * x_mask  # (bs, n, dx)
						flow_matching_utils.assert_correctly_masked(Q, x_mask)
						
						# 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
						Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
						K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))
						
						Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
						K = K.unsqueeze(1)  # (bs, n, 1, n head, df)
						
						# Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
						Y = Q * K
						Y = Y / math.sqrt(Y.size(-1))
						
						# # Compute the distance based on the Gaussian kernel
						# Y_exp = torch.exp(
						#     -(Q - K) * (Q - K) / math.sqrt(Y.size(-1))
						# )  # bs, n, n, n_head, df
						# Y = Y + Y_exp
						flow_matching_utils.assert_correctly_masked(
							Y, (e_mask1 * e_mask2).unsqueeze(-1)
						)
						
						# Incorporate edge features to the self attention scores.
						E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
						E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
						E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
						E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
						Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)
						
						# Incorporate y to E
						newE = Y.flatten(start_dim=3)  # bs, n, n, dx
						ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
						ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
						newE = ye1 + (ye2 + 1) * newE
						
						# Output E
						newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de
						flow_matching_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)
						
						# Compute attentions. attn is still (bs, n, n, n_head, df)
						softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
						attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head
						
						V = self.v(X) * x_mask  # bs, n, dx
						V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
						V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)
						
						# Compute weighted values
						weighted_V = attn * V
						weighted_V = weighted_V.sum(dim=2)
						
						# Send output to input dim
						weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx
						
						# Incorporate y to X
						yx1 = self.y_x_add(y).unsqueeze(1)
						yx2 = self.y_x_mul(y).unsqueeze(1)
						newX = yx1 + (yx2 + 1) * weighted_V
						
						# Output X
						newX = self.x_out(newX) * x_mask
						flow_matching_utils.assert_correctly_masked(newX, x_mask)
						
						# Process y based on X axnd E
						y = self.y_y(y)
						e_y = self.e_y(E)
						x_y = self.x_y(X)
						new_y = y + x_y + e_y
						new_y = self.y_out(new_y)  # bs, dy
						
						# newX = self.dropout_X(newX)
						# newE = self.dropout_E(newE)
						
						return newX, newE, new_y
				```
		1. 图 Transfomer 前向传播