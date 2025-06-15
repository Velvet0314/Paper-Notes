1. 将数据按批次（batch）读取
	batch 中加载的数据：
	- **蛋白质结构**相关数据 
	```python
	batch['protein_pos']               # [N_protein_atoms, 3] - 所有蛋白质原子的3D坐标[x,y,z]
	batch['protein_atom_feature']      # [N_protein_atoms, feat_dim] - 蛋白质原子特征向量
	batch['protein_atom_name']         # [N_protein_atoms] - 蛋白质原子名称编码(CA,CB,N,O等)
	batch['protein_edit_residue']      # [N_residues] - 布尔掩码，标记需要编辑的口袋残基(True=需要编辑)
	batch['amino_acid']                # [N_residues] - 氨基酸类型编码(1-20对应20种标准氨基酸)
	batch['residue_natoms']            # [N_residues] - 每个样本中残基的原子数量
	batch['protein_atom_to_aa_type']   # [N_protein_atoms] - 每个原子所属的氨基酸类型映射
	batch['res_idx']                   # [N_residues] - 残基在蛋白质序列中的位置索引
	```
	
	- **配体结构**相关数据
	```python
	batch['ligand_element']        # [N_ligand_atoms] - 配体原子的元素类型编码(6=碳, 7=氮, 8=氧等)
	batch['ligand_bond_type']      # [N_ligand_edges] - 配体键类型编码(1=单键, 2=双键等)
	batch['edge_batch']            # [N_edges] - 边连接的批次索引
	batch['ligand_batch']          # [N_ligand_atoms] - 每个配体原子属于哪个batch样本
	batch['ligand_bond_index']     # [2, N_ligand_edges] - 配体原子间的键连接关系索引
	batch['ligand_natoms']         # [batch_size] - 每个样本中配体的原子数量
	batch['ligand_pos']            # [N_ligand_atoms, 3] - 配体所有原子的3D坐标[x,y,z]
	batch['ligand_feat']           # [N_ligand_atoms, feat_dim] - 配体原子特征向量
	batch['ligand_mask']           # [N_ligand_atoms] - 配体原子有效性掩码(1.0=有效, 0.0=填充)
	```
	
	- **序列和批次**相关数据
	```python
	batch['backbone_pos']           # [N_residues, 4, 3] - 主链原子位置(N,CA,C,O)
	batch['amino_acid_processed']   # [N_residues] - 处理后的氨基酸序列，用于ESM模型输入
	batch['atom2residue']           # [N_protein_atoms] - 每个原子所属的残基索引
	batch['residue_pos']            # [N_residues, 14, 3] - 每个残基的14个重原子坐标(N,CA,C,O,CB等)
	batch['amino_acid_batch']       # [N_residues] - 每个残基属于哪个batch样本的索引
	batch['edit_residue_num']       # [batch_size] - 每个样本中需要编辑的残基数量
	```
	
	- **序列掩码和特殊数据**
	```python
	batch['seq']                   # [batch_size, max_seq_len] - 完整的蛋白质序列编码
	batch['full_seq_mask']         # [batch_size, max_seq_len] - 完整序列的有效位置掩码
	batch['r10_mask']              # [batch_size, max_seq_len] - r10区域(结合位点周围10Å)的掩码

	```
	
	- 文件标识信息
	```python
	batch['protein_filename']      # 蛋白质文件名列表
	batch['pocket_filename']       # 口袋文件名列表
	batch['ligand_filename']       # 配体文件名列表
	```
2. 准备训练初始化需要的数据
	```python
	residue_mask = batch['protein_edit_residue']    # 需要编辑的蛋白质残基掩码
	label_ligand = copy.deepcopy(batch['ligand_pos'])   # 真实配体位置
	atom_mask = model.residue_atom_mask[batch['amino_acid'][residue_mask]].bool()   # 原子掩码
	label_X = copy.deepcopy(batch['residue_pos'])   # 真实残基位置
	res_S = copy.deepcopy(batch['amino_acid_processed'])    # 处理后的氨基酸序列
	```
3. 进行训练初始化 `model.init(batch)`
	1. 获取编辑残基掩码
		```python
		residue_mask = batch['protein_edit_residue'] 
		```
		- 含义: 布尔掩码，标记哪些残基需要被编辑/优化
		- 作用: 只对口袋中的关键残基进行设计，其他残基保持固定
	2. 配体位置初始化
		```python
		label_ligand, pred_ligand = copy.deepcopy(batch['ligand_pos']), copy.deepcopy(batch['ligand_pos'])
		pred_ligand = label_ligand + torch.randn_like(label_ligand).to(self.device) * 0.5
		```
		- `label_ligand`: 真实配体位置（用作训练标签）
		- `pred_ligand`: 预测配体位置，通过在真实位置基础上添加高斯噪声初始化
		- 噪声强度: 0.5Å的标准差，模拟配体位置的不确定性
	3. 蛋白质结构初始化
		```python
		res_X = copy.deepcopy(batch['residue_pos'])  # 复制残基位置
		res_X = interpolation_init_new(res_X, residue_mask, copy.deepcopy(batch['backbone_pos']), batch['amino_acid_batch'])
		```
		- `interpolation_init_new()`: 对需要编辑的残基进行主链坐标插值初始化
			- **为需要编辑的残基（口袋残基）通过插值方法初始化主链原子坐标**，确保这些残基在优化开始前就有一个合理的几何结构
			1. 标准主链模板定义
				```python
				backbone = torch.tensor(...)
				```
			2. 批次处理逻辑
				```python
				num_protein = residue_batch.max().item() + 1
				offset = 0
				
				for i in range(num_protein):  # 遍历每个蛋白质样本
				    residue_mask_i = residue_mask[residue_batch == i]
				    backbone_pos_i = backbone_pos[residue_batch == i]
				    if (~residue_mask_i).sum() <= 2:  # 如果固定残基太少，跳过
				        offset += len(residue_mask_i)
				        continue
				```
			3. 参考点识别
				```python
					else:
						residue_index = torch.arange(len(residue_mask_i)).to(res_X.device)
						front = residue_index[~residue_mask_i][:2]    # 前面的两个固定残基
						end = residue_index[~residue_mask_i][-2:]     # 后面的两个固定残基
						near = nearest(residue_mask_i)                # 每个残基最近的固定残基对
				```
				-  `nearest()` 函数:
					- 作用：**为每个残基（包括编辑残基和固定残基）找到其==最近的前一个和后一个固定残基的索引==**，用于后续的插值计算
					- 输入输出：
						```python
						def nearest(residue_mask):
							# 输入: residue_mask [N_residues] - 布尔掩码，0表示固定残基，1表示需要编辑的残基
							# 输出: index [[前一个固定残基索引, 后一个固定残基索引], ...]
							# 长度为 N_residues 的列表，每个元素是长度为2的列表
						```
					- 算法步骤分析：
						1. 初始化
							```python
							index = [[0, 0] for _ in range(len(residue_mask))]  # 初始化结果数组
							p, q = 0, len(residue_mask)  # p记录最近的前固定残基，q记录最近的后固定残基
							```
						2. 第一次遍历：寻找前固定残基
							```python
							for i in range(len(residue_mask)):
							    if residue_mask[i] == 0:  # 遇到固定残基
							        p = i                 # 更新最近的前固定残基索引
							    else:                     # 遇到编辑残基
							        index[i][0] = p       # 记录当前最近的前固定残基
							```
						3. 第二次遍历：寻找后固定残基
							```python
							for i in range(len(residue_mask) - 1, -1, -1):  # 从后往前遍历
							    if residue_mask[i] == 0:  # 遇到固定残基
							        q = i                 # 更新最近的后固定残基索引
							    else:                     # 遇到编辑残基
							        index[i][1] = q       # 记录当前最近的后固定残基
							```
			4. 三种插值策略
				对于每个需要编辑的残基，根据其位置采用不同的插值策略
				1. **策略A: 前端外推** (`k < front[0]`) 
					```python
					for k in range(len(residue_mask_i)):
						if residue_mask_i[k]:
							ind = k + offset
							if k < front[0]:
							    alpha = (backbone_pos_i[front[0]] + (k - front[0]) / (front[0] - front[1]) * (backbone_pos_i[front[0]] - backbone_pos_i[front[1]]))[1: 2]
					```
					- **适用**: 残基位于最前面的固定残基之前
					- **方法**: 基于前两个固定残基的方向进行线性外推
					- **取`[1:2]`**: 只取CA原子位置作为插值中心
				2. **策略B: 后端外推** (`k < end[1]`)
					```python
							elif k > end[1]:
							    alpha = (backbone_pos_i[end[1]] + (k - end[1]) / (end[1] - end[0]) * (backbone_pos_i[end[1]] - backbone_pos_i[end[0]]))[1: 2]
					```
					- **适用**: 残基位于最后面的固定残基之后
					- **方法**: 基于后两个固定残基的方向进行线性外推
				3. **策略C: 中间插值 (前后都有固定残基)**
					```python
							else:
							    alpha = (((k - near[k][0]) * backbone_pos_i[near[k][1]] + (near[k][1] - k) * backbone_pos_i[near[k][0]]) * 1 / (near[k][1] - near[k][0]))[1: 2]
					```
					- **适用**: 残基位于两个固定残基之间
					- **方法**: 在最近的两个固定残基之间进行线性插值
					- **公式**: 加权平均，权重与距离成反比
			5. 主链坐标生成
				```python
				res_X[ind][:4] = alpha + backbone @ quaternion_to_matrix(q=torch.randn(4, device=res_X.device)).t()
				```
				1. ==**alpha**: 插值得到的CA原子位置 `[1, 3]`==
				2. **backbone**: 标准主链模板 `[4, 3]`
				3. **quaternion_to_matrix**: 随机四元数生成的旋转矩阵 `[3, 3]`
				4. **最终结果**: **alpha** 平移 + **quaternion_to_matrix** 旋转后的标准主链模板
		- 作用: 基于相邻固定残基的主链位置，通过线性插值估算编辑残基的主链坐标
	4. 侧链原子随机初始化
		```python
		for k in range(len(batch['amino_acid'])):  # 遍历所有残基
		    if residue_mask[k]:  # 如果是需要编辑的残基
		        pos = res_X[k]
		        pos[4:] = (pos[1].repeat(10, 1) + 0.1 * torch.randn(10, 3, device=self.device))
		        res_X[k] = pos
		```
		- `pos[1]` : CA原子位置（第1个位置是CA，第0个是N）
		- `pos[4:]`: 侧链原子位置（前4个是主链原子N,CA,C,O）
		- 初始化策略: 将所有侧链原子放在CA附近，添加 0.1Å 的随机扰动
	5. 配体特征嵌入
		```python
		ligand_feat = self.ligand_atom_emb(batch['ligand_feat'])
		```
		- 作用: 将配体原子特征通过线性层映射到隐藏维度
		- 输入: 原始配体原子特征（原子类型、键连接等）
		- 输出: 统一的隐藏维度特征向量
	6. 蛋白质原子特征构建
		```python
		atom_emb = self.protein_atom_emb(self.res_atom_type[res_S])  # 原子类型嵌入
		atom_pos_emb = self.atom_pos_embedding(torch.arange(14).to(self.device)).unsqueeze(0).repeat(res_S.shape[0], 1, 1)  # 原子位置嵌入
		res_emb = self.residue_embedding(res_S).unsqueeze(-2).repeat(1, 14, 1)  # 残基类型嵌入
		res_pos_emb = self.pe(batch['res_idx']).unsqueeze(-2).repeat(1, 14, 1)  # 残基位置嵌入
		res_H = torch.cat([atom_emb, atom_pos_emb, res_emb, res_pos_emb], dim=-1)
		```
		这里构建了4层嵌入特征：
		1. 原子类型嵌入 (`atom_emb`)
			- `self.res_atom_type[res_S]`: 根据残基类型获取其14个原子的类型编码
			- 维度: `[N_residues, 14, emb_dim1]`
		2. 原子位置嵌入 (`atom_pos_emb`)
			- 作用: 区分同一残基内不同原子的位置（N,CA,C,O,CB,...）
			- 维度: `[N_residues, 14, 8]`
		3. 残基类型嵌入 (`res_emb`)
			- 作用: 编码残基的氨基酸类型信息
			- 重复: 同一残基的14个原子共享相同的残基嵌入
			- 维度: `[N_residues, 14, emb_dim2]`
		4. 残基位置嵌入 (`res_pos_emb`)
			- 作用: 编码残基在蛋白质序列中的位置信息
			- self.pe: 位置编码函数，类似Transformer中的位置编码
			- 维度: `[N_residues, 14, 16]`
	7. 特殊序列信息存储
		```python
		self.seq = batch['seq']
		self.full_seq_mask = batch['full_seq_mask'] 
		self.r10_mask = batch['r10_mask']
		```
		- 用途: 存储 ESM 模型需要的序列信息，用于后续的序列-结构联合建模\
	8. 返回初始化结果
		```python
		return res_H, res_X, res_S, batch['amino_acid_batch'], pred_ligand, ligand_feat, batch['ligand_mask'], batch['edit_residue_num'], residue_mask
		```
		返回的9个张量：
		- **res_H**: 蛋白质原子特征 `[N_residues, 14, hidden_dim]`
		- **res_X**: 蛋白质原子坐标 `[N_residues, 14, 3]`
		- **res_S**: 残基序列 `[N_residues]`
		- **batch**: 批次索引
		- **pred_ligand**: 初始化的配体位置 `[N_ligand_atoms, 3]`
		- **ligand_feat**: 配体特征 `[N_ligand_atoms, hidden_dim]`
		- **ligand_mask**: 配体掩码
		- **edit_residue_num**: 每个样本的编辑残基数量
		- **residue_mask**: 残基编辑掩码
4. ==**随机循环训练策略**==
	- 优点：
		1. ==**模拟蛋白质自然折叠过程**==
			```python
			# 自然蛋白质折叠：展开态 → 中间态1 → 中间态2 → 天然态
			# 循环训练模拟：初始化 → 无梯度优化1 → 无梯度优化2 → 梯度训练
			for t in range(total_steps, -1, -1):    # total_steps: 1-3 随机
			    if t == 0:
			        # 最终状态：精确优化
			        model.train()
			        res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(...)
			    else:
			        # 中间状态：结构探索
			        model.eval()
			        with torch.no_grad():
				        res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(...)
			```
			**生物学启发**：蛋白质折叠是一个多步骤的渐进过程，不是一步到位的
		2. **渐进式结构优化**
			- **前期无梯度步骤**：相当于结构"退火"过程，允许大幅度结构调整
			- **最后梯度步骤**：基于更合理的结构进行精确优化
			- **训练有效性**：大范围地结构探索，保证一个良好的初始化起点 —— 梯度更有意义，避免局部最优，增强训练的稳定性和收敛性
			- **训练多样性**：随机训练步数（对应了不同次数的初始化探索），增加了随机性和鲁棒性
	1. 等变双层图 Transformer
		1. 原子级别注意力：
			- 输入：
				- **res_H**: `[N_residue, 14, hidden_dim]` - 残基的原子特征
				- **res_X**: `[N_residue, 14, 3]` - 残基的原子坐标
				- **atom_mask**: `[N_residue, 14]` - 残基中有效的原子掩码，标识有效原子位置（不同氨基酸包含的原子类型不同）。`res_S` 是残基序列，包含每个残基的类型编码
				- **batch**: `[N_residue]` - 批次索引
				- **edge_index**: `[2, N_edges]` - K近邻图的边索引
			- 核心计算：
				```python
				# 查询、键、值矩阵 - 每个残基的每个原子都有独立的Q,K,V
				Q = self.W_Q(res_H).view([n_nodes, n_channels, self.num_heads, self.d])  # [N_residue, 14, heads, d]
				K = self.W_K(res_H).view([n_nodes, n_channels, self.num_heads, self.d])  # [N_residue, 14, heads, d]
				V = self.W_V(res_H).view([n_nodes, n_channels, self.num_heads, self.d])  # [N_residue, 14, heads, d]
				
				# 原子级别注意力分数 - 残基i的原子p对残基j的原子q的注意力
				attend_logits = torch.matmul(Q[row].transpose(1, 2), K[col].permute(0, 2, 3, 1))
				# shape: [n_edges, num_heads, 14, 14] - 每条边上14x14的原子间注意力矩阵
				```
			- 输出更新：
				- **res_H**: 更新后的残基原子特征 `[N_residue, 14, hidden_dim]`
				- **res_X**: 更新后的残基原子坐标 `[N_residue, 14, 3]`
		2. 残基-配体级别注意力：
			- 输入：
				- **res_H**: `[N_residue, 14, hidden_dim]` - 残基的原子特征
				- **res_X**: `[N_residue, 14, 3]` - 残基的原子坐标
				- **ligand_pos**: `[batch_size, max_ligand_atoms, 3]` - 配体原子坐标
				- **ligand_feat**: `[batch_size, max_ligand_atoms, hidden_dim]` - 配体原子特征
				- **ligand_mask**: `[batch_size, max_ligand_atoms]` - 配体原子掩码
				- **edit_residue_num**: `[batch_size]` - 每个样本中可编辑残基数量
				- **residue_mask**: `[N_residue]` - 可编辑残基掩码
			- 核心计算：
				```python
				# 只考虑可编辑的残基
				row = torch.arange(n_nodes).to(self.device)[residue_mask]  # 可编辑残基索引
				col = torch.repeat_interleave(torch.arange(batch_size).to(self.device), edit_residue_num)  # 对应配体索引
				
				# 残基-配体注意力矩阵
				Q_lig = self.W_Q_lig(res_H).view([n_nodes, n_channels, self.num_heads, self.d])  # 残基查询
				K_lig = self.W_K_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])  # 配体键
				V_lig = self.W_V_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])  # 配体值
				
				# 原子级别交互注意力 - 残基原子对配体原子的注意力
				attend_logits = torch.matmul(Q_lig[row].transpose(1, 2), K_lig[col].permute(0, 2, 3, 1))
				# shape: [n_edges, num_heads, 14, max_ligand_atoms] - 残基14个原子对配体原子的注意力
				```
			- 输出双向更新：
				- 残基更新：
					- **res_H**: 基于配体信息更新残基特征 `[N_residue, 14, hidden_dim]`
					- **res_X**: 基于配体位置更新残基坐标 `[N_residue, 14, 3]`
				- 配体更新：
					- **ligand_feat**: 基于残基信息更新配体特征 `[batch_size, max_ligand_atoms, hidden_dim]`
					- **ligand_pos**: 基于残基位置更新配体坐标 `[batch_size, max_ligand_atoms, 3]`
	2. 等变FFN
	3. 流程：双层图 Transformer → LayerNorm → FFN → LayerNorm
	4. ESM with Adapter
		1. Adapter 设计
			1. 只在最后一层插入 adapter
				```python
				args.adapter_layer_indices = [-1]  # 默认只在最后一层插入
				```
			2. 只保留模型 alphabet 部分
				```python
				model = cls(args, deepcopy(alphabet))
				model.load_state_dict(pretrained_model.state_dict(), strict=False)
	
				del pretrained_model
				```
			3. 冻结所有不包括 adapter 的参数
				```python
				# freeze pretrained parameters
				for pname, param in model.named_parameters():
					if 'adapter' not in pname:
						param.requires_grad = False
				```
		2. 标准自注意力 —— 保持原有的序列建模能力
			```python
			# 步骤1: 层归一化 + 自注意力
			residual = x  # 保存残差连接的输入
			x = self.self_attn_layer_norm(x)  # 层归一化
			x, attn = self.self_attn(
			    query=x,   # 序列表示作为查询
			    key=x,     # 序列表示作为键  
			    value=x,   # 序列表示作为值
			    key_padding_mask=self_attn_padding_mask,
			    need_weights=True,
			    need_head_weights=need_head_weights,
			    attn_mask=self_attn_mask,
			)
			x = residual + x  # 残差连接
			```
		3. 标准FFN —— 增强非线性表达能力
		4. 带有结构信息的 Adapter 结合 CrossAttention
			```python
			def forward_adapter(self, x, encoder_out, attn_mask, attn_padding_mask):
			    # 获取3D结构特征
			    encoder_feats = encoder_out['feats']  # [batch_size, seq_len, encoder_dim]
			    encoder_feats = encoder_feats.transpose(0, 1)  # [seq_len, batch_size, encoder_dim]
			    
			    # 结构感知交叉注意力
			    x = self.structural_adapter_attn(
			        x,                          # 查询：ESM2的序列表示 [seq_len, batch_size, embed_dim]
			        key=encoder_feats,          # 键：3D结构特征 [seq_len, batch_size, encoder_dim]
			        value=encoder_feats,        # 值：3D结构特征 [seq_len, batch_size, encoder_dim]
			        key_padding_mask=attn_padding_mask,
			        attn_mask=attn_mask,
			        need_weights=False
			    )[0]
			    
			    # 瓶颈前馈网络
			    x = self.structural_adapter_ffn(x)
				
			    return x
			```
			- Bottleneck FFN
				```python
				# structural_adapter_ffn 也是一个 NormalizedResidualBlock
				def structural_adapter_ffn_forward(x):
				    residual = x  # [seq_len, batch_size, embed_dim]
				    
				    # 层归一化
				    x_norm = layer_norm(x)
				    
				    # 瓶颈FFN: embed_dim -> embed_dim//2 -> embed_dim
				    # 对于ESM2-650M: 1280 -> 640 -> 1280
				    x_ffn = fc2(gelu(fc1(x_norm)))  # 瓶颈设计减少参数
				    
				    # 残差连接
				    return residual + x_ffn
				```
5. 通过采样进一步预测氨基酸类型
	```python
	def sample_from_categorical(logits=None, temperature=3.0):
		# 采样
		if temperature:
			# 温度缩放 + 分类分布采样
			dist = torch.distributions.Categorical(logits=logits.div(temperature))
			tokens = dist.sample()
			scores = dist.log_prob(tokens)
		# 贪婪选择
		else:
			scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
			
		return tokens, scores
	```
	- 作用：
		1. 增加探索性和随机性（temperature 越高越随机）
		2. 用于后续评估指标的计算
6. 损失函数定义