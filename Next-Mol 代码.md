1. **De novo 3D molecule generation**
	- 通过允许的数据集进行不同下游任务的训练：
		-  从头开始生成 3D 分子 
			- QM9 2014：包含约 134,000 个小分子的量子化学计算数据 —— 包含分子的几何结构、电子性质、热力学性质等信息
			- GeomDrugs：包含约 430 000 个药物类有机分子包含药物分子的多构象几何数据 —— 专注于药物相关分子的3D几何构象生成
			- GeomDrugs-JODO：论文 [Learning Joint 2D & 3D Diffusion Models for Complete Molecule Generation](https://arxiv.org/abs/2305.12347) 使用的数据集
		- 3D Conformer prediction
			- Geom-QM9
			- GeomDrugs
	- MoLlama 生成 1D 序列，然后利用 RDkit 解析为 2D 图 （包含**原子特征** $\mathbf{H}$ 和**完整的原子对特征** $\mathbf{E}$），然后利用 DMT 进行生成
2. **Conditional 3D molecule generation**
	- 类似于从头开始 3D 生成
3. **3 Stage 联合进行分子预测**
	- Stage 1：只对 DMT 进行训练，冻结 MoLlama 和 Projector
	- Stage 2：针对 MoLlama 和 Projector 进行 warm up，使用训练好的 DMT 作为损失函数来优化 Projector，以实现：
		- **避免扰乱**：保护预训练的 DMT 表示不被随机梯度破坏
		- **特征对齐**：让 Projector 学会产生 DMT 期望的特征格式
		- 这里是训练一个 ExtendedProjector 加到 Projector（LLMProjector） 上（冻结原始参数，只训练新增的层）
			- 原始 Projector
				```python
				if self.use_llm:
					if self.llm_cond:
						self.projector = nn.Sequential(
							nn.Linear(in_dim, 4 * hidden_dim),
							nn.GELU(),
							nn.Linear(4 * hidden_dim, hidden_dim),
							)
				else:
					if self.delta_train:
						self.extended_node_emb = ExtendedProjector(self.node_emb, in_dim, hidden_dim, disable_extra_gelu=self.disable_extra_gelu)
				```
			- 启用 warm up 后采用 ExtendedProjector
				```python
				class ExtendedProjector(nn.Module):
				"""Extend an existing projector with a new input."""
					def __init__(self, projector, extend_dim, hidden_dim, disable_extra_gelu):
						super().__init__()
						# the following is weight tying
						
						# 权重绑定：复用原始的 projector 权重（冻结）
						self.linear1 = projector[0]
						self.act = projector[1]
						self.linear2 = projector[2]
						self.disable_extra_gelu = disable_extra_gelu
						
						# 新增的可训练层
						if self.disable_extra_gelu:
							self.projector = nn.Sequential(
								nn.Linear(extend_dim, 4 * hidden_dim),
								nn.GELU(),
								nn.Linear(4 * hidden_dim, self.linear1.out_features),
							)
						else:
							self.projector = nn.Sequential(
								nn.Linear(extend_dim, 4 * hidden_dim),
								nn.GELU(),
								nn.Linear(4 * hidden_dim, self.linear1.out_features),
								nn.GELU(),
							)
							
					def forward(self, x, new_x):
						x = self.linear1(x) + self.projector(new_x)
						return self.linear2(self.act(x))
				```
	- Stage 3：三者联合训练优化, MoLlama 进行 LoRA 微调以更适配当前任务，Projector 来自 **MoLlama 的输出的隐藏层高维度信息**转换为 DMT 所需的特征信息作为输入，DMT 结合前面两者和**已经解析成 2D 图的 1D MoLlama 输出序列** 并从 2D 分子图中提取特征作为输入进行优化
		- 提取原子特征与边特征
			```python
			def featurize_mol(mol, types=drugs_types):
				"""
				Part of the featurisation code taken from GeoMol https://github.com/PattanaikL/GeoMol
				Returns:
					x:  node features
					z: atomic numbers of the nodes (the symbol one hot is included in x)
					edge_index: [2, E] tensor of node indices forming edges
					edge_attr: edge features
				"""
				if type(types) is str:
					if types == 'qm9':
						types = qm9_types
					elif types == 'drugs':
						types = drugs_types
					
				N = mol.GetNumAtoms()
				atom_type_idx = []
				atomic_number = []
				atom_features = []
				chiral_tag = []
				ring = mol.GetRingInfo()
				
				for i, atom in enumerate(mol.GetAtoms()):
					atom_type_idx.append(types[atom.GetSymbol()])
					chiral_tag.append(chirality[atom.GetChiralTag()])
					# 提取原子特征
					atomic_number.append(atom.GetAtomicNum())
					atom_features.extend([atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0])
					atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
					atom_features.extend(one_k_encoding(atom.GetHybridization(), [
											Chem.rdchem.HybridizationType.SP,
											Chem.rdchem.HybridizationType.SP2,
											Chem.rdchem.HybridizationType.SP3,
											Chem.rdchem.HybridizationType.SP3D,
											Chem.rdchem.HybridizationType.SP3D2]))
					atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
					atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
					# 环结构特征
					atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
										int(ring.IsAtomInRingOfSize(i, 4)),
										int(ring.IsAtomInRingOfSize(i, 5)),
										int(ring.IsAtomInRingOfSize(i, 6)),
										int(ring.IsAtomInRingOfSize(i, 7)),
										int(ring.IsAtomInRingOfSize(i, 8))])
					atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))
					
				z = torch.tensor(atomic_number, dtype=torch.long)
				
				# 提取边特征
				row, col, edge_type = [], [], []
				for bond in mol.GetBonds():
				    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
				    row += [start, end]
				    col += [end, start]
				    edge_type += 2 * [bonds[bond.GetBondType()]]  # 双向边
				
				edge_index = torch.tensor([row, col], dtype=torch.long)
				edge_type = torch.tensor(edge_type, dtype=torch.long)
				
				# 边特征的one-hot编码
				edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
			```
		- 构建完整的原子对特征
			```python
			class QM9UniMolVersion(Dataset):
			'''
			transform the data into uni-mol version
			'''
				## obtain edge types; which is defined as the combination of two atom types
				edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
				# edge_type: [num_atoms, num_atoms]，每个(i, j)位置表示原子i和原子j的类型组合编码
				# 例如，如果num_types=5，原子类型分别为1和3，则edge_type[i, j]=1*5+3=8
				# 这样可以唯一编码每对原子的类型组合，便于后续模型区分不同原子对
				
				dist = distance_matrix(coordinates, coordinates).astype(np.float32)
				# dist: [num_atoms, num_atoms]，每个(i, j)位置表示原子i和原子j之间的欧氏距离
				# 用于提供分子的几何结构信息
				
				tgt_dist = distance_matrix(tgt_coordinates, tgt_coordinates).astype(np.float32)
				# tgt_dist: [num_atoms, num_atoms]，目标构象的距离矩阵（如真实分子的参考坐标）
				
				coordinates, dist, tgt_coordinates, tgt_dist = torch.from_numpy(coordinates), torch.from_numpy(dist), torch.from_numpy(tgt_coordinates), torch.from_numpy(tgt_dist)
				
				## prepare the bond type matrix
				# data.edge_index # shape = [2, num_edges]; data.edge_type # shape = [num_edges]
				num_atoms = dist.shape[0]
				sp_bond_type = torch.sparse_coo_tensor(data.edge_index + 1, data.edge_type, size=(num_atoms, num_atoms), dtype=torch.long) # +1 because of the bos token in uni-mol atoms
				# 构建稀疏的键类型矩阵，索引加1是因为UniMol格式在原子序列前加了BOS token
				# data.edge_index: [2, num_edges]，每列表示一个键的两个原子索引
				# data.edge_type: [num_edges]，每个键的类型（如单键、双键等）
				
				bond_type = sp_bond_type.to_dense()
				# 将稀疏矩阵转为密集矩阵，得到[num_atoms, num_atoms]的bond_type
				# bond_type[i, j]表示原子i和原子j之间的化学键类型（0表示无键，1-4表示不同类型的键）
			```