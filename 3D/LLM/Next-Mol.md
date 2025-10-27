### 论文讲解

**核心**：使用 MoLlama 生成 1D 的分子序列（即其一维子集），然后使用 DMT 扩散模型预测其 3D 构象（3D 坐标）
模型流程：
- 从头开始生成 3D 分子：
	- **MoLlama (语言模型)**  → (自回归地生成)  **SELFIES 字符串 (1D 表示)**  → (通过解析库确定性地转换)  **2D 分子图 (原子 + 键)**  → (提取特征)  **DMT 的输入(原子特征 $\mathbf{H}$ 和完整的原子对特征 $\mathbf{E}$)** → (DMT进行去噪预测)  **最终的3D分子构象 (原子坐标)**
- 3D 构象预测：
	- 已有的数据集 **SELFIES 字符串** → 通过 MoLlama 提取高维度语义特征 → (通过解析库确定性地转换)  **2D 分子图 (原子 + 键)** → (提取特征)  **DMT 的输入(原子特征 $\mathbf{H}$ 和完整的原子对特征 $\mathbf{E}$，以及MoLlama提取的特征)** → (DMT进行去噪预测)  **最终的3D分子构象 (原子坐标)**

1. 使用 MoLlama 进行 1D 分子生成
	- 数据准备：ZNIC-15数据库（分子数量为 **1.8B**），经过预处理后得到 90B 大小的 SELFIES tokens 
	- 预训练 MoLlama：使用 960M 参数量的 **Llama-2**，共进行了 **55.5 万个全局步骤**，处理了 **1450 亿个标记**，大约相当于对整个预训练数据集进行了 **1.6 次完整遍历**
	- 对 SELFIES 进行随机增强：
		- 由于分子可以通过不同方式遍历其二维分子图来生成多个合法的 SELFIES 表达，因此**一个分子对应多个有效 SELFIES**
		- 通过 **随机遍历分子图** 的顺序来生成这些随机化的 SELFIES
		- 相比使用 **规范遍历顺序**（canonical traversal order），这种方法能够提升样本多样性，缓解过拟合现象
		- 其背后的直觉是：**分子中的原子本质上是无序的，因此理想的语言模型应当以相似的概率生成同一个分子的不同排列顺序**
2. 通过 DMT（Diffusion Molecular Transformer）预测 3D 构象（3D 坐标）
	 - 核心：
		1. 控制训练与推理过程的**扩散过程**
		2. 用于建模的**神经网络架构**
		3. 提高模型泛化能力的**旋转增强**策略
	1. 扩散过程
		- **公式 1：前向加噪过程 (Forward Process)**$$q(\mathbf{x}^{(t)} | \mathbf{x}^{(0)}) = \mathcal{N}(\mathbf{x}^{(t)}; \sqrt{\bar{\alpha}(t)}\mathbf{x}^{(0)}, (1-\bar{\alpha}(t))\mathbf{I})$$
			**变量定义：**
		    *   $\mathbf{x}^{(0)} \in \mathbb{R}^{N \times 3}$：原始的、真实的分子三维坐标（N个原子，每个原子3个坐标）
		    *   $\mathbf{x}^{(t)} \in \mathbb{R}^{N \times 3}$：在时间步 `t` 的加噪后的分子坐标
		    *   $t \in (0, 1]$：连续的时间步，表示加噪的程度。$t=0$是原始数据，$t=1$是纯噪声
		    *   $\bar{\alpha}(t)$：一个预先设定的函数（称为noise schedule，噪声表），它随着 `t` 从0到1而单调从 1 递减到 0。它控制了信号与噪声的比例
		    *   $\mathcal{N}(\cdot; \mu, \sigma^2\mathbf{I})$：一个高斯分布，均值为 $\mu$，协方差矩阵为 $\sigma^2\mathbf{I}$
			**直观解释：**
		    - 这个公式描述了如何从一个干净的分子坐标 $\mathbf{x}^{(0)}$ 直接得到任意时刻 $t$ 的噪声版本 $\mathbf{x}^{(t)}$。可以把它想象成一个“信号衰减”和“噪声注入”的过程：
			    *   $\sqrt{\bar{\alpha}(t)}\mathbf{x}^{(0)}$：这是“信号”部分。当 $t \to 0$, $\bar{\alpha}(t) \to 1$，信号很强。当 $t \to 1$, $\bar{\alpha}(t) \to 0$，信号几乎消失
			    *   $(1-\bar{\alpha}(t))\mathbf{I}$：这是噪声的方差。当 $t \to 0$, 方差为 0，没有噪声。当 $t \to 1$, 方差为1，噪声最强
			所以，这个过程就是平滑地将数据点 $\mathbf{x}^{(0)}$ 转换成一个标准高斯噪声分布中的样本
		- **公式 2：重参数技巧 (Reparameterization Trick)**$$\mathbf{x}^{(t)} = \sqrt{\bar{\alpha}(t)}\mathbf{x}^{(0)} + \sqrt{1 - \bar{\alpha}(t)}\boldsymbol{\epsilon}^{(t)}, \quad \text{where} \quad \boldsymbol{\epsilon}^{(t)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
			**变量定义:**
		    *   $\boldsymbol{\epsilon}^{(t)}$：一个从标准高斯分布中采样的随机噪声
			
			**直观解释:**
			- 这是实现公式1的一种更实用的方法。它将随机性从分布的定义中分离出来。我们不再从一个复杂的、依赖于$\mathbf{x}^{(0)}$的高斯分布中采样，而是从一个简单的、固定的标准高斯分布中采样噪声 $\boldsymbol{\epsilon}^{(t)}$，然后通过确定性的变换得到 $\mathbf{x}^{(t)}$
			
			**推导过程:**
		    - 这本质上是高斯分布的标准化。如果一个随机变量 $Z \sim \mathcal{N}(\mu, \sigma^2)$，那么它可以被写成 $Z = \mu + \sigma \cdot E$，其中 $E \sim \mathcal{N}(0, 1)$。这里就是应用了这个原理
		    
			**与模型的联系:**
		    - 这个技巧至关重要，因为它使得整个加噪过程对于 $\mathbf{x}^{(0)}$ 是可微的，**允许梯度在训练时能够顺利反向传播**。在代码实现中，即是用这个公式来生成带噪样本的
		- **公式 3：损失函数 (Loss Function)**$$\mathcal{L} = ||\boldsymbol{\epsilon}^{(t)} - \text{DMT}(\mathbf{G}^{(t)}, t)||^2_2$$
		**变量定义:**
		- $\text{DMT}(\cdot, \cdot)$: 我们的神经网络模型，即 Diffusion Molecular Transformer
		-  $\mathbf{G}^{(t)} = (\mathbf{x}^{(t)}, \mathbf{h}, \mathbf{e})$: 在时间步 $t$ 的整个分子图，包括带噪的坐标 $\mathbf{x}^{(t)}$，原子特征 $\mathbf{h}$（如原子类型），和键/对特征 $\mathbf{e}$（如键类型）
		
		**直观解释:**
		- 这是训练的核心。我们给模型 $\text{DMT}$ 一个带噪的分子 $\mathbf{G}^{(t)}$ 和当前的时间步 `t`，并要求它**预测出当初我们加入的那个标准高斯噪声 $\boldsymbol{\epsilon}^{(t)}$**
		- 损失函数就是计算模型预测的噪声和真实噪声之间的均方误差（MSE）
		
		**推导过程:**
		- 这个简单的MSE损失是扩散模型完整变分下界（VLB）目标的一个简化但非常有效的版本。理论上，最小化这个损失等价于优化数据似然
		
		**与模型的联系:**
		- 这是在训练循环中被优化的最终目标。我们通过小批量随机梯度下降（mini-batch SGD）来近似这个损失的期望值（在所有可能的 $\mathbf{x}^{(0)}$, `t`, $\boldsymbol{\epsilon}^{(t)}$ 上的期望）
	2. 网络架构
		- 思想：MCF 提出：在 AI 领域，长远来看，那些**充分利用计算能力**的、**通用的、可扩展的**方法，最终会胜过那些依赖于人类领域知识、进行精巧设计的复杂方法（Bitter Lesson）
			- **“精巧的复杂方法” (Old Way):** 构建具有内置**3D旋转平移等变性 (Equivariance)** 的图神经网络（如 E(n)-GNN）。这些网络在数学上很优美，能保证你旋转输入分子，输出也会相应地旋转。但它们通常结构复杂，难以扩展到非常大的模型
			- **“通用的可扩展方法” (MCF's Way):** **放弃内置的等变性！** 使用一个相对“笨”但**极其强大且可扩展**的架构——标准的 Transformer。然后通过海量的数据和强大的算力，让模型**自己学会**几何规律，而不是强行把规律写进网络结构里
			- 通过**数据增强**来学习几何。在训练时，它会对输入的分子进行随机旋转，然后将这个旋转矩阵的信息作为一个**条件**输入给模型。这样，模型就必须学会：“哦，当输入被这样旋转时，我预测的噪声也应该被同样地旋转”。通过这种方式，模型从数据中学会了等变性
		- 核心：DMT 的输入同时包含**原子特征** $\mathbf{H}$ 和**完整的原子对特征** $\mathbf{E}$，然后通过 **关系多头自注意力 (RMHA, Relational Multi-Head Self-Attention)** 使得模型在处理原子时能充分考虑原子对之间的关系
		- **公式 (1)-(4): RMHA的核心计算（这里是单头）**$$\begin{align}[\mathbf{Q}; \mathbf{K}; \mathbf{V}] &= [\mathbf{W}_q; \mathbf{W}_k; \mathbf{W}_v]\mathbf{H}^T \quad &(1) \\ [\mathbf{Q}^E; \mathbf{V}^E] &= \text{tanh}([\mathbf{W}_{eq}; \mathbf{W}_{ev}]\mathbf{E}^T) \quad &(2) \\ a_{i,j} &= \text{softmax}_j\left(\frac{(\mathbf{Q}_i^E \odot \mathbf{Q}_i)\mathbf{K}_j^T}{\sqrt{d}}\right) \quad &(3) \\ \mathbf{O}_i &= \sum_{j=1}^N a_{i,j} (\mathbf{V}_j \odot \mathbf{V}_{i,j}^E) \quad &(4) \end{align}$$
		**变量定义:**
	    *   $\mathbf{H} \in \mathbb{R}^{N \times d}$：原子（节点）的特征表示
	    *   $\mathbf{E} \in \mathbb{R}^{N \times N \times d_e}$： 原子对（边）的特征表示
	    *   $\mathbf{Q}, \mathbf{K}, \mathbf{V}$：从原子特征 $\mathbf{H}$ 投影得到的查询、键、值矩阵
	    *   $\mathbf{Q}^E, \mathbf{V}^E$：从原子对特征 $\mathbf{E}$ 投影得到的查询、值矩阵
	    *   $\odot$：逐元素相乘 (Element-wise product)
	    *   $a_{i,j}$：原子 $i$ 对原子 $j$ 的注意力权重
	    *   $\mathbf{O}_i$：RMHA模块为原子 $i$ 输出的更新后的表示
		**直观解释:**
	    1.  **公式(1)和(2)**：和标准Transformer一样，我们先把原子特征 $\mathbf{H}$ 和原子对特征 $\mathbf{E}$ 通过线性层（$\mathbf{W}$矩阵）投影成各自的查询、键、值
	    2.  **公式(3) (注意力分数计算)**： 这是第一个关键
		    - 在计算原子 $i$ 和 $j$ 之间的注意力分数时，标准做法是计算 $\mathbf{Q}_i \mathbf{K}_j^T$
		    - 这里采用原子对的查询特征 $\mathbf{Q}_i^E$ 来“调制”（通过逐元素相乘）原子 $i$ 的查询向量 $\mathbf{Q}_i$。**直观上，这意味着：原子 $i$ 在“看”原子 $j$ 的时候，它“看”的方式（即它的查询Q）会根据它们之间的关系（如化学键）而改变**
	    3.  **公式(4) (值聚合):** 这是第二个关键
		    - 在聚合来自邻居 $j$ 的信息时，标准做法是直接用注意力分数 $a_{i,j}$ 加权求和 $\mathbf{V}_j$
		    - 这里采用原子对的值特征 $\mathbf{V}_{i,j}^E$ 来“过滤”或“门控”原子 $j$ 的值向量 $\mathbf{V}_j$。**直观上，这意味着：从原子 $j$ 传递给原子 $i$ 的信息内容（即它的值V），也会根据它们之间的关系而调整**
		**与模型的联系:**
		- 这四个公式定义了 DMT 模型中每个Transformer层的核心计算单元。通过这种方式，模型在更新每个原子的3D坐标预测时，能够同时考虑全局的原子环境和局部的、精确的键合与距离信息，这对于生成化学上合理的几何结构至关重要
3. 结合 MoLlama 表征提升 DMT 预测分子 3D 构象的能力
	1. **跨模态投影器（Cross-Modal Projector）**
		- 挑战：
			- **MoLlama 使用因果自注意力（causal self-attention）机制**，**每个 token 只能看到它前面的 token**，这限制了分子表示的表达能力
			- **SELFIES token 并不能直接对应到单个原子上**，因此需要额外的映射机制
		- 方法：
			- 将 MoLlama 输出的 SELFIES 表征输入到一个**单层双向自注意力模块**中，以**扩展每个 SELFIES token 的感受野（receptive field）**
			- 借助 SELFIES 格式和 RDKit 工具，实现了 SELFIES 到原子的映射策略：
				1. 对于一个原子对应多个 SELFIES token 的情况，采用**平均池化（mean pooling）作为该原子的表示**
				2. 对于 **没有对应 SELFIES token 的氢原子**，使用一个**可学习的替代 token** 进行表示
			最终，SELFIES-to-atom 映射的输出将通过一个 **多层感知机（MLP）**，并与 DMT 原始的原子表示进行拼接，用于后续的 3D 构象预测
	2. **训练策略（Training Strategy）**
		- 核心：在一个已经预训练好的 DMT 基础上引入 MoLlama 表征
		- 优化：整个过程中，MoLlama 使用 **LoRA（Low-Rank Adaptation）** 方法进行微调，以节省显存
		- 策略：
			1. 独立训练一个不包含 MoLlama 的 DMT 模型，直至收敛
			2. 将 MoLlama 及其跨模态投影器连接至已训练好的 DMT 模型，在保持 DMT 参数冻结的情况下，训练10 个 epoch，以**预热（warm up） projector 和 LoRA 中的随机参数**，同时**避免梯度扰乱 DMT 的预训练表示**
			3. 对整个集成模型进行联合微调，直至收敛
		- 注：在将 MoLlama 的表征集成到 DMT 中时，使用 **canonical SELFIES（规范形式）** 比使用 randomized SELFIES 更有效。这可能是因为 1D MoLlama 与 3D DMT 之间存在表示空间的差异，用固定的 canonical 表达方式有助于更快收敛，降低跨模态对齐的难度

### 一些问题

1. SELFIES-to-Atom Mapping 是如何做的？
	- 流程：
		 1. **SELFIES → SMILES 映射**
		    - 使用 **SELFIES 软件**
		    - 将 SELFIES 的每个 token 映射为对应的 SMILES token 
		2. **SMILES → 原子索引 映射**
		    - 使用 **RDKit** 生成 SMILES 时提供的 **原子顺序**
		    - 手动将 SMILES 中的原子位置对应到原子索引
		3. 将上述两个映射结果 **组合**
		4. 处理缺失的氢原子
			- 映射过程中需要 **显式处理氢的补全与对应关系**
	1.  SMILES-to-Atom 映射构建
		```python
		def build_rdkit2cano_smiles_withoutH_mapping(rdmol):
		    """
		    构建从3D分子结构(RDKit原子)到1D SMILES字符串的映射
		    
		    返回:
		        rdmol_wh2smiles: 原子索引到SMILES字符位置的映射 [N_atoms]
		        canonical_smiles: 标准化SMILES字符串
		    """
		    # 步骤1: 为每个原子添加索引标记
		    rdmol = copy.deepcopy(rdmol)
		    for atom in rdmol.GetAtoms():
		        atom.SetProp("atom_index", str(atom.GetIdx()))
		    
		    # 步骤2: 移除氢原子
		    rdmol_woh = Chem.RemoveHs(rdmol)
		    
		    # 步骤3: 生成标准化SMILES字符串
		    canonical_smiles = Chem.MolToSmiles(rdmol_woh, canonical=True)
		    
		    # 步骤4: 重新排列原子顺序以匹配SMILES
		    smiles_atom_order = rdmol_woh.GetPropsAsDict(True,True)['_smilesAtomOutputOrder']
		    rdmol_woh = Chem.RenumberAtoms(rdmol_woh, list(smiles_atom_order))
		
		    # 步骤5: 建立3D原子到1D SMILES字符的映射
		    # rdmol_wh2rdmol_woh: 含氢分子 -> 去氢分子的原子映射
		    # rdmol_woh2smiles: 去氢分子原子 -> SMILES字符位置的映射
		    
		    # 最终得到: rdmol_wh2smiles [原子索引 -> SMILES字符位置]
		    rdmol_wh2smiles = []
		    for i, j in enumerate(rdmol_wh2rdmol_woh):
		        j = int(j)
		        if j == invalid_int:  # 氢原子，无对应SMILES位置
		            rdmol_wh2smiles.append(invalid_int)
		        else:
		            rdmol_wh2smiles.append(rdmol_woh2smiles[j])
		    
		    return rdmol_wh2smiles, canonical_smiles
		```
	2. SELFIES-to-SMILES 映射构建
		```python
		def get_smiles2selfies_mapping(cano_smiles):
		    """
		    构建从SMILES字符串到SELFIES token的映射
		    
		    返回:
		        smiles2selfies: SMILES字符位置到SELFIES token索引的映射
		        selfies_tokens: SELFIES token列表
		        selfies: SELFIES字符串
		    """
		    # 步骤1: SELFIES编码并获取归因信息
		    selfies, attribution, selfies_tokens = sf_encode_and_attribute(cano_smiles)
		    smiles_tokens, start_poses = split_smiles(cano_smiles, True)
		    
		    # 步骤2: 构建原子集合和映射
		    atom_set = set()
		    for attr in attribution:
		        if attr.attribution is None:
		            continue
		        for item in attr.attribution:
		            atom_set.add((item.index, item.token))
		    atom_list = list(atom_set)
		    atom_list.sort(key=lambda x: x[0])
		    atom_mapping = {atom: i for i, atom in enumerate(atom_list)}
		
		    # 步骤3: 建立SELFIES token到SMILES字符位置的映射
		    selfies2smiles = []
		    for attr in attribution:
		        selfies2smiles.append([])
		        if attr.attribution is None:
		            continue
		        for item in attr.attribution:
		            atom_index = atom_mapping[(item.index, item.token)]
		            sp = start_poses[atom_index]  # SMILES中的起始位置
		            for i in range(len(item.token)):
		                if item.token[i].isalpha():  # 只考虑字母字符（原子符号）
		                    selfies2smiles[-1].append(sp + i)
		
		    # 步骤4: 反转映射 - 从SMILES位置到SELFIES token
		    smiles2selfies = {}
		    for i, smiles_id_list in enumerate(selfies2smiles):
		        for smiles_id in smiles_id_list:
		            if smiles_id not in smiles2selfies:
		                smiles2selfies[smiles_id] = [i]  # SMILES位置 -> SELFIES token索引列表
		            else:
		                smiles2selfies[smiles_id].append(i)
		    
		    return smiles2selfies, selfies_tokens, selfies
		```
2. 模型流程
	1. MoLlama 处理 SELFIES 序列
		```python
		class DiffussionPL(L.LightningModule):
			def forward_llm(self, data_batch, selfies_batch, context=None):
				targets = selfies_batch.input_ids.masked_fill(~selfies_batch.attention_mask.bool(), -100)
				outputs = self.llm_model(input_ids=selfies_batch.input_ids,
				                        attention_mask=selfies_batch.attention_mask,
				                        return_dict=True,
				                        labels=targets,
				                        output_hidden_states=True)
				outputs_hidden_states = outputs.hidden_states
		```
	2.  LLMProjector 处理隐藏状态（ExtendedProjector 加到 Projector（LLMProjector） 上，是为了让 Projector 学会产生 DMT 期望的特征格式）
		```python
		if self.use_llm_projector:
			lm_x = self.llm_projector(hidden_states, data_batch.rdmol2selfies, selfies_batch)
			return lm_x, lm_loss
		```
	3. bmm 映射转换（SELFIES-to-Atom Mapping）
		```python
		class LLMProjector(nn.Module):
			def forward(self, hidden_states, rdmol2selfies, selfies_batch):
				if self.llm_jk == 'last':
					lm_embeds = hidden_states[-1] # shape = [batch_size, seq_len, hidden_size]
				elif self.llm_jk == 'mean':
					lm_embeds = torch.stack(hidden_states[1:], dim=2) # shape = [batch_size, seq_len, num_layers, hidden_size]
					lm_embeds = (self.mean_weight.softmax(dim=-1) @ lm_embeds).squeeze(2) # shape = [batch_size, seq_len, hidden_size]
					lm_embeds = self.mean_ln(lm_embeds)
				else:
					raise NotImplementedError
					
				if self.use_self_att_proj:
					lm_embeds = self.self_att_proj(self.linear_proj(lm_embeds), src_key_padding_mask=~selfies_batch.attention_mask.bool())
				
				lm_x = torch.bmm(rdmol2selfies.to(lm_embeds.dtype), lm_embeds) # shape = [batch_size, rdmol_len, selfies_len], [batch_size, selfies_len, hidden_size] -> [batch_size, rdmol_len, hidden_size]
				norm = torch.clamp(torch.sum(rdmol2selfies, dim=-1, keepdim=True), min=1) # shape = [batch_size, 1, 1]
				lm_x = lm_x / norm # shape = [batch_size, rdmol_len, hidden_size]
			return lm_x
			
			# 解释：
			# 输入维度
			rdmol2selfies: [batch_size, n_atoms, seq_len]     # 映射矩阵
			lm_embeds:     [batch_size, seq_len, hidden_size]  # 序列特征
			# bmm操作
			lm_x = torch.bmm(rdmol2selfies, lm_embeds)
			# 输出: [batch_size, n_atoms, hidden_size]
		```