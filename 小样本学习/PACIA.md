### Few-Shot MPP（分子性质预测）

1. 问题定义
	1. **任务** (Task $\mathcal{T}_\tau$)
		- 每一个任务 $\mathcal{T}_\tau$ 对应一种特定的**生化性质**
	2. **数据样本** ($\mathcal{X}_{\tau, i}$ 和 $y_{\tau, i}$)
		* $\mathcal{X}_{\tau, i}$：一个分子图（Graph），包含原子节点和化学键边
		* $y_{\tau, i} \in \{0, 1\}$：标签。$1$ 代表该分子在这个性质 $\tau$ 上是 **Active（有活性的/有效的）**，$0$ 代表 **Inactive（无活性的/无效的）**
	3. **支持集** (Support Set $\mathcal{S}_\tau$)
		$$\mathcal{S}_\tau = \{(\mathcal{X}_{\tau, s}, y_{\tau, s})\}_{s=1}^{N_\tau}$$
		- 任务中的训练集：参考分子，哪些是有效的，哪些是无效的
	4. **查询集** (Query Set $\mathcal{Q}_\tau$)
		$$\mathcal{Q}_\tau = \{(\mathcal{X}_{\tau, q}, y_{\tau, q})\}_{q=1}^{M_\tau}$$
		- 任务中的测试集：需要预测的分子，用来计算 Loss 并更新模型
	5. **N-way K-shot**
		- N-way：分类的类别
		- K-shot：每一类有 $K$ 个样本
2. 编码器-预测器框架
	1. 标准 GNN 在**原子层面**更新特征
		$$\mathbf{h}_v^l = \text{UPD}^l \left( \mathbf{h}_v^{l-1}, \text{AGG}^l (\{ (\mathbf{h}_v^{l-1}, \mathbf{h}_u^{l-1}, \mathbf{b}_{vu}) | u \in \mathcal{H}(v) \}) \right) \tag{1}$$
		- **变量定义**：
			* **$\mathbf{h}_v^l$**：第 $l$ 层网络中，原子（节点）$v$ 的特征向量（Embedding）
			* **$\mathbf{h}_v^{l-1}$**：上一层（$l-1$层）该原子的特征。如果是第0层，就是原子的初始物理化学特征（如原子序数、杂化类型等）
			* **$\mathcal{H}(v)$**：原子 $v$ 的邻居集合（通过化学键相连的其他原子）
			* **$\mathbf{b}_{vu}$**：原子 $v$ 和 $u$ 之间的**边特征**（Edge Feature），例如键的类型（单键、双键、芳香键）
			* **$\text{AGG}^l(\cdot)$**：**聚合函数 (Aggregation Function)**。负责把周围邻居的信息“收集”起来
			* **$\text{UPD}^l(\cdot)$**：**更新函数 (Update Function)**。负责结合“自己原来的特征”和“收集到的邻居特征”，算出新的特征
	2. 读出机制 —— 通过一个向量来表示整个分子（**分子层面**）
		$$\mathbf{r} = \text{READOUT}(\{\mathbf{h}_v^L | v \in \mathcal{V}\}) \tag{2}$$
		- 变量定义：
			* $\mathbf{r}$：整个分子的表示向量（Molecular Representation），是一个固定长度的向量
			* $\mathbf{h}_v^L$：经过 $L$ 层 GNN 处理后，最终层的原子特征
			* $\mathcal{V}$：该分子图中所有原子的集合
			* $\text{READOUT}(\cdot)$：读出函数，也叫全局池化（Global Pooling）
	3. 预测机制
		$$\hat{\mathbf{y}}_{\tau, q} = f(\mathbf{r}_{\tau, q} | \{ \mathbf{r}_{\tau, s} \}_{s \in \mathcal{S}_\tau}) \tag{3}$$
		- 变量定义：
			* $\hat{y}_{\tau, q}$：针对查询分子 $q$ 在任务 $\tau$ 上的预测结果（标签）
			* $\mathbf{r}_{\tau, q}$：查询分子（Query）的向量表示（由公式 2 得到）
			* $\{\mathbf{r}_{\tau, s}\}_{s \in \mathcal{S}_\tau}$：支持集中所有分子的向量表示集合
			* $f(\cdot | \cdot)$：条件预测器
		- 区别：
			- 传统模型：$\hat{y} = f(\mathbf{r}; W)$。预测只依赖于输入 $r$ 和固定的权重 $W$
			- Few-Shot： $\hat{y}$ 不仅取决于查询分子 $\mathbf{r}_{\tau, q}$ 长什么样，还取决于参考书（支持集）里的信息
	4. 关系图预测器
		* **目标**：在预测阶段，不仅仅看单个分子，而是构建一个由“支持集分子”和“查询集分子”组成的**关系图**，通过图卷积来细化特征，利用相似性进行分类
		- **关系图（Relation Graph）**：
			* **节点（Nodes）**：不再是原子，而是**整个分子**（包括支持集中的 $N$ 个分子和查询集中的 1 个分子）
			* **边（Edges）**：分子之间的相似度
		- 动态构建关系边
			$$a_{ij}^l = \begin{cases} \text{MLP}(|\mathbf{h}_{\tau, i}^{l-1} - \mathbf{h}_{\tau, j}^{l-1}|) & \text{if } i \neq j \\ 1 & \text{otherwise} \end{cases} \tag{4}$$
			- **变量定义**：
				* $i, j$：代表第 $i$ 个和第 $j$ 个分子
				* $\mathbf{h}_{\tau, i}^{l-1}$：第 $l-1$ 层关系图网络输出的**分子 $i$ 的特征向量**
					* 初始状态 ($l=0$)：这就是公式 (2) 输出的 $\mathbf{r}$（由 Encoder 提取的分子指纹）
				* $|\mathbf{h}_i - \mathbf{h}_j|$：绝对差值，衡量两个向量距离最简单有效的方法之一。它捕捉了分子 $i$ 和 分子 $j$ 在哪些特征维度上不一样
				* $\text{MLP}(\cdot)$：一个多层感知机。它的作用是将“特征差值”映射为一个标量权重 $a_{ij}$
				* $a_{ij}^l$：边权重（Edge Weight）。代表分子 $i$ 和 $j$ 的**相似度**，或者说信息传递的强度
		- 关系图上的特征更新
			$$\mathbf{h}_{\tau, i}^l = \text{MLP} \left( \sum_{j=1}^{N_\tau+1} a_{ij}^l \mathbf{h}_{\tau, j}^{l-1} \right) \tag{5}$$
			- **变量定义**：
				* $\sum_{j=1}^{N_\tau+1}$：遍历当前任务中的所有分子（支持集 + 查询集共 N + 1 个）
				* $a_{ij}^l \mathbf{h}_{\tau, j}^{l-1}$：加权聚合。分子 $i$ 会吸收分子 $j$ 的特征，吸收多少取决于它们的相似度 $a_{ij}$
				* $\mathbf{h}_{\tau, i}^l$：更新后的分子 $i$ 的特征
3. 编码器-预测器框架的分层适应
	1. 统一的 GNN 适配器
		1. **调制节点嵌入**
			* **目标**：动态调整特征分布，相当于一种条件式的 LayerNorm 或 FiLM（Feature-wise Linear Modulation
			    $$\hat{\mathbf{h}}^l = e(\mathbf{h}^l, \boldsymbol{\gamma}^l) \tag{6}$$
				* **变量定义**：
				    * $\mathbf{h}^l$：原始 GNN 层的输出特征
				    * $\boldsymbol{\gamma}^l$：由超网络生成的**自适应参数**（Adaptive Parameters）
				    * $e(\cdot)$：逐元素操作函数（通常是仿射变换，即 Scale & Shift）$\hat{\mathbf{h}} = \mathbf{h} \odot \boldsymbol{\gamma}_{scale} + \boldsymbol{\gamma}_{shift}$
		2. **调制传播深度**
			- 目标：解决 GNN 的过平滑问题，并针对不同难度的分子选择最佳的感受野
			    $$\boldsymbol{p} = \text{softmax}([p^1, p^2, \dots, p^L]) \tag{7}$$
			    $$\tilde{\mathbf{h}} = \sum_{l=1}^L [\boldsymbol{p}]_l \mathbf{h}^l \tag{8}$$
				* **变量定义**：
				    * $p^l$：第 $l$ 层的非归一化置信度分数（由超网络生成）
				    * $\boldsymbol{p}$：归一化后的层选择概率分布，代表了模型认为“在这个深度停止消息传递”的概率
				    * $\tilde{\mathbf{h}}$：最终的加权混合特征
				- 注：在推理（Inference/Testing）阶段，使用 $l' = \text{argmax } p^l$（公式13），即直接选择概率最大的一层，这是为了减少计算量并获得确定的推理路径
	2. 层次化自适应参数生成
		1. 提取类原型
			- 目标：把一堆零散的支持集样本（Support Set），压缩成具有代表性的向量
				$$\mathbf{r}_{\tau, +}^l = \frac{1}{|S_\tau^+| |\mathcal{V}_{\tau, s}|} \sum_{X_{\tau, s} \in S_\tau^+} \text{MLP} \left( \sum_{v \in X_{\tau, s}} \mathbf{h}_v^l \bigg| \mathbf{y}_{\tau, s} \right) \tag{9}$$
				- 变量定义：
					* $\mathbf{r}_{\tau, +}^l$：第 $l$ 层 **正类（Active）的原型向量**
					* $S_\tau^+$：任务 $\tau$ 支持集中所有正样本的集合
					* $|\mathcal{V}_{\tau, s}|$：分子 $s$ 中的原子数量（用于归一化）
					* $\sum_{v \in X_{\tau, s}} \mathbf{h}_v^l$：**分子内聚合**。把分子 $s$ 的所有原子特征加起来，得到该分子的初步表示
					* $\bigg| \mathbf{y}_{\tau, s}$：**条件拼接**。将标签信息 $\mathbf{y}$（one-hot 向量）拼接到特征上，让 MLP 明确知道“现在处理的是正样本”
					* $\sum_{X_{\tau, s} \in S_\tau^+}$：**集合间聚合**。把所有正样本的特征加起来
				- 注：$\mathbf{r}_{\tau, -}^l$ 同理，只是针对负样本集 $S_\tau^-$
		2. 任务级适应
			- 目标：生成用于 **Encoder（原子级 GNN）** 的参数
				$$[\boldsymbol{\gamma}_\tau^l, p_\tau^l] = \text{MLP} \left( [\mathbf{r}_{\tau, +}^l \parallel \mathbf{r}_{\tau, -}^l] \right) \tag{10}$$
				- **变量定义**：
					* $\parallel$：**拼接操作 (Concatenation)**。将正类原型和负类原型拼在一起
					* $[\boldsymbol{\gamma}_\tau^l, p_\tau^l]$：输出的适配参数
					    * $\boldsymbol{\gamma}$：调制特征（公式 6）
					    * $p$：调制深度（公式 7）
					* **MLP**：这是一个可学习的神经网络（即超网络的一部分）
				- 注：这里的参数只与任务（支持集）有关，与具体的 Query 无关。因为 Encoder 的作用是提取通用的化学特征
		3. 查询级适应
			- 目标：生成用于 **Predictor（关系图 GNN）** 的参数
				$$[\boldsymbol{\gamma}_{\tau, q}^l, p_{\tau, q}^l] = \text{MLP} \left( [\mathbf{r}_{\tau, +}^l \parallel \mathbf{r}_{\tau, -}^l \parallel \sum_{v \in X_{\tau, q}} \mathbf{h}_v^l ] \right) \tag{11}$$
				- 变量定义：
					* $\sum_{v \in X_{\tau, q}} \mathbf{h}_v^l$：**当前查询分子（Query）** 的特征表示
					* 输入：正类原型 || 负类原型 || 当前查询分子
				- 解释：
					- **动态难度调整**：
					    * 如果是简单的 Query（一眼就能看出和正类很像），MLP 可能会输出一个倾向于浅层的 $p$ 分布
					    * 如果是困难的 Query（结构处于边界，模棱两可），MLP 会输出倾向于深层的 $p$ 分布，并调整 $\gamma$ 来关注更细微的差异
					* **个性化服务**：每一个 Query 分子，享有一套专门为它定制的 GNN 参数
		4. 元训练目标函数
			- 训练整个架构（主网络 + 超网络）的损失函数
				$$\min_\Theta \sum_{\tau=1}^N \mathcal{L}_\tau, \quad \text{with } \mathcal{L}_\tau = - \sum_{x_{\tau, q} \in Q_\tau} \mathbf{y}_{\tau, q}^\top \log (\hat{\mathbf{y}}_{\tau, q}) \tag{13}$$
				- 变量定义：
					* $\Theta$：**所有可学习参数的集合**
					    * 包括：GNN 主干网参数、Relation Graph 参数、**以及生成 $\gamma, p$ 的那些 MLP（超网络）的参数**。
					    * 注意：$\gamma$ 和 $p$ 本身不是 $\Theta$，它们是 $\Theta$ 的输出
				* $\mathcal{L}_\tau$：单个任务 $\tau$ 的损失
				* $\mathbf{y}^\top \log (\hat{\mathbf{y}})$：标准的**交叉熵损失 (Cross-Entropy Loss)**
4. 元学习的训练流程
	<div style="text-align: center;"> <img src="meta-PACIA.png" width="400"> </div>
	
	- 输入：元学习的任务集 $\mathcal{T}_{train}$
	- 阶段一：Encoder 适应 (Task-Level Adaptation)
		1. line 1：初始化 ——  GNN 的权重、Relation Graph 的权重，以及**生成适配参数的超网络的权重**
		2. line 2~8：从任务集中采样一个任务
			- 遍历 GNN Encoder 的每一层
			- 通过**超网络**调用公式10，输入公式9得到的 Support 的正/负原型向量，得到当前层的适配参数 $\gamma_\tau^l$ 和 $p_\tau^l$ 
			- 通过适配参数调制 **GNN** 的原子嵌入（公式6）
			- 通过 **主干网络（GNN）** 更新原子的 Embedding
		3. line 9：利用生成的 $p$，进行深度调制，得到最终的原子特征，并使用公式2将原子特征聚合成分子特征
	- 阶段二：Predictor 适应 (Query-Level Adaptation)
		1. line 11~16：从 Query 中采样一个需要预测的分子
			- 遍历关系图网络的每一层
			- 通过**超网络**调用公式11，输入 Support + 当前 Query 的特征，得到针对当前 Query 定制的适配参数 ${\gamma}_{\tau, q}^l, p_{\tau, q}^l$
			- 通过适配参数调制**关系图**的节点嵌入（公式6）
			- 通过 **主干网络（关系图）** 计算 Query 和 Support 的相似度（公式4），并更新分子的特征（公式5）
		2. line 17：利用生成的 $p$，进行深度调制
		3. line 18：输出最终预测结果