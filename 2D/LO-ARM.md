1. 背景知识
	1. 问题定义
		- 假设数据向量 $\boldsymbol{x} = (x_1, \ldots, x_L)$,其中每个维度 $x_i$ 从集合 $\mathcal{X}$ 中取值,该集合可以是实数或离散的。在不失一般性的情况下，假设 $\boldsymbol{x}$ 是一个包含 $L$ 个分类变量的向量，因此 $\mathcal{X}$ 是一个包含 $m = |\mathcal{X}|$ 个类别的离散集合
		- 对于分子图，$\mathcal{X} = \mathcal{V} \cup \mathcal{E}$，其中 $\mathcal{V}$ 和 $\mathcal{E}$ 分别是原子和键的类型。一个自回归模型是 $x$ 上的一个联合概率分布，其因式分解为：
			$$p_\theta(x) = \prod_{i=1}^{L} p_\theta(x_i | x_{<i}) \tag{1}$$
			- 其中 $x_i$ 表示 $\boldsymbol{x}$ 的第 $i$ 个维度，$x_{<i} = (x_1, \ldots, x_{i-1})$ 表示前 $i-1$ 个元素的向量，$x$ 和 $p_\theta(x_i | x_{<i})$ 是条件分布，约定为 $p_\theta(x_1 | x_{<1}) = p_\theta(x_1)$
		- 从模型中采样会按顺序生成数据维度，从 $x_1$ 开始到 $x_L$ 结束。拥有固定或预先指定的顺序通常会产生次优结果，并且在对没有自然顺序的数据建模时可能引入不适当的归纳偏差
		- AO-ARMs 通过训练一个可以在从索引 $\{1, \ldots, L\}$ 的 $L!$ 个排列中**均匀抽取的随机顺序**下生成数据维度的模型来解决这个问题
		- 给定一个排列 $\sigma$，模型的联合分布因式分解为：
			$$p_\theta(x|\sigma) = \prod_{i=1}^{L} p_\theta(x_{\sigma_i} | x_{\sigma_{<i}}) 
			\tag{2}$$
			- 其中 $\sigma_{<i}$ 表示排列 $\sigma$ 下前 $i-1$ 个元素的索引
		- 如果 $p(\sigma)$ 表示 $L!$ 个排列上的均匀分布，参数 $\theta$ 通过最大化期望对数似然(每个数据点)来找到：
			$$\mathbb{E}_{p(\sigma)} [\log p_\theta(x|\sigma)] \tag{3}$$
			- 解释上述目标的一种方法是作为对数似然的变分下界，该对数似然来自一个概率潜变量模型。具体来说，如果 $\sigma$ 是与数据点 $\boldsymbol{x}$ 对应的潜变量（所以在训练时对于每个样本 $\boldsymbol{x}^{(n)}$ 都有一个不同的 $\sigma^{(n)}$），对数似然可以下界为 $\log p_\theta(x) = \log \sum_\sigma p(\sigma)p(x|\sigma) \geq \mathbb{E}_{p(\sigma)}[\log p(x|\sigma)]$，即方程 (3) 中的训练目标
			- 注：在实践中，这个目标是随机优化的
	2. 自回归生成作为去掩码过程
		- 由于每个离散数据维度（或 token）$x_i, i \in \{1, \ldots, L\}$ 取 $m$ 个分类值，用一个额外的辅助类别或掩码来扩充空间。因此，将每个 $x_i$ 表示为一个 $m+1$ 维（而不是 $m$ 维）的 $\text{one-hot}$ 向量，其中最后的第 $m+1$ 个值表示 $x_i$ 被掩码
		- 用 ARMs 将生成过程建模为去掩码过程
			1. 具体来说，从一个完全掩码的状态 $\bar{\boldsymbol{x}} = (\bar{x}_1, \ldots, \bar{x}_L)$ 开始，其中 $\bar{x}_i$ 表示掩码
			2. 在每一步，选择一个维度 $\bar{x}_i$ 并"去掩码"，这意味着在 $m$ 个类别中采样一个分类值 $x_i$
			3. 重复这个过程直到所有维度都被去掩码，这产生一个最终生成的数据点 $\boldsymbol{x}$
			- 在这种表示下，我们的模型也与最近的掩码离散扩散模型相联系
			- 注：与离散扩散模型不同，我们在框架中不指定前向过程，而只指定去掩码或后向过程
2. 学习顺序的自回归模型 (LO-ARM)
	- 将 AO-ARM 中的均匀先验分布 $p(\sigma)$ 替换为有序的可学习分布。我们称这种分布为顺序策略，因为它通过对 $\boldsymbol{x}$ 的已生成数据维度进行调节来动态决定要生成的下一个维度
	1. 模型设定与顺序策略（Order-Policy）
		- 对于一个排列 $\sigma$ 进行采样，可以用一组 $L$ 个潜在变量 $z_i$ 来表示，$i = 1, \cdots, L$：
			$$p(z) = \prod_{i=1}^{L} p(z_i | x_{z_{<i}}) 
			\tag{4}$$
			- 其中 $z_1 \sim p(z_1) = p(z_1|z_{<1})$ 是一个从集合 $\{1, \ldots, L\}$ 中取 $L$ 个值的分类变量，并且每个后续的 $z_i \sim p(z_i|z_{<i})$ 从集合 $z_{>i} = \{1, \ldots, L\} \setminus z_{<i}$ 中取 $L - i + 1$ 个值，$z_{<i} = (z_1, \ldots, z_{i-1})$
		- 概率分布 $p_\theta(x)$ 可以写为：
			$$p_\theta(\boldsymbol{x}) = \sum_{\boldsymbol{z}} p(\boldsymbol{z})p_\theta(\boldsymbol{x}|\boldsymbol{z}) = \sum_{\boldsymbol{z}} \prod_{i=1}^{L} p(z_i|\boldsymbol{z}_{<i})p_\theta(x_{z_i}|x_{\boldsymbol{z}_{<i}})$$

			- 其中两个条件 $p(z_i|z_{<i})$ 和 $p_\theta(x_{z_i}|x_{z_{<i}})$ 并列遵循一个自回归结构，从 $i = 1$ 到 $i = L$ 展开。当每个 $p(z_i|z_{<i})$ 是均匀的时,则 $p(z)$ 是在 $L!$ 个排列上的自回归表示的均匀分布。在这种情况下，模型简化为标准的 AO-ARMs
		- 在论文提出的方法中，命名为学习顺序 ARM (LO-ARM)，同样使用潜变量 $z$ ，但用一个更有信息量的分布替换 $p(z_i|z_{<i})$，这个分布称为 order-policy（顺序策略）
		- 定义3.1 order-policy 是 $\boldsymbol{z}$ 上的一个遵循因式分解的分布
			$$p_\theta^x(\boldsymbol{z}) = \prod_{i=1}^{L} p_\theta(z_i|\boldsymbol{z}_{<i}, \boldsymbol{x}_{\boldsymbol{z}_{<i}}) \tag{5}$$
			- 其中每个因子 $p_\theta(z_i|z_{<i}, x_{z_{<i}})$ 是在索引 $z_i$ 上的参数化分类分布，该索引读取自回归顺序中的下一个数据维度
				- 条件于索引 $z_{<i}$ （AO-ARM）
					- 任何一种生成顺序 $\boldsymbol{z}$ (一个排列) 都是等可能的。这等价于一个**均匀的先验** $p(\boldsymbol{z}) = \prod_{i=1}^{L} p_\theta(z_i|\boldsymbol{z}_{<i})$
					- 在第 $i$ 步，已经选择了 $i-1$ 个维度，还剩下 $L - (i-1) = L-i+1$ 个维度未被选择
					* 由于所有排列都是等可能的，那么从剩下的维度中选择任何一个的概率都是相同的
				- **条件于相应的数据维度** $x_{z_{<i}} = (x_{z_1}, \ldots, x_{z_{i-1}})$
	2. 使用变分推断训练
		- 为了从一组训练样本 $\mathcal{D} = {\boldsymbol{x}^{(n)}}_{n=1}^N$ 训练 LO-ARM 模型，去最大化对数似然
			$$\sum_{n=1}^{N} \log p_\theta(\boldsymbol{x}^{(n)}) = \sum_{n=1}^{N} \log \sum_{\boldsymbol{z}^{(n)}} p_\theta(\boldsymbol{z}^{(n)}, \boldsymbol{x}^{(n)})$$
		- 为简单起见，现在只考虑一个数据点，并省略索引 $n$。联合分布 $p_\theta(\boldsymbol{z}, \boldsymbol{x})$ 可以因式分解为：
				$$p_\theta(\boldsymbol{z}, \boldsymbol{x}) = \prod_{i=1}^{L} p_\theta(z_i|\boldsymbol{z}_{<i}, \boldsymbol{x}_{\boldsymbol{z}_{<i}})p_\theta(x_{z_i}|x_{\boldsymbol{z}_{<i}}) \tag{6}$$
			* 变量定义：
				* $\boldsymbol{z} = (z_1, ..., z_L)$：维度的生成顺序，是一个 $1, ..., L$ 的排列
				* $z_i$：在第 $i$ 步被选择生成的维度的**索引**
				* $\boldsymbol{z}_{<i}$：前 $i-1$ 步选择的维度索引集合
				* $\boldsymbol{x}_{\boldsymbol{z}_{<i}}$：前 $i-1$ 步已生成的**数据值**
				* $\boldsymbol{x}_{z_i}$：在第 $i$ 步生成的维度 $z_i$ 上的数据值
			- 解释：在每一步 $i$，生成过程包含两个动作：
			    1.  **$p_\theta(z_i | \boldsymbol{z}_{<i}, \boldsymbol{x}_{\boldsymbol{z}_{<i}})$ (顺序策略, Order-Policy)**：这是模型的核心创新。它是一个策略网络，负责“决策”。它观察已经选择过的位置 $\boldsymbol{z}_{<i}$ 和已经生成的值 $\boldsymbol{x}_{\boldsymbol{z}_{<i}}$，然后从**剩余的、未被选择**的位置中，决定下一步应该生成哪一个位置 $z_i$
			    2. **$p_\theta(\boldsymbol{x}_{z_i} | \boldsymbol{x}_{\boldsymbol{z}_{<i}})$ (数据分类器, Classifier)**：这是标准的自回归预测。一旦顺序策略决定了要生成的位置 $z_i$，这个分类器就负责预测该位置的具体值 $\boldsymbol{x}_{\boldsymbol{z}_i}$
		- 目标：最大化边缘对数似然 $\log p_\theta(\boldsymbol{x}) = \log \sum_\boldsymbol{z} p_\theta(\boldsymbol{x}, \boldsymbol{z})$。由于需要对所有 $L!$ 种顺序 $\boldsymbol{z}$ 求和，这个目标是难以直接计算的。借由变分推断，引入一个**近似后验分布** $q_\theta(\boldsymbol{z}|\boldsymbol{x})$，来近似真实的后验 $p_\theta(\boldsymbol{z}|\boldsymbol{x})$
			$$q_\theta(\boldsymbol{z}|\boldsymbol{x}) = \prod_{i=1}^L q_\theta(z_i | \boldsymbol{z}_{<i}, \boldsymbol{x}) \tag{7}$$
		- ELBO 推导：
			$$
				\begin{align}
				\log p_\theta(x) & \ge \sum_z q_\theta(z|x) \log \frac{p_\theta(z, x)}{q_\theta(z|x)} \quad (\text{ELBO 定义}) \\[5pt]
				& = \sum_z q_\theta(z|x) \sum_{i=1}^L \log \frac{p_\theta(z_i | z_{<i}, x_{z_{<i}}) p_\theta(x_{z_i} | x_{z_{<i}})}{q_\theta(z_i | z_{<i}, x)} \\[5pt]
				& = \sum_{i=1}^L \mathbb{E}_{q_\theta(z_{<i}|x)} \left[ \textcolor{blue}{\mathbb{E}_{q_\theta(z_i|z_{<i}, x)} \left[ \log \frac{p_\theta(z_i | z_{<i}, x_{z_{<i}}) p_\theta(x_{z_i} | x_{z_{<i}})}{q_\theta(z_i | z_{<i}, x)} \right]} \right] \\[5pt]
				& = \sum_{i=1}^L \mathbb{E}_{q_\theta(z_{<i}|x)} [ \textcolor{blue}{F_\theta(z_{<i}, x)}] \tag{8}
				\end{align}
			$$
			- 推导过程：
			    1. 第一行是 ELBO 的标准定义，利用了 Jensen 不等式
				2. 第二行将公式 (6) 和 (7) 的分解形式代入
				3. 第三行是关键的数学变换。它利用了期望的线性性质和链式法则，将对整个序列 $z$ 的期望，分解为对每个时间步 $i$ 的期望的加和。具体来说，外层期望 $\mathbb{E}_{q_\theta(z_{<i}|x)}$ 是对前 $i-1$ 步的顺序进行采样，内层期望 $\mathbb{E}_{q_\theta(z_i|z_{<i}, x)}$ 是在给定前 $i-1$ 步的基础上，对第 $i$ 步的维度选择进行采样
				4. 第四行用 $F_\theta(z_{<i}, x)$ 简化了内层期望的表示
		- 最终损失函数
			- 边缘化掉所有未来的潜变量 $z_{>i} = \{1, \ldots, L\} \setminus z_{<i}$，因为函数 $F_\theta(\boldsymbol{z}_{<i}, \boldsymbol{x})$ 不依赖于这些。计算完整的 ELBO 代价太高，因此在实践中我们通过在总和 $\sum_{i=1}^{L}$ 中采样一项以及一个 $\boldsymbol{z}_{<i} \sim q_\theta(\boldsymbol{z}_{<i}|\boldsymbol{x})$ 来构造一个无偏估计：
				$$\mathcal{L}(\theta) = -LF_\theta(\boldsymbol{z}_{<i}, \boldsymbol{x}), \quad \boldsymbol{z}_{<i} \sim q_\theta(\boldsymbol{z}_{<i}|\boldsymbol{x}) \tag{9}$$
		- 与 AO-ARM 的关系：
			$$\mathcal{L}(\theta) = -\frac{L}{L-i+1} \sum_{z_i \in \boldsymbol{z}_{\ge i}} \log p_\theta(x_{z_i} | \boldsymbol{x}_{\boldsymbol{z}_{<i}}) \tag{10}$$
			- 解释：这个公式旨在说明，LO-ARM 的训练目标是一个更**通用 (general)** 的框架。当特定的简化假设成立时，它会**退化 (degenerate)** 为标准的 AO-ARM 目标
			- **关键假设**：这个退化发生的前提是：
			    1.  变分策略 $q_\theta(z_i | z_{<i}, x)$ 是一个**固定的均匀分布**，即 $q_\theta(z_i | \dots) = \frac{1}{L-i+1}$
			    2.  模型顺序策略 $p_\theta(z_i | z_{<i}, x_{z_{<i}})$ 也是一个**固定的均匀分布**
			- 推导：
			    * 回顾 ELBO 中的核心项 $F_\theta(z_{<i}, x) = \mathbb{E}_{q_\theta(z_i|z_{<i},x)} \left[ \log p_\theta(x_{z_i} | x_{z_{<i}}) - \log \frac{q_\theta(z_i | z_{<i}, x)}{p_\theta(z_i | z_{<i}, x_{z_{<i}})} \right]$
			    * 在上述假设下，$\log \frac{q_\theta(\dots)}{p_\theta(\dots)} = \log \frac{1/(L-i+1)}{1/(L-i+1)} = \log(1) = 0$。策略部分的 KL 散度项消失了
			    * 因此，$F_\theta$ 只剩下分类器项的期望：$F_\theta(z_{<i}, x) = \mathbb{E}_{z_i \sim \text{Uniform}}[\log p_\theta(x_{z_i} | x_{z_{<i}})]$
			    * 这个期望是对所有 $L-i+1$ 个剩余维度的均匀平均，即 $\frac{1}{L-i+1} \sum_{z_i \in \boldsymbol{z}_{\ge i}} \log p_\theta(x_{z_i} | x_{z_{<i}})$
			    * 代入随机损失 $\mathcal{L}(\theta) = -L \cdot F_\theta(\dots)$，就得到了公式(10)的形式
			* 意义：当**放弃学习顺序**（即让所有顺序选择都变成纯随机）时，LO-ARM 的学习信号就完全来自于“在任意给定的部分数据下，预测所有剩下部分”的能力，这正是 AO-ARM 的训练方式。这证明了 LO-ARM 是对 AO-ARM 的一个严格推广
		- 带来的问题：相较于 AO-ARMs，损失函数中的随机目标函数更为复杂，因此具有**更高的方差**
		- 解决方法：RLOO 梯度估计
			- 由于需要从离散的 $q_\theta$ 中采样路径 $\boldsymbol{z}_{<i}$，必须使用 REINFORCE 算法来估计 $\theta$ 的梯度。但朴素的 REINFORCE 算法方差极大，会导致训练极其不稳定。RLOO（REINFORCE Leave-One-Out） 是一种有效的方差缩减技巧
			- 首先选择一个单一的随机索引 $i \sim \text{Uniform}[1, \ldots, L]$。对于这个固定的 $i$
				- **加入基线 (Baseline)**： 梯度变为 $\nabla_\theta \log q_\theta(\text{路径}) \times (\text{奖励} - \text{基线})$。如果基线接近奖励的平均值，那么乘积项就会变小，梯度方差也随之减小
				- **用另一个样本的奖励作为当前样本的基线**
				    * 对于样本路径 $z_{<i}^1$，它的基线就是 $F_\theta(z_{<i}^2, x)$
				    * 对于样本路径 $z_{<i}^2$，它的基线就是 $F_\theta(z_{<i}^1, x)$
				$$\frac{L}{2} \left\{ (\nabla_\theta \log q_\theta(z_{<i}^1|x) - \nabla_\theta \log q_\theta(z_{<i}^2|x)) \Delta F + \nabla_\theta F_\theta(z_{<i}^1, x) + \nabla_\theta F_\theta(z_{<i}^2, x) \right\} \tag{11}$$
				- 其中 $\Delta F = F_\theta(z_{<i}^1, x) - F_\theta(z_{<i}^2, x)$ 是两条路径奖励值的差
	3. 参数化的分布
		1. 变分分布
			$$q_\theta(z_i = k | z_{<i}, x) = \frac{e^{g_{\theta,k}(x)}}{\sum_{k' \in \boldsymbol{z}_{\ge i}} e^{g_{\theta,k'}(x)}} \tag{12}$$
			- 解释：定义在训练时，如何从一个完整数据点 $x$ **反向推断**出“最优”的生成顺序
				- Plackett-Luce 模型：这个形式称为 Plackett-Luce。它的一个巨大优势是，我们只需要用网络一次性计算出所有维度的logits 向量 $g_\theta(x) \in \mathbb{R}^L$。之后，自回归地采样整个路径 $z=(z_1, \dots, z_L)$ 就变得非常高效，只需要不断地对剩余 logits 做softmax 采样即可
			    * $g_{\theta,k}(x)$：一个神经网络，为每个维度 $k$ 输出一个代表其生成优先级的 logit。这个 logit 是基于对整个 $x$ 的分析得出的
			- 两种实现 $g_\theta$ 的方式：
			    - **共享躯干 (Shared-torso)**：在主干网络上再加一个输出头，用于输出 $g_\theta$ 的logits
			    - **独立网络 (Separate NN)**：使用一个完全独立的神经网络来参数化 $q_\theta$。这给予了变分分布更大的灵活性和容量，使其能更好地近似真实的后验。论文的实验结果（`st-sep`模型效果最好）表明，这种额外的灵活性对于提升模型性能至关重要
		2. 分类器
			$$p_\theta(x_{z_i}=k | \boldsymbol{x}_{\boldsymbol{z}_{<i}}) = \text{softmax}(f_{\theta,k}(\bar{\boldsymbol{x}}_{z_{<i}})) \tag{13}$$
			- 解释：定义如何从部分生成的数据中，预测下一个维度的具体值
				* $x_{z_{<i}}$：这是一个部分生成的分子图（一些节点和边有确定类型，其余被mask）
			    * $f_{\theta,k}(\cdot)$：这是一个大型神经网络（如 GNN）的 **第 k 个输出头**。它专门负责预测**第 k 个维度**（比如第3个原子，或第(2,4)条边）的类型。输入是部分分子图，输出是该维度所有可能类型的 logits
			    * $\text{softmax}$：将 logits 转换为标准的概率分布
		3. 模型顺序策略
			$$p_\theta(z_i = k | z_{<i}, x_{z_{<i}}) = \frac{e^{h_{\theta,k}(x_{z_{<i}})}}{\sum_{k' \in \boldsymbol{z}_{\ge i}} e^{h_{\theta,k'}(x_{z_{<i}})}} \tag{14}$$
			- 解释：定义生成时如何决策下一步生成哪个维度
				* $h_{\theta,k}(\cdot)$：这是一个神经网络，它为**选择第 k 个维度**这个动作打分。分数越高，代表模型认为下一步生成第 k 个维度越好
			    * $\sum_{k' \in \boldsymbol{z}_{\ge i}}$：softmax 的分母**只对当前还未生成的维度** $(k' \in \boldsymbol{z}_{\ge i})$ 进行归一化
			* **两种实现 $h_{\theta,k}$ 的方式:**
			    - **基于熵 (Entropy-based)**： 这是一个引入归纳偏置的巧妙方法。$h_{\theta,k}$ 的值不直接由网络输出，而是由分类器 $p_\theta(x_k|x_{z_{<i}})$ 的**信息熵**决定，通常是 $h_{\theta,k} \propto -\text{Entropy}(\dots)$。这意味着模型被鼓励**优先生成那些它最有把握的维度**（熵越低，不确定性越小，得分越高）
				- **共享躯干 (Shared-torso)**： 这是论文中更主流的做法。与分类器 $f_\theta$ 共享同一个网络主干，但在最后引出一个额外的线性层作为“策略头”，直接输出 $h_{\theta,k}$ 的 logits。这让模型完全从数据中学习决策逻辑

