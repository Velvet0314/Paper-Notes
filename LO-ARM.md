1. 背景知识
	1. 问题定义
		- 假设数据向量 $\boldsymbol{x} = (x_1, \ldots, x_L)$,其中每个维度 $x_i$ 从集合 $\mathcal{X}$ 中取值,该集合可以是实数或离散的。在不失一般性的情况下，假设 $\boldsymbol{x}$ 是一个包含 $L$ 个分类变量的向量，因此 $\mathcal{X}$ 是一个包含 $m = |\mathcal{X}|$ 个类别的离散集合
		- 对于分子图，$\mathcal{X} = \mathcal{V} \cup \mathcal{E}$，其中 $\mathcal{V}$ 和 $\mathcal{E}$ 分别是原子和键的类型。一个自回归模型是 $x$ 上的一个联合概率分布，其因式分解为：
			$$p_\theta(x) = \prod_{i=1}^{L} p_\theta(x_i | x_{<i}) \tag{1}$$
			- 其中 $x_i$ 表示 $\boldsymbol{x}$ 的第 $i$ 个维度，$x_{<i} = (x_1, \ldots, x_{i-1})$ 表示前 $i-1$ 个元素的向量，$x$ 和 $p_\theta(x_i | x_{<i})$ 是条件分布，约定为 $p_\theta(x_1 | x_{<1}) = p_\theta(x_1)$
		- 从模型中采样会按顺序生成数据维度，从 $x_1$ 开始到 $x_L$ 结束。拥有固定或预先指定的顺序通常会产生次优结果，并且在对没有自然顺序的数据建模时可能引入不适当的归纳偏差
		- AO-ARMs 通过训练一个可以在从索引 $\{1, \ldots, L\}$ 的 $L!$ 个排列中均匀抽取的随机顺序下生成数据维度的模型来解决这个问题。给定一个排列 $\sigma$，模型的联合分布因式分解为：
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
			$$p_\theta^x(\boldsymbol{z}) = \prod_{i=1}^{L} p_\theta(z_i|\boldsymbol{z}_{<i}, \boldsymbol{x}_{\boldsymbol{z}_{<i}}) \tag{5)}$$
			- 其中每个因子 $p_\theta(z_i|z_{<i}, x_{z_{<i}})$ 是在索引 $z_i$ 上的参数化分类分布，该索引读取自回归顺序中的下一个数据维度，不仅条件于索引 $z_{<i}$ 而且条件于相应的数据维度 $x_{z_{<i}} = (x_{z_1}, \ldots, x_{z_{i-1}})$
	2. 

