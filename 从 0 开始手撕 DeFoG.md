1. 超参数与各类配置
	- 通过 Hydra 和 OmegaConf 进行配置
2. 数据集加载 —— 以 QM9_no_H 为例
	- 暂时空着
3. 额外图特征提取 —— extra_features
	- 目的：为图神经网络补充结构特征 —— 代码中给出了七种方式，默认采用的是 `rrwp`，针对不同的分子集的特征应该选用不同的方式
		1. `cycles`：仅使用循环特征 (k-cycles)
		2. `eigenvalues`：使用图拉普拉斯矩阵的特征值
		3. `rrwp`：随机游走核心特征（Regularized Random Walk Kernels）
		4. `rrwp_double`：RRWP 特征 + 非归一化版本
		5. `rrwp_only`：仅 RRWP（不含循环特征）
		6. `rrwp_comp`：RRWP 补集版本（考虑图的补图）
		7. `all`：所有特征的组合（循环 + 特征值 + 特征向量）
	- 补充的特征
		1. 图的最大节点数
		2. 循环特征计算
			- 概念：
				1. k-cycles：**k-圈（k-cycle）** 是指长度为 k 的环，即由 k 个节点和 k 条边组成的闭合路径
			- 步骤：
		3. 提取特征的方式 —— 对应 `rrwp`
		4. `rrwp` 的步数
		5. 归一化的随机游走核
		6. 非归一化的随机游走核