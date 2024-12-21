## 项目进度

李青阳：TargetDiff论文阅读一遍+TargetDiff代码阅读
	1. 训练部分代码 train_diffusion.py
	2. 主要模型代码 molopt_score_model.py —— 算法部分对应的代码 
	3. 数据输入代码 transform.py
	4. 评估指标  evaluation_diifusion.py
	5. 能够进行模型训练、单步调试
王琪皓：深度学习课程 —— softmax回归
潘若溪：Python入门 —— 基础语法学习

## 后期规划

进度提速
- 预计是先快速完成TargetDiff的代码阅读和实验
- 然后进行24ICLR Protein-Ligand Interaction Prior for Binding-aware 3D Molecule Diffusion Models的代码阅读和实验

## TargetDiff 的目的

- 靶标感知分子生成
- 生成分子的亲和力预测 —— 生成分子的质量评估

## TargetDiff 理论
### TargetDiff 原理简述

#### 生成模型

- 学习一个分布 distribution —— 如何学习？
	- 已知一个简单的分布（高斯分布、均匀分布...），从中采样（sample）$z$
	- 利用 $Network\ G$ 将简单分布映射到一个复杂分布
	- 生成样本 $G(z)=y$，$y$ 近似于复杂分布（我们无法从复杂分布中直接采样）
- 利用学习到的分布，从中采样得到结果

#### DDPM 

##### Denoising Diffusion Probabilistic Models 去噪扩散概率模型

##### 核心思想

- **前向扩散过程（Forward Diffusion Process）**
	- 逐步加噪，记录噪声和中间产物，训练网络预测噪声 
- **反向去噪过程（Reverse Denoising Process）**
	- 从纯噪声开始，逐步去噪，恢复出想要的目标数据

### TargetDiff 在干什么 —— 训练算法流程

1. 输入：蛋白质-配体的结合数据集
2. 扩散条件初始化：采样时间步 —— 从均匀分布 $U(0, \dots, T)$ 中采样扩散时间 $t$
3. 预处理：将蛋白质原子的质心移动到原点，以对齐配体和蛋白质的位置，确保数据在空间上的一致性
4. 加噪：网络中主要是针对 位置 $x$ 和 原子类型 $v$ 进行扰动，逐步加噪
	- $x_t = \sqrt{\bar{\alpha}_t} x_0 + (1 - \bar{\alpha}_t) \epsilon$，其中 $\epsilon$  是从正态分布 $\mathcal{N}(0, I)$  中采样的噪声
	- $$\begin{align}log \mathbf{c} &= \log \left( \bar{\alpha}_t \mathbf{v}_0 + \frac{(1 - \bar{\alpha}_t)}{K} \right) \\ \mathbf{v}_t &= \text{one\_hot} \left( \arg \max_i [g_i + \log c_i] \right), \text{ where } g \sim \text{Gumbel}(0, 1)\end{align}$$
5. 预测：$[\hat{x}_0,\hat{v}_0]=\phi_\theta([xt, vt], t, \mathcal{P})$ ，预测扰动位置和类型，即 $\hat{x}_0$  和 $\hat{v}_0$ ，条件是当前的 $x_t$、$v_t$、时间步 $t$ 和蛋白质信息 $\mathcal{P}$
6. 计算后验类型分布：根据公式计算原子类型的后验分布 $c(v_t, v_0)$ 和 $c(v_t, \hat{v}_0)$
7. 损失函数：
	- 均方误差 MSE：度量原子坐标的偏差
	- KL 散度（KL-divergence）：度量类型分布的差异
8. 更新参数： 最小化损失函数 $L$  来更新模型参数 $\theta$
![[train_algorithm.png]]
### TargetDiff 在干什么 —— 采样算法流程

1. 输入：蛋白质结合位点（binding site）$\mathcal{P}$ 与 训练好的模型 $\phi_\theta$
2. 输出：由模型生成的能与蛋白质口袋结合的配体分子 $\mathcal{M}$
3. 确定原子数量：基于口袋大小，从一个先验分布中采样一个生成的配体分子的原子数量
4. 预处理：移动蛋白质原子的质心至坐标原点，使位置标准化，以确保生成的配体与蛋白质结合位点对齐
5. 初始化：采样一个初始的原子坐标（coordinates）$\mathbf{x}_T$ 和 原子类型 $\mathbf{v}_T$
	- $\mathbf{x}_T \in \mathcal{N}(0,\boldsymbol{I})$ —— 从标准正态分布 $\mathcal{N}(0,\boldsymbol{I})$ 中采样
	- $\mathbf{v}_T = \text{one\_hot} \left( \arg \max_i g_i \right), \text{ where } g \sim \text{Gumbel}(0, 1)$
- $\textbf{for}\ t\ \text{in}\ T,T-1,\cdots,1\ \textbf{do}$ （反向去噪）
6. 预测：$[\hat{x}_0,\hat{v}_0]=\phi_\theta([xt, vt], t, \mathcal{P})$ ，预测扰动位置和类型，即 $\hat{x}_0$  和 $\hat{v}_0$ ，条件是当前的 $x_t$、$v_t$、时间步 $t$ 和蛋白质信息 $\mathcal{P}$
7. 根据后验分布 $p_\theta(x_{t-1} | x_t, \hat{x}_0)$ 对原子位置 $\mathbf{x}_{t-1}$进行采样
8. 根据后验分布 $p_\theta(v_{t-1} | v_t, \hat{v}_0)$ 对原子类型 $\mathbf{v}_{t-1}$ 进行采样
![[sample_algorithm.png]]

## TargetDiff 代码

### 代码解读：[Velvet0314/targetdiff at 4LearnOnly](https://github.com/Velvet0314/targetdiff/tree/4LearnOnly)

### 环境安装 Tips

- 推荐在 Linux 下进行环境安装（可以用 WSL） —— Vina 需要 Linux 环境
- 注意 Pytorch, Cuda, Python 的版本对应
- 需要安装对应版本的 cudatoolkit 实现 Pytorch 中利用 cuda 进行 GPU 的加速
- 我的环境在 myenvironment.yaml 中，可以跑通

### 额外内容

- test_cuda.py 用于测试 cuda 是否启用
- viewlmdb.py 用于可视化输入数据

### 训练流程



### 采样流程

### 验证流程

## 一些疑问

1. 什么是蛋白质口袋？
	蛋白质口袋指的是蛋白质表面或内部的**三维结构凹陷区域**，该区域通常是其他分子（如配体、小分子药物或离子）与蛋白质发生结合或相互作用的地方。
2. 代码
3. 数学推导
