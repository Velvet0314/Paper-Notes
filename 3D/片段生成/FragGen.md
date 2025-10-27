1. 蛋白质-配体相互作用学习模块
	$$\begin{aligned} \left(n^\prime_{p_i}, \vec{n_{p_i}}^\prime \right) &= \text{Emb}(n_{p_i}, \vec{n_{p_i}}) \\[6pt] \left(n^\prime_{l}, \vec{n_{l_i}}^\prime \right) &= \text{Emb}(n_{l_i}, \vec{n_{p_i}}) \\[6pt] \left(h_i, \vec{h_i} \right) &= \text{GeomEncoder}(n_{l_i}, n_{p_i}, \vec{n_{l_i}}, \vec{n_{p_i}}, e_{ij}, \vec{e_{ij}})\end{aligned}$$
	* **变量定义:**
	    * $n_p, n_l$：蛋白质和配体的结点特征
	    * $\vec{n_p}, \vec{n_l}$：蛋白质和配体的矢量特征
	    * $e_{ij}, \vec{e_{ij}}$：结点 $i, j$ 之间边的标量和矢量特征
	    * $\text{Emb}$：一个嵌入层，将蛋白质和配体的原始特征映射到相同维度的高维空间
	    * $\text{GeomEncoder}$：一个几何等变图神经网络（Geometric Equivariant GNN）
	    * $h_i, \vec{h_i}$：蛋白质-配体图的隐藏特征
2. 前沿预测（Frontier prediction）
	$$\begin{aligned} (n_{f_i}, \vec{n_{f_i}}) &= f\left( \text{SL}_{f_1}(h_i), \text{VL}_{f_1} \left(\vec{h_i} \right) \right) \\[6pt] p_{f_i} &= \sigma\left(\text{SL}_{f_3} \left(||\vec{n_{f_i}}||_2 + f(\text{SL}_{f_2}(n_{f_i}))\right) \right) \end{aligned}$$
	- **变量定义：**
		- $p_{f_i}$：表示结点 $i$ 作为前沿的概率
		- $\sigma$：为 sigmoid 函数
		- SL 和 VL 分别表示标量层（scalar layers）和向量层（vector layers）
		- $n_{f_i}, \overrightarrow{n_{f_i}}$ 为中间的标量与向量特征
	- **什么是前沿（Frontier）？**
		- 模型下一步应该**从哪里开始添加新的化学片段**
3. 腔体检测（Cavity detection）—— 采用**混合密度网络 (Mixture Density Network, MDN)**
	$$\begin{aligned} \left(r_i, \vec{r_i}\right) &= \text{GVP}_r\left(\text{SL}_{x1}(h_i), \text{VL}_{x1}\left(\vec{h_i}\right)\right) \\ \left(w_i, \vec{w_i}\right) &= \text{GVP}_w\left(\text{SL}_{x2}(h_i), \text{VL}_{x2}\left(\vec{h_i}\right)\right) \\ \left(\Sigma_i, \vec{\Sigma_i}\right) &= \text{GVP}_{\Sigma}\left(\text{SL}_{x3}(h_i)\\ \text{VL}_{x3}\left(\vec{h_i}\right)\right) \end{aligned}$$
	- **变量定义：**
		- $\text{GVP}$: Geometric Vector Perceptron，一个能够处理标量和矢量数据并保持**SE(3)等变性**的神经网络层
		- $(r_i,\vec{r}_i)$: 第 $i$ 个高斯分量的**均值**。$\vec{r}_i$ 是一个3D**相对位置矢量**，表示该高斯分量中心相对于前沿原子位置的位移（方向和距离）。$r_i$ 是与之相关的标量输出
	    * $(w_i, \vec{w}_i)$: 第 $i$ 个高斯分量的**权重**。$w_i$ 是一个标量，表示**混合系数 (mixing coefficient)**。它量化了该分量在整个混合分布中所占的比重
	    * $(\Sigma_i, \vec{\Sigma}_i)$: 第 $i$ 个高斯分量的**方差**。$\Sigma_i$ 是一个标量，用于描述该分布的扩展程度或不确定性
    $$\vec{x_i} = \vec{x_{ai}} + \sum_{k=1}^{K} w_i^k \vec{r_i}^k$$
	- **变量定义：**
	    * $\vec{x}_i$：一个三维矢量，代表该模块的最终输出 —— 预测的腔体中心的**绝对坐标 (absolute coordinate)**
	    * $\vec{x}_{ai}$：一个三维矢量，代表前沿原子 $a$ 在当前步骤 $i$ 的**绝对坐标**。这是所有相对计算的**参考原点**
	    * $K$：一个整数超参数，表示高斯混合模型中**分量的总数**
	    * $w_i^k$：第 $k$ 个高斯分量的**归一化权重**。它是通过对所有 $K$ 个未归一化权重 $w_i$ 应用 Softmax 函数得到的，确保 $\sum_{k=1}^{K} w_i^k = 1$
	    * $\vec{r}_i^k$：GVP 网络为第 $k$ 个高斯分量预测的**相对位置矢量 (均值)**
	- 这是 FragGen 几何处理协议的**第一步关键创新**。模型不是直接预测新原子的坐标，而是预测一个目标“区域”
4. 片段查询（Fragment query）
	$$\begin{aligned} y_{m_{ij}}, \vec{y}_{m_{ij}} &= \text{GeomMessage}\left(h_i, \vec{h_i}, c_{ij}, \vec{c}_{ij}\right) \\ y_{h_i}, \vec{y}_{h_i} &= \sum_{k=1}^{i}\left(y_{m_{ik}}, \vec{y}_{m_{ik}}\right) \\ p_{y_i} &= \sigma\left(\text{SL}_{t2}\left(\|\vec{y}_{h_i}\|_2 + f\left(\text{SL}_{t1}\left(y_{h_i}\right)\right)\right)\right) \end{aligned}$$
	- **变量定义：**
		- $y_{m_{ij}}, \vec{y}_{m_{ij}}$：结点 $i$（腔节点）和结点 $i$ 的 $K$ 个最近邻结点 $j$ 之间的消息
		- $y_{h_i}, \vec{y}_{h_i}$：腔体结点 $i$ 上的聚类类型隐藏特征
		- GeomMessage：消息块，它使腔结点 $i$ 与其口袋环境融合
		- $p_{y_i}$：**下一个片段类型的概率分布**。这是一个向量，其维度等于片段库中片段的总数
5. 片段的连接选择（Attachment selection）
	$$\begin{aligned} h_{a_i}, \vec{h}_{a_i} &= \text{GVP}_{\text{atta}}(\vec{h}_i, \tilde{h}_i) \\[8pt] h^\prime_{f_j} &= \text{GAT}(h_{f_j}, e_{f_j}) \\[8pt] y_{\text{cr}}^{\text{emb}}, y_{\text{nx}}^{\text{emb}} &= \text{Embed}(y_{\text{cr}}, y_{\text{nx}}) \\[8pt] h^\prime_{a_j} &= (h^\prime_{f_j} \ || \ y_{\text{cr}}^{\text{emb}} \ || \ y_{\text{nx}}^{\text{emb}} \ || \ h_{a_i}) \\[8pt] p_{a_j} &= \sigma(\text{MLP}(h^\prime_{a_j}))\end{aligned}$$
	- **变量定义：**
		- $h_{a_i}, \vec{h}_{a_i}$：第 $i$ 个结点所连接原子的隐藏特征，即选中的前沿原子 —— 提取3D几何信息
		- $h^\prime_{f_j}. e_{f_j}$：新片段中原子 $j$ 的初始原子特征和它参与的化学键（边）的特征
		- GAT：图注意力网络 (Graph Attention Network) —— 片段内部的化学性质主要由其 2D 拓扑决定
		- $y_{\text{cr}}, y_{\text{nx}}$：当前（current）片段和下一个（next）片段的类型
		- $||$：拼接操作
		- $h^\prime_{a_j}$：为新片段中的原子 $j$ 构建的、用于最终决策的综合特征向量
		- $p_{a_j}$：原子 $j$ 被选为连接点的最终概率
6. 形成化学键连接（Bond linking）
	$$\begin{aligned} h_{b_i}, \vec{h}_{b_i} &= \text{GVP}_{\text{bond}}(h_i, \vec{h}_i) \\[8pt] h_{d_{ij}}, h_{n_x} &= \text{MLP}(d_{ij}, n_{nx}) \\[8pt] y_{\text{cr}}^{\text{emb}}, y_{\text{nx}}^{\text{emb}} &= \text{Embed}(y_{\text{cr}}, y_{\text{nx}}) \\[8pt] h_{\text{valen}} &= \text{MLP}(\text{valen}_{\text{cr}} \ || \ \text{valen}_{\text{nx}}) \\[8pt] p_{b_{ij}} &= \sigma(\text{MLP}(h_{b_i} h_{d_{ij}} \ || \ y_{\text{cr}}^{\text{emb}} \ || \ y_{\text{nx}}^{\text{emb}} \ || \ h_{\text{valen}})) \end{aligned}$$
	- **变量定义：**
		- $h_{b_i}, \vec{h}_{b_i}$：经过 GVP 处理后，专门用于键预测任务的**前沿原子的特征**
		- $d_{ij}$：前沿结点 $i$ 和空腔结点 $j$ 之间的欧氏距离
		- $n_{nx}$：下一个要连接的片段的连接原子 $j$ 的原子
		- $\text{valen}_{\text{cr}}, \text{valen}_{\text{nx}}$：当前原子 $i$ 和下一个原子 $j$ 的**当前价态**（即已经形成的价键数）
		-  $p_{b_{ij}}$：原子 $i$ 和 $j$ 之间形成某种特定类型化学键的概率向量
7. 化学初始化（Chemical initialization）
	- 下一个片段的几何形状可以分为四个组成部分
		1. 前沿原子预测（模型下一步应该**从哪里开始添加新的化学片段**）
		2. 片段查询（模型下一步应该**添加的片段是什么**）
		3. 片段的连接选择（**添加的片段上哪个点和现在的分子进行结合**）
		4. 预测化学键连接（两个连接的点间形成的**化学键是什么**）
	- 目标：确定新片段的==**整体朝向 (Overall Orientation)**==
	- 方法：对齐两个 3D 向量 —— **罗德里格旋转公式**，通过将 $\mathbf{a}$ 旋转到 $\mathbf{b}$ 实现目标
		- 向量 $\mathbf{a}$：**在片段的局部坐标系中**，存在一个理想的成键向量。例如，对于一个 sp3 杂化的碳原子，它有四个呈四面体构型的理想成键方向。向量 $\mathbf{a}$ 指的就是新片段连接原子上，那个**即将用于形成新化学键的、理想化的“价键向量”**。这个向量是固化在片段的初始几何构象中的==（待考量）==
		- 向量 $\mathbf{b}$：起点是“前沿原子 (focal atom)，终点是一个指定的口袋节点 (designated pocket node)，即模块中预测出的**腔体中心**
	- 归一化
	$$\mathbf{a}_{\text{norm}} = \frac{\mathbf{a}}{||\mathbf{a}||} \quad , \quad \mathbf{b}_{\text{norm}} = \frac{\mathbf{b}}{||\mathbf{b}||}$$
	- 计算旋转轴和旋转角
		$$\begin{aligned}\mathbf{v} &= \mathbf{a}_{\text{norm}} \times \mathbf{b}_{\text{norm}} \\ c &= \mathbf{a}_{\text{norm}} \cdot \mathbf{b}_{\text{norm}} \end{aligned}$$
		- **变量定义：**
			- $\mathbf{v}$ 是旋转轴
			- $c = cos(\theta)$ 得到旋转角
	- 构建旋转矩阵
		$$\begin{aligned} \left[\mathbf{v}\right]_{x} &= \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix} \\[10pt] \mathbf{R}_{ab} &= \mathbf{I} + [\mathbf{v}]_{x} + [\mathbf{v}]_{x}^2 \frac{1}{1+c} \end{aligned}$$
		- 对于任何向量 $\mathbf{p}$，计算叉乘 $\mathbf{v} \times \mathbf{p}$ 的结果与矩阵 $\left[\mathbf{v}\right]_{x} \cdot \mathbf{p}$ 的结果相同
		- 公式 $\mathbf{R}_{ab}$ 详细见 [[Frag2Seq 数学推导]]
		$$\mathbf{r}_f^\prime = \mathbf{R}_{ab} \mathbf{r}_f$$
8. 二面角处理（Dihedral handling）
	- 目标：**通过绕新形成的化学键进行旋转，来微调其姿态，以达到能量最优、与环境最匹配的构象**
	$$\begin{aligned} (h^{\text{tor}}_i, \vec{h}^{\text{tor}}_i) &= \text{GeomEncoder}(n_l, n_p, \vec{n}_l, \vec{n}_p, e_{ll,pp,pl}\vec{e}_{ll,pp,pl}) \\[8pt] h_{\text{mol}} &= \sum_{i=1}^{N} h^{\text{tor}}_i \\[8pt] \theta &= \text{MLP}(h_a||h_b ||h_{mol}) \\[8pt] r_f^{\prime \prime} &= \mathbf{R}(\mathbf{u}, \theta)r_f^\prime \end{aligned}$$
	- **变量定义：**
		- $h^{\text{tor}}_i, \vec{h}^{\text{tor}}_i$：分别是配体和蛋白质的结点和边特征，$ll,pp,pl$ 表示配体内部、蛋白质内部以及它们之间的边
		- $h_{\text{mol}}$：配体特征总和
		- $h_a, h_b$：前沿原子和新片段的连接原子的特征
		- $\mathbf{u}$：**旋转轴**。在这里，它就是新形成的化学键的方向向量（即从**前沿原子 $a$** 指向**新片段的连接原子 $b$** 的单位向量）
	    * $\theta$：上一步 MLP 预测出的**旋转角度**
9. 损失函数
	- 前沿原子预测损失 (Frontier Atom Prediction Loss)
	$$- \frac{1}{n} \sum_{i=1}^{n} (f_i \cdot \log p_{f_i} + (1-f_i) \cdot \log(1-p_{f_i}))$$
		* **类型**：**二元交叉熵损失 (Binary Cross-Entropy Loss)**
		* **变量定义：**
		    * $n$：当前分子中所有原子的总数
		    * $f_i$：真实标签。如果原子 $i$ 是正确的前沿原子，则 $f_i=1$，否则 $f_i=0$
		    * $p_{f_i}$：模型预测原子 $i$ 是前沿原子的概率
	- 连接原子预测损失 (Attachment Atom Prediction Loss)
	$$- \frac{1}{m} \sum_{j=1}^{m} (a_j \cdot \log p_{a_j} + (1-a_j) \cdot \log(1-p_{a_j}))$$
		* **类型**：**二元交叉熵损失**
		* **变量定义：**
		    * $m$：新片段中所有原子的总数
		    * $a_j$：真实标签。如果新片段中的原子 $j$ 是正确的连接原子，则 $a_j=1$，否则 $a_j=0$
		    * $p_{a_j}$：模型预测原子 $j$ 是连接原子的概率
	- 腔体位置预测损失 (Cavity Location Prediction Loss)
	$$- \log \sum_{k=1}^{K} w_i^{(k)} \mathcal{N}(\mathbf{x}^{(k)} | \mathbf{r}_{a_i}, \mathbf{\Sigma}_i^{(k)})$$
		* **类型**：**混合密度网络 (MDN) 的负对数似然损失 (Negative Log-Likelihood Loss)**
		* **变量定义：**
		    * $K$：混合高斯分布的分量数
		    * $w_i^{(k)}$：模型预测的第$k$个高斯分量的权重
		    * $\mathcal{N}(\cdot | \mu, \Sigma)$：均值为$\mu$、协方差为$\Sigma$的高斯概率密度函数
		    * $\mathbf{x}^{(k)}$：真实的待预测目标向量（从前沿原子指向真实腔体位置的向量）
		    * $\mathbf{r}_{a_i}, \mathbf{\Sigma}_i^{(k)}$：模型预测的第$k$个高斯分量的均值和协方差
	- 片段类型与化学键类型损失
	$$- \sum_i y_i \log p_{y_i} - \sum_j b_{ij} \log p_{b_{ij}}$$
	* **类型**：**分类交叉熵损失 (Categorical Cross-Entropy Loss)**
	* **变量定义：**
	    * $y_i, p_{y_i}$：片段类型的 one-hot 真实标签和模型预测的概率分布
	    * $b_{ij}, p_{b_{ij}}$：化学键类型的 one-hot 真实标签和模型预测的概率分布
	- 二面角预测损失 (Dihedral Angle Prediction Loss)
	$$- \log \left( \frac{e^{\kappa \cos(\theta - \mu)}}{2\pi I_0(\kappa)} \right)$$
	* **类型**: **冯·米塞斯分布 (von Mises distribution) 的负对数似然损失**
	* **变量定义：**
	    * $\mu$：真实的二面角
	    * $\theta$：模型预测的二面角
	    * $\kappa$：**集中度参数 (concentration parameter)**，也由模型预测。它控制了分布的“胖瘦”。$\kappa$越大，分布越尖锐，表示模型对自己的预测越有信心
	    * $I_0(\kappa)$：零阶修正贝塞尔函数，作为归一化常数
	*   **直观解释：**
	    这是处理**周期性数据**（如角度）的标准和最佳方法
	    * **为什么不用MSE？** 均方误差 $(\theta-\mu)^2$ 会错误地认为 $1^\circ$ 和 $359^\circ$ 相差很远，但实际上它们只差 $2^\circ$
	    * **von Mises 如何工作？** 它的核心是 $\cos(\theta - \mu)$。当预测值 $\theta$ 和真实值 $\mu$ 完全相同时，$\cos(0)=1$，损失最小。当它们相差180度时，$\cos(180^\circ)=-1$，损失最大。它完美地捕捉了角度的周期性
	    * **$\kappa$ 的作用？** 允许模型学习预测的不确定性。对于一些非常灵活、可以自由旋转的键，模型可以学习预测一个较小的 $\kappa$，得到一个平坦的分布。对于一个有很强空间位阻的键，模型可以学习预测一个大的 $\kappa$，得到一个尖锐的分布