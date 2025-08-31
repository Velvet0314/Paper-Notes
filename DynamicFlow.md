1. 背景与准备工作
	- **目标**：在给定蛋白质口袋的 **apo（游离）状态** 的前提下，生成其 **holo（配体结合）状态** 以及相应的配体分子
	- **配体表示**
		- 配体分子可以表示为一个包含 $N_m$ 个原子的集合：
			$$
              \mathcal{M} = \left\{ \left( \mathbf{x}_m^{(i)}, \mathbf{v}_m^{(i)}, \mathbf{b}_m^{(ij)} \right) \right\}_{i,h \in \{1, ..., N_m\}}
             $$
			其中：
			- $\mathbf{x}_m^{(i)} \in \mathbb{R}^3$：原子位置
			- $\mathbf{v}_m^{(i)} \in \mathbb{R}^K$：原子类型
			- $\mathbf{b}_m^{(ij)} \in \mathbb{R}^B$：化学键类型
	- **Apo 状态蛋白口袋表示**
		- **原子层级（Atom Level）**
			$$
             \mathcal{P}_0 = \left\{ \left( \mathbf{x}_{p0}^{(i)}, \mathbf{v}_{p0}^{(i)} \right) \right\}_{i=1}^{N_p}
             $$
			包含 $N_p$ 个原子的三维坐标和原子类型
		- **残基层级（Residue Level）**
			$$
              \mathcal{S}_0 = \left[ \left( \mathbf{a}_0^{(i)}, \mathbf{t}_0^{(i)}, \mathbf{r}_0^{(i)}, \chi_0^{(i)}, \chi_{10}^{(i)}, \chi_{20}^{(i)}, \chi_{30}^{(i)}, \chi_{40}^{(i)} \right) \right]_{i=1}^{D_p}
             $$
			其中：
			- $D_p$：残基数量
			- $\mathbf{a}_0^{(i)} \in \mathbb{R}^S$：氨基酸类型
			- $\mathbf{t}_0^{(i)} \in \mathbb{R}^3$：残基的平移向量
			- $\mathbf{r}_0^{(i)} \in \mathrm{SO}(3)$：残基框架旋转
			- $\chi_0^{(i)} \in \mathrm{SO}(2)$：主链扭转角
			- $\chi_{j0}^{(i)} \in \mathrm{SO}(2),\ j=1\dots4$：侧链扭转角
	- 注意事项（Notes）
		- holo 状态中所有下标由 0 变为 1
		- 蛋白构象变化不改变氨基酸类型和原子类型，即：
			$$
              \mathbf{a}_0^{(i)} = \mathbf{a}_1^{(i)},\quad \mathbf{v}_0^{(i)} = \mathbf{v}_1^{(i)}\quad \forall i
            $$
	- 建模目标
		SBDD（结构基础药物设计）任务可形式化为建模以下条件概率分布：
		$$
         q(\mathcal{P}_1, \mathcal{M} \mid \mathcal{P}_0)
         $$
        其中：
		* $\mathcal{P}_0$：Apo状态的蛋白质口袋。它由原子坐标、残基类型、骨架框架（平移和旋转）和扭转角等一系列变量描述
	    * $\mathcal{P}_1$：Holo 状态的蛋白质口袋。其结构与$P_0$类似，但坐标、框架和扭转角发生了变化
	    * $\mathcal{M}$：与 $\mathcal{P}_1$ 结合的配体分子，由原子坐标、原子类型、键类型等描述
		在已知 apo 状态下，生成 holo 状态的蛋白结构 $\mathcal{P}_1$ 与其结合的配体结构 $\mathcal{M}$
2. 流模型知识补充
	- **连续情形下的流匹配（Flow Matching for Continuous Variables）**
		- 在流匹配中，我们考虑一个**边缘概率路径** $p_t$，该路径连接初始分布 $p_0$ 和目标分布 $p_1$，训练一个生成模型使得从 $p_0$ 采样的样本 $\mathbf{x}_0 \sim p_0$ 能够被传输到 $\mathbf{x}_1 \sim q_1$
		- 定义联合分布 $\pi(\mathbf{x}_0, \mathbf{x}_1)$（也称为数据耦合 data coupling），该路径对应一个基于时间的流 $\psi_t：[0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ 和一个相关的向量场 $\mathbf{u}_t ：[0,1] \times \mathbb{R}^d \to \mathbb{R}^d$，可通过如下常微分方程（ODE）建模：
			$$
			\dot{\mathbf{x}}_t = \frac{d \mathbf{x}_t}{dt} = \mathbf{u}_t(\mathbf{x}_t), \quad \text{where} \quad \mathbf{x}_t = \psi_t(\mathbf{x}_0)
			$$
			- **变量定义**：
			    * $\mathbf{x}_t$：系统在时间 $t$ 的状态（比如所有原子的坐标）
			    * $\dot{\mathbf{x}}_t$：状态随时间变化的**速度**
			    * $\mathbf{u}_t(\mathbf{x}_t)$：一个**向量场 (vector field)**。你可以想象在空间的每一点 $\mathbf{x}_t$ 都有一个箭头 $\mathbf{u}_t$，指明了该点应该“流动”的方向和快慢
				- $\psi_t(\mathbf{x}_0)$ 表示路径
			* **直观解释:** 这个公式描述了一个**确定性的演化过程**。一旦你定义了向量场 $\mathbf{u}_t$（即流动的规则），并给定一个初始状态 $\mathbf{x}_0$，那么未来的所有状态 $\mathbf{x}_t$ 都会沿着这个向量场指定的轨迹精确地流动，最终到达 $\mathbf{x}_1$
			* **与模型的联系:** 目标是训练一个神经网络 $\mathbf{v}_\theta$ 去**模仿**这个理想的、但未知的向量场 $\mathbf{u}_t$
			我们可以使用神经网络 $\mathbf{v}_\theta(\mathbf{x}_t, t)$ 来回归建模真实向量场 $\mathbf{u}_t(\mathbf{x}_t)$
			但是，由于我们不知道生成边缘分布路径 $p_t$ 的向量场 $\mathbf{u}_t$ 的闭合形式，因此流匹配目标很难实现
		- 一种近似方式：
			- 通过学习一个条件向量场 $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$，使得其产生的条件概率路径 $p_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$ 去逼近真实的**边缘概率分布路径** （后续数学证明是精确的）
			- **边缘概率分布路径**：
				$$ 
				p_t(\mathbf{x}_t) = \int_{\mathbf{x}_0, \mathbf{x}_1}p_t(\mathbf{x}_t|\mathbf{x}_0, \mathbf{x}_1) \pi(\mathbf{x}_0, \mathbf{x}_1) d\mathbf{x}_0 d\mathbf{x}_1
				$$
				- 推导：
	                $$
		            \begin{align}
	                p_t(\mathbf{x}_t) &= \int_{\mathbf{x}_0} \int_{\mathbf{x}_1} p_t(\mathbf{x}_t, \mathbf{x}_0, \mathbf{x}_1) d\mathbf{x}_1 d\mathbf{x}_0 \\
	                &= \int_{\mathbf{x}_0} \int_{\mathbf{x}_1} p_t(\mathbf{x}_0) p_t(\mathbf{x}_1|\mathbf{x}_0) p_t(\mathbf{x}_t|\mathbf{x}_0, \mathbf{x}_1) d\mathbf{x}_1 d\mathbf{x}_0 \\
	                &= \int_{\mathbf{x}_0} \int_{\mathbf{x}_1} \pi(\mathbf{x}_0, \mathbf{x}_1) p_t(\mathbf{x}_t|\mathbf{x}_0, \mathbf{x}_1) d\mathbf{x}_1 d\mathbf{x}_0 \\
	                &= \int_{\mathbf{x}_0, \mathbf{x}_1} p_t(\mathbf{x}_t|\mathbf{x}_0, \mathbf{x}_1) \pi(\mathbf{x}_0, \mathbf{x}_1)  d\mathbf{x}_0 d\mathbf{x}_1
	                \end{align}
	                $$
	                - 由**富比尼定理**：
		                对于一个“行为良好”的函数（在我们的概率密度函数场景下，这个条件总是满足的），其在某个区域上的多重积分，**等于**按照任意顺序进行的逐层积分，并且其结果与积分顺序无关，即：
		                $$\iint_{D} f(x,y) \,dA = \int_a^b \int_c^d f(x,y) \,dy \,dx = \int_c^d \int_a^b f(x,y) \,dx \,dy$$
			- **边缘向量场定义**：
				$$
				\mathbf{u}_t(\mathbf{x}_t) := \int_{\mathbf{x}_0 \mathbf{x}_1} \, \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1) \, \frac{p_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)\pi(\mathbf{x}_0, \mathbf{x}_1)}{p_t(\mathbf{x}_t)} d\mathbf{x}_0 d\mathbf{x}_1
				$$
				* **左边:**
				    * $\mathbf{u}_t(\mathbf{x}_t)$：这是我们最终想要求解的**边缘向量场 (marginal vector field)**。其含义是：在时间 $t$，当整个数据分布演化到 $p_t$ 时，位于点 $\mathbf{x}_t$ 的样本“平均”应该朝哪个方向流动。这是描述**整个数据云团**如何运动的宏观规律
				* **右边 (积分内部):**
				    * $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$：这是我们预先定义的、简单的**条件向量场 (conditional vector field)**。其含义是：如果我们**已经知道**一个样本的起点是 $\mathbf{x}_0$、终点是 $\mathbf{x}_1$，那么当它在时间 $t$ 途经 $\mathbf{x}_t$ 点时，它应该朝哪个方向流动。这描述的是**单个样本点**如何从起点到终点的微观规律
				    * $p_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$：这是**条件概率路径**。它描述了在给定的起点 $\mathbf{x}_0$ 和终点 $\mathbf{x}_1$ 下，在时间 $t$ 恰好处于 $\mathbf{x}_t$ 的概率密度。对于确定性路径（如线性插值），这是一个狄拉克函数 $\delta(\mathbf{x}_t - ((1-t)\mathbf{x}_0 + t\mathbf{x}_1))$
				    * $\pi(\mathbf{x}_0, \mathbf{x}_1)$：这是起点 $\mathbf{x}_0$ 和终点 $\mathbf{x}_1$ 的**联合分布**，也叫**数据耦合 (data coupling)**。它描述了我们如何配对源分布和目标分布中的样本。在监督学习或成对数据集（如本文的Apo-Holo对）中，它就是数据集中给定的配对
				    *   $p_t(\mathbf{x}_t)$：这是**边缘概率路径**。它描述了在时间 $t$，从所有可能的起点出发、流向所有可能的终点的所有样本，汇集在 $\mathbf{x}_t$ 这一点上的总概率密度。它是我们不知道的、复杂的分布
				- 推导：（条件概率公式）
					$$ \frac{p_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)\pi(\mathbf{x}_0, \mathbf{x}_1)}{p_t(\mathbf{x}_t)} = \frac{p_t(\mathbf{x}_t, \mathbf{x}_0, \mathbf{x}_1)}{p_t(\mathbf{x}_t)} = p_t(\mathbf{x}_0, \mathbf{x}_1 | \mathbf{x}_t)
					$$
					* $p_t(\mathbf{x}_0, \mathbf{x}_1 | \mathbf{x}_t)$：这是一个**后验概率 (posterior probability)**。其含义是：已知在时间 $t$ 有一个样本粒子位于 $\mathbf{x}_t$ 这个位置，那么这个粒子当初是从 $\mathbf{x}_0$ 出发、并将要前往 $\mathbf{x}_1$ 的概率是多少？
				* 公式改写：重写成一个更直观的期望形式
					$$
					\mathbf{u}_t(\mathbf{x}_t) = \mathbb{E}_{p_t(\mathbf{x}_0, \mathbf{x}_1 | \mathbf{x}_t)} \left[ \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1) \right]
					$$
					- 含义：**边缘向量场** $\mathbf{u}_t(\mathbf{x}_t)$ 是所有可能的个体向量场 $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$ 的加权平均，权重是这些“个体路径”穿过点 $\mathbf{x}_t$ 的后验概率
			- 条件流匹配的训练损失函数为：
				$$
				\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}(0,1), \pi(\mathbf{x}_0, \mathbf{x}_1), p_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)} \left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0) \right\|^2 \tag{1}
				$$
				* **变量定义:**
				    *   $\mathbf{v}_\theta(\mathbf{x}_t, t)$：神经网络对向量场的**预测**
				    *   $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0)$：**目标向量场**。这是流匹配最巧妙的地方。由于不知道驱动整个数据分布演化的复杂边缘向量场 $\mathbf{u}_t(\mathbf{x}_t)$，但可以**定义**一个简单的、从单个样本 $\mathbf{x}_0$ 流向 $\mathbf{x}_1$ 的**条件向量场**。最简单的例子就是线性插值路径 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，其对应的向量场就是恒定的 $\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0$
				    *   $\mathbb{E}(\cdot)$：在所有可能的时间 $t$，所有数据对 $(\mathbf{x}_0, \mathbf{x}_1)$ 以及它们定义的路径点 $\mathbf{x}_t$ 上求期望
				- 期望与损失函数的关系：
					- 定理：
						设随机变量 $x,y$，其联合分布已知，若以均方误差为损失函数：
						$$\mathcal{L}(f) = \mathbb{E}_{x,y} \left[ (y - f(x))^2 \right]$$
						则最小化该损失的最优预测函数为：
						 $$f^*(x) = \mathbb{E}[y \mid x]$$
					- 条件期望的定义：
						如果 $Z = f(X, Y)$ 是两个随机变量的函数，则其在 $X$ 条件下的条件期望为：
							$$
							\mathbb{E}[f(X, Y) \mid X] = \int f(X, y)\, p(y \mid X)\, dy
							$$
					由定理与条件期望的定义得到：
						$$
						\begin{align}
						f^*(x) &= \mathbb{E}[ \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0) \mid \mathbf{x}_t] \\
						&= \int f(\mathbf{x}_t, \mathbf{x}_0, \mathbf{x}_1)p(\mathbf{x}_0, \mathbf{x}_1|\mathbf{x}_t) d\mathbf{x}_0 \,d\mathbf{x}_1\\
						&= \int_{\mathbf{x}_0, \mathbf{x}_1} \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1) \cdot p_t(\mathbf{x}_0, \mathbf{x}_1 | \mathbf{x}_t) \,d\mathbf{x}_0 \,d\mathbf{x}_1 \\
						&= \mathbb{E}_{p_t(\mathbf{x}_0, \mathbf{x}_1 | \mathbf{x}_t)} \left[ \mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1) \right] = \mathbf{u}_t(\mathbf{x}_t)
						\end{align}
						$$
					即，$\mathcal{L}$ 最小化得到的最优网络 $\mathbf{v}_\theta(\mathbf{x}_t, t)$ 为学习目标 $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_1, \mathbf{x}_0)$ 的条件期望，而该期望恰好是最初想要的学习的、但无法计算的 **边缘向量场** $\mathbf{u}_t(\mathbf{x}_t)$
		- ⭐**总结**：
			1. **目标：** 学习**边缘向量场** $\mathbf{u}_t(\mathbf{x}_t)$
			2. **困境：** $\mathbf{u}_t(\mathbf{x}_t)$ 和其对应的**边缘概率路径** $p_t(\mathbf{x}_t)$ 无法直接计算
			3. **策略：**
			    * **定义**一个简单的、可计算的**条件向量场** $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$
			    * **证明**一个理论关系：我们想求的 $\mathbf{u}_t(\mathbf{x}_t)$ 正是这个简单的 $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$ 在后验概率 $p_t(\mathbf{x}_0, \mathbf{x}_1 | \mathbf{x}_t)$ 下的期望
			    * **设计**一个损失函数，让神经网络 $\mathbf{v}_\theta$ 直接去**学习/回归**这个简单的 $\mathbf{u}_t(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_1)$
			    * 利用统计学原理，保证通过这个学习过程，$\mathbf{v}_\theta$ **最终会收敛**到我们真正的目标 $\mathbf{u}_t(\mathbf{x}_t)$
	- **离散变量的流匹配（Discrete Flow Matching）**
		对于离散变量，我们采用 **Discrete Flow Matching（DFM）** 框架，该方法基于 **连续时间马尔可夫链（CTMC）**
		设：
		- $\mathbf{x} \in \{1, ..., S\}^D$：一个长度为 $D$ 的离散变量序列
		- 每个元素取 $S$ 个可能的状态
		- 用 $j$ 表示状态，$x_i = j$ 表示 $\mathbf{x}$ 的第 $i$ 个维度处于状态 $j$
		引入一个**速率矩阵** $R_t \in \mathbb{R}^{S \times S}$，其作用与连续情况下的向量场 $\mathbf{u}_t$ 相似，定义跳转概率如下：
			$$
              P(\mathbf{x}_t \text{ 跳转到状态 } j) = R_t(\mathbf{x}_t, j) dt
            $$
		即：
			$$
			p_{t+dt \mid t}(\boldsymbol{j}|\mathbf{x}) = \delta(\mathbf{x}, j) + R_t(\mathbf{x}_t, j) dt
			$$
		其中 $\delta(i,j)$ 是 Kronecker delta 函数（当 $i = j$ 时为 1，否则为 0）
		- **公式解读**：
			- 含义：已知在时间 $t$，系统的状态是 $\mathbf{x}_t$，那么在下一个瞬间 $t+dt$，它的状态变成 $j$ 的概率是多少？
			- 所有可能的情况：
				* **情况一：状态保持不变 ($\boldsymbol{j} = \mathbf{x}_t$)**
				    * 公式变为：$p(\mathbf{x}_t | \mathbf{x}_t) = δ{\mathbf{x}_t, \mathbf{x}_t} + R_t(\mathbf{x}_t, \mathbf{x}_t) dt = 1 + R_t(\mathbf{x}_t, \mathbf{x}_t) dt$
				    * **直观解释：** 在极短的时间 $dt$ 内，系统**几乎肯定**会保持在原来的状态。所以这个概率非常接近 1
				    * 数学细节：对角线元素 $R_t(i, i)$ 被定义为 $-∑_{k≠i} R_t(i, k)$，即离开状态 $i$ 的所有速率之和的负数。所以 $1 + R_t(\mathbf{x}_t, \mathbf{x}_t) dt$ 是一个略小于1的数，这保证了所有概率加起来等于 1
				* **情况二：状态跳转到一个新状态 ($\boldsymbol{j} ≠ \mathbf{x}_t$)**
				    * 公式变为：$p(\boldsymbol{j} | \mathbf{x}_t) = δ{\mathbf{x}_t, \boldsymbol{j}} + R_t(\mathbf{x}_t, \boldsymbol{j}) dt = 0 + R_t(\mathbf{x}_t, \boldsymbol{j}) dt$
				    * **直观解释：** 正如我们上面分析的，在极短时间 $dt$ 内，系统从 $\mathbf{x}_t$ 跳转到 $\boldsymbol{j}$ 的概率就是 **速率 × 时间**，即 $R_t(\mathbf{x}_t, \boldsymbol{j}) dt$
				**所以公式用一种统一的形式，描述了两种可能性：**
				1.  以**极大的概率** ($≈ 1$) **保持不变**
				2.  以**极小的概率** ($\propto dt$) **跳转**到一个新的状态，跳转到哪个新状态，由速率矩阵 $R_t$ 决定
	- **离散流建模目标**
		- 构建离散流时，我们使用条件概率路径来构建边缘分布 $p_t(\mathbf{x}_t) := \mathbb{E}_{p(\mathbf{x}_1)} [p_t(\mathbf{x}_t | \mathbf{x}_1)]$
		- 我们定义**条件速率矩阵**：$$R_t(\mathbf{x}_t, \boldsymbol{j} | \mathbf{x}_1)$$  用于逼近条件概率分布 $p_t(\mathbf{x}_t | \mathbf{x}_1)$ —— 极其简单的 **个体** 跳转规则
			- 注：$R_t(\mathbf{x}_t, \boldsymbol{j} | \mathbf{x}_1)$ 通常可以用一个更简单的公式表示。例如：$R_t(\mathbf{x}_t, \boldsymbol{j} | \mathbf{x}_1) := \delta\{\mathbf{x}_1, \boldsymbol{j}\} \delta\{\mathbf{x}_t, M\}/(1 - t)$，其中 $M$ 是掩码标记
		- 可以证明，**边缘速率矩阵（marginal rate matrix）** —— 其实与连续形式类似，是 **个体的期望**
			$$R_t(\mathbf{x}_t, \boldsymbol{j}) := \mathbb{E}_{p(x_1 \mid \mathbf{x}_t)}[R_t(\mathbf{x}_t, \boldsymbol{j} \mid x_1)]$$
			与我们上面定义的边缘概率路径 $p_t(\mathbf{x}_t)$ 对应，其中：
			$$
			\begin{align}
			p(\mathbf{x}_1 \mid \mathbf{x}_t) &= \frac{p(\mathbf{x}_1, \mathbf{x}_t)}{p(\mathbf{x}_t)} \\
			&= \frac{p(\mathbf{x}_t \mid \mathbf{x}_1) \cdot p(\mathbf{x}_1) }{p(\mathbf{x}_t)} \\
			&=\frac{p_t(\mathbf{x}_t \mid x_1) \, q(x_1)}{p_t(\mathbf{x}_t)} \\
			\end{align}
			$$
			- 后验概率 $p(\mathbf{x}_1 \mid \mathbf{x}_t)$：已知在时间 $t$ 出现了一个状态 $\mathbf{x}_t$，那么这个状态是由哪个"干净"的目标  $\mathbf{x}_1$ 演化而来的概率是多少？
		- 我们可以使用一个神经网络 $p_t^\theta(x_1 \mid \mathbf{x}_t)$ 来近似后验分布 $p_t(x_1 \mid \mathbf{x}_t)$。我们将 $R_t^\theta(\mathbf{x}_t, \cdot) := \mathbb{E}_{p_t^\theta(x_1 \mid \mathbf{x}_t)}[R_t(\mathbf{x}_t, \cdot \mid x_1)]$ 表示为通过神经网络生成的估计速率
			- 通过迭代采样的方式从如下过程生成样本：
				$$
				p_{t + dt}(x_{t + dt} \mid \mathbf{x}_t) = \delta\{\mathbf{x}_{t + dt}, \mathbf{x}_t\} + R_t^\theta(\mathbf{x}_t, x_{t + dt}) \, dt + o(dt) \tag{2}
				$$
				*   **变量定义:**
				    * $\mathbf{x}_t$：在时间 $t$ 的离散状态（比如一个包含所有原子类型的序列）
				    * $\mathbf{x}_{t+dt}$：在下一时刻 $t+dt$ 的状态
				    * $R_t^\theta(\mathbf{x}_t, \mathbf{x}_{t+dt})$：神经网络预测的**速率矩阵**元素，表示从状态 $\mathbf{x}_t$ 跳转到 $\mathbf{x}_{t+dt}$ 的**速率**
				    * $\delta\{i, j\}$：克罗内克函数，当 $i=j$ 时为 1，否则为 0
				    * $o(dt)$：高阶无穷小项，当 $dt$ 趋于 0 时可以忽略
		- ⭐**总结**：
			1. **目标：** 学习**边缘速率矩阵** $R_t(\mathbf{x}_t, \boldsymbol{j})$
			2. **困境：** $R_t(\mathbf{x}_t, \boldsymbol{j})$ 和其对应的**边缘概率路径** $p_t(x_t)$ 无法直接计算
			3. **策略：**
			    * **定义**一个简单的、可计算的**条件速率矩阵 $R_t(\mathbf{x}_t, \boldsymbol{j} | \mathbf{x}_1)$
			    * **证明**一个理论关系：我们想求的 $R_t(\mathbf{x}_t, \boldsymbol{j})$ 正是这个简单的 $R_t(\mathbf{x}_t, \boldsymbol{j} | \mathbf{x}_1)$ 在后验概率 $p(\mathbf{x}_1|\mathbf{x}_t)$ 下的期望
			    * **设计**一个**代理任务**。由于无法直接计算上述期望中的权重（后验概率），转而训练一个神经网络 $p_t^\theta(x_1 \mid \mathbf{x}_t)$ 去**近似**这个后验概率
			    * **利用** $p_t^\theta(x_1 \mid \mathbf{x}_t)$ 代替原本的后验概率去计算 $R_t(\mathbf{x}_t, \boldsymbol{j} | \mathbf{x}_1)$
3. 用于 SBDD 的全原子流
	- 模型作用：将 Apo 状态转换为 Holo 状态，并从一个带噪声的先验分布中生成配体分子
	- 源分布
		$$ p_0(\mathcal{P}, \mathcal{M} | \mathcal{P}_0) := \delta(\mathcal{P}, \mathcal{P}_0) \prod_{i=1}^{N_m} \mathcal{N}(x_m^{(i)}; \text{CoM}(\mathcal{P}_0), \mathbf{I}) \delta{\{v_m^{(i)}, M_v\}} \prod_{i,j=1}^{N_m} \delta{\{b_m^{(ij)}, M_b\}} \tag{3}$$
		- 变量定义：
		    * $p_0(\mathcal{P}, \mathcal{M}|\mathcal{P}_0)$：这是在给定 Apo 口袋 $\mathcal{P}_0$ 条件下，时间 $t=0$ 时的联合状态 $(\mathcal{P}, \mathcal{M})$ 的概率分布
		    * $\delta(\mathcal{P}, \mathcal{P}_0)$：狄拉克 $\delta$ 函数。它表示在 $t=0$ 时，口袋部分 $\mathcal{P}$ **精确地等于**给定的 Apo 口袋 $\mathcal{P}_0$（狄拉克函数的概率密度只在 $\mathcal{P}=\mathcal{P}_0$ 处为无穷大，其他地方为 0，这样保证了**一定**取到 Apo 口袋 $\mathcal{P}_0$）
		    * $\prod\limits_{i=1}^{N_m}$：连乘符号，表示对配体中的所有 $N_m$ 个原子进行操作
		    * $\mathcal{N}(x_m^{(i)}; \text{CoM}(\mathcal{P}_0), \mathbf{I})$：一个标准正态（高斯）分布。它表示第 $i$ 个配体原子的初始位置 $x_m^{(i)}$ 是从以Apo口袋的质心 ($\text{CoM}(\mathcal{P}_0)$) 为均值、单位矩阵 $\mathbf{I}$ 为协方差的分布中采样的。直观上，就是一团**以口袋为中心的随机点云**
		    * $\delta{\{v_m^{(i)}, M_v\}}$：离散的克罗内克 $\delta$ 函数。它表示第 $i$ 个配体原子的初始类型 $v_m^{(i)}$ **精确地等于**一个特殊的“原子类型掩码” $M_v$
		    * $\delta{\{b_m^{(ij)}, M_b\}}$：离散的克罗内克 $\delta$ 函数。它表示配体中任意一对原子 $(i,j)$ 之间的初始键类型 $b_m^{(ij)}$ **精确地等于**一个特殊的“化学键类型掩码” $M_b$
		- 含义：这是生成过程的起点($t=0$)，在这个起点：
		    1.  蛋白质口袋部分是确定的，就是我们输入的 Apo 口袋 $\mathcal{P}_0$
		    2.  配体的位置是完全随机的，像一团噪声
		    3.  配体的原子类型和化学键是完全未知的，全部用“掩码”符号表示
	- 数据集的构建：
		- $\pi (\underbrace{\mathcal{M}_0, \mathcal{P}_0}_{\text{起点,即公式(3)}}, \underbrace{\mathcal{M}_1, \mathcal{P}_1}_{\text{终点,真实分布采样}}|\mathcal{P}_0)$
		- 一个 Apo 状态可能会对应多个 Holo 状态（一个起点对应多个终点）
	- 训练方式：
		- 使用预定义的条件概率路径 $p_t(\mathcal{M}_t,\mathcal{P}_t|\mathcal{M}_0, \mathcal{P}_0, \mathcal{M}_1, \mathcal{P}_1)$ 对**插值**进行采样，以训练我们的流模型
		- 使用插值的优点：
			- **提供一个可解的训练框架：** 将一个复杂的生成问题，转化为一个简单的、有监督的回归/分类问题 —— 原本的生成任务是无监督的（从噪声生成数据），现在变成了一个标准的**有监督学习**任务。输入是 $(\mathcal{M}_t,\mathcal{P}_t)$，标签是 $(\mathcal{M}_1,\mathcal{P}_1)$
			- **分而治之**：从中间状态到终点，而非从零到一
			- 契合之前流模型分析：将无法直接计算的复杂变量转化为可计算的简单变量，两者之间关联已由数学推导证明
	- 连续变量的流匹配
		- 技巧：**不直接让网络预测向量场 $\mathbf{u}_t$，而是让网络预测最终的“干净”样本 $\mathbf{x}_1$，然后通过一个固定的数学公式从预测的 $\mathbf{x}_1$ 反推出向量场。** 这样做通常能让网络学习更稳定，因为目标 $\mathbf{x}_1$ 是固定的，而向量场 $\mathbf{u}_t$ 是随时间变化的
		- 注：推导中以 $\mathbf{t}$ 为例，给定源和目标分布的样本，即 $\mathbf{t}_0$ 和 $\mathbf{t}_1$，我们将条件概率路径定义为这两个样本之间的线性插值，并推导出条件向量场如下（上标中的 $t$ 表示“平移”，下标中的 $t$ 代表“时间”）
		1. **平移**
			- **欧氏空间中的路径与向量场**
				$$  p_t^t(\mathbf{t}_t^{(i)}|\mathbf{t}_0^{(i)}, \mathbf{t}_1^{(i)}) := \delta(\mathbf{t}_t^{(i)}, t \mathbf{t}_1^{(i)} + (1-t)\mathbf{t}_0^{(i)}) \quad \text{and} \quad u_t^t(\mathbf{t}_t^{(i)}|\mathbf{t}_1^{(i)}, \mathbf{t}_0^{(i)}) = \mathbf{t}_1^{(i)} - \mathbf{t}_0^{(i)} = \frac{\mathbf{t}_1^{(i)} - \mathbf{t}_0^{(i)}}{1-t} \tag{4}$$
				- 变量定义：
				    * $p_t^t(\dots)$：这是第 $i$ 个残基的平移向量 $\mathbf{t}^{(i)}$ 在时间 $t$ 的**条件概率路径**
					* $\delta(\mathbf{t}_t^{(i)}, t \mathbf{t}_1^{(i)} + (1-t)\mathbf{t}_0^{(i)})$：狄拉克 $\delta$ 函数表明这是一个**确定性路径**。在时间 $t$，平移向量 $\mathbf{t}_t^{(i)}$ **精确地等于**起点 $\mathbf{t}_0^{(i)}$ 和终点 $\mathbf{t}_1^{(i)}$ 的线性插值（只有在两者相等时概率密度为无限大，使得概率为 1，否则为 0）
					* $\mathbf{u}_t^t(\dots)$：这是上述路径对应的**条件向量场**（即速度）
					* $\mathbf{t}_1^{(i)} - \mathbf{t}_0^{(i)}$：对于线性插值，从起点到终点的速度是一个**常数向量**
					* $\frac{\mathbf{t}_1^{(i)} - \mathbf{t}_t^{(i)}}{1-t}$：这是速度的另一种等价写法。将 $\mathbf{t}_t^{(i)} = (1-t)\mathbf{t}_0^{(i)} + t\mathbf{t}_1^{(i)}$ 变形可得 $\mathbf{t}_1^{(i)} - \mathbf{t}_0^{(i)} = (\mathbf{t}_1^{(i)} - \mathbf{t}_t^{(i)})/(1-t)$。这个形式在后续定义损失函数时更方便
				- 含义：这个公式为欧氏空间中的变量（如原子坐标、平移向量）定义了最简单的演化路径：一条直线。并给出了沿着这条直线运动的速度
			- **平移的 CFM 损失（条件流匹配损失）**
				$$ \mathcal{L}_t^t = \mathbb{E}_{\pi(\mathbf{t}_0, \mathbf{t}_1), p_t^t(\mathbf{t}_t^{(i)}|\mathbf{t}_0^{(i)}, \mathbf{t}_1^{(i)})} \left\| v_\theta(t_t^{(i)}, t) - (\mathbf{t}_1^{(i)} - \mathbf{t}_0^{(i)}) \right\|^2_2  \tag{5}$$
				- 变量定义：
				    * $\mathcal{L}_t^t$：平移部分的损失函数
					* $\mathbb{E}_{\dots}$：在所有数据对 $(\mathbf{t}_0, \mathbf{t}_1)$ 和所有时间 $t$ 上求期望
					* $\mathbf{t}_\theta(\mathbf{t}_t^{(i)}, t)$：神经网络的**输出**。它的任务是接收在时间 $t$ 的“含噪”样本 $\mathbf{t}_t^{(i)}$，并**直接预测**出最终的“干净”目标 $\mathbf{t}_1^{(i)}$
					* $\mathbf{t}_1^{(i)}$：训练数据中真实的**目标值**
					* $\| \cdot \|^2$：均方误差 (MSE)，计算预测值和真实目标之间的欧氏距离的平方
				- 含义：这是对网络 $v_θ$ 的一个简单直接的**监督学习**任务。损失函数要求网络学会“去噪”或“修复”：无论给它一个什么样的中间状态 $\mathbf{t}_t$，它都要能准确地猜出这个状态的终点 $\mathbf{t}_1$
		2. **旋转**
			 - **SO(3)流形上的路径与向量场**
				$$  p_t^{(\mathrm{r})}(\mathbf{r}_t^{(i)}|\mathbf{r}_0^{(i)}, \mathbf{r}_1^{(i)}) := \delta(\mathbf{r}_t^{(i)}, \exp_{\mathbf{r}_0^{(i)}}(t \log_{\mathbf{r}_0^{(i)}}(\mathbf{r}_1^{(i)}))) \quad \text{and} \quad \mathbf{u}_t^{(\mathrm{r})}(\mathbf{r}_t^{(i)}|\mathbf{r}_0^{(i)}, \mathbf{r}_1^{(i)}) = \frac{\log_{\mathbf{r}_t^{(i)}}(\mathbf{r}_1^{(i)})}{1-t} \tag{6}$$
				- 变量定义：
				    * $\mathbf{r} \in \text{SO(3)}$：旋转矩阵，存在于一个弯曲的**SO(3)流形**上，不能用简单的线性插值
					    * 什么是 **SO(3)**
						    * SO(3) 是一个数学群，表示：
								$$ SO(3) = \{R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \det(R) = 1\} $$
								变量定义：
								- $R^T R = I$，其中 $R$ 是正交矩阵
								- $\det(R) = 1$，其中 $R$ 的行列式为 1，即是“纯旋转矩阵”，不包含翻转
								所以 SO(3) 表示的是**所有 3D 空间中的旋转操作组成的集合**，是一个李群
				    * $\log_{\mathbf{r}_0^{(i)}}(\mathbf{r}_1^{(i)})$：SO(3)上的**对数映射 (Logarithmic Map)**。它将从 $\mathbf{r}_0^{(i)}$ 到 $\mathbf{r}_1^{(i)}$ 的旋转，映射到 $\mathbf{r}_0^{(i)}$ 点的 **切空间** 上，得到一个三维向量（旋转轴乘以角度）。可以理解为把弯曲的路径“拉直”成一个向量
				    * $\exp_{\mathbf{r}_0^{(i)}}(\dots)$：SO(3)上的**指数映射 (Exponential Map)**。它是对数映射的逆过程，将切空间上的一个向量映射回SO(3)流形上的一个点（旋转矩阵）
				    * **路径部分:** $\text{exp}(\dots)$ 的含义是：
					    1. 将 $\mathbf{r}_0^{(i)}$ 到 $\mathbf{r}_1^{(i)}$ 的旋转“拉直”成向量
					    2. 在这个向量方向上走 $t$ 这么长的比例
					    3. 再把这个新位置“弯”回到流形上。这定义了SO(3)流形上两点间的**测地线 (geodesic)**，即“最短路径”
				    * **向量场部分:** $log_{r_t}(r_1)/(1-t)$ 是在时间 $t$ 的点 $\mathbf{r}_t^{(i)}$ 的切空间上表示的速度向量
				- 含义：这个公式将公式(4)中的直线路径推广到了旋转所在的弯曲流形上，使用了流形上的“直线”——测地线——来定义演化路径
			- **旋转的 CFM 损失**
				$$ \mathcal{L}_t^r = \mathbb{E}_{\pi(\mathbf{r}_0, \mathbf{r}_1), p_t^t(\mathbf{r}_t^{(i)}|\mathbf{r}_0^{(i)}, \mathbf{r}_1^{(i)})} \left\| v_\theta(\mathbf{r}_t^{(i)}, t) - \frac{\log_{\mathbf{r}_t^{(i)}}(\mathbf{r}_1^{(i)})}{1-t} \right\|_{\text{SO(3)}}^2 \tag{7} $$
				- 变量定义：
				    * $\mathcal{L}_t^r$：旋转部分的损失
				    * $\mathbf{r}_\theta(\mathbf{r}_t^{(i)}, t)$：神经网络对最终“干净”旋转 $\mathbf{r}_1^{(i)}$ 的**预测**
				    * $\log_{\mathbf{r}_t^{(i)}}(\cdot)$：将预测的旋转 $\mathbf{r}_\theta$ 和真实的旋转 $\mathbf{r}_1^{(i)}$ 都映射到当前点 $\mathbf{r}_1^{(i)}$ 的切空间上，这样它们就都变成了普通的三维向量，可以进行比较
				    * $\| \cdot \|_{\text{SO(3)}}^2$：计算这两个在切空间上的向量的均方误差
				- 含义：与公式(5)的思想完全一样。网络不直接预测速度，而是预测最终的“干净”旋转 $\mathbf{r}_1^{(i)}$。损失函数通过对数映射，在切空间上衡量预测值和真实目标之间的距离
		3. **扭转角**
			- **环面上的指数和对数映射**
				$$ \exp_x(u) = (x+u)\%(k\pi) \quad \text{and} \quad \log_x(y) = \text{arctan2}(\sin(y-x), \cos(y-x)) \tag{8} $$
				- 变量定义：
				    * $x, y \in \mathcal{M}$：环面上的两个点，即两个角度值
				    * $u \in T_x\mathcal{M}$：在点 $x$ 的切空间上的一个向量，即一个角度的改变量
				    * $\exp_x(u)$：指数映射。在角度 $x$ 的基础上，增加一个改变量 $u$，然后用模运算 $\%(k\pi)$ 来处理周期性，确保结果仍在 $[0, k\pi]$ 的区间内
				    * $\log_x(y)$：对数映射。计算从角度 $x$ 到角度 $y$ 的最短弧长（带方向）。$\text{arctan2}(sin, cos)$ 函数可以正确地处理象限，确保结果在 $[-\pi, π]$ 之间，这代表了最短路径
				- 含义：这个公式为扭转角所在的环面（一维圆周）定义了与SO(3)流形上类似的几何运算，核心是处理角度的**周期性**
			- **环面上的路径与向量场**
				$$ p_t^{(\chi)}(\chi_t^{(i)}|\chi_0^{(i)}, \chi_1^{(i)}) := \delta(\chi_t^{(i)}, (\chi_0^{(i)} + t \cdot \text{wrap}(\chi_1^{(i)} - \chi_0^{(i)}))\%(k\pi)) \tag{9} $$
				$$ \mathbf{u}_t^{(\chi)}(\chi_t^{(i)}|\chi_0^{(i)}, \chi_1^{(i)}) = \text{wrap}(\chi_1^{(i)} - \chi_t^{(i)}) = \frac{\text{wrap}(\chi_1^{(i)} - \chi_t^{(i)})}{1-t} \tag{10} $$
				- 变量定义：
				    * $\chi$：代表扭转角
				    * $\text{wrap}(\cdot)$：这个函数的作用是将一个任意大小的角度差，包装到 $[-\pi, π]$ 区间内，代表了最短的角位移
				    * **路径 (Eq. 9):** 与欧氏空间类似，它定义了一个线性插值。起点是 $\chi_0$，方向是 $\text{wrap}(\chi_1-\chi_0)$ (最短角位移)，走了 $t$ 的比例。最后的模运算 $\%(kπ)$ 确保结果仍在合法的角度范围内
				    * **向量场 (Eq. 10):** 同样，这是速度的另一种等价写法，$\text{wrap}(\chi_1-\chi_t)$ 是从当前点 $\chi_t$ 到终点 $\chi_1$ 的最短角位移
				- 含义：这两个公式为扭转角定义了在环面上的“直线”演化路径，关键在于使用 $\text{wrap}$ 函数正确处理了角度的周期性
			- **扭转角的 CFM 损失**
				$$ \mathcal{L}_t^{\mathcal{X}} = \mathbb{E}_{\pi(\chi_0, \chi_1), \, p_t^{\mathcal{X}}(\chi_t^{(i)} \mid \chi_0^{(i)}, \chi_1^{(i)})} \left\| \mathrm{wrap} \left( v_\theta(\chi_t^{(i)}, t) - \frac{\chi_1^{(i)} - \chi_0^{(i)}}{1 - t} \right) \right\|^2 \tag{11}$$
				- 变量定义：
				    * $\mathcal{L}^\chi_t$：扭转角部分的损失
				    * $\chi_\theta(\chi_t, t)$：神经网络对最终“干净”角度 $\chi_1$ 的**预测**
				    * $\text{wrap}(\cdot)$：对预测值和真实目标值与当前值的差都进行 $\text{wrap}$ 操作，确保我们比较的是两个在 $[-\pi, π]$ 区间内的最短角位移
				    * $\| \cdot \|^2$：计算这两个角位移之间的均方误
				- 含义：与平移和旋转的损失函数思想一致，网络预测最终的干净角度，损失函数衡量的是预测所隐含的角位移与真实角位移之间的差距
	- 离散变量的流匹配
		- **离散变量的路径与速率矩阵**
			$$ p_t^b(\mathbf{b}_t^{(ij)}|\mathbf{b}_0^{(ij)}, \mathbf{b}_1^{(ij)}) := \text{Cat}(t\delta\{\mathbf{b}_t^{(ij)}, \mathbf{b}_1^{(ij)}\} + (1-t)\delta\{\mathbf{b}_t^{(ij)}, \mathcal{M}\}) \tag{12} $$
			$$ R_t^b(\mathbf{b}_t^{(ij)}, n|\mathbf{b}_1^{(ij)}) = \frac{\delta\{\mathbf{b}_1^{(ij)}, n\}\delta\{\mathbf{b}_t^{(ij)}, \mathcal{M}\}}{1-t} \tag{13} $$
			- 变量定义：
			    * $\mathbf{b}$：代表键类型（离散变量）
			    * $\mathcal{M}$：掩码符号
			    * **路径 (12):** 这是一个概率密度的线性插值。在时间 $t$，状态有 $1-t$ 的概率是掩码 $\mathcal{M}$，有 $t$ 的概率是最终的真实键类型 $\mathbf{b}_1$
			    * **速率矩阵 (13):** 这是从路径(12)推导出的条件速率矩阵。$n$ 代表一个具体的目标键类型。这个公式的含义是：只有当当前状态 $\mathbf{b}_t$ 是掩码 $\mathcal{M}$，并且最终目标 $\mathbf{b}_1$ 恰好是 $n$  时，才存在一个从 $\mathcal{M}$ 跳转到 $n$  的非零速率，其速率为 $1/(1-t)$
			- 含义：定义了从一个完全未知的掩码状态，演化到一个确定的目标状态的最简单的离散路径
		- **离散变量的损失**
			$$ \mathcal{L}_t^b = \mathbb{E}_{\pi(\mathbf{b}_0, \mathbf{b}_1), p_t(\mathbf{b}_t^{(ij)}|\mathbf{b}_0^{(ij)}, \mathbf{b}_1^{(ij)})}[-\log p_\theta^b(\mathbf{b}_1^{(ij)}|\mathbf{b}_t^{(ij)})] \tag{14} $$
			- 变量定义：
			    * $\mathcal{L}_t^b$：键类型部分的损失
			    * $p_\theta(\mathbf{b}_1^{(ij)}|\mathbf{b}_t^{(ij)})$：神经网络的输出。它是一个分类器，输入在时间 $t$ 的状态 $\mathbf{b}_t$，输出一个对最终干净状态 $\mathbf{b}_1$ 的概率分布
			    * $-\log(\cdot)$：负对数似然，是交叉熵损失的核心
			- 含义：训练一个神经网络，使其能够根据一个可能被掩码的中间状态 $\mathbf{b}_t$，准确地预测出它最终应该是什么键类型 $\mathbf{b}_1$
		- **估计的边际速率矩阵**
			$$ R_t^b(\mathbf{b}_t^{(ij)}, n) = \mathbb{E}_{p_\theta^b(\mathbf{b}_1^{(ij)}|\mathbf{b}_t^{(ij)})}[R_t(\mathbf{b}_t^{(ij)}, n|\mathbf{b}_1^{(ij)})] = \frac{p_\theta^b(\mathbf{b}_1^{(ij)}|\mathbf{b}_t^{(ij)})}{1-t}\delta\{\mathbf{b}_t^{(ij)}, \mathcal{M}\} \tag{15} $$
		- **在采样过程中，从 $t$ 到 $t+dt$ 的转移步骤**
			$$ p_{t+dt}(n|\mathbf{b}_t^{(ij)}) = \delta\{\mathbf{b}_t^{(ij)}, n\} + R_t^b(\mathbf{b}_t^{(ij)}, n)dt \tag{16} $$
	- 相互作用损失
		- 对于时间 $t$，将预测的“干净” holo 状态和配体分子表示为：
			$$ \hat{\mathcal{S}}_1 := \left[ \left( \mathbf{a}_0^{(i)}, t_\theta^{(i)}, \mathbf{r}_\theta^{(i)}, \chi_{1\theta}^{(i)}, \chi_{2\theta}^{(i)}, \chi_{3\theta}^{(i)}, \chi_{4\theta}^{(i)} \right) \right]_{i=1}^{D_p}\quad \text{and} \quad\hat{\mathcal{M}}_1 := \left\{ \left( \hat{\mathbf{x}}_{m_1}^{(i)}, \hat{\mathbf{v}}_{m_1}^{(i)} \right) \right\}_{i=1}^{N_p} \tag{17} $$
		- 损失具体表示
			$$ \mathcal{L}_t^{\text{int}} = \sum\nolimits_{i=1}^{N_p+N_m}\sum\nolimits_{j=i+1}^{N_p+N_m} \|\hat{d}_1^{(ij)} - d_1^{(ij)}\|^2 \cdot \mathbb{1}\{d_1^{(ij)} < 3.5\} \cdot \mathbb{1}\{t > 0.65\} \tag{18} $$
			- 变量定义：
			    *   $\mathcal{L}_t^{\text{int}}$：相互作用损失
				*   $\hat{d}_1^{(ij)}$：神经网络在时间 $t$ **预测**的“干净”构象中，原子 $i$ 和 $j$ 之间的**距离**
				*   $d_1^{(ij)}$：**真实**的 Holo 构象中，原子 $i$ 和 $j$ 之间的**距离**
				*   $\mathbb{I}\{d_1^{(ij)} < 3.5\}$：指示函数。表示这个损失只对距离小于3.5埃的**局部原子对**生效
				*   $\mathbb{I}\{t > 0.65\}$：指示函数。表示这个损失只在生成过程的**后期** ($t>0.65$) 才被激活
4. 推广到随机全原子流
	- 口袋的 Apo 和 Holo 状态之间的线性插值所诱导的概率路径其集合论支撑是有限的，因此可能导致鲁棒性较差
	- **SDE（随机微分方程）** 在高维空间中对噪声具有更强的鲁棒性
	- 欧氏空间中的随机路径
		$$ p_t^t(\mathbf{t}_t^{(i)}|\mathbf{t}_0^{(i)}, \mathbf{t}_1^{(i)}) := \mathcal{N}(\mathbf{t}_t^{(i)}; t\mathbf{t}_1^{(i)} + (1-t)\mathbf{t}_0^{(i)}, \gamma^2t(1-t)\mathbf{I}) \tag{19} $$
		 - 区别：对平移引入高斯分布添加随机噪声
			- 协方差 $\Sigma$：$\gamma^2t(1-t)\mathbf{I}$：这是与 ODE 的核心区别。它为路径加入了**随机噪声**
	        * $\gamma$：一个常数超参数，用于控制噪声的整体**强度**
	        * $t(1-t)$：一个随时间变化的**方差调度 (variance schedule)**。当 $t=0$ 或 $t=1$ 时，这一项为0，意味着在起点和终点，路径是**没有噪声的**，精确地收敛到 $t₀$ 和 $t₁$。当 $t=0.5$ 时，这一项达到最大值 0.25，意味着在路径的**中点，随机性最强**
	        * $\mathbf{I}$：单位矩阵，表示在各个维度上的噪声是独立的
	- SO(3)流形上的随机路径
		$$ p_t^t(\mathbf{r}_t^{(i)}|\mathbf{r}_0^{(i)}, \mathbf{r}_1^{(i)}) := \mathcal{IG}_{\text{SO(3)}}(\mathbf{r}_t^{(i)}; \exp_{\mathbf{r}_0^{(i)}}(t\log_{\mathbf{r}_0^{(i)}}(\mathbf{r}_1^{(i)})), \gamma t(1-t)) \tag{20} $$
		- 区别：对SO(3)旋转引入高斯分布添加随机噪声
			* $\mathcal{IG}_{\text{SO(3)}}(\cdot; \mu, \sigma^2)$：**SO(3)上的各向同性高斯分布 (Isotropic Gaussian distribution on SO(3))**。这是正态分布在旋转群上的推广。可以想象成在一个旋转状态 $\mu$ 周围，向所有“方向”均匀地弥散开来的概率分布
		    * 均值 $\mu$：$\exp_{\mathbf{r}_0^{(i)}}(t\log_{\mathbf{r}_0^{(i)}}(\mathbf{r}_1^{(i)}))$。这与确定性路径（公式6）的中心完全一样，即 $\mathbf{r}_0^{(i)}$ 和 $\mathbf{r}_0^{(i)}$ 之间的**测地线**上的点
		    * 方差 $\sigma^2$：$\gamma t(1-t)$。这同样是一个时间调度的方差，控制了在均值周围的“弥散”程度。$\gamma$ 是一个与欧氏空间情况不同的噪声强度参数
	- 环面上的随机路径
		$$ p^\chi_t(\chi_t^{(i)}|\chi_0^{(i)}, \chi_1^{(i)}) \propto \sum\nolimits_{d \in \mathbb{Z}} \exp\left(-\frac{\|\chi_t^{(i)} - (\chi_0^{(i)} + t \cdot \text{wrap}(\chi_1^{(i)} - \chi_0^{(i)})) \%(k\pi) + k\pi d\|^2}{2\gamma^2t(1-t)}\right) \tag{21}$$
		- 区别：对环面引入高斯分布添加随机噪声
		    * 这个公式定义了一个**包裹正态分布 (Wrapped Normal Distribution)**
		    * 内部结构：$\text{exp}(- (x - \mu)^2 / (2\sigma^2))$ 这是标准正态分布的概率密度函数形式
		        * 均值 $\mu$：$(\chi_0^{(i)} + t \cdot \text{wrap}(\chi_1^{(i)} - \chi_0^{(i)})) \%(k\pi)$。这与确定性路径（公式9）的中心完全一样，即环面上的测地线插值点
		        * 方差 $\sigma^2$：$\gamma^2t(1-t)$。同样是时间调度的方差
		    * $+ kπd$ 和 $\Sigma_{d∈\mathbb{Z}}$：这是“包裹”操作的核心。想象一下在一条直线上有一个标准正态分布，然后把这条直线像卷尺一样，以 $kπ$ 为周长卷起来形成一个圆环（环面）。原来直线上所有相隔 $kπ$ 整数倍的点（$\dots, x-kπ, x, x+kπ, x+2kπ, \dots$）都会被卷到同一个位置。这个求和 $\Sigma$ 就是把所有这些点上的概率密度加起来，得到环面上该点的总概率密度
5. 多尺度全原子模型架构
	- 原子级别的 SE(3) 等变图神经网络
		- EGNN —— 在神经网络中同时保留了**节点级别**和**边级别**的隐藏状态
		- **蛋白质-配体复合物图中的消息传递**
			- 在训练过程中，对于时间 $t$，我们在蛋白质-配体复合物的原子之上构建一个 $k$ 近邻图（$knn$） $G_c$ 来捕捉蛋白质-配体的相互作用
				$$ \Delta \mathbf{h}_c^{(i)} \leftarrow \sum\nolimits_{j \in \mathcal{N}_c(i)} \phi_e(\mathbf{h}^{(i)}, \mathbf{h}^{(j)}, \|\mathbf{x}^{(i)} - \mathbf{x}^{(j)}\|, E^{(ij)}, t) \tag{22} $$
				- 变量定义：
				    * $\Delta \mathbf{h}_c^{(i)}$：节点 $i$ 的隐藏状态 $\mathbf{h}_c^{(i)}$ 的**更新量**
				    * $\sum_{j \in \mathcal{N}_c(i)}$：对节点 $i$ 在复合物图 $G_c$ 中的所有邻居 $j$ 进行求和
				    * $\phi_e$：一个可学习的神经网络，通常是一个多层感知机(MLP)，被称为**边操作 (edge operation)** 或消息函数
				    * $\mathbf{h}_c^{(i)}, \mathbf{h}_c^{(j)}$：节点 $i$ 和其邻居 $j$ 的当前隐藏状态（节点特征）
				    * $\|\mathbf{x}^{(i)} - \mathbf{x}^{(j)}\|$：节点 $i$ 和 $j$ 之间的**欧氏距离**。这是将几何信息融入消息的关键
				    * $E^{(ij)}$：边 $(i,j)$ 的特征。这里它指示了这条边是蛋白质-蛋白质，配体-配体，还是蛋白质-配体之间的边
				    * $t$：时间 $t$，作为条件输入
				- 含义：图神经网络中标准的**消息传递 (Message Passing)** 步骤。对于图中的每一个原子 $i$，它会“环顾”四周的邻居 $j$，从每个邻居那里收集信息。收集的信息包括：邻居自身的特征 $\mathbf{h}_c^{(j)}$，与邻居的距离 $||x-x||$，以及它们之间边的类型 $E^{(ij)}$。这些信息被一个神经网络 $\phi_e$ 处理成一个“消息”，然后所有邻居的消息被累加起来，形成对中心节点 $i$ 特征的更新。
		- **配体内部的消息传递**
			- 在所有配体原子之上构建一个全连接的配体图 $G_m$ 来建模配体分子内部的化学相互作用
				$$ \mathbf{m}^{(ij)} \leftarrow \phi_m(\|\mathbf{x}^{(i)} - \mathbf{x}^{(j)}\|, e^{(ij)}) \quad \text{and} \quad \Delta \mathbf{h}_m^{(i)} \leftarrow \sum_{j \in \mathcal{N}_m(i)} \phi_m(\mathbf{h}_m^{(i)}, \mathbf{h}_m^{(j)}, \mathbf{m}^{(ij)}, t) \tag{23} $$
				- 变量定义：
					* $G_m$：一个只包含配体原子的图，且这个图是**全连接**的，即每个配体原子都与其他所有配体原子相连
					* $\mathbf{m}^{(ij)}$：边 $(i,j)$ 上的消息。在EGNN的原始框架中，这部分通常是 $φ_e$ 的一部分，但这里可能为了突出配体内部的化学键信息而分开表示
					* $e^{(ij)}$：配体原子 $i$ 和 $j$ 之间的**化学键的隐藏状态**（边特征）
					* $\Delta \mathbf{h}_m^{(i)}$：配体原子 $i$ 的隐藏状态的更新量，但这次的消息是**只来自于其他配体原子**
		- **节点与边特征的更新**
			$$ 
			\begin{align}
			&\mathbf{h}^{(i)} \leftarrow \mathbf{h}^{(i)} + \phi_h(\Delta \mathbf{h}_c^{(i)} + \Delta \mathbf{h}_m^{(i)}) \tag{24} \\  &\mathbf{e}^{(ji)} \leftarrow \sum_{k \in \mathcal{N}_m(j) \setminus \{i\}} \phi_e(\mathbf{h}^{(i)}, \mathbf{h}^{(j)}, \mathbf{h}^{(k)}, \mathbf{m}^{(ik)}, \mathbf{m}^{(ji)}, t) \tag{25} 
			\end{align}
			$$
			* **公式 (24):** 这是**节点更新 (Node Update)** 步骤。将从复合物图 $G_c$ 和配体图 $G_m$ 中收集到的所有消息更新量 $Δh$，通过一个MLP $φ_h$ 处理后，加回到原始的节点隐藏状态 $\mathbf{h}^{(i)}$ 上。
			* **公式 (25):** 这是一个**边更新 (Edge Update)** 步骤。它更新了配体内部化学键的特征 $e$。更新的依据是这条边所连接的两个原子 $i$, $j$，以及它们各自的邻居 $k$ 的信息。这允许模型学习更复杂的化学环境依赖关系
		- **原子坐标的等变更新**
			$$ \Delta \mathbf{x}_c^{(i)} \leftarrow \sum_{j \in \mathcal{N}_c(i)} \frac{\mathbf{d}^{(ij)}}{d^{(ij)}} \phi_x^c(\mathbf{h}^{(i)}, \mathbf{h}^{(j)}, d^{(ij)}, E^{(ij)}, t) \quad \Delta \mathbf{x}_m^{(i)} \leftarrow \sum_{j \in \mathcal{N}_m(i)} \frac{\mathbf{d}^{(ij)}}{d^{(ij)}} \phi_x^m(\mathbf{h}^{(i)}, \mathbf{h}^{(j)}, d^{(ij)}, m^{(ij)}, t) \tag{26} $$
			- 变量定义：
			    * $\Delta \mathbf{x}^{(i)}$：原子 $i$ 坐标的**更新向量**
			    * $\mathbf{d}^{(ij)} = \mathbf{x}^{(i)} - \mathbf{x}^{(j)}$：从邻居 $j$ 指向中心原子 $i$ 的**方向向量**
			    * $d^{(ij)} = \|\mathbf{d}^{(ij)}\|$：距离
			    * $\frac{\mathbf{d}^{(ij)}}{d^{(ij)}}$：**单位方向向量**
			    * $\phi_x$：一个MLP，它的输出是一个**标量**，表示在 $j→i$ 方向上的“力”的大小
	- 残基级别的 Transformer
		将上述全原子GNN输出的最终蛋白质原子隐藏状态 $\mathbf{h}$，根据 atom37模板 聚合成残基级别。聚合后的残基级别隐藏状态，连同氨基酸类型 $\mathbf{a}$、5个扭转角 $\chi$、以及框架 $(\mathbf{t}, \mathbf{r})$，被输入到一个残基级别的Transformer模型中。该模型由节点嵌入层、边嵌入层和多个不变点注意力(IPA)块组成。在每个 IPA 块中，残基级别的隐藏状态 $\mathbf{h}$ 和框架 $(\mathbf{t}, \mathbf{r})$ 都被更新。最终更新的框架被用作预测。扭转角则基于最终的残基级别隐藏状态进行预测