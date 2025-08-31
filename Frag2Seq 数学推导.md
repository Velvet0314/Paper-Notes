### **附录 B: 片段与分子坐标系之间旋转向量的推导**

**【翻译】**
在3.3.2节中，我们得到了片段和分子局部坐标系之间的旋转矩阵 $R_{g \to m}$。然后，我们可以推导出更紧凑的旋转向量表示。具体来说，我们可以通过以下方式获得旋转角 $\psi$：

**【公式 (9)】**
$$
\psi = \arccos\left(\frac{\text{tr}(R_{g \to m}) - 1}{2}\right) \tag{9}
$$
其中 $\text{tr}(\cdot)$ 表示矩阵的迹（trace）。假设 $R_{g \to m}$ 具有以下形式：

**【公式 (10)】**
$$
R_{g \to m} = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} & r_{32} & r_{33} \end{bmatrix}
$$
旋转轴 $\mathbf{a} = (a_x, a_y, a_z)$ 可以通过以下方式获得：

**【公式 (11)-(13)】**
$$
\begin{align}
a_x &= \frac{r_{32} - r_{23}}{2\sin\psi} \tag{11} \\
a_y &= \frac{r_{13} - r_{31}}{2\sin\psi} \tag{12} \\
a_z &= \frac{r_{21} - r_{12}}{2\sin\psi} \tag{13}
\end{align}
$$
然后，我们可以通过 $\mathbf{m} = \psi\mathbf{a}$ 计算出旋转向量 $\mathbf{m} = (m_x, m_y, m_z)$。

**【详细解析与推导过程】**

**1. 核心目标**
将一个表示三维旋转的 **3x3 旋转矩阵 $R$** 转换为一个信息等价但表示更紧凑的 **3维旋转向量 $\mathbf{m}$**

**2. 推导旋转角 $\psi$ (公式 9)**
*   **理论基础**: 旋转矩阵的**迹（trace）** 与其旋转角 $\psi$ 之间存在一个固定关系，该关系源于罗德里格斯旋转公式：$$\text{tr}(R) = 1 + 2\cos(\psi)$$
* **推导**:
    1.  从 $\text{tr}(R) = 1 + 2\cos(\psi)$ 开始
    2.  移项得： $\text{tr}(R) - 1 = 2\cos(\psi)$
    3.  两边除以2：$\frac{\text{tr}(R) - 1}{2} = \cos(\psi)$
    4.  取反余弦：$$\psi = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right) \tag{9}$$
		

- 罗德里格斯旋转公式（罗德里格斯公式是连接**旋转矩阵**和**轴-角表示**的桥梁）：
	给定一个单位向量旋转轴 $\mathbf{k}$ 和一个旋转角度 $\theta$，一个任意向量 $\mathbf{v}$ 绕着轴 $\mathbf{k}$ 旋转 $\theta$ 角度后得到的新向量 $\mathbf{v}_{\text{rot}}$ 可以通过以下公式计算$$\mathbf{v}_{\text{rot}} = \mathbf{v} \cos\theta + (\mathbf{k} \times \mathbf{v}) \sin\theta + \mathbf{k}(\mathbf{k} \cdot \mathbf{v})(1 - \cos\theta)$$
	其中：
		*   $\mathbf{v} \in \mathbb{R}^3$: 需要被旋转的原始向量
		*   $\mathbf{k} \in \mathbb{R}^3$: 定义旋转轴的**单位向量** (即 $\|\mathbf{k}\|_2 = 1$)
		*   $\theta$ : 旋转的角度（弧度制）
		*   $\times$: 向量的叉乘 (Cross Product)
		*   $\cdot$ : 向量的点乘 (Dot Product)
	罗德里格斯公式可以被写成矩阵形式 $\mathbf{v}_{\text{rot}} = R \mathbf{v}$，其中旋转矩阵 $R$ 可以表示为：$$R = I + (\sin\theta)K_{\mathbf{k}} + (1-\cos\theta)K_{\mathbf{k}}^2 \tag{26}$$
	
	其中 $I$ 是单位矩阵，$K_{\mathbf{k}}$ 是由旋转轴 $\mathbf{k}=(k_x, k_y, k_z)$ 构成的**斜对称矩阵 (skew-symmetric matrix)**：$$K = \begin{bmatrix} 0 & -k_z & k_y \\ k_z & 0 & -k_x \\ -k_y & k_x & 0 \end{bmatrix} \tag{25}$$
	$K^2$ 是矩阵 $K$ 的平方
	
	目标是计算 $\text{tr}(R)$。迹（trace）是一个线性算子，这意味着 $\text{tr}(A+B) = \text{tr}(A) + \text{tr}(B)$ 并且 $\text{tr}(cA) = c \cdot \text{tr}(A)$
	
	应用到 $R$ 的表达式上：$$\text{tr}(R) = \text{tr}(I + (\sin\theta)K + (1-\cos\theta)K^2)$$
	利用迹的线性性质，可以拆分成三部分：$$\text{tr}(R) = \text{tr}(I) + \text{tr}((\sin\theta)K) + \text{tr}((1-\cos\theta)K^2)$$$$\text{tr}(R) = \text{tr}(I) + (\sin\theta)\text{tr}(K) + (1-\cos\theta)\text{tr}(K^2)$$
	 1.  计算 $\text{tr}(I)$
		单位矩阵 $I$ 的形式是：$$I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$它的迹是主对角线元素之和：$$\text{tr}(I) = 1 + 1 + 1 = 3$$
	2. 计算 $\text{tr}(K)$
		斜对称矩阵 $K$ 的形式是：$$K = \begin{bmatrix} 0 & -k_z & k_y \\ k_z & 0 & -k_x \\ -k_y & k_x & 0 \end{bmatrix}$$它的主对角线元素全部为零：$$\text{tr}(K) = 0 + 0 + 0 = 0$$
	3. 计算 $\text{tr}(K^2)$
		首先需要计算出矩阵 $K^2$$$K^2 = K \cdot K = \begin{bmatrix} 0 & -k_z & k_y \\ k_z & 0 & -k_x \\ -k_y & k_x & 0 \end{bmatrix} \begin{bmatrix} 0 & -k_z & k_y \\ k_z & 0 & -k_x \\ -k_y & k_x & 0 \end{bmatrix}$$只需要计算 $K^2$ 的对角线元素来求迹：
		 - **第(1,1)个元素**: $(0)(0) + (-k_z)(k_z) + (k_y)(-k_y) = -k_z^2 - k_y^2$
		 - **第(2,2)个元素**: $(k_z)(-k_z) + (0)(0) + (-k_x)(k_x) = -k_z^2 - k_x^2$
		 - **第(3,3)个元素**: $(-k_y)(k_y) + (k_x)(-k_x) + (0)(0) = -k_y^2 - k_x^2$
		得到 $\text{tr}(K^2)$：$$\text{tr}(K^2) = (-k_z^2 - k_y^2) + (-k_z^2 - k_x^2) + (-k_y^2 - k_x^2)$$$$\text{tr}(K^2) = -2(k_x^2 + k_y^2 + k_z^2)$$**核心关键点**: 因为 $\mathbf{k}$ 是一个**单位向量**，所以它的模长平方为1：$$\|\mathbf{k}\|^2 = k_x^2 + k_y^2 + k_z^2 = 1$$代入得到：$$\text{tr}(K^2) = -2(1) = -2$$
	4. 组合所有结果
		将计算出的三个值代入：$$\text{tr}(R) = \text{tr}(I) + (\sin\theta)\text{tr}(K) + (1-\cos\theta)\text{tr}(K^2)$$$$\text{tr}(R) = 3 + (\sin\theta)(0) + (1-\cos\theta)(-2)$$$$\text{tr}(R) = 3 + 0 - 2(1-\cos\theta)$$$$\text{tr}(R) = 3 - 2 + 2\cos\theta$$$$\text{tr}(R) = 1 + 2\cos\theta$$
**3. 推导旋转轴 $\mathbf{a}$ (公式 11-13)**
*   **理论基础**: 旋转矩阵 $R$ 与其旋转轴 $\mathbf{a}$ 构成的斜对称矩阵 $K_{\mathbf{a}}$ 之间存在关系：$$R - R^T = 2\sin(\psi) K_{\mathbf{a}}$$其中，$R^T$ 是 $R$ 的转置矩阵，$K_{\mathbf{a}}$ 定义为：$$K_{\mathbf{a}} = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0 \end{bmatrix}$$
*   **推导**:
    1.  计算 $R - R^T$：$$R - R^T = \begin{bmatrix} 0 & r_{12}-r_{21} & r_{13}-r_{31} \\ r_{21}-r_{12} & 0 & r_{23}-r_{32} \\ r_{31}-r_{13} & r_{32}-r_{23} & 0 \end{bmatrix}$$
    2.  计算 $2\sin(\psi) K_{\mathbf{a}}$：$$2\sin(\psi)K_{\mathbf{a}} = \begin{bmatrix} 0 & -2\sin(\psi)a_z & 2\sin(\psi)a_y \\ 2\sin(\psi)a_z & 0 & -2\sin(\psi)a_x \\ -2\sin(\psi)a_y & 2\sin(\psi)a_x & 0 \end{bmatrix}$$
    3.  通过逐元素比较 $R - R^T$ 和 $2\sin(\psi)K_{\mathbf{a}}$，可得：$$r_{32} - r_{23} = 2\sin(\psi)a_x \implies a_x = \frac{r_{32} - r_{23}}{2\sin\psi} \tag{11}$$$$r_{13} - r_{31} = 2\sin(\psi)a_y \implies a_y = \frac{r_{13} - r_{31}}{2\sin\psi} \tag{12}$$   $$r_{21} - r_{12} = 2\sin(\psi)a_z \implies a_z = \frac{r_{21} - r_{12}}{2\sin\psi} \tag{13}$$
**4. 最终的旋转向量 $\mathbf{m}$**
*   旋转向量 $\mathbf{m}$ 的方向是旋转轴 $\mathbf{a}$ (单位向量)，模长是旋转角 $\psi$
*   计算: $\mathbf{m} = \psi \mathbf{a}$

---
### **附录 C: 证明 (PROOFS)**

#### **C.1 引理3.2的证明 (PROOF OF LEMMA 3.2)**

**引理 3.2 (规范排序与3D分子图同构的关系)**. 令 $M_1$ 和 $M_2$ 为两个3D分子图。令 $L$ 是一个将分子 $M$ 映射到其规范SMILES产生的规范原子顺序 $L(M)$ 的函数。那么以下等价关系成立：
$$
L(M_1) = L(M_2) \iff M_1 \cong_{3D} M_2
$$
**证明：**
*   **证明 (⇐) 方向**:
    1.  **假设**: $M_1 \cong_{3D} M_2$
    2.  根据 3D 同构的定义， $M_1$ 和 $M_2$ 具有完全相同的分子图结构
    3.  规范SMILES算法为任何一个给定的分子图结构提供一个**唯一的**字符串表示
    4.  因此，$M_1$ 和 $M_2$ 的规范SMILES字符串必然相等
    5.  函数 $L(M)$ 是从这个唯一的SMILES字符串中导出的原子顺序，因此 $L(M_1) = L(M_2)$

*   **证明 (⇒) 方向**:
    1.  **假设**: $L(M_1) = L(M_2)$
    2.  这意味着它们的规范SMILES字符串是相同的。根据规范化算法的性质，这保证了 $M_1$ 和 $M_2$ 具有相同的分子图结构
    3.  因此，存在一个原子间的一一对应（双射）$b: \text{ver}(M_1) \to \text{ver}(M_2)$
    4.  因为两个分子结构相同，一个分子的三维坐标必然可以通过对另一个分子进行一系列平移和旋转得到
    5.  任何平移和旋转的组合都是一个刚体变换 $\tau \in SE(3)$
    6.  因此，存在一个 $\tau \in SE(3)$ 使得 $M_1$ 和 $M_2$ 的原子坐标可以重合
    7.  根据定义，$M_1 \cong_{3D} M_2$

**【引理3.2证明完毕】**

---
#### **C.2 引理3.3的证明 (PROOF OF LEMMA 3.3)**

**引理 3.3 (几何表示的不变性与可逆性)**. 我们构建的球坐标表示 $S=f(V)$ 和旋转向量表示 $\mathbf{m}=g(\mathbf{m},g)$ 对于任意3D刚体变换 $\tau \in SE(3)$ 都是**不变的**。并且，该表示是**可逆的**

**证明：**
* **Part 1: 证明球坐标 $S$ 的SE(3)不变性**
    1.  设任意SE(3)变换为 $\tau(\mathbf{v}) = R\mathbf{v} + \mathbf{t}$，其中 $R$ 是旋转矩阵，$\mathbf{t}$ 是平移向量
    2.  变换后的原子坐标为 $\mathbf{v}' = R\mathbf{v} + \mathbf{t}$
    3.  首先证明分子局部坐标系 $m=(\mathbf{x}, \mathbf{y}, \mathbf{z})$ 的等变性$$x = \text{normalize}(v_{\ell_2} - v_{\ell_1}), \quad y = \text{normalize}(v_{\ell_m} - v_{\ell_1}) \times x), \quad z = x \times y \tag{14}$$$$\begin{align*}x' &= \text{normalize}(Rv_{\ell_2} + t - (Rv_{\ell_1} + t)) = R(\text{normalize}(v_{\ell_2} - v_{\ell_1})) = Rx, \\y' &= \text{normalize}((Rv_{\ell_m} + t - (Rv_{\ell_1} + t)) \times x'), \\&= \text{normalize}(R((v_{\ell_m} - v_{\ell_1}) \times x)), \\ \tag{15}&= R(\text{normalize}((v_{\ell_m} - v_{\ell_1}) \times x)), \\ &= Ry, \\z' &= x' \times y' = Rx \times Ry = R(x \times y) = Rz.\end{align*}$$
    4.  计算变换后的球坐标 $(d', \theta', \phi')$$$\begin{align} \\d_i &= \|v_{\ell_i} - v_{\ell_1}\|_2, \\ \theta_i &= \arccos\left(\frac{(v_{\ell_i} - v_{\ell_1}) \cdot z}{d_{\ell_i}}\right), \tag{16} \\  \phi_i &= \operatorname{atan2}((v_{\ell_i} - v_{\ell_1}) \cdot y, (v_{\ell_i} - v_{\ell_1}) \cdot x) \end{align}$$$$\begin{align*}d'_{\ell_i} &= \|Rv_{\ell_i} + t - (Rv_{\ell_1} + t)\|_2 = \|R(v_{\ell_i} - v_{\ell_1})\|_2 = \|v_{\ell_i} - v_{\ell_1}\|_2 = d_{\ell_i}, \\\theta'_{\ell_i} &= \arccos \left( \frac{(Rv_{\ell_i} + t - (Rv_{\ell_1} + t)) \cdot z'}{d'_{\ell_i}} \right) = \arccos \left( \frac{R(v_{\ell_i} - v_{\ell_1}) \cdot Rz/d_{\ell_i}}{1} \right) = \theta_{\ell_i}, \\\phi'_{\ell_i} &= \operatorname{atan2}((Rv_{\ell_i} + t - (Rv_{\ell_1} + t)) \cdot y', (Rv_{\ell_i} + t - (Rv_{\ell_1} + t)) \cdot x'), \tag{17} \\&= \operatorname{atan2}(R(v_{\ell_i} - v_{\ell_1}) \cdot Ry, R(v_{\ell_i} - v_{\ell_1}) \cdot Rx), \\&= \operatorname{atan2}((v_{\ell_i} - v_{\ell_1}) \cdot y, (v_{\ell_i} - v_{\ell_1}) \cdot x), \\&= \phi_{\ell_i}\end{align*}$$
    5.  因此，球坐标表示 $S$ 是SE(3)不变的。其可逆性由笛卡尔-球坐标变换的标准公式保证$$f^{-1}(\cdot)=[d_{\ell_i}\sin(\theta_{\ell_i})\cos(\phi_{\ell_i}), d_{\ell_i}\sin(\theta_{\ell_i})\sin(\phi_{\ell_i}), d_{\ell_i}\cos(\theta_{\ell_i})] \tag{18}$$
	6. 由该逆变换，得到世界坐标系坐标$$
	    v_{\ell_i} = R_{m\to w} f^{-1}(S)^T + t_{m\to w} \cdot w \tag{19}$$因此，存在一个变换 $\tau \in SE(3)$，使得 $f^{-1}(S)=\tau(V)$
* **Part 2: 证明旋转向量 $\mathbf{m}$ 的SE(3)不变性**
    1.  相对旋转矩阵定义为 $R_{g \to m} = R_{m \to w}^T R_{g \to w}$
    2.  在世界坐标系的变换 $\tau \in SE(3)$ 下，分子和片段的局部坐标系到世界坐标系的旋转矩阵变为 $R'_{m \to w} = R R_{m \to w}$ 和 $R'_{g \to w} = R R_{g \to w}$
		$R$ 是世界坐标系的刚体旋转变换矩阵
    3.  计算变换后的相对旋转矩阵 $R_{g \to m}$：$$\begin{align*}R_{g \to m} &= (R'_{m \to w})^T R'_{g \to w} \\&= (R R_{m \to w})^T (R R_{g \to w}) \tag{21} \\&= (R_{m \to w}^T R^T) (R R_{g \to w}) \tag{22} \\&= R_{m \to w}^T (R^T R) R_{g \to w}  \\&= R_{m \to w}^T I R_{g \to w} \quad (\text{因为 } R^T R = I) \tag{23}\\&= R_{m \to w}^T R_{g \to w} = R_{g \to m} \tag{24} \end{align*}$$
    4.  由于 $R_{g \to m}$ 在世界坐标系的变换下保持不变，那么由它唯一确定的旋转向量 $\mathbf{m}$ 也必然是不变的
    5.  其可逆性由罗德里格斯旋转公式保证

**【引理3.3证明完毕】**

---
#### **C.3 定理3.4的证明 (PROOF OF THEOREM 3.4)**

在证明定理3.4之前，我们需要建立两个关键的前置引理

**【引理3.2: 规范排序与3D同构的等价性】**
令 $L(M)$ 为分子 $M$ 的规范原子顺序。那么，$L(M_1) = L(M_2) \iff M_1 \cong_{3D} M_2$

**【引理C.1: 几何表示与3D同构的等价性】**
令 $f \circ g$ 为将 3D 分子映射到其 SE(3) 不变几何表示（球坐标和旋转向量）的函数。那么，对于两个分子 $M_1$ 和 $M_2$ 的所有对应片段 $i$，以下等价关系成立：
$$
(f \circ g)(M_1)_i = (f \circ g)(M_2)_i \iff M_1 \cong_{3D} M_2
$$
**【引理 2 的证明】**
* **证明 (⇐) 方向: $M_1 \cong_{3D} M_2 \implies (f \circ g)(M_1) = (f \circ g)(M_2)$**
    1.  **假设**: $M_1 \cong_{3D} M_2$。这意味着存在一个刚体变换 $\tau \in SE(3)$，使得 $V_2 = \tau(V_1)$
    2.  **调用不变性**: 根据引理3.3，函数 $f \circ g$ 是 SE(3) 不变的，即 $f \circ g(M) = f \circ g(\tau(M))$
    3.  **推论**: 因此，$(f \circ g)(M_2) = (f \circ g)(\tau(M_1)) = (f \circ g)(M_1)$。这意味着它们的几何表示对于所有对应的片段都是相同的

* **证明 (⇒) 方向: $(f \circ g)(M_1) = (f \circ g)(M_2) \implies M_1 \cong_{3D} M_2$**
    1.  **假设**: 对于所有片段，$M_1$ 和 $M_2$ 的几何表示相同
    2.  **调用可逆性**: 根据引理3.3，函数 $f \circ g$ 是可逆的，即存在逆函数 $(f \circ g)^{-1}$，可以从几何表示重建分子的3D结构，其结果与原始结构最多相差一个未知的全局刚体变换
    3.  **推论**: 既然输入到逆函数 $(f \circ g)^{-1}$ 的几何表示是相同的，其输出也必然相同。这意味着存在变换 $\tau_1, \tau_2 \in SE(3)$，使得 $\tau_1(V_1) = \tau_2(V_2)$。
    4.  **得出结论**: 对上式进行变换可得 $V_2 = (\tau_2^{-1} \circ \tau_1)(V_1)$。由于 SE(3) 群的闭包性，令 $\tau'' = \tau_2^{-1} \circ \tau_1$，则 $\tau'' \in SE(3)$。这正是 3D 同构的定义，因此 $M_1 \cong_{3D} M_2$
**【引理 2 证明完毕】**

**【定理陈述】**
Frag2Seq 函数是一个满射函数，并且以下等价关系成立：
$$
\text{Frag2Seq}(M_1) = \text{Frag2Seq}(M_2) \iff M_1 \cong_{3D} M_2
$$
**证明：**
- **第一部分: 证明 Frag2Seq 的满射性 (Proof of Surjectivity)**
	1.  **目标**: 证明对于输出空间 $\mathcal{U}$ 中任意一个合法的序列 $q$，都存在一个输入空间的3D分子 $M \in \mathcal{M}$，使得 $q = \text{Frag2Seq}(M)$
	2.  **构造性证明**:
	    *   给定任意一个合法的序列 $q = \text{concat}(\mathbf{x}_1, \dots, \mathbf{x}_k)$
	    *   对于序列中的每一个元素 $\mathbf{x}_i = [s_i, d_i, \theta_i, \phi_i, m_{xi}, m_{yi}, m_{zi}]$，我们可以执行以下构造步骤：
	        a.  **重建几何表示**: 从向量中提取出球坐标 $S_i = (d_i, \theta_i, \phi_i)$ 和旋转向量 $\mathbf{m}_i = (m_{xi}, m_{yi}, m_{zi})$
	        b.  **调用可逆性 (引理 3.3)**: 函数 $f$ 和 $g$ 都是可逆的。我们可以从 $S_i$ 和 $\mathbf{m}_i$ 中唯一地重建出该片段相对于其所属分子的局部坐标系的相对位姿，即相对旋转矩阵 $R_{g_i \to m}$ 和相对平移向量 $\mathbf{t}_{g_i \to m}$
	        c.  **重建 3D 坐标**: 从片段类型 $s_i$ 可在字典中查到其内部原子在片段局部坐标系下的坐标 $V^{g_i}$。应用论文公式(7)中描述的坐标系逆变换链 ($g_i \to m \to w$)，我们可以计算出该片段所有原子在世界坐标系下的3D坐标 $V_i^w$
	    *   将所有片段的原子坐标 $\{V_1^w, \dots, V_k^w\}$ 组合起来，我们便成功构造出了一个完整的3D分子 $M$
	    *   由于这个构造过程对于任何合法的 $q$ 都是可行的，因此 **Frag2Seq** 函数是**满射**的
	**【满射性证明完毕】**
- **第二部分: 证明等价**
	- **证明 (⇐) 方向**: $M_1 \cong_{3D} M_2 \implies \text{Frag2Seq}(M_1) = \text{Frag2Seq}(M_2)$
		1.  **假设**: $M_1 \cong_{3D} M_2$
		2.  **调用引理 3.1**: 由 $M_1 \cong_{3D} M_2$ 可知，$L(M_1) = L(M_2)$。这保证了两个序列的片段类型 $s_i$ 和排列顺序是完全相同的
		3.  **调用引理 3.2**: 由 $M_1 \cong_{3D} M_2$ 可知，它们的几何表示 $(f \circ g)(M_1)$ 和 $(f \circ g)(M_2)$ 是完全相同的。这保证了两个序列中所有的几何数值 $(d_i, \theta_i, \dots)$ 都是相同的
		4.  **结论**: 因为序列的两个组成部分（片段类型/顺序 和 几何表示）都完全相同，所以拼接成的最终序列 $\text{Frag2Seq}(M_1)$ 和 $\text{Frag2Seq}(M_2)$ 必然相同
	-  **证明 (⇒) 方向**: $\text{Frag2Seq}(M_1) = \text{Frag2Seq}(M_2) \implies M_1 \cong_{3D} M_2$
		1.  **假设**: $\text{Frag2Seq}(M_1) = \text{Frag2Seq}(M_2)$
		2.  **解析假设**: 这个假设意味着两个序列逐元素相等，我们可以从中拆分出两个事实：
		    *   **(事实 A)**: 两个序列的片段类型和顺序部分相同，即 $L(M_1) = L(M_2)$
		    *   **(事实 B)**: 两个序列的几何表示部分相同，即 $(f \circ g)(M_1) = (f \circ g)(M_2)$
		3.  **应用引理得出结论**:
		    *   我们聚焦于**事实 B**
		    *   根据我们已经证明的**引理 2** 的 (⇒) 方向，即：
		        $$
		        (f \circ g)(M_1) = (f \circ g)(M_2) \implies M_1 \cong_{3D} M_2
		        $$
		    *   我们可以直接从**事实 B** 推导出最终的结论：$M_1 \cong_{3D} M_2$
**【定理3.4完整证明完毕】**