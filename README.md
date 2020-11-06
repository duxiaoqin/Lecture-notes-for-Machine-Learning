# Lecture notes for Machine Learning (机器学习讲义)

Python 3.5

### 主要内容

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter-CrossProduct.html">Chapter 叉积</a>, <a href="Chapter-CrossProduct.pdf">(Download PDF, 9 Pages)</a>
   - 定义；
   - 方向；
   - 模；
   - 性质；
   - 应用；

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter-Gaussian.html">Chapter 高斯分布</a>, <a href="Chapter-Gaussian.pdf">(Download PDF, 17 Pages)</a>
   - 一元高斯分布；
   - 多元高斯分布；
      - 多元独立标准高斯分布；
      - 一般多元高斯分布；
      - 几何解释；
   - 高斯分布的矩；
      - 一元高斯分布情形；
      - 多元高斯分布情形；
   - 高斯分布的KL散度；
      - 一元高斯分布情形；
      - 多元高斯分布情形；
      
- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter-Jacobian.html">Chapter 雅可比矩阵</a>, <a href="Chapter-Jacobian.pdf">(Download PDF, 9 Pages)</a>
   - 定积分的换元法；
   - 极坐标换元法；
   - 雅可比矩阵；
   - 雅可比行列式与通用换元法；

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter1-CN.html">Chapter 基础知识</a>, <a href="Chapter1-CN.pdf">(Download PDF, 79 Pages)</a>
   - 模型、策略和算法；
   - 实例：多项式曲线拟合；
      - 正则项的意义；
      - 测试数据集与验证数据集的引入；
   - 模型学习的概率解释；
      - 概率学派与贝叶斯学派；
      - 最大似然估计与最大后验估计的引入；
      - 高斯分布及其性质；
      - 基于高斯分布的似然函数及最大化；
      - 最大似然方法的局限性；
      - 多项式曲线拟合问题的概率解释；
      - 多项式曲线拟合问题的改进；
      - 贝叶斯曲线拟合；
   - 附录；
      - 概率论基础知识；
      - 矩阵与向量；
      - 特征值与奇异值分解；
      - 梯度下降法；
      - 牛顿法与拟牛顿法；
      - 拉格朗日乘数法；
   - 实验：最小二乘法之直线拟合与多项式拟合，<a href="Chapter1-SourceCode.zip">Download Source Code</a>；

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter2-CN.html">Chapter 判别函数的线性分类基础</a>, <a href="Chapter2-CN.pdf">(Download PDF, 47 Pages)</a>
   - 二分类与多分类基础；
   - 基本判别函数；
      - 最小平方分类方法；
      - Fisher二分类方法；
      - 最小平方分类方法与Fisher分类方法；
      - Fisher多分类方法；
      - 感知机；
   - 附录；
      - 标量函数对矩阵的求导；
   - 实验：感知机，<a href="Perceptron.ipynb">Download Source Code</a>，<a href="Fitting-Perceptron.mp4">Result(Video)</a>；

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter3-CN.html">Chapter K近邻</a>, <a href="Chapter3-CN.pdf">(Download PDF, 20 Pages)</a>
   - kNN模型；
      - 距离度量；
      - k值；
      - 决策规则；
   - kd树；
      - kd树的生成；
      - kd树的搜索；
   - 实验：<a href="Chapter3-SourceCode.zip">Download Source Code</a>；

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter4-CN.html">Chapter 决策树</a>, <a href="Chapter4-CN.pdf">(Download PDF, 47 Pages)</a>
   - 分类决策树；
      - 信息论基础：自信息、信息熵、条件熵、联合熵、互信息、相对熵、交叉熵；
      - 特征选择；
      - 决策树的生成：ID3与C4.5算法；
      - 剪枝算法；
   - 分类回归树(CART算法)
      - 回归树的生成；
      - 分类树的生成；
      - 剪枝算法；
   - 实验：<a href="Chapter4-SourceCode.zip">Download Source Code</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter5-CN.html">Chapter 朴素贝叶斯</a>, <a href="Chapter5-CN.pdf">(Download PDF, 17 Pages)</a>
   - 分类模型；
   - 类别推理的依据——期望风险最小化；
   - 参数估计；
   - 学习算法；
   - 实验：<a href="Chapter5-SourceCode.zip">Download Source Code</a>；

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter6-CN.html">Chapter 逻辑回归</a>, <a href="Chapter6-CN.pdf">(Download PDF, 36 Pages)</a>
   - 从线性回归到二分类逻辑回归：Sigmoid函数；
   - 二分类模型的似然函数与损失函数；
   - 二分类模型中梯度的计算；
   - 代理损失函数；
   - Logistic分布；
   - Sigmoid函数与Tanh函数；
   - 对数几率与线性决策面；
   - 从二分类到多分类：Sigmoid与Softmax函数；
   - 多分类模型的似然函数与损失函数；
   - 多分类模型中梯度的计算；
   - 神经网络的视角；
   - 广义线性模型与指数族分布；
   - 逻辑回归与高斯朴素贝叶斯；
   - 实验：<a href="Chapter6-SourceCode.zip">Download Source Code</a>，<a href="Fitting-LR.mp4">Result(Video)</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter7-CN.html">Chapter 最大熵模型</a>, <a href="Chapter7.pdf">(Download PDF, 29 Pages)</a>
   - 最大熵原理；
   - 最大熵模型；
      - 条件熵；
      - 模型的定义与推导；
      - 梯度下降法；
      - 改进的迭代尺度法(IIS)；
      - 拟牛顿法；
   - 最大熵模型与逻辑回归；
   - 实验：<a href="ME.ipynb">Download Source Code</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter8-CN.html">Chapter 支持向量机</a>, <a href="Chapter8-CN.pdf">(Download PDF, 61 Pages)</a>
   - 线性可分支持向量机；
      - 几何间隔与函数间隔；
      - 学习策略：硬间隔最大化；
      - 最优分离超平面的存在性与唯一性；
      - 原始学习算法的求解实例；
      - 学习的对偶算法；
      - 对偶算法的求解实例；
   - 一般线性支持向量机；
      - 学习策略：软间隔最大化；
      - 学习的对偶算法；
      - 合页损失函数的视角；
   - 非线性支持向量机；
   - SMO算法；
      - 解析法：求解2变量二次规划子问题；
      - 启发式：2变量的选择方法；
      - E与b值的更新；
   - 实验：<a href="Chapter8-SourceCode.zip">Download Source Code</a>，<a href="Fitting-SVM.mp4">Result(Video)</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter9-CN.html">Chapter 提升方法</a>, <a href="Chapter9-CN.pdf">(Download PDF, 32 Pages)</a>
   - AdaBoost模型；
      - 加法模型与前向分步算法；
      - AdaBoost模型的推导；
      - AdaBoost算法；
      - 算法示例；
      - 训练误差分析；
   - 回归提升树；
      - 回归提升树算法；
      - 算法示例；
   - 梯度提升树
   - 实验：<a href="Chapter9-SourceCode.zip">Download Source Code</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter10-CN.html">Chapter EM算法</a>, <a href="Chapter10-CN.pdf">(Download PDF, 32 Pages)</a>
   - EM算法的引入；
   - 通用EM算法；
   - EM算法的收敛性；
   - 朴素贝叶斯的EM算法；
   - 高斯混合模型的EM算法；
   - EM算法与F函数；
   - 实验：<a href="Chapter10-SourceCode.zip">Download Source Code</a>，<a href="Fitting-EM.mp4">Result(Video)</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter11-CN.html">Chapter HMM模型</a>, <a href="Chapter11-CN.pdf">(Download PDF, 28 Pages)</a>
   - 概率计算：前向算法、后向算法及其它概率的计算；
   - 预测算法：贪心算法、Viterbi算法；
   - 学习算法：监督学习与Baum-Welch算法(EM算法)；
   - 实验：<a href="Chapter11-SourceCode.zip">Download Source Code</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter-KalmanFiltering.html">Chapter 卡尔曼滤波基础</a>, <a href="Chapter-KalmanFiltering.pdf">(Download PDF, 22 Pages)</a>
   - 卡尔曼滤波的算法推导；
   - 滤波算法的通用形式；
      - 标量情形；
      - 向量情形；
   - 实验：<a href="PositonVelocityKalmanFiltering.zip">Download Source Code</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter12-CN.html">Chapter 线性链条件随机场</a>, <a href="Chapter12-CN.pdf">(Download PDF, 67 Pages)</a>
   - 概率图模型：有向图与无向图模型；
   - 线性链条件随机场：基本形式、简化形式与矩阵形式；
   - 概率计算：前向-后向算法、常见概率的计算、特征函数期望值的计算；
   - 预测算法：贪心算法与Viterbi算法；
   - 学习算法；
      - 改进的迭代尺度法(IIS)；
      - 拟牛顿法；
   - 实验：<a href="Chapter12-SourceCode.zip">Download Source Code</a>

- <a href="https://duxiaoqin.github.io/Lecture-notes-for-Machine-Learning/Chapter-KMeans.html">Chapter k均值聚类</a>, <a href="Chapter-KMeans.pdf">(Download PDF, 39 Pages)</a>
   - 无监督学习；
   - 相似度准则；
      - 数据点之间的相似度；
         - Minkowski距离；
         - Mahalanobis距离；
         - 夹角余弦；
         - 相关系数；
      - 簇之间的相似度；
   - k均值算法的推导；
   - 与高斯混合模型EM算法的关系；
   - 附录：Mahalanobis距离变换的演示；
   - 实验：<a href="KMeans.ipynb">Download Source Code</a>
