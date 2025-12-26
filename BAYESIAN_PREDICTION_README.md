# 泰坦尼克号乘客生还概率预测 - 贝叶斯推断方法

## Bayesian Inference for Titanic Survival Prediction

本项目使用贝叶斯推断方法来预测泰坦尼克号乘客的生还概率。通过贝叶斯逻辑回归模型，我们评估了不同特征（如年龄、性别、船舱等级等）对生还概率的影响，并通过后验分布来处理参数的不确定性。

This project implements a Bayesian inference approach to predict the survival probability of Titanic passengers. Using a Bayesian logistic regression model, we evaluate the impact of different features (such as age, gender, cabin class, etc.) on survival probability and handle parameter uncertainty through posterior distributions.

---

## 功能特点 (Features)

1. **贝叶斯逻辑回归模型**: 使用PyMC构建完整的贝叶斯推断框架
2. **参数不确定性量化**: 通过后验分布评估每个参数的不确定性
3. **特征重要性分析**: 评估各个特征对生还概率的影响程度
4. **预测不确定性**: 为每个预测提供置信区间
5. **可视化分析**: 生成后验分布图和特征重要性图

---

## 技术实现 (Technical Implementation)

### 核心方法

#### 1. 贝叶斯逻辑回归模型

模型使用以下贝叶斯框架：

```
先验分布 (Prior):
  α ~ Normal(0, 2)        # 截距
  β ~ Normal(0, 2)        # 特征系数向量

似然函数 (Likelihood):
  logit(p) = α + X·β
  y ~ Bernoulli(p)        # 观测数据

后验分布 (Posterior):
  P(α, β | y, X) ∝ P(y | α, β, X) × P(α, β)
```

#### 2. MCMC采样

使用NUTS (No-U-Turn Sampler) 采样器进行后验采样：
- 默认采样2000个样本
- 1000步调优（tune）
- 2条独立的MCMC链

#### 3. 特征工程

模型考虑以下特征：
- **性别 (Sex)**: 男性/女性
- **船舱等级 (Pclass)**: 1等舱/2等舱/3等舱
- **年龄 (Age)**: 乘客年龄
- **票价 (Fare)**: 船票价格
- **兄弟姐妹/配偶数 (SibSp)**: 同船的兄弟姐妹或配偶数量
- **父母/子女数 (Parch)**: 同船的父母或子女数量
- **登船港口 (Embarked)**: S=南安普顿, C=瑟堡, Q=皇后镇

---

## 安装和使用 (Installation and Usage)

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行模型

#### 使用示例数据：

```bash
python bayesian_survival_prediction.py
```

#### 在Python代码中使用：

```python
from bayesian_survival_prediction import BayesianSurvivalPredictor
import pandas as pd

# 创建模型实例
model = BayesianSurvivalPredictor()

# 加载数据（可以使用自己的CSV文件）
# data = pd.read_csv('titanic.csv')
# X, y, feature_names = model.load_and_preprocess_data(data=data)

# 或使用内置示例数据
X, y, feature_names = model.load_and_preprocess_data()

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
model.fit(X_train, y_train, samples=2000, tune=1000, chains=2)

# 预测
prob_mean, prob_std = model.predict_proba(X_test)
predictions = model.predict(X_test)

# 查看模型摘要
model.summary()

# 生成可视化
model.plot_posterior_distributions()
model.plot_feature_importance()
```

---

## 输出结果 (Output)

运行程序后，将生成以下输出：

### 1. 终端输出
- 数据加载和预处理信息
- 模型训练进度
- 后验分布统计摘要
- 预测准确率和分类报告
- 特征影响分析
- 预测不确定性分析

### 2. 可视化文件
- `posterior_distributions.png`: 后验分布图，展示每个参数的不确定性
- `feature_importance.png`: 特征重要性图，展示各特征对预测的影响

---

## 贝叶斯方法的优势 (Advantages of Bayesian Approach)

1. **不确定性量化**: 
   - 为每个参数提供完整的后验分布，而不仅仅是点估计
   - 可以评估模型对参数值的置信程度

2. **小样本友好**: 
   - 通过先验知识可以在小样本情况下获得更稳定的估计
   - 减少过拟合风险

3. **预测区间**: 
   - 提供预测的不确定性（标准差）
   - 可以识别模型对哪些样本的预测更有信心

4. **可解释性**: 
   - 后验分布直观展示了参数的可能取值范围
   - 特征系数的后验分布反映了特征的真实影响及其不确定性

5. **稳健性**: 
   - 对异常值和噪声数据更加稳健
   - 自然地集成了正则化（通过先验分布）

---

## 示例输出解读 (Interpreting Results)

### 特征系数的含义：

- **正系数**: 该特征增加时，生还概率提高
  - 例如：如果"Sex_male"的系数为负值，说明男性生还概率更低（女性优先）
  
- **负系数**: 该特征增加时，生还概率降低
  - 例如：如果"Pclass"的系数为负值，说明船舱等级越高（数字越大），生还概率越低

### 不确定性分析：

- **后验标准差**: 反映了对参数估计的不确定性
  - 标准差越小，说明对该参数的估计越确定
  
- **预测标准差**: 反映了对个体预测的不确定性
  - 预测标准差大的样本，模型的预测不太确定

---

## 依赖项 (Dependencies)

- Python >= 3.8
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- PyMC >= 5.0.0
- ArviZ >= 0.15.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Scikit-learn >= 1.3.0
- SciPy >= 1.10.0

---

## 扩展和改进 (Extensions and Improvements)

可以考虑的扩展方向：

1. **分层贝叶斯模型**: 考虑不同组（如不同船舱等级）的分层结构
2. **非线性效应**: 使用样条或高斯过程捕获非线性关系
3. **特征交互**: 在模型中加入特征交互项
4. **模型比较**: 使用WAIC或LOO进行贝叶斯模型比较
5. **后验预测检查**: 验证模型的拟合质量

---

## 参考资料 (References)

- PyMC Documentation: https://www.pymc.io/
- Bayesian Data Analysis (Gelman et al.)
- Statistical Rethinking (McElreath)

---

## 许可证 (License)

MIT License

---

## 作者 (Author)

GitHub Copilot - Bayesian Inference Implementation
