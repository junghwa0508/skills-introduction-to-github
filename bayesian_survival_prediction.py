"""
贝叶斯推断预测泰坦尼克号乘客生还概率
Bayesian Inference for Titanic Survival Prediction

使用贝叶斯方法评估不同特征（如年龄、性别、船舱等级等）对生还概率的影响，
并通过后验分布来处理参数的不确定性。
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BayesianSurvivalPredictor:
    """
    贝叶斯生存预测模型类
    使用PyMC实现贝叶斯逻辑回归来预测泰坦尼克号乘客的生还概率
    """
    
    def __init__(self):
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_preprocess_data(self, filepath=None, data=None):
        """
        加载和预处理数据
        
        Parameters:
        -----------
        filepath : str, optional
            CSV文件路径
        data : pd.DataFrame, optional
            直接传入的DataFrame
            
        Returns:
        --------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
        feature_names : list
            特征名称列表
        """
        if data is None:
            if filepath is None:
                # 如果没有提供数据，创建示例数据
                data = self._create_sample_data()
            else:
                data = pd.read_csv(filepath)
        
        # 数据预处理
        df = data.copy()
        
        # 处理缺失值
        if 'Age' in df.columns:
            df['Age'].fillna(df['Age'].median(), inplace=True)
        if 'Fare' in df.columns:
            df['Fare'].fillna(df['Fare'].median(), inplace=True)
        if 'Embarked' in df.columns:
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        # 特征工程
        features = []
        feature_names = []
        
        # 性别特征 (Sex)
        if 'Sex' in df.columns:
            df['Sex_male'] = (df['Sex'] == 'male').astype(int)
            features.append('Sex_male')
            feature_names.append('Sex_male')
        
        # 船舱等级 (Pclass)
        if 'Pclass' in df.columns:
            features.append('Pclass')
            feature_names.append('Pclass')
        
        # 年龄 (Age)
        if 'Age' in df.columns:
            features.append('Age')
            feature_names.append('Age')
        
        # 票价 (Fare)
        if 'Fare' in df.columns:
            features.append('Fare')
            feature_names.append('Fare')
        
        # 兄弟姐妹/配偶数量 (SibSp)
        if 'SibSp' in df.columns:
            features.append('SibSp')
            feature_names.append('SibSp')
        
        # 父母/子女数量 (Parch)
        if 'Parch' in df.columns:
            features.append('Parch')
            feature_names.append('Parch')
        
        # 登船港口 (Embarked)
        if 'Embarked' in df.columns:
            df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
            df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
            features.extend(['Embarked_S', 'Embarked_C'])
            feature_names.extend(['Embarked_S', 'Embarked_C'])
        
        X = df[features].values
        y = df['Survived'].values if 'Survived' in df.columns else None
        
        self.feature_names = feature_names
        
        return X, y, feature_names
    
    def _create_sample_data(self):
        """创建示例数据用于演示"""
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'Survived': np.random.binomial(1, 0.4, n_samples),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 15, n_samples).clip(1, 80),
            'SibSp': np.random.poisson(0.5, n_samples),
            'Parch': np.random.poisson(0.3, n_samples),
            'Fare': np.random.exponential(30, n_samples),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
        }
        
        # 根据特征调整生存概率，使其更真实
        df = pd.DataFrame(data)
        for i in range(n_samples):
            prob = 0.4
            if df.loc[i, 'Sex'] == 'female':
                prob += 0.3
            if df.loc[i, 'Pclass'] == 1:
                prob += 0.2
            elif df.loc[i, 'Pclass'] == 3:
                prob -= 0.2
            if df.loc[i, 'Age'] < 10:
                prob += 0.15
            
            prob = np.clip(prob, 0.1, 0.9)
            df.loc[i, 'Survived'] = np.random.binomial(1, prob)
        
        return df
    
    def build_model(self, X_train, y_train):
        """
        构建贝叶斯逻辑回归模型
        
        Parameters:
        -----------
        X_train : np.ndarray
            训练特征
        y_train : np.ndarray
            训练标签
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_train)
        
        with pm.Model() as self.model:
            # 先验分布
            # 为每个特征的系数设置正态分布先验
            # 均值为0，标准差为2（弱信息先验）
            beta = pm.Normal('beta', mu=0, sigma=2, shape=X_scaled.shape[1])
            
            # 截距项的先验
            alpha = pm.Normal('alpha', mu=0, sigma=2)
            
            # 线性组合
            logit_p = alpha + pm.math.dot(X_scaled, beta)
            
            # 似然函数（使用逻辑函数）
            p = pm.Deterministic('p', pm.math.sigmoid(logit_p))
            
            # 观测数据的似然
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)
    
    def fit(self, X_train, y_train, samples=2000, tune=1000, chains=2):
        """
        拟合贝叶斯模型
        
        Parameters:
        -----------
        X_train : np.ndarray
            训练特征
        y_train : np.ndarray
            训练标签
        samples : int
            后验采样数量
        tune : int
            调优步数
        chains : int
            MCMC链数量
        """
        self.build_model(X_train, y_train)
        
        with self.model:
            # 使用NUTS采样器进行后验采样
            self.trace = pm.sample(
                samples, 
                tune=tune, 
                chains=chains,
                return_inferencedata=True,
                random_seed=42
            )
    
    def predict_proba(self, X_test):
        """
        预测生还概率
        
        Parameters:
        -----------
        X_test : np.ndarray
            测试特征
            
        Returns:
        --------
        prob_mean : np.ndarray
            平均预测概率
        prob_std : np.ndarray
            预测概率的标准差（不确定性）
        """
        X_scaled = self.scaler.transform(X_test)
        
        # 从后验分布中提取参数
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, X_scaled.shape[1])
        
        # 计算每个样本的预测概率
        n_samples = len(alpha_samples)
        predictions = np.zeros((n_samples, X_test.shape[0]))
        
        for i in range(n_samples):
            logit_p = alpha_samples[i] + np.dot(X_scaled, beta_samples[i])
            predictions[i] = 1 / (1 + np.exp(-logit_p))
        
        # 计算平均概率和不确定性
        prob_mean = predictions.mean(axis=0)
        prob_std = predictions.std(axis=0)
        
        return prob_mean, prob_std
    
    def predict(self, X_test, threshold=0.5):
        """
        预测类别
        
        Parameters:
        -----------
        X_test : np.ndarray
            测试特征
        threshold : float
            分类阈值
            
        Returns:
        --------
        predictions : np.ndarray
            预测的类别
        """
        prob_mean, _ = self.predict_proba(X_test)
        return (prob_mean >= threshold).astype(int)
    
    def plot_posterior_distributions(self):
        """绘制后验分布"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # 绘制截距的后验分布
        az.plot_posterior(self.trace, var_names=['alpha'], ax=axes[0])
        axes[0].set_title('Intercept (alpha)')
        
        # 绘制每个特征系数的后验分布
        for i, feature_name in enumerate(self.feature_names):
            if i + 1 < len(axes):
                az.plot_posterior(
                    self.trace, 
                    var_names=['beta'], 
                    coords={'beta_dim_0': i},
                    ax=axes[i + 1]
                )
                axes[i + 1].set_title(f'{feature_name}')
        
        plt.tight_layout()
        plt.savefig('posterior_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("后验分布图已保存为 posterior_distributions.png")
    
    def plot_feature_importance(self):
        """
        绘制特征重要性（基于后验分布的绝对值均值）
        """
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, len(self.feature_names))
        
        # 计算每个特征系数的平均绝对值
        importance = np.abs(beta_samples).mean(axis=0)
        
        # 排序
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance)), importance[indices])
        plt.yticks(range(len(importance)), [self.feature_names[i] for i in indices])
        plt.xlabel('特征重要性（系数绝对值均值）')
        plt.title('贝叶斯模型特征重要性')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("特征重要性图已保存为 feature_importance.png")
    
    def summary(self):
        """打印模型摘要"""
        print("\n" + "="*60)
        print("贝叶斯逻辑回归模型摘要")
        print("="*60)
        print(az.summary(self.trace, var_names=['alpha', 'beta']))
        
        # 打印特征名称对应关系
        print("\n特征对应关系:")
        for i, name in enumerate(self.feature_names):
            print(f"  beta[{i}]: {name}")
        print("="*60)


def main():
    """主函数：演示贝叶斯生存预测模型的使用"""
    
    print("="*60)
    print("泰坦尼克号乘客生还概率预测 - 贝叶斯推断方法")
    print("="*60)
    
    # 创建模型实例
    model = BayesianSurvivalPredictor()
    
    # 加载和预处理数据
    print("\n1. 加载和预处理数据...")
    X, y, feature_names = model.load_and_preprocess_data()
    print(f"   数据形状: X={X.shape}, y={y.shape}")
    print(f"   特征: {feature_names}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    # 训练模型
    print("\n2. 训练贝叶斯模型...")
    print("   使用NUTS采样器进行后验采样...")
    model.fit(X_train, y_train, samples=2000, tune=1000, chains=2)
    print("   训练完成！")
    
    # 显示模型摘要
    print("\n3. 模型摘要:")
    model.summary()
    
    # 预测
    print("\n4. 进行预测...")
    y_pred_proba, y_pred_std = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # 评估模型
    print("\n5. 模型评估:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   准确率: {accuracy:.4f}")
    
    print("\n   分类报告:")
    print(classification_report(y_test, y_pred, target_names=['未生还', '生还']))
    
    print("\n   混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 显示预测不确定性
    print("\n6. 预测不确定性分析:")
    print(f"   平均预测不确定性（标准差）: {y_pred_std.mean():.4f}")
    print(f"   最大预测不确定性: {y_pred_std.max():.4f}")
    print(f"   最小预测不确定性: {y_pred_std.min():.4f}")
    
    # 显示一些具体预测案例
    print("\n7. 示例预测（前5个测试样本）:")
    print("   索引 | 真实值 | 预测概率 | 不确定性 | 预测值")
    print("   " + "-"*50)
    for i in range(min(5, len(y_test))):
        print(f"   {i:4d} | {y_test[i]:6d} | {y_pred_proba[i]:8.4f} | "
              f"{y_pred_std[i]:8.4f} | {y_pred[i]:6d}")
    
    # 绘制后验分布
    print("\n8. 生成可视化...")
    model.plot_posterior_distributions()
    model.plot_feature_importance()
    
    print("\n9. 特征影响分析:")
    beta_samples = model.trace.posterior['beta'].values.reshape(-1, len(feature_names))
    for i, name in enumerate(feature_names):
        beta_mean = beta_samples[:, i].mean()
        beta_std = beta_samples[:, i].std()
        print(f"   {name:15s}: 系数均值={beta_mean:7.4f}, "
              f"标准差={beta_std:.4f}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
    
    return model, X_test, y_test, y_pred_proba, y_pred_std


if __name__ == "__main__":
    model, X_test, y_test, y_pred_proba, y_pred_std = main()
