"""
可视化贝叶斯推断结果
Visualization for Bayesian Inference Results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_results(filepath='bayesian_results.json'):
    """加载结果文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_feature_coefficients(results):
    """绘制特征系数图"""
    features = results['coefficients']['features']
    names = list(features.keys())
    values = list(features.values())
    
    # 根据系数值排序
    sorted_indices = np.argsort(values)
    names_sorted = [names[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]
    
    # 创建颜色：正值为绿色，负值为红色
    colors = ['green' if v > 0 else 'red' for v in values_sorted]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(values_sorted)), values_sorted, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel('Coefficient Value (System Shuzhi)', fontsize=12)
    ax.set_title('Feature Coefficients for Survival Prediction\n(Shengcun Yuce Tezheng Xishu)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Increase Survival (Zengjia Shengcun)'),
        Patch(facecolor='red', alpha=0.7, label='Decrease Survival (Jiangdi Shengcun)')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
    print("Feature coefficients plot saved as 'feature_coefficients.png'")
    plt.close()


def plot_feature_importance(results):
    """绘制特征重要性图"""
    importance = results['feature_importance']
    names = list(importance.keys())
    values = list(importance.values())
    
    # 排序
    sorted_indices = np.argsort(values)[::-1]
    names_sorted = [names[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(values_sorted)), values_sorted, color='steelblue', alpha=0.7)
    
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=45, ha='right')
    ax.set_ylabel('Importance (Zhongyaoxing)', fontsize=12)
    ax.set_title('Feature Importance Ranking\n(Tezheng Zhongyaoxing Paiming)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱子上添加数值
    for i, (bar, val) in enumerate(zip(bars, values_sorted)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance_ranking.png', dpi=300, bbox_inches='tight')
    print("Feature importance plot saved as 'feature_importance_ranking.png'")
    plt.close()


def plot_summary_dashboard(results):
    """创建汇总仪表板"""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. 模型准确率
    ax1 = fig.add_subplot(gs[0, 0])
    accuracy = results['accuracy']
    ax1.bar(['Accuracy'], [accuracy], color='green', alpha=0.7)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.set_title('Model Accuracy\n(Moxing Zhunquelv)', fontweight='bold')
    ax1.text(0, accuracy + 0.02, f'{accuracy:.2%}', ha='center', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 特征系数
    ax2 = fig.add_subplot(gs[0, 1])
    features = results['coefficients']['features']
    names = list(features.keys())
    values = list(features.values())
    sorted_indices = np.argsort(values)
    names_sorted = [names[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]
    colors = ['green' if v > 0 else 'red' for v in values_sorted]
    
    ax2.barh(range(len(values_sorted)), values_sorted, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(names_sorted)))
    ax2.set_yticklabels(names_sorted, fontsize=8)
    ax2.set_xlabel('Coefficient')
    ax2.set_title('Feature Coefficients\n(Tezheng Xishu)', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. 特征重要性
    ax3 = fig.add_subplot(gs[1, :])
    importance = results['feature_importance']
    imp_names = list(importance.keys())
    imp_values = list(importance.values())
    sorted_indices = np.argsort(imp_values)[::-1]
    imp_names_sorted = [imp_names[i] for i in sorted_indices]
    imp_values_sorted = [imp_values[i] for i in sorted_indices]
    
    bars = ax3.bar(range(len(imp_values_sorted)), imp_values_sorted, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(imp_names_sorted)))
    ax3.set_xticklabels(imp_names_sorted, rotation=45, ha='right')
    ax3.set_ylabel('Importance')
    ax3.set_title('Feature Importance Distribution\n(Tezheng Zhongyaoxing Fenbu)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, imp_values_sorted):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('Bayesian Survival Prediction - Summary Dashboard\n(Beiyesi Shengcun Yuce - Zongjie Yibiaopan)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("Summary dashboard saved as 'summary_dashboard.png'")
    plt.close()


def create_interpretation_text(results):
    """创建结果解释文本"""
    text = []
    text.append("="*70)
    text.append("贝叶斯生还预测模型结果解释")
    text.append("Bayesian Survival Prediction Model - Result Interpretation")
    text.append("="*70)
    text.append("")
    
    # 模型性能
    text.append("1. 模型性能 (Model Performance):")
    text.append(f"   准确率 (Accuracy): {results['accuracy']:.2%}")
    text.append("")
    
    # 特征影响分析
    text.append("2. 特征影响分析 (Feature Impact Analysis):")
    text.append("")
    text.append("   正系数特征 (Positive Coefficients - Increase Survival):")
    features = results['coefficients']['features']
    for name, value in sorted(features.items(), key=lambda x: x[1], reverse=True):
        if value > 0:
            text.append(f"   - {name:25s}: {value:+.4f}")
    text.append("")
    
    text.append("   负系数特征 (Negative Coefficients - Decrease Survival):")
    for name, value in sorted(features.items(), key=lambda x: x[1]):
        if value < 0:
            text.append(f"   - {name:25s}: {value:+.4f}")
    text.append("")
    
    # 特征重要性排名
    text.append("3. 特征重要性排名 (Feature Importance Ranking):")
    importance = results['feature_importance']
    for i, (name, imp) in enumerate(sorted(importance.items(), 
                                           key=lambda x: x[1], reverse=True), 1):
        text.append(f"   {i}. {name:25s} - 重要性: {imp:.4f}")
    text.append("")
    
    # 关键发现
    text.append("4. 关键发现 (Key Findings):")
    
    # 找出最重要的特征
    max_imp_feature = max(importance.items(), key=lambda x: x[1])
    text.append(f"   - 最重要特征: {max_imp_feature[0]} (重要性={max_imp_feature[1]:.4f})")
    
    # 找出影响最大的正面和负面特征
    pos_features = {k: v for k, v in features.items() if v > 0}
    neg_features = {k: v for k, v in features.items() if v < 0}
    
    if pos_features:
        max_pos = max(pos_features.items(), key=lambda x: x[1])
        text.append(f"   - 最增加生还概率的特征: {max_pos[0]} (系数={max_pos[1]:.4f})")
    
    if neg_features:
        max_neg = min(neg_features.items(), key=lambda x: x[1])
        text.append(f"   - 最降低生还概率的特征: {max_neg[0]} (系数={max_neg[1]:.4f})")
    
    text.append("")
    text.append("="*70)
    
    interpretation = "\n".join(text)
    
    # 保存到文件
    with open('model_interpretation.txt', 'w', encoding='utf-8') as f:
        f.write(interpretation)
    
    print("\nModel interpretation saved as 'model_interpretation.txt'")
    print(interpretation)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("贝叶斯推断结果可视化")
    print("Bayesian Inference Results Visualization")
    print("="*70 + "\n")
    
    # 加载结果
    print("Loading results from 'bayesian_results.json'...")
    results = load_results()
    print("Results loaded successfully!\n")
    
    # 创建可视化
    print("Creating visualizations...")
    plot_feature_coefficients(results)
    plot_feature_importance(results)
    plot_summary_dashboard(results)
    
    # 创建解释文本
    print("\nGenerating interpretation...")
    create_interpretation_text(results)
    
    print("\n" + "="*70)
    print("All visualizations and interpretations generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - feature_coefficients.png")
    print("  - feature_importance_ranking.png")
    print("  - summary_dashboard.png")
    print("  - model_interpretation.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
