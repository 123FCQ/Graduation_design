import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, idx_to_label):
    """
    绘制混淆矩阵
    
    参数:
    y_true - 真实标签
    y_pred - 预测标签
    idx_to_label - 索引到标签的映射
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 获取类别标签 - 确保是字符串类型
    class_names = [str(idx_to_label[i]) for i in range(len(idx_to_label))]
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制热图
    sns.heatmap(
        cm, 
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # 保存并显示图表
    plt.savefig('confusion_matrix.png')
    plt.show()

