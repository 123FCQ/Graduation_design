import numpy as np

import numpy as np
from sklearn import svm
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
from feature_extractor import apply_neighborhood_smoothing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
class HybridSVMClassifier:
    """使用混合特征的SVM分类器"""
    
    def __init__(self, num_classes, time_weight=0.3, freq_weight=0.7, use_time=True, idx_to_label=None, random_state=42):
        """
        初始化SVM分类器
        
        参数:
        num_classes - 类别数量
        time_weight - 时域特征权重
        freq_weight - 频域特征权重
        use_time - 是否使用时域特征
        """
        self.num_classes = num_classes
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.use_time = use_time
        self.model = None
        self.idx_to_label = idx_to_label
        self.random_state = random_state
    def prepare_features(self, time_features, freq_features):
        """
        准备混合特征
        
        参数:
        time_features - 时域特征
        freq_features - 频域特征
        neighborhoods - 邻域信息
        
        返回:
        混合特征矩阵
        """
        
        # 归一化特征
        if self.use_time:
            time_norm = time_features / (np.linalg.norm(time_features, axis=1, keepdims=True) + 1e-8)
            freq_norm = freq_features / (np.linalg.norm(freq_features, axis=1, keepdims=True) + 1e-8)
            
            # 加权组合
            hybrid_features = np.concatenate([
                self.time_weight * time_norm,
                self.freq_weight * freq_norm
            ], axis=1)
        else:
            freq_norm = freq_features / (np.linalg.norm(freq_features, axis=1, keepdims=True) + 1e-8)
            hybrid_features = freq_norm
        
        return hybrid_features
    def get_target_frequencies(self):
        """
        返回目标变量（表面类型）的期望频率分布
        
        这些频率可能代表实际应用中不同表面类型的预期分布，
        用于调整分类器的类别权重，以处理类别不平衡问题。
        
        返回:
        按类别索引排列的目标频率列表
        """
        # 确保idx_to_label被定义，将类别索引映射到标签名称
        # 这个映射应该在分类器初始化时提供
        # 如果没有提供，可以使用均等频率
        if not hasattr(self, 'idx_to_label'):
            return [1.0/self.num_classes for _ in range(self.num_classes)]
        
        # 每种表面类型的目标频率
        tf = {
            'carpet': 0.06,                    # 地毯
            'concrete': 0.16,                  # 混凝土
            'fine_concrete': 0.09,             # 细混凝土
            'hard_tiles': 0.06,                # 硬瓷砖
            'hard_tiles_large_space': 0.10,    # 大空间硬瓷砖
            'soft_pvc': 0.17,                  # 软质PVC
            'soft_tiles': 0.23,                # 软瓷砖
            'tiled': 0.03,                     # 瓷砖
            'wood': 0.06,                      # 木地板
        }
        
        # 计算总和，确保归一化
        s = sum(tf.values())
        
        # 返回按类别索引排列的归一化频率
        return [tf.get(self.idx_to_label.get(i, ''), 1.0/self.num_classes) / s 
                for i in range(self.num_classes)]
    def compute_class_weights(self, y_train, is_test=False):
        """
        计算类别权重
        
        基于训练集中各类别的频率和目标频率，计算类别权重。
        类别权重用于处理类别不平衡问题，使模型对样本少的类别更加敏感。
        
        参数:
        y_train - 训练集的标签
        is_test - 是否为测试模式，如果是则使用均等权重
        
        返回:
        类别权重字典，将类别索引映射到权重
        """
        import collections
        
        # 计算每个类别的样本数
        counter = collections.Counter(y_train)
        num_samples = len(y_train)
        
        # 如果有类别没有样本，返回None（使用默认权重）
        if min(counter.values()) == 0:
            return None
        
        # 如果是测试模式，使用均等权重
        if is_test:
            return {i: 1.0 for i in range(self.num_classes)}
        
        # 获取目标频率分布
        tf = self.get_target_frequencies()
        
        # 计算每个类别的权重：目标频率 * 总样本数 / 该类别样本数
        weights = {i: tf[i] * num_samples / counter[i] if counter[i] > 0 else 0.0 
                for i in range(self.num_classes)}
        
        return weights


    
    def fit(self, X_train, y_train, C=1.2, kernel='rbf'):
        """
        训练SVM模型
        
        参数:
        X_train - 训练特征
        y_train - 训练标签
        C - SVM正则化参数
        kernel - 核函数类型
        
        返回:
        训练准确率
        """
        # 计算类别权重
        class_weights = self.compute_class_weights(y_train)
        
        # 创建SVM分类器
        self.model = svm.SVC(
            C=C,
            kernel=kernel,
            gamma='scale',
            class_weight=class_weights,
            probability=True,
            random_state=self.random_state,
            cache_size=2000,
            # decision_function_shape='ovo',
            decision_function_shape='ovr'  # 使用one-vs-rest决策函数
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估训练准确率
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"训练准确率: {train_acc:.4f}")
        
        return train_acc
    
    def predict(self, X_test):
        """
        预测类别
        
        参数:
        X_test - 测试特征
        
        返回:
        预测标签
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法训练模型")
        
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test, idx_to_label=None):
        """
        评估模型性能
        
        参数:
        X_test - 测试特征
        y_test - 测试标签
        idx_to_label - 索引到标签的映射（可选）
        
        返回:
        评估指标字典, 预测标签
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法训练模型")
        
        # 预测类别
        y_pred = self.model.predict(X_test)
      
        
        # 计算其他指标
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        # 将标签转换为one-hot编码形式
        y_bin = label_binarize(y_test, classes=range(self.num_classes))

        # 获取预测概率
        y_score = self.model.predict_proba(X_test)

        # 计算多类AUC
        auc_score = roc_auc_score(y_bin, y_score, multi_class='ovr', average='weighted') * 100
        # 创建评估指标字典
        metrics = {
            "accuracy": accuracy,
            "auc_score": auc_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # 打印评估结果
        print("\n模型评估指标:")
        print(f"准确率 (Accuracy): {accuracy:.2f}%")
        print(f"AUC Score: {auc_score:.2f}%")
        print(f"精确率 (Precision): {precision:.2f}%")
        print(f"召回率 (Recall): {recall:.2f}%")
        print(f"F1 Score: {f1:.2f}%")
        
        return metrics, y_pred
    
    def save_model(self, model_path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法训练模型")
        
        joblib.dump(self.model, model_path)
        print(f"SVM模型已保存至 {model_path}")
    
    def load_model(self, model_path):
        """加载模型"""
        self.model = joblib.load(model_path)
        print(f"SVM模型已从 {model_path} 加载")