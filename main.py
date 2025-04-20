import torch
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_data_from_csv
from feature_extractor import prepare_time_data
from contrastive_tcn_trainer import train_tcn_extractor
from hybrid_svm_classifier import HybridSVMClassifier
from feature_processor import extract_and_smooth_features
from model_evaluate import plot_confusion_matrix
from excel_logger import log_experiment_to_excel

def main():
    """主函数，执行训练、评估和预测流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Surface Classification with SVM')
    parser.add_argument('--all_x', type=str, default='output.csv', help='所有特征CSV路径')
    parser.add_argument('--train_y', type=str, default='y_train.csv', help='训练标签CSV路径')
    parser.add_argument('--test_size', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, default=170, help='TCN训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--tcn_model_path', type=str, default='tcn_extractor.pth', help='TCN模型保存路径')
    parser.add_argument('--svm_model_path', type=str, default='svm_classifier.joblib', help='SVM模型保存路径')
    parser.add_argument('--mode', type=str, default='both', help='模式: train, evaluate, or both')
    parser.add_argument('--time_weight', type=float, default=0.3, help='时域特征权重')
    parser.add_argument('--freq_weight', type=float, default=0.7, help='频域特征权重')
    parser.add_argument('--use_time', type=bool, default=True, help='是否使用时域特征')
    parser.add_argument('--use_smoothing', type=bool, default=True, help='是否使用邻域平滑')
    parser.add_argument('--svm_c', type=float, default=1.2, help='SVM的C参数')
    parser.add_argument('--svm_kernel', type=str, default='rbf', help='SVM核函数类型')
    parser.add_argument('--excel_log', type=str, default='experiment_results.xlsx', help='Excel格式实验记录文件路径')
    parser.add_argument('--train_tcn', type=bool, default=True, help='是否训练TCN')
    parser.add_argument('--plot_tcn_loss', type=bool, default=True, help='是否绘制TCN训练Loss曲线')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载和预处理数据...")
    data_df, labels, label_to_idx = load_data_from_csv(
        args.all_x, args.train_y
    )
    num_classes = len(label_to_idx)
    print(f"数据加载完成: {num_classes}个类别")
    
    # 初始化TCN训练历史
    tcn_history = None
    
    # 训练时域特征提取器（如果需要）
    if args.use_time and args.mode in ['train', 'both']:
        print("\n开始训练时域特征提取器...")
        # 准备时域数据
        time_data = prepare_time_data(data_df)
        
        # 训练TCN特征提取器并保存训练历史
        tcn_model, tcn_history = train_tcn_extractor(
            time_data, 
            labels, 
            num_epochs=args.epochs, 
            learning_rate=args.lr,
            model_path=args.tcn_model_path,
            plot_loss=args.plot_tcn_loss
        )
    elif args.use_time:
        # 加载预训练的TCN模型
        from model import TCNFeatureExtractor
        input_channels = 6  # 时域输入通道数
        tcn_model = TCNFeatureExtractor(input_channels=input_channels, output_features=32)
        tcn_model.load_state_dict(torch.load(args.tcn_model_path))
        print(f"已加载TCN模型从 {args.tcn_model_path}")
    else:
        tcn_model = None
    
    # 提取特征并应用平滑
    time_features, freq_features = extract_and_smooth_features(
        data_df, 
        tcn_model,
        time_weight=args.time_weight,
        freq_weight=args.freq_weight,
        use_time=args.use_time,
        use_smoothing=args.use_smoothing
    )
    
    # 准备混合特征
    features = np.zeros((len(data_df), 0))  # 创建空特征矩阵
    
    # 添加时域特征
    if args.use_time and time_features is not None:
        # 归一化时域特征
        time_norm = time_features / (np.linalg.norm(time_features, axis=1, keepdims=True) + 1e-8)
        features = np.concatenate([features, args.time_weight * time_norm], axis=1)
    
    # 添加频域特征
    if freq_features is not None:
        # 归一化频域特征
        freq_norm = freq_features / (np.linalg.norm(freq_features, axis=1, keepdims=True) + 1e-8)
        features = np.concatenate([features, args.freq_weight * freq_norm], axis=1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=args.test_size, random_state=args.random_state
    )
    
    # 创建训练集的标签到索引的映射
    train_labels = np.unique(y_train)
    train_label_to_idx = {label: idx for idx, label in enumerate(train_labels)}
    
    # 反向映射: 索引到标签
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    train_idx_to_label = {idx: label for idx, label in enumerate(train_labels)}
    
    # 将标签转换为适合SVM的格式
    y_train_idx = np.array([train_label_to_idx[label] for label in y_train])
    
    # 现在我们确保只使用在训练集中出现的类别
    mask = np.array([label in train_label_to_idx for label in y_test])
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
    y_test_idx = np.array([train_label_to_idx[label] for label in y_test_filtered])
    
    # 初始化SVM分类器
    svm_classifier = HybridSVMClassifier(
        num_classes=len(train_labels),  # 使用训练集中的类别数
        time_weight=args.time_weight,
        freq_weight=args.freq_weight,
        use_time=args.use_time,
        idx_to_label={idx: idx_to_label[label_idx] for idx, label_idx in train_idx_to_label.items()},  # 传递训练集的标签映射
        random_state=args.random_state
    )
    
    # 变量初始化
    train_metrics = {
        'accuracy': 0.0,
        'auc_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    test_metrics = train_metrics.copy()
    predictions = None
    
    # 训练模式
    if args.mode in ['train', 'both']:
        print("\n开始训练SVM分类器...")
        train_accuracy = svm_classifier.fit(
            X_train, y_train_idx, C=args.svm_c, kernel=args.svm_kernel
        )
        
        # 保存SVM模型
        svm_classifier.save_model(args.svm_model_path)
        
        # 记录训练准确率
        train_metrics['accuracy'] = train_accuracy
    
    # 评估模式
    if args.mode in ['evaluate', 'both']:
        # 如果只进行评估，加载保存的模型
        if args.mode == 'evaluate':
            print(f"加载SVM模型从 {args.svm_model_path}...")
            svm_classifier.load_model(args.svm_model_path)
        
        print("\n开始评估...")
        test_metrics, predictions = svm_classifier.evaluate(X_test_filtered, y_test_idx, train_idx_to_label)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(y_test_idx, predictions, train_idx_to_label)
        
    
    # 记录实验结果
    if args.mode in ['both', 'evaluate'] and test_metrics['accuracy'] > 0:
        
        # 修改excel_logger.py来支持更多指标
        log_experiment_to_excel(
            args, 
            train_metrics['accuracy'], 
            test_metrics['accuracy'], 
            train_label_to_idx, 
            tcn_history=tcn_history, 
            metrics=test_metrics,
            excel_file=args.excel_log
        )

if __name__ == '__main__':
    main()