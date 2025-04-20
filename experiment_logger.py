"""
实验记录模块 - 用于记录模型训练和评估的结果
"""

import datetime
import os

def log_experiment_results(args, train_best_accuracy, test_accuracy, label_to_idx, log_file="experiment_results.txt"):
    """
    记录实验结果到文本文件
    
    参数:
    args - 命令行参数
    train_best_accuracy - 训练中获得的最佳准确率
    test_accuracy - 测试集上的准确率
    label_to_idx - 标签到索引的映射（用于记录类别数量）
    log_file - 日志文件路径
    """
    # 如果文件不存在，创建并添加标题行
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("实验结果日志\n")
            f.write("=" * 100 + "\n\n")
    
    # 添加新的实验记录
    with open(log_file, 'a', encoding='utf-8') as f:
        # 记录日期和时间
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"实验时间: {current_time}\n")
        f.write("-" * 80 + "\n")
        
        # 记录性能指标
        f.write("性能指标:\n")
        f.write(f"  训练最佳准确率: {train_best_accuracy:.2f}%\n")
        f.write(f"  测试准确率: {test_accuracy:.2f}%\n")
        f.write(f"  准确率差异 (训练-测试): {train_best_accuracy - test_accuracy:.2f}%\n")
        f.write("\n")
        
        # 记录数据集信息
        f.write("数据集配置:\n")
        f.write(f"  测试集比例: {args.test_size}\n")
        f.write(f"  随机种子: {args.random_state}\n")
        f.write(f"  类别数量: {len(label_to_idx)}\n")
        f.write("\n")
        
        # 记录模型参数
        f.write("模型配置:\n")
        f.write(f"  分类器类型: {args.classifier}\n")
        f.write(f"  使用时域特征: {args.use_time}\n")
        f.write(f"  时域特征权重: {args.time_weight}\n")
        f.write(f"  频域特征权重: {args.freq_weight}\n")
        f.write(f"  使用邻域平滑: {args.use_smoothing}\n")
        f.write(f"  训练TCN特征提取器: {args.train_tcn}\n")
        if args.classifier == 'svm':
            f.write(f"  SVM C参数: {args.svm_c}\n")
            f.write(f"  SVM核函数: {args.svm_kernel}\n")
        f.write("\n")
        
        # 记录训练参数
        f.write("训练配置:\n")
        f.write(f"  训练轮数: {args.epochs}\n")
        f.write(f"  学习率: {args.lr}\n")
        # f.write(f"  模型保存路径: {args.model_path}\n")
        f.write("\n")
        
        # 添加分隔线
        f.write("=" * 80 + "\n\n")
    
    print(f"实验结果已记录到 {log_file}")