import numpy as np
import torch
from feature_extractor import extract_frequency_domain_features, prepare_time_data, apply_neighborhood_smoothing
from contrastive_tcn_trainer import extract_tcn_features

def extract_and_smooth_features(data_df, tcn_model, time_weight=0.3, freq_weight=0.7, 
                               use_time=True, use_smoothing=True, batch_size=32):
    """
    提取并平滑特征
    
    参数:
    data_df - 数据DataFrame
    tcn_model - 训练好的TCN特征提取器
    time_weight - 时域特征权重
    freq_weight - 频域特征权重
    use_time - 是否使用时域特征
    use_smoothing - 是否应用邻域平滑
    batch_size - 批次大小
    
    返回:
    时域特征, 频域特征, 邻域信息
    """
    # 获取邻域信息
    neighborhoods = data_df['neigh'].values
    
    # 提取频域特征
    print("提取频域特征...")
    freq_features_raw = extract_frequency_domain_features(data_df) # (3810, 6, 32) 不过是列表的形式
    
    # 重塑频域特征为2D
    n_samples = len(freq_features_raw)
    freq_features = np.zeros((n_samples, 6 * 32)) # (3810, 192)
    
    for i in range(n_samples):
        flat_features = []
        for channel in range(6):
            flat_features.extend(freq_features_raw[i][channel])
        freq_features[i] = np.array(flat_features)
    
    # 如果使用时域特征，提取时域特征
    if use_time:
        print("提取时域特征...")
        # 准备时域数据
        time_data = prepare_time_data(data_df)
        
        # 使用训练好的TCN提取特征
        time_features = extract_tcn_features(tcn_model, time_data, batch_size) # (3810, 32)
    else:
        time_features = None
    
    # 应用邻域平滑
    if use_smoothing:
        print("应用邻域平滑...")
        # 平滑频域特征
        freq_features = apply_neighborhood_smoothing(neighborhoods, freq_features)
        
        # 如果使用时域特征，也平滑时域特征
        if use_time:
            time_features = apply_neighborhood_smoothing(neighborhoods, time_features)
    
    return time_features, freq_features