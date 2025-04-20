import pandas as pd
import numpy as np
import re

def parse_sensor_data(s):
    """解析形如'[-0.74857    0.33995   -0.26429...'的传感器数据"""
    if not isinstance(s, str):
        return s
    
    # 移除引号和换行符
    s = s.strip('"').replace('\n', ' ')
    
    # 使用正则表达式提取所有数字
    numbers = re.findall(r'-?\d+\.?\d*(?:e[-+]?\d+)?', s)
    
    # 转换为浮点数数组
    return np.array([float(num) for num in numbers])

def parse_neigh_list(s):
    """解析形如'[0, 475, 1496, 111, ...]'的邻域列表"""
    if not isinstance(s, str):
        return s
    
    # 直接使用eval安全地解析这种格式的列表字符串
    try:
        return eval(s)
    except:
        # 如果eval失败，使用正则表达式提取数字
        return [int(num) for num in re.findall(r'\d+', s)]

def load_data_from_csv(all_x_path, all_y_path):
    """
    从CSV文件加载数据
    
    参数:
    train_x_path - 训练特征CSV路径
    train_y_path - 训练标签CSV路径
    test_x_path - 测试特征CSV路径
    test_y_path - 测试标签CSV路径
    
    返回:
    数据DataFrame, 标签数组, 标签映射字典
    """
    # 读取标签数据
    all_y = pd.read_csv(all_y_path)
    
    # 读取预处理好的特征数据（包含邻域信息）
    all_x = pd.read_csv(all_x_path)
    
    # 转换传感器数据列
    sensor_cols = ['lx', 'ly', 'lz', 'ax', 'ay', 'az', 'ox', 'oy', 'oz', 'ow']
    for col in sensor_cols:
        if col in all_x.columns:
            all_x[col] = all_x[col].apply(parse_sensor_data)
    
    # 转换邻域列表
    if 'neigh' in all_x.columns:
        all_x['neigh'] = all_x['neigh'].apply(parse_neigh_list)
    
    # 确保数值列保持正确类型
    numeric_cols = ['series_id', 'len', 'stable_id', 'y']
    for col in numeric_cols:
        if col in all_x.columns:
            all_x[col] = pd.to_numeric(all_x[col])
    
    # 创建标签映射
    unique_labels = all_y['surface'].unique()
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 将标签转换为数字
    labels = np.array([label_to_idx[label] for label in all_y['surface']])
    
    return all_x, labels, label_to_idx