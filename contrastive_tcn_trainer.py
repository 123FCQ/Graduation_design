import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import TCNFeatureExtractor

class TimeDomainDataset(Dataset):
    """用于时域特征提取器训练的数据集"""
    def __init__(self, time_data, labels):
        self.time_data = time_data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.time_data[idx], self.labels[idx]

def contrastive_loss(features, labels, margin=1.0):
    """
    对比损失函数 - 同类样本特征接近，不同类样本特征远离
    
    参数:
    features - 特征批次 [batch_size, feature_dim]
    labels - 对应的标签 [batch_size]
    margin - 不同类别特征之间的最小距离
    
    返回:
    对比损失值
    """
    batch_size = features.size(0)
    
    # 计算特征对之间的欧氏距离矩阵
    dist_matrix = torch.cdist(features, features, p=2)
    
    # 创建标签对矩阵（True表示同类，False表示不同类）
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # 计算损失: 同类样本距离最小化，不同类样本距离大于margin
    same_class_loss = label_matrix.float() * (dist_matrix ** 2)
    diff_class_loss = (1 - label_matrix.float()) * torch.clamp(margin - dist_matrix, min=0) ** 2
    
    # 计算不包括对角线元素（自身与自身比较）的损失和
    n_pairs = batch_size * (batch_size - 1)
    loss = (same_class_loss.sum() + diff_class_loss.sum()) / n_pairs
    
    return loss
# 在contrastive_tcn_trainer.py文件开头导入matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  # 添加这一行
from torch.utils.data import Dataset, DataLoader
from model import TCNFeatureExtractor

# 修改train_tcn_extractor函数，添加绘制loss曲线的功能
def train_tcn_extractor(time_data, labels, num_epochs=50, batch_size=32, learning_rate=0.001,
                      model_path='tcn_extractor.pth', plot_loss=True):
    """
    训练时域特征提取器
    
    参数:
    time_data - 时域数据，形状为 [n_samples, seq_len, features]
    labels - 类别标签
    num_epochs - 训练轮数
    batch_size - 批次大小
    learning_rate - 学习率
    model_path - 模型保存路径
    plot_loss - 是否绘制loss曲线
    
    返回:
    训练好的特征提取器, 训练历史
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练TCN特征提取器，使用设备: {device}")
    
    # 创建数据集和加载器
    dataset = TimeDomainDataset(
        torch.FloatTensor(time_data),
        torch.LongTensor(labels)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建特征提取器
    input_channels = time_data.shape[2]  # 时域输入通道数6 (3810, 128, 6)
    model = TCNFeatureExtractor(input_channels=input_channels, output_features=32)
    model.to(device)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 记录训练历史
    history = {
        'loss': []
    }
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 进度条
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for time_data_batch, labels_batch in pbar:
            # 将数据移至设备
            time_data_batch = time_data_batch.to(device) # torch.Size([32, 128, 6])
            labels_batch = labels_batch.to(device) # torch.Size([32])
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播提取特征
            features = model(time_data_batch) # torch.Size([32, 32]) 32个样本 每个样本提取32个特征
            
            # 计算对比损失
            loss = contrastive_loss(features, labels_batch)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        epoch_loss = running_loss / len(dataloader)
        
        # 保存历史
        history['loss'].append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}')
        
        # 更新学习率
        scheduler.step(epoch_loss)
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            print(f'新的最佳损失: {best_loss:.6f} - 模型已保存到 {model_path}')
    
    # 绘制loss曲线
    if plot_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('TCN Feature Extractor Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('tcn_training_loss.png')
        plt.show()
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    
    return model, history

def extract_tcn_features(model, time_data, batch_size=32):
    """
    使用训练好的TCN模型提取特征
    
    参数:
    model - 训练好的TCN特征提取器
    time_data - 时域数据
    batch_size - 批次大小
    
    返回:
    提取的特征
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 转换为张量
    time_tensor = torch.FloatTensor(time_data)
    features = []
    
    # 批处理提取特征
    with torch.no_grad():
        for i in range(0, len(time_tensor), batch_size):
            batch = time_tensor[i:min(i+batch_size, len(time_tensor))].to(device)
            batch_features = model(batch).cpu().numpy()
            features.append(batch_features)
    
    # 合并所有批次结果
    features = np.vstack(features)
    
    return features