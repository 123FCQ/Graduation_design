import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    """时间卷积网络(TCN)的基本构建块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        
        # 计算填充大小，使输出长度与输入相同
        padding = (kernel_size - 1) * dilation
        
        # 第一个扩张卷积层
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二个扩张卷积层
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接，如果输入输出通道数不同，用1x1卷积调整
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x):
        # 保存输入尺寸，用于后续裁剪
        original_size = x.size(2)
        
        # 残差连接
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 裁剪输出，确保与原始输入大小相同
        if out.size(2) > original_size:
            out = out[:, :, :original_size]
        
        # 残差相加和激活
        return F.relu(out + residual)

class TCNFeatureExtractor(nn.Module):
    """时间卷积网络特征提取器"""
    def __init__(self, input_channels=6, output_features=32):
        super(TCNFeatureExtractor, self).__init__()
        
        # 定义参数
        hidden_units = 64
        kernel_size = 3
        
        # 创建TCN块，使用不同的扩张率
        self.tcn_layers = nn.ModuleList()
        num_levels = 6
        for i in range(num_levels):
            dilation = 2**i
            in_channels = input_channels if i == 0 else hidden_units
            self.tcn_layers.append(
                TCNBlock(in_channels, hidden_units, kernel_size, dilation)
            )
        
        # 全局池化层后的全连接层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_units, output_features)
    
    def forward(self, x):
        # 输入形状: [batch_size, sequence_length, features]
        # 需要转换为 [batch_size, features, sequence_length]
        x = x.permute(0, 2, 1)
        
        # 通过TCN层
        for layer in self.tcn_layers:
            x = layer(x)
        
        # 全局池化
        x = self.global_pool(x).squeeze(-1)
        
        # 全连接层
        x = self.fc(x)
        
        return x