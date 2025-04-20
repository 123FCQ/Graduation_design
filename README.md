# 地面介质分类算法 - 基于时频域融合与邻域平滑驱动

## 项目概述

本项目提出了一种创新的地面介质分类算法，旨在解决移动机器人自主导航系统中地面介质识别的关键挑战。算法采用时频域融合与邻域平滑驱动的方法，有效应对传感器数据高噪声性和不同地面介质特征相似性的问题。



## 技术架构

1. **时域特征提取**：
   - 采用具有多尺度感受野的TCN（时间卷积网络）
   - 设计对比损失函数优化特征提取性能

2. **频域特征提取**：
   - 利用快速傅里叶变换(FFT)获取频域特征

3. **特征融合与平滑**：
   - 融合时域和频域特征
   - 应用四元数距离度量构建邻域网络进行特征平滑

4. **分类器**：
   - 采用支持向量机(SVM)进行最终分类

## 实验结果

- 在包含9种地面类型的数据集上达到了**99.13%**的分类准确率
- 比传统方法提高了约**9个百分点**
- 通过消融实验验证了各模块的有效性

## 环境配置

### 依赖安装

```bash
# 安装PyTorch和相关库（使用阿里云镜像加速）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 \
-i https://mirrors.aliyun.com/pypi/simple/ \
-f https://mirrors.aliyun.com/pytorch-wheels/torch_stable.html

# 安装其他依赖
pip install numpy scipy scikit-learn matplotlib pandas tqdm \
-i https://mirrors.aliyun.com/pypi/simple/
```

### 系统要求

- Python 3.8+
- CUDA 11.8+ (GPU加速)
- 8GB+ RAM

