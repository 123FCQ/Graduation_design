import numpy as np
import pandas as pd
import math
import torch
import pyquaternion as pq
from sklearn.neighbors import KernelDensity
# from model import TCNFeatureExtractor

def prepare_neigh(t):
    """
    为每个序列创建基于四元数距离的邻域图
    
    这个函数计算每个序列的邻居关系，基于机器人方向（四元数）之间的距离。
    它使用空间分区技术加速邻居搜索，并使用核密度估计计算连接概率。
    
    参数:
    t - 包含四元数方向数据的DataFrame，具有ox, oy, oz, ow列（表示四元数的四个分量）
    
    返回:
    neigh_discrete - 列表的列表，每个子列表包含与该序列相似的序列索引
    """
    # 将方向数据转换为四元数对象的列表
    q = [[pq.Quaternion(x2) for x2 in zip(*x)] for x in zip(t['ox'].values, t['oy'].values, t['oz'].values, t['ow'].values)]
    
    # 定义四元数之间的度量函数
    def metric(p0, p1, p2, p3):
        return min(
            pq.Quaternion.distance(p1, p2),
            max(
                pq.Quaternion.distance(2*p1-p0, p2),
                pq.Quaternion.distance(2*p2-p3, p1)
            )
        )
    
    # 计算两个四元数序列端点之间的距离函数
    def dist(a0, a, b, b0):
        return max(-(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w), metric(a0, a, b, b0))

    # 计算相邻四元数度量的概率分布
    d_values = []
    for e in q:
        for i in range(8):
            d = metric(e[i*16], e[i*16+1], e[i*16+2], e[i*16+3])
            d_values.append(d)

    # 设置阈值，使用99.9%分位数
    thresh = np.quantile(d_values, 0.999)

    # 定义网格大小用于空间划分
    mesh = 100

    # 将四元数坐标映射到网格位置的辅助函数
    def pos(x):
        return math.floor((x+1)/2*mesh)
    
    def pos2(x):
        return np.unique([
            max(math.floor((x+1)/2*mesh-0.5), 0),
            min(math.floor((x+1)/2*mesh+0.5), mesh-1)
        ])
    
    def ibucket(q):
        return (pos(q.x), pos(q.y), pos(q.z), pos(q.w))
    
    def gbuckets(q):
        for i in pos2(q.x):
            for j in pos2(q.y):
                for k in pos2(q.z):
                    for l in pos2(q.w):
                        yield i, j, k, l
    
    # 创建空间索引，用于快速邻域查找
    ind = [{}, {}]
    for i in range(len(q)):
        for j in [0, 1]:
            h = ind[j]
            k = ibucket(q[i][j*127])
            if k in h:
                h[k].append(i)
            else:
                h[k] = [i]

    # 初始化前向和后向邻居列表
    neigh_fwd = [[] for x in q]
    neigh_rev = [[] for x in q]
    
    # 在第一轮中，仅存储距离；之后才计算概率
    for i in range(len(q)):
        for k in gbuckets(q[i][127]):
            if k in ind[0]:
                for j in ind[0][k]:
                    if j == i: continue
                    d = dist(q[i][126], q[i][127], q[j][0], q[j][1])
                    if d >= thresh: continue
                    neigh_fwd[i].append((j, d))
                    neigh_rev[j].append((i, d))

    # 设置核密度估计的带宽
    bandwidth = 0.00005
    
    # 连接情况的核密度估计
    kd = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kd = kd.fit([[x] for x in d_values])
    
    # 连接的对数概率，减去log(len(q))作为先验概率调整
    delta0 = -math.log(len(q))
    
    # 定义函数，计算给定距离的连接概率对数
    p_connected = lambda x: kd.score_samples([[x]])[0] + delta0
    
    # 所有情况(连接和非连接)的核密度估计
    total = len(q) * len(q)
    
    # 收集所有可能是邻居样本之间的距离
    d_values2 = []
    for i in range(len(q)):
        for n, d in neigh_fwd[i]:
            d_values2.append(d)

    # 调整因子，考虑观察到的连接相对于可能连接总数的比例
    delta = math.log(len(d_values2) / total)
    
    # 为所有情况创建核密度估计
    kd2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kd2 = kd2.fit([[x] for x in d_values2])
    
    # 定义函数，计算给定距离在所有情况下的概率对数
    p_all = lambda x: kd2.score_samples([[x]])[0] + delta
    
    # 给定距离计算连接概率的函数
    prob = lambda x: math.exp(p_connected(x) - p_all(x))

    # 将距离转换为概率，并按概率排序
    for neigh in [neigh_fwd, neigh_rev]:
        for i in range(len(q)):
            na = [(n, prob(d)) for n, d in neigh[i]]
            na.sort(key=(lambda x: x[1]), reverse=True)
            neigh[i] = na

    # 初始化离散邻域列表
    neigh_discrete = [[] for x in q]

    # 设置概率和长度阈值
    p_cutoff = 0.5
    len_cutoff = 30
    
    for i in range(len(q)):
        visited = {i}
        visited_o = [i]
        front = [i, i]
        front_p = [1.0, 1.0]
        
        while (len(visited)) < len_cutoff:
            pmax = -math.inf
            nmax = None
            dmax = None
            
            for d in [0, 1]:
                for n in neigh[d][front[d]]:
                    if n[0] in visited: continue
                    p = n[1] * front_p[d]
                    if p <= pmax: continue
                    pmax = p
                    nmax = n[0]
                    dmax = d
                    
            if nmax is None: break
            if pmax < p_cutoff: break
            
            visited.add(nmax)
            visited_o.append(nmax)
            front[dmax] = nmax
            front_p[dmax] = pmax
            
        neigh_discrete[i] = visited_o

    return neigh_discrete

def add_neigh(t):
    """
    计算并将邻域信息添加到数据框中
    
    参数:
    t - 需要添加邻域信息的DataFrame
    """
    neigh = prepare_neigh(t)
    t['neigh'] = neigh
    t['stable_id'] = [i for i in range(len(neigh))]

# def extract_time_domain_features(t, model=None):
#     """
#     使用TCN模型提取时域特征
    
#     参数:
#     t - 包含传感器数据的DataFrame
#     model - 预训练的TCN模型（如果为None则创建新模型）
    
#     返回:
#     时域特征矩阵
#     """
#     # 准备数据
#     features = ['ax', 'ay', 'az', 'lx', 'ly', 'lz']
#     X = np.zeros((len(t), 128, len(features)))
    
#     for i, row in t.iterrows():
#         for j, feat in enumerate(features):
#             sensor_data = row[feat][:128]
#             # 确保数据长度为128
#             if len(sensor_data) < 128:
#                 sensor_data = np.pad(sensor_data, (0, 128 - len(sensor_data)), 'constant')
#             X[i, :, j] = sensor_data
    
#     # 转换为PyTorch张量
#     X_tensor = torch.tensor(X, dtype=torch.float32)
    
#     # 创建或使用TCN模型
#     if model is None:
#         model = TCNFeatureExtractor(output_features=32)
    
#     # 将模型设置为评估模式
#     model.eval()
    
#     # 批处理数据
#     batch_size = 32
#     n_samples = len(X_tensor)
#     tcn_features = []
    
#     with torch.no_grad():
#         for i in range(0, n_samples, batch_size):
#             batch = X_tensor[i:min(i+batch_size, n_samples)]
#             # 需要将batch的形状从[batch, seq, features]转换为[batch, features, seq]
#             batch = batch.permute(0, 2, 1)
#             batch_features = model(batch).numpy()
#             tcn_features.append(batch_features)
    
#     # 合并所有批次的结果
#     tcn_features = np.vstack(tcn_features)
    
#     return tcn_features

def extract_frequency_domain_features(t):
    """
    提取频域特征（FFT）
    
    参数:
    t - 包含传感器数据的DataFrame
    
    返回:
    FFT特征向量列表
    """
    def mfft(x):
        # 计算FFT并提取前64个复数分量（除直流分量外）
        # 除以sqrt(128.0)进行归一化
        return [np.fft.fft(v)[1:65] / math.sqrt(128.0) for v in x]

    # 要处理的特征列表
    features = ['ax', 'ay', 'az', 'lx', 'ly', 'lz']  # 角速度和线性加速度
    
    tlen = len(t)  # 总序列数

    # 计算每个特征的FFT
    fft = {f: mfft(t[f]) for f in features}

    # nfft仍然是复数数组
    nfft = {}
    
    # 对每个特征进行归一化处理
    for f in features:
        v = fft[f]  # 该特征的所有FFT结果

        lf = range(len(v[0]))  # 频率索引范围
        
        # 计算每个频率分量的平均幅度，用于归一化
        norm = [sum([np.absolute(v[i][j]) for i in range(tlen)]) / tlen for j in lf]
        
        # 对每个序列的FFT结果进行归一化
        nfft[f] = [[l[j] / norm[j] for j in lf] for l in v]

    def trans(x):
        # 将复数转换为其幅度
        return np.absolute(x)

    # 获取邻域信息 - 由prepare_neigh函数生成的每个序列的邻居列表
    neigh = t['neigh'].values

    # 初始化结果列表 - 将存储每个序列的特征向量
    r = [0 for i in range(tlen)]

    # 对每个序列提取特征
    for i in range(tlen): 
        rline = []  # 两个特征组：平均值，偶数频率
        
        for f in features:
                
            line = [trans(v) for v in nfft[f][i]]

            # 归一化特征
            avg = sum(line) / len(line)  # 计算平均值
            scale = 1 / (2 * avg)        # 计算缩放因子
            
            # 存储平均值作为特征
            # rline[0].append([avg])
            
           # 提取频率对并计算平均值
            paired_features = [scale * (line[2 * i] + line[2 * i + 1]) for i in range(len(line) // 2)]
            
            rline.append(paired_features)

        r[i] = rline  # 存储该序列的特征

    return r  # 一共有3810行，每行下面有三个子列表，分别代表该样本的平均值，偶数频率，奇数频率特征

def apply_neighborhood_smoothing(neighborhoods, features):
    """
    应用邻域平滑
    
    参数:
    neighborhoods - 邻域信息列表
    features - 要平滑的特征矩阵
    
    返回:
    平滑后的特征矩阵
    """
    # 检查输入是否为NumPy数组，如果是则转换为张量
    is_numpy = isinstance(features, np.ndarray)
    if is_numpy:
        features = torch.FloatTensor(features)
    smoothed_features = torch.zeros_like(features)
    
    for i in range(len(neighborhoods)):
        # 获取当前样本的所有邻居索引
        neighborhood = neighborhoods[i]
        # 提取邻居特征
        neighborhood_features = features[neighborhood]
        # 计算邻居特征的平均值
        smoothed_features[i] = torch.mean(neighborhood_features, dim=0)
    
    return smoothed_features

def prepare_time_data(t):
    """准备适合TCN处理的时域数据"""
    features = ['ax', 'ay', 'az', 'lx', 'ly', 'lz']
    n_samples = len(t)
    X = np.zeros((n_samples, 128, len(features)))
    
    for i, row in t.iterrows():
        for j, feat in enumerate(features):
            sensor_data = row[feat][:128]
            # 确保数据长度为128
            if len(sensor_data) < 128:
                sensor_data = np.pad(sensor_data, (0, 128 - len(sensor_data)), 'constant')
            X[i, :, j] = sensor_data
    
    return X

def extract_hybrid_features(t, time_weight=0.3, freq_weight=0.7, use_time=True, use_smoothing=True):
    """
    提取混合特征（时域 + 频域）
    
    参数:
    t - 包含传感器数据和邻域信息的DataFrame
    time_weight - 时域特征权重
    freq_weight - 频域特征权重
    use_time - 是否使用时域特征
    use_smoothing - 是否应用邻域平滑
    
    返回:
    混合特征矩阵
    """
    # 提取频域特征
    freq_features_raw = extract_frequency_domain_features(t)
     # 重塑频域特征为2D
    n_samples = len(freq_features_raw)
    freq_features = np.zeros((n_samples, 6 * 32))  # 展平为(3810, 192)
    
    for i in range(n_samples):
        flat_features = []
        for channel in range(6):
            flat_features.extend(freq_features_raw[i][channel])
        freq_features[i] = np.array(flat_features)
    time_data = prepare_time_data(t)
    # 如果需要时域特征，提取并组合
    # if use_time:
    #     # 提取时域特征
    #     # time_features = extract_time_domain_features(t)
    #      # 提取原始时域数据
    #     time_data = prepare_time_data(t)  # 新函数，准备适合TCN的时域数据
    #     # 归一化特征
    #     # time_norm = time_features / (np.linalg.norm(time_features, axis=1, keepdims=True) + 1e-8)
    #     # freq_norm = freq_features / (np.linalg.norm(freq_features, axis=1, keepdims=True) + 1e-8)
        
    #     # 加权组合
    #     # hybrid_features = np.concatenate([
    #     #     time_weight * time_norm,
    #     #     freq_weight * freq_norm
    #     # ], axis=1)
    # else:
    #     # 只使用频域特征
    #     hybrid_features = freq_features
    
    # 应用邻域平滑
    if use_smoothing:
        # hybrid_features = apply_neighborhood_smoothing(t, hybrid_features)
        # 只对频域特征进行邻域平滑 
        freq_features = apply_neighborhood_smoothing(t['neigh'].values, freq_features)
    
    return time_data, freq_features