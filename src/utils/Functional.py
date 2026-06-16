import numpy.random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from numpy.random import shuffle

from .Config import CONFIG


def self_conv(signal, moment):
    # 确保输入张量是2维的
    if signal.dim() != 2:
        raise ValueError(f'signal should be 2-dimensional, got shape {signal.shape}')
    if moment.dim() != 2:
        raise ValueError(f'moment should be 2-dimensional, got shape {moment.shape}')

    # 确保两个张量的特征维度一致，但不进行任何可能导致错误扩展的操作
    if signal.shape[1] != moment.shape[1]:
        # 取较小的维度以避免扩展
        min_features = min(signal.shape[1], moment.shape[1])
        signal = signal[:, :min_features]
        moment = moment[:, :min_features]

    # 确保两个张量的节点数量一致，但同样避免错误扩展
    if signal.shape[0] != moment.shape[0]:
        # 取较小的节点数以避免扩展
        min_nodes = min(signal.shape[0], moment.shape[0])
        signal = signal[:min_nodes, :]
        moment = moment[:min_nodes, :]

    # 对于self_conv(z, z)这种特殊情况，我们不能简单地分批处理矩阵乘法
    # 因为矩阵乘法的结果是一个完整的邻接矩阵，不能被分批处理后再拼接

    # 使用更小的批处理大小来减少显存使用，但仅用于FFT操作
    batch_size = min(200, signal.shape[0])  # 使用较大的批处理大小进行FFT

    # 如果张量足够小，直接处理
    if signal.shape[0] <= batch_size:
        signal_fft = torch.fft.fft2(input=signal, norm='ortho')
        moment_fft = torch.fft.fft2(input=moment, norm='ortho')
        result = torch.matmul(moment_fft, signal_fft.t())
        out = torch.fft.ifft2(input=result, norm='ortho').to(torch.float32)
        return out
    else:
        # 对于大张量，使用显存优化策略
        # 首先计算FFT，但分批进行以减少显存使用
        signal_fft = torch.fft.fft2(input=signal, norm='ortho')  # 整体计算，但后续分批使用
        moment_fft = torch.fft.fft2(input=moment, norm='ortho')  # 整体计算，但后续分批使用

        # 分批进行矩阵乘法，减少中间结果的显存占用
        result_parts = []
        for i in range(0, moment_fft.shape[0], batch_size):
            moment_batch = moment_fft[i:i + batch_size]
            # 与整个signal_fft的转置相乘
            result_batch = torch.matmul(moment_batch, signal_fft.t())
            # 立即进行逆FFT变换
            out_batch = torch.fft.ifft2(input=result_batch, norm='ortho').to(torch.float32)
            # 转移到CPU以释放GPU显存
            result_parts.append(out_batch.cpu())
            # 删除临时变量以释放显存
            del moment_batch, result_batch, out_batch

        # 在CPU上合并结果，然后移回GPU
        out = torch.cat(result_parts, dim=0).to(signal.device)

        # 清理临时变量
        del signal_fft, moment_fft, result_parts

        return out


def compute_loss_para(adj):
    # 避免除以零的情况
    total_elements = adj.shape[0] * adj.shape[0]
    num_pos = adj.sum().item()
    num_neg = total_elements - num_pos
    
    # 防止除以零
    if num_pos == 0 or num_neg == 0:
        pos_weight = torch.tensor(1.0)
        norm = torch.tensor(1.0)
    else:
        pos_weight = torch.tensor(num_neg / num_pos)  # 平衡正负样本
        norm = torch.tensor(total_elements / float(num_neg * 2))
    
    # 限制权重范围，避免数值不稳定
    pos_weight = torch.clamp(pos_weight, max=100.0)
    norm = torch.clamp(norm, max=1e6)
    
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(adj.device)
    
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def info_nec_loss(query, pos_key, neg_keys, temperature):
    # 计算正样本的相似度
    pos_sim = torch.matmul(query, pos_key.t()) / temperature
    
    # 计算所有负样本的相似度
    neg_sim_list = []
    for tensor in neg_keys:
        neg_sim = torch.matmul(query, tensor.t()) / temperature
        neg_sim_list.append(neg_sim)
    
    # 合并所有相似度分数
    all_sim = torch.cat([pos_sim.unsqueeze(1), torch.stack(neg_sim_list, dim=1)], dim=1)
    
    # 使用logsumexp稳定数值计算
    logits = all_sim
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def compute_epr_score(predictions, true_edges, false_edges, k=None):
    """
    计算EPR (Edge Prediction Ratio) 指标

    Args:
        predictions: 模型预测的邻接矩阵 [N, N]
        true_edges: 真实边的索引 [(i, j), ...]
        false_edges: 虚假边的索引 [(i, j), ...]
        k: 要评估的前K条边，如果为None则使用真实边数

    Returns:
        epr_score: EPR分数
        precision_at_k: 前K条边的精确率
        recall_at_k: 前K条边的召回率
    """
    if k is None:
        k = len(true_edges)

    # 获取所有边的预测分数
    all_edge_scores = []
    edge_labels = []

    # 真实边的分数和标签
    for i, j in true_edges:
        score = predictions[i, j].item()
        all_edge_scores.append(score)
        edge_labels.append(1)

    # 虚假边的分数和标签
    for i, j in false_edges:
        score = predictions[i, j].item()
        all_edge_scores.append(score)
        edge_labels.append(0)

    # 转换为numpy数组
    all_edge_scores = np.array(all_edge_scores)
    edge_labels = np.array(edge_labels)

    # 按预测分数降序排序
    sorted_indices = np.argsort(all_edge_scores)[::-1]
    sorted_labels = edge_labels[sorted_indices]

    # 计算前K条边的指标
    top_k_labels = sorted_labels[:k]

    # 计算真阳性、假阳性等
    tp = np.sum(top_k_labels == 1)
    fp = np.sum(top_k_labels == 0)
    fn = len(true_edges) - tp
    tn = len(false_edges) - fp

    # 计算EPR (真阳性比例)
    epr_score = tp / k if k > 0 else 0.0

    # 计算精确率和召回率
    precision_at_k = tp / k if k > 0 else 0.0
    recall_at_k = tp / len(true_edges) if len(true_edges) > 0 else 0.0

    return epr_score, precision_at_k, recall_at_k


def compute_epr_at_different_k(predictions, true_edges, false_edges, k_values=None):
    """
    计算不同K值下的EPR指标

    Args:
        predictions: 模型预测的邻接矩阵 [N, j], ...]
        false_edges: 虚假边的索引 [(i, j), ...]
        k_values: K值列表，如果为None则使用默认值

    Returns:
        results: 包含不同K值下EPR指标的字典
    """
    if k_values is None:
        # 默认K值：真实边数的不同比例
        total_true_edges = len(true_edges)
        k_values = [
            int(total_true_edges * 0.1),  # 10%
            int(total_true_edges * 0.2),  # 20%
            int(total_true_edges * 0.3),  # 30%
            int(total_true_edges * 0.4),  # 40%
            int(total_true_edges * 0.5),  # 50%
            int(total_true_edges * 0.6),  # 60%
            int(total_true_edges * 0.7),  # 70%
            int(total_true_edges * 0.8),  # 80%
            int(total_true_edges * 0.9),  # 90%
            total_true_edges  # 100%
        ]
        # 过滤掉0值
        k_values = [k for k in k_values if k > 0]

    results = {}
    for k in k_values:
        if k <= 0:
            continue
        epr_score, precision_at_k, recall_at_k = compute_epr_score(
            predictions, true_edges, false_edges, k
        )
        results[f'K={k}'] = {
            'EPR': epr_score,
            'Precision@K': precision_at_k,
            'Recall@K': recall_at_k
        }

    return results


def compute_network_centrality_features(g):
    """
    计算网络的多种中心性指标，为每个顶点构建网络特征

    Args:
        g: DGL图

    Returns:
        centrality_features: [N, 6] 张量，每行包含一个顶点的6个中心性指标
    """
    import networkx as nx

    # 转换为NetworkX图（无向图）
    nx_g = g.cpu().to_networkx().to_undirected()
    N = nx_g.number_of_nodes()

    # 计算各种中心性指标
    try:
        # 介数中心性
        betweenness = nx.betweenness_centrality(nx_g)
    except:
        print("警告：介数中心性计算失败，使用随机值")
        betweenness = {i: np.random.random() for i in range(N)}

    try:
        # 度中心性
        degree_cent = nx.degree_centrality(nx_g)
    except:
        print("警告：度中心性计算失败，使用随机值")
        degree_cent = {i: np.random.random() for i in range(N)}

    try:
        # 聚类系数
        clustering = nx.clustering(nx_g)
    except:
        print("警告：聚类系数计算失败，使用随机值")
        clustering = {i: np.random.random() for i in range(N)}

    try:
        # 接近中心性
        closeness = nx.closeness_centrality(nx_g)
    except:
        print("警告：接近中心性计算失败，使用随机值")
        closeness = {i: np.random.random() for i in range(N)}

    try:
        # 特征向量中心性
        eigenvector = nx.eigenvector_centrality(nx_g, max_iter=5000, tol=1e-9)
    except:
        try:
            print("警告：特征向量中心性计算失败，尝试使用PageRank")
            eigenvector = nx.pagerank(nx_g, max_iter=5000)
        except Exception as e:
            print("使用度中心性作为特征向量中心性的备份")
            # 最终回退到度中心性
            degree_cent = nx.degree_centrality(nx_g)
            eigenvector = degree_cent

    # 度（原始度数，不是归一化的）
    degrees = dict(nx_g.degree())

    # 构建特征矩阵 [N, 6]
    centrality_features = torch.zeros(N, 6)

    for i in range(N):
        centrality_features[i, 0] = betweenness.get(i, 0.0)  # 介数中心性
        centrality_features[i, 1] = degrees.get(i, 0.0)  # 度
        centrality_features[i, 2] = clustering.get(i, 0.0)  # 聚类系数
        centrality_features[i, 3] = degree_cent.get(i, 0.0)  # 度中心性
        centrality_features[i, 4] = closeness.get(i, 0.0)  # 接近中心性
        centrality_features[i, 5] = eigenvector.get(i, 0.0)  # 特征向量中心性

    # 归一化特征（可选）
    # centrality_features = (centrality_features - centrality_features.mean(dim=0)) / (centrality_features.std(dim=0) + 1e-8)

    return centrality_features


def compute_network_centrality_features_robust(g, max_iter=1000):
    """
    更鲁棒的网络中心性计算，包含错误处理和回退策略

    Args:
        g: DGL图
        max_iter: 特征向量中心性的最大迭代次数

    Returns:
        centrality_features: [N, 6] 张量，每行包含一个顶点的6个中心性指标
    """
    import networkx as nx

    # 转换为NetworkX图（无向图）
    nx_g = g.cpu().to_networkx().to_undirected()
    N = nx_g.number_of_nodes()

    # 初始化特征矩阵
    centrality_features = torch.zeros(N, 6)

    # 1. 介数中心性
    try:
        betweenness = nx.betweenness_centrality(nx_g)
        for i in range(N):
            centrality_features[i, 0] = betweenness.get(i, 0.0)
    except Exception as e:
        print(f"介数中心性计算失败: {e}，使用度中心性作为替代")
        try:
            degree_cent = nx.degree_centrality(nx_g)
            for i in range(N):
                centrality_features[i, 0] = degree_cent.get(i, 0.0)
        except:
            print("度中心性也失败，使用随机值")
            centrality_features[:, 0] = torch.rand(N)

    # 2. 度
    try:
        degrees = dict(nx_g.degree())
        for i in range(N):
            centrality_features[i, 1] = degrees.get(i, 0.0)
    except:
        print("度计算失败，使用随机值")
        centrality_features[:, 1] = torch.rand(N)

    # 3. 聚类系数
    try:
        clustering = nx.clustering(nx_g)
        for i in range(N):
            centrality_features[i, 2] = clustering.get(i, 0.0)
    except:
        print("聚类系数计算失败，使用随机值")
        centrality_features[:, 2] = torch.rand(N)

    # 4. 度中心性
    try:
        degree_cent = nx.degree_centrality(nx_g)
        for i in range(N):
            centrality_features[i, 3] = degree_cent.get(i, 0.0)
    except:
        print("度中心性计算失败，使用随机值")
        centrality_features[:, 3] = torch.rand(N)

    # 5. 接近中心性
    try:
        closeness = nx.closeness_centrality(nx_g)
        for i in range(N):
            centrality_features[i, 4] = closeness.get(i, 0.0)
    except:
        print("接近中心性计算失败，使用随机值")
        centrality_features[:, 4] = torch.rand(N)

    # 6. 特征向量中心性
    try:
        eigenvector = nx.eigenvector_centrality(nx_g, max_iter=max_iter)
        for i in range(N):
            centrality_features[i, 5] = eigenvector.get(i, 0.0)
    except Exception as e:
        print(f"特征向量中心性计算失败: {e}，使用度中心性作为替代")
        try:
            degree_cent = nx.degree_centrality(nx_g)
            for i in range(N):
                centrality_features[i, 5] = degree_cent.get(i, 0.0)
        except:
            print("度中心性也失败，使用随机值")
            centrality_features[:, 5] = torch.rand(N)

    return centrality_features


class TemporalPositionWeightedInfoNCE(nn.Module):
    """
    时序位置加权InfoNCE损失函数
    结合时间信息和位置相似性来计算对比损失
    """
    def __init__(self, temperature=0.07, position_weight=0.5, temporal_weight=0.5):
        super(TemporalPositionWeightedInfoNCE, self).__init__()
        self.temperature = temperature
        self.position_weight = position_weight
        self.temporal_weight = temporal_weight
        
    def forward(self, query, positive_key, negative_keys, position_sim,
                pseudotime_query=None, pseudotime_pos=None, pseudotime_neg=None):
        """
        前向传播
        
        Args:
            query: 查询向量 [D] 或 [1, D]
            positive_key: 正样本键向量 [D] 或 [1, D]
            negative_keys: 负样本键向量列表，每个 [D] 或 [1, D]
            position_sim: 位置相似性 [1] 或标量
            pseudotime_query: 查询的时间信息
            pseudotime_pos: 正样本的时间信息
            pseudotime_neg: 负样本的时间信息
        """
        # 确保输入是2D张量 [1, D]
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, D]
        if positive_key.dim() == 1:
            positive_key = positive_key.unsqueeze(0)  # [1, D]
        
        # 计算查询和正样本的相似度
        pos_sim = torch.matmul(query, positive_key.t()) / self.temperature  # [1, 1]
        
        # 计算查询和所有负样本的相似度
        neg_sims = []
        if negative_keys and len(negative_keys) > 0:
            for neg_key in negative_keys:
                if neg_key.dim() == 1:
                    neg_key = neg_key.unsqueeze(0)  # [1, D]
                neg_sim = torch.matmul(query, neg_key.t()) / self.temperature  # [1, 1]
                neg_sims.append(neg_sim)
        else:
            # 如果没有负样本，创建一个零向量作为占位符
            zero_neg_key = torch.zeros_like(query)
            neg_sim = torch.matmul(query, zero_neg_key.t()) / self.temperature
            neg_sims.append(neg_sim)
        
        # 组合所有相似度 [1, 1 + num_neg]
        all_sims = torch.cat([pos_sim] + neg_sims, dim=1)
        
        # 应用位置权重
        weights = torch.ones_like(all_sims)
        # 对正样本应用位置权重（第一列）
        if position_sim.numel() == 1:
            weights[:, 0] = 1.0 + self.position_weight * position_sim
        else:
            avg_position_sim = torch.mean(position_sim)
            weights[:, 0] = 1.0 + self.position_weight * avg_position_sim
        
        # 加权相似度
        weighted_sims = all_sims * weights  # [1, 1 + num_neg]
        
        # 计算InfoNCE损失
        labels = torch.zeros(1, dtype=torch.long, device=query.device)  # 正样本索引为0
        
        loss = F.cross_entropy(weighted_sims, labels)
        
        return loss