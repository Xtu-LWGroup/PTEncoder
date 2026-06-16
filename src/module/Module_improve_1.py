"""
改进版基因调控网络模块
包含性能优化和动态可视化功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import math
import copy
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn.pytorch import SAGEConv
from src.utils.Functional import TemporalPositionWeightedInfoNCE
from src.utils.Functional import self_conv, compute_loss_para
from src.utils.Config import CONFIG
import torch.nn.functional as func
import torch.nn.init as init

class TemporalPositionalEncoding(nn.Module):
    """
    时间位置编码层，用于处理伪时间信息
    """

    def __init__(self, d_model, max_time=1.0):
        super(TemporalPositionalEncoding, self).__init__()
        # 确保d_model是整数
        d_model = int(d_model)

        # 创建位置编码矩阵
        pe = torch.zeros(1000, d_model)  # 预设1000个时间点
        position = torch.arange(0, 1000).unsqueeze(1).float()

        # 计算分母项，确保维度匹配
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))

        # 对于奇数维度的d_model，需要特殊处理
        sin_input = position * div_term[:min(len(div_term), (d_model + 1) // 2)]
        cos_input = position * div_term[:min(len(div_term), d_model // 2)]

        pe[:, 0::2] = torch.zeros_like(pe[:, 0::2])  # 初始化偶数列
        pe[:, 1::2] = torch.zeros_like(pe[:, 1::2])  # 初始化奇数列

        # 填充偶数列 (0, 2, 4, ...)
        if sin_input.size(1) > 0:
            pe[:, 0::2][:, :sin_input.size(1)] = torch.sin(sin_input)
        # 填充奇数列 (1, 3, 5, ...)
        if cos_input.size(1) > 0:
            pe[:, 1::2][:, :cos_input.size(1)] = torch.cos(cos_input)

        # 注册为buffer，这样它会被移动到正确的设备上
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, time_seq):
        """
        前向传播

        Args:
            time_seq: 时间序列张量，形状为 [batch_size, seq_len] 或 [seq_len]

        Returns:
            位置编码张量
        """
        # 确保时间值在有效范围内
        time_seq = torch.clamp(time_seq, 0, 1.0)

        # 将时间映射到位置索引
        positions = (time_seq * (self.pe.size(0) - 1)).long()

        # 根据输入维度获取编码
        if time_seq.dim() == 1:
            # 单个序列
            return self.pe[positions]
        else:
            # 批处理序列
            batch_size, seq_len = time_seq.shape
            encoded = self.pe[positions.view(-1)].view(batch_size, seq_len, -1)
            return encoded


class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class PGNN_layer(nn.Module):
    """
    改进的PGNN层，优化显存使用和数值稳定性
    """

    def __init__(self, input_dim, feature_dim, dist_trainable=True, act=F.relu):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.dist_trainable = dist_trainable
        self.act = act
        
        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, feature_dim, 1)
        
        # 使用更小的隐藏层维度以减少参数
        self.linear_hidden = nn.Linear(input_dim * 2, feature_dim)
        self.linear_out_position = nn.Linear(feature_dim, 1)

    def ensure_tensor_device(self, tensor, device):
        """确保张量在正确的设备上"""
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor

    def ensure_graph_device(self, graph, device):
        """确保图在正确的设备上"""
        if graph.device != device:
            graph = graph.to(device)
        return graph

    def forward(self, feature, dists_max, dists_argmax):
        device = feature.device

        # 确保所有张量都在正确的设备上
        dists_max = self.ensure_tensor_device(dists_max, device)
        dists_argmax = self.ensure_tensor_device(dists_argmax, device)
        
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()
        
        # 检查张量形状
        if dists_argmax.dim() == 3:
            dists_argmax = dists_argmax.squeeze(2)  # 移除多余的维度

        # 确保维度匹配
        if dists_argmax.shape[0] != feature.shape[0]:
            min_nodes = min(dists_argmax.shape[0], feature.shape[0])
            dists_argmax = dists_argmax[:min_nodes]
            feature = feature[:min_nodes]

        if dists_max.shape[0] != feature.shape[0]:
            min_nodes = min(dists_max.shape[0], feature.shape[0])
            dists_max = dists_max[:min_nodes]
            feature = feature[:min_nodes]

        # 处理dists_argmax的维度
        if dists_argmax.dim() == 1:
            dists_argmax = dists_argmax.unsqueeze(1)  # 添加维度使其成为二维
        elif dists_argmax.dim() == 3:
            # 如果dists_argmax是三维的，将其压缩为二维
            dists_argmax = dists_argmax.squeeze(1)  # 移除中间的维度

        # 确保dists_argmax的第二个维度不超过feature的节点数
        max_neighbors = feature.shape[0]
        dists_argmax = torch.clamp(dists_argmax, 0, max_neighbors - 1)

        # 获取邻居特征
        # 检查dists_argmax的形状，并确保其与feature的形状兼容
        dists_argmax_flat = dists_argmax.flatten()

        # 限制索引范围以防止越界
        dists_argmax_clamped = torch.clamp(dists_argmax_flat, 0, feature.shape[0] - 1)

        # 获取对应的特征
        subset_features = feature[dists_argmax_clamped, :]

        # 计算期望的形状
        expected_nodes = dists_argmax.shape[0]  # 保持原始的节点数
        expected_neighbors = dists_argmax.shape[1] if len(dists_argmax.shape) > 1 else 1  # 保持原始的邻居数
        expected_features = feature.shape[1]  # 保持原始的特征数

        # 检查是否有足够的元素进行重塑
        required_elements = expected_nodes * expected_neighbors * expected_features
        available_elements = subset_features.numel()

        if available_elements >= required_elements:
            # 如果有足够的元素，直接重塑
            subset_features = subset_features[:required_elements].reshape(
                (expected_nodes, expected_neighbors, expected_features)
            )
        else:
            # 如果元素不足，需要调整形状
            # 首先尝试调整邻居数量
            max_possible_neighbors = available_elements // (
                        expected_nodes * expected_features) if expected_nodes * expected_features > 0 else 0

            if max_possible_neighbors > 0:
                # 如果至少可以有一个邻居
                adjusted_neighbors = max(1, max_possible_neighbors)
                adjusted_elements = expected_nodes * adjusted_neighbors * expected_features
                subset_features = subset_features[:adjusted_elements].reshape(
                    (expected_nodes, adjusted_neighbors, expected_features)
                )
            else:
                # 如果连一个完整的邻居都不够，尝试调整节点数量
                max_possible_nodes = available_elements // expected_features if expected_features > 0 else 0
                if max_possible_nodes > 0:
                    adjusted_nodes = max(1, max_possible_nodes)
                    adjusted_elements = adjusted_nodes * 1 * expected_features  # 使用1个邻居
                    subset_features = subset_features[:adjusted_elements].reshape(
                        (adjusted_nodes, 1, expected_features)
                    )
                else:
                    # 最后手段：创建最小尺寸的张量
                    subset_features = subset_features[:expected_features].reshape(
                        (1, 1, expected_features)
                    )
                    # 扩展到期望的形状
                    subset_features = subset_features.expand(expected_nodes, 1, expected_features)

        # 计算消息
        messages = subset_features * dists_max.unsqueeze(-1)

        # 确保messages和dists_max形状匹配
        if messages.shape[0] != dists_max.shape[0]:
            min_nodes = min(messages.shape[0], dists_max.shape[0])
            messages = messages[:min_nodes]
            dists_max = dists_max[:min_nodes]

        # 添加自特征
        # 确保messages和self_feature在最后一个维度上具有相同的大小
        # 调整self_feature的形状以匹配messages的形状
        self_feature = feature.unsqueeze(1).repeat(1, messages.shape[1], 1)

        # 确保messages和self_feature的形状兼容
        if messages.shape[-1] != self_feature.shape[-1]:
            # 如果特征维度不匹配，调整到最小维度
            min_features = min(messages.shape[-1], self_feature.shape[-1])
            messages = messages[:, :, :min_features]
            self_feature = self_feature[:, :, :min_features]

        messages = torch.cat((messages, self_feature), dim=-1)

        # 如果维度不匹配，调整messages的维度以匹配期望的input_dim*2
        expected_input_dim = self.input_dim * 2
        if messages.shape[-1] != expected_input_dim:
            if messages.shape[-1] > expected_input_dim:
                # 如果输入维度太大，截取前expected_input_dim个特征
                messages = messages[..., :expected_input_dim]
            else:
                # 如果输入维度太小，用零填充
                padding = torch.zeros(
                    messages.shape[:-1] + (expected_input_dim - messages.shape[-1],),
                    device=messages.device,
                    dtype=messages.dtype
                )
                messages = torch.cat([messages, padding], dim=-1)

        # 使用改进的激活函数处理以减少显存使用
        messages = self.linear_hidden(messages)

        # 使用更小的批处理大小处理激活函数
        if messages.dim() > 2:
            original_shape = messages.shape
            messages = messages.view(-1, original_shape[-1])

            # 使用极小的批处理大小
            batch_size = 5  # 极小的批处理大小以节省显存
            result_parts = []

            for i in range(0, messages.shape[0], batch_size):
                batch = messages[i:i + batch_size]
                # 使用torch.no_grad来减少显存使用
                with torch.no_grad():
                    batch_act = self.act(batch)
                # 立即将处理完的批次转移到CPU以释放GPU显存
                result_parts.append(batch_act.cpu())
                # 删除临时变量以释放GPU显存
                del batch, batch_act

                # 强制垃圾回收以释放显存
                if i % 20 == 0:  # 每处理20个批次后清理一次
                    torch.cuda.empty_cache()

            # 在CPU上合并结果，然后移回GPU
            messages = torch.cat(result_parts, dim=0).to(messages.device)
            messages = messages.view(original_shape)
            # 清理临时变量
            del result_parts
            # 再次清理显存
            torch.cuda.empty_cache()
        else:
            messages = self.act(messages)

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


class GlobalPathEncoder(nn.Module):
    """
    全局路径编码器，使用改进的PGNN
    """

    def __init__(self, input_dim, hidden_dim, num_anchors, centrality_method='degree'):
        super(GlobalPathEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors
        self.centrality_method = centrality_method

        # 使用改进的PGNN层
        self.pgnn_layer = PGNN_layer(input_dim, hidden_dim)
        
        # 网络中心性计算所需的库
        try:
            import networkx as nx
            self.nx_available = True
        except ImportError:
            self.nx_available = False
            print("Warning: NetworkX not available. Centrality-based anchor selection disabled.")
        
        # 用于存储选定的锚点索引
        self.anchor_indices = None
        
        # 距离计算器的最大距离
        self.max_distance = 10
        
        # 位置嵌入层
        self.position_embedding = nn.Linear(num_anchors, hidden_dim)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 添加时间位置编码
        self.temporal_pos_encoding = TemporalPositionalEncoding(hidden_dim)

    def select_anchors_by_centrality(self, graph):
        """
        基于网络中心性选择锚点
        """
        if not self.nx_available:
            # 如果没有NetworkX，则随机选择锚点
            num_nodes = graph.number_of_nodes()
            anchor_indices = torch.randperm(num_nodes)[:min(self.num_anchors, num_nodes)]
            return anchor_indices
        
        import networkx as nx
        
        # 将DGL图转换为NetworkX图
        nx_graph = graph.cpu().to_networkx().to_undirected()
        
        if self.centrality_method == 'degree':
            # 度中心性
            centrality = nx.degree_centrality(nx_graph)
        elif self.centrality_method == 'betweenness':
            # 介数中心性
            centrality = nx.betweenness_centrality(nx_graph)
        elif self.centrality_method == 'closeness':
            # 接近中心性
            centrality = nx.closeness_centrality(nx_graph)
        elif self.centrality_method == 'eigenvector':
            # 特征向量中心性
            centrality = nx.eigenvector_centrality(nx_graph, max_iter=1000)
        elif self.centrality_method == 'pagerank':
            # PageRank中心性
            centrality = nx.pagerank(nx_graph)
        else:
            # 默认使用度中心性
            centrality = nx.degree_centrality(nx_graph)
        
        # 将中心性值转换为张量
        num_nodes = graph.number_of_nodes()
        centrality_values = torch.tensor([centrality.get(i, 0.0) for i in range(num_nodes)])
        
        # 选择中心性最高的节点作为锚点
        _, anchor_indices = torch.topk(centrality_values, min(self.num_anchors, num_nodes))
        
        return anchor_indices

    def _compute_distances_to_anchors(self, graph, anchor_indices):
        """计算所有节点到锚点集的距离"""
        num_nodes = graph.number_of_nodes()
        num_anchors = len(anchor_indices)
        device = graph.device

        # 初始化距离矩阵
        distances = torch.full((num_nodes, num_anchors), float('inf'),
                               device=device, dtype=torch.float32)

        # 对每个锚点计算最短路径
        for i, anchor in enumerate(anchor_indices):
            # 使用BFS计算到锚点的距离
            dist_to_anchor = self._bfs_distance(graph, anchor.item(), num_nodes, device)
            distances[:, i] = dist_to_anchor

        # 将无穷大距离设为最大距离
        distances = torch.clamp(distances, max=self.max_distance)

        return distances

    def _bfs_distance(self, graph, start_node, num_nodes, device):
        """使用BFS计算到起始节点的距离"""
        distances = torch.full((num_nodes,), float('inf'),
                               device=device, dtype=torch.float32)
        distances[start_node] = 0

        # 使用队列进行BFS
        queue = [start_node]
        visited = set([start_node])

        while queue:
            current = queue.pop(0)
            # 获取邻居节点
            neighbors = graph.successors(current) if hasattr(graph, 'successors') else []
            if len(neighbors) == 0:
                # 如果没有直接后继节点，尝试使用邻接矩阵
                row = graph.adjacency_matrix().to_dense()[current]
                neighbors = torch.nonzero(row).squeeze(-1)

            for neighbor in neighbors:
                neighbor_idx = neighbor.item() if hasattr(neighbor, 'item') else int(neighbor)
                if neighbor_idx not in visited:
                    distances[neighbor_idx] = distances[current] + 1
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        return distances

    def forward(self, graph, features, pseudotime=None):
        # num_nodes = graph.number_of_nodes()
        device = graph.device
        # 确保device是有效的
        if device is None:
            device = torch.device('cpu')

        # 确保特征在正确的设备上
        features = features.to(device)
        
        # 选择锚点（如果尚未选择）
        if self.anchor_indices is None:
            self.anchor_indices = self.select_anchors_by_centrality(graph).to(device)

        # 计算所有节点到锚点的距离
        distances = self._compute_distances_to_anchors(graph, self.anchor_indices)
        
        # 归一化距离
        distances_norm = distances / (distances.max() + 1e-8)
        
        # 找到每个节点到锚点的最大距离和对应锚点
        dists_max, dists_argmax = torch.max(distances_norm, dim=1)
        # 确保所有张量在同一设备上
        dists_max = dists_max.to(device)
        dists_argmax = dists_argmax.to(device)
        
        # 使用改进的PGNN层处理
        _, global_features = self.pgnn_layer(features, dists_max, dists_argmax.unsqueeze(1))
        
        # 位置嵌入
        position_emb = self.position_embedding(distances_norm)
        
        # 融合全局特征和位置嵌入
        global_features = global_features + position_emb
        
        # 如果提供了伪时间，将其编码并融合到特征中
        if pseudotime is not None and pseudotime.nelement() > 0:
            # 确保pseudotime形状正确
            if pseudotime.dim() == 1:
                pseudotime = pseudotime.unsqueeze(1)

            # 获取时间编码
            time_encoded = self.temporal_pos_encoding(pseudotime)
            
            # 调整time_encoded的形状以匹配global_features
            if time_encoded.shape[0] != global_features.shape[0]:
                min_nodes = min(time_encoded.shape[0], global_features.shape[0])
                time_encoded = time_encoded[:min_nodes]
                global_features = global_features[:min_nodes]

            if time_encoded.shape[1] != global_features.shape[1]:
                # 如果维度不匹配，使用线性层调整
                time_encoded = F.interpolate(
                    time_encoded.unsqueeze(0).transpose(1, 2),
                    size=global_features.shape[1],
                    mode='linear',
                    align_corners=True
                ).squeeze(0).transpose(0, 1)

            # 融合时间信息和全局特征
            global_features = global_features + time_encoded

        # 通过输出层
        global_features = self.output_layer(global_features)
        global_features = self.norm(global_features)

        return global_features

    def get_selected_anchors(self):
        """
        获取所选的锚点索引
        """
        return self.anchor_indices


class AttentionFusion(nn.Module):
    """注意力融合层 - 结合局部和全局特征"""

    def __init__(self, feature_dim=128):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim

        # 注意力权重计算
        self.attention_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),  # 2个权重：局部和全局
            nn.Softmax(dim=-1)
        )

        # 融合后的特征变换
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, local_features, global_features):
        # 确保两个特征在同一设备上
        device = local_features.device
        global_features = global_features.to(device)
        # 计算注意力权重
        combined = torch.cat([local_features, global_features], dim=-1)
        attention_weights = self.attention_mlp(combined)  # [num_nodes, 2]

        # 加权融合
        weighted_local = local_features * attention_weights[:, 0:1]
        weighted_global = global_features * attention_weights[:, 1:2]

        # 拼接并变换
        fused_features = torch.cat([weighted_local, weighted_global], dim=-1)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.norm(fused_features)

        return fused_features, attention_weights


class DualPathEncoder(nn.Module):
    """
    双路径编码器，结合局部和全局信息，支持时间核
    """

    def __init__(self, input_dim, hidden_dim, num_anchors, centrality_method='degree',
                 use_temporal_kernel=True, temporal_sigma=0.1):
        super(DualPathEncoder, self).__init__()
        self.local_encoder = LocalPathEncoder(
            input_dim, hidden_dim,
            use_temporal_kernel=use_temporal_kernel,
            temporal_sigma=temporal_sigma
        )
        self.global_encoder = GlobalPathEncoder(input_dim, hidden_dim, num_anchors, centrality_method)

        # 注意力融合层
        self.attention_fusion = AttentionFusion(hidden_dim)

    def forward(self, graph, features, motif_adjs=None, pseudotime=None):
        # 确保特征在正确的设备上
        device = graph.device
        if device is None:
            device = torch.device('cpu')
        features = features.to(device)
        
        # 局部特征（启用时间核）
        if motif_adjs is None:
            raise ValueError('motif_adjs must be provided for LocalPathEncoder')
        local_features, _ = self.local_encoder(features, motif_adjs, pseudotime=pseudotime)
        
        # 全局特征
        global_features = self.global_encoder(graph, features, pseudotime=pseudotime)
        
        # 确保两个特征在同一设备上
        local_features = local_features.to(device)
        global_features = global_features.to(device)

        # 注意力融合
        fused_features, attention_weights = self.attention_fusion(local_features, global_features)

        return fused_features, attention_weights


class EnhancedCVGAE(nn.Module):
    """
    增强版CVGAE，集成改进的编码器和Neural ODE
    """

    def __init__(self, n_gene, n_feat, hidden_dim, num_anchors, centrality_method='degree', 
                 use_neural_ode=True, ode_method='euler', ode_steps=5,
                 use_temporal_kernel=True, temporal_sigma=0.1):
        super(EnhancedCVGAE, self).__init__()
        self.n_gene = n_gene
        self.n_feat = n_feat
        self.hidden_dim = hidden_dim
        self.use_neural_ode = use_neural_ode

        # 使用双路径编码器（启用时间核）
        self.encoder = DualPathEncoder(
            n_feat, hidden_dim, num_anchors, centrality_method,
            use_temporal_kernel=use_temporal_kernel,
            temporal_sigma=temporal_sigma
        )

        # Neural ODE层（如果启用）
        if self.use_neural_ode:
            self.neural_ode = NeuralODELayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                method=ode_method,
                num_steps=ode_steps
            )

        # 变分推断层
        self.mean_layer = nn.Linear(hidden_dim, hidden_dim)
        self.var_layer = nn.Linear(hidden_dim, hidden_dim)

        # 解码器
        from .model import BaseBlock  # BaseBlock在model.py中
        self.decoder = BaseBlock(
            n_feat=n_gene, layer_depth=4, dropout=0.1, activation='elu'
        )

        self.h_norm = nn.BatchNorm1d(hidden_dim)
        self.z_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, block, motif_adjs=None, pseudotime=None):
        # 获取编码特征
        h, attention_weights = self.encoder(block, block.ndata['feature'], motif_adjs, pseudotime)

        h = self.h_norm(h)
        block.ndata['feature'] = h

        # 如果启用了Neural ODE且提供了伪时间，通过ODE演化状态
        if self.use_neural_ode and pseudotime is not None:
            # 确保pseudotime形状正确
            if pseudotime.dim() == 0:
                pseudotime_expanded = pseudotime.unsqueeze(0).expand(h.size(0))
            elif pseudotime.dim() == 1:
                if pseudotime.size(0) != h.size(0):
                    # 如果长度不匹配，广播到所有节点
                    pseudotime_expanded = pseudotime[0].unsqueeze(0).expand(h.size(0))
                else:
                    pseudotime_expanded = pseudotime
            else:
                pseudotime_expanded = pseudotime.view(-1)[:h.size(0)]
            
            # 通过Neural ODE演化：从t=0到t=pseudotime
            h = self.neural_ode(h, pseudotime_expanded)

        # 变分推断
        mean = self.mean_layer(h)
        var = self.var_layer(h)

        # 重参数化技巧
        gaussian_noise = torch.randn(mean.size(0), mean.size(1), device=mean.device)
        z = mean + gaussian_noise * torch.exp(var)

        z = self.z_norm(z)

        # 生成预测的邻接矩阵
        inferred_net = self_conv(z, z)
        inferred_net = self.decoder(inferred_net)

        return inferred_net, mean, var, z, h, attention_weights


class EnhancedCLGVAE(pl.LightningModule):
    """
    改进版的增强对比学习变分图自编码器，集成Neural ODE和时间核
    """

    def __init__(self, n_gene, n_feat, hidden_dim, num_anchors, device='cuda:0', 
                 motif_adjs=None, centrality_method='degree',
                 use_neural_ode=True, ode_method='euler', ode_steps=5,
                 use_temporal_kernel=True, temporal_sigma=0.1):
        super(EnhancedCLGVAE, self).__init__()
        self.n_gene = n_gene
        self.n_feat = n_feat
        self.hidden_dim = hidden_dim
        self.device1 = torch.device(device) if isinstance(device, str) else device
        self.use_neural_ode = use_neural_ode
        self.use_temporal_kernel = use_temporal_kernel

        # 使用改进的编码器（启用Neural ODE和时间核）
        self.q_encoder = EnhancedCVGAE(
            n_gene, n_feat, hidden_dim, num_anchors, centrality_method,
            use_neural_ode=use_neural_ode,
            ode_method=ode_method,
            ode_steps=ode_steps,
            use_temporal_kernel=use_temporal_kernel,
            temporal_sigma=temporal_sigma
        )
        self.k_encoder = copy.deepcopy(self.q_encoder)

        # 评估指标
        self.au_roc = BinaryAUROC()
        self.au_prc = BinaryAveragePrecision()

        # 负样本池
        self.neg_samples = []
        self.neg_samples_maxlen = 100

        # 位置加权InfoNCE损失（支持时序加权）
        self.position_weighted_info_nce = TemporalPositionWeightedInfoNCE(
            temperature=CONFIG.TEMPERATURE, position_weight=0.5, temporal_weight=1.0
        )
        # 存储伪时间数据（可选）
        self.pseudotime = None  # 保持为None，时间信息已编码在特征中

        # 存储motif adjacency matrices
        self.motif_adjs = motif_adjs if motif_adjs is not None else {}

        # 改进的损失权重
        self.bce_weight = 1.0
        self.kl_weight = 0.01  # 降低KL散度权重
        self.info_nce_weight = 0.1
        self.n_bce_weight = 1.0

        # 添加梯度裁剪参数
        self.gradient_clip_val = 1.0

    def ensure_tensor_device(self, tensor, device):
        """确保张量在正确的设备上"""
        if isinstance(tensor, torch.Tensor) and tensor.device != device:
            tensor = tensor.to(device)
        return tensor

    def ensure_graph_device(self, graph, device):
        """确保图在正确的设备上"""
        if graph.device != device:
            graph = graph.to(device)
        return graph

    def forward(self, block, motif_adjs=None, pseudotime=None):
        # 确保图在正确的设备上
        device = self.device1
        if device is None:
            device = torch.device('cpu')
        block = self.ensure_graph_device(block, device)
        raw_block = self.ensure_graph_device(copy.deepcopy(block), device)

        # 安全处理pseudotime参数以避免numpy数组访问.device属性
        if pseudotime is not None:
            if isinstance(pseudotime, np.ndarray):
                pseudotime = torch.from_numpy(pseudotime).to(device).to(torch.float32)
            elif isinstance(pseudotime, torch.Tensor):
                pseudotime = pseudotime.to(device)
        # 同样安全处理self.pseudotime，以防万一
        if hasattr(self, 'pseudotime') and self.pseudotime is not None:
            if isinstance(self.pseudotime, np.ndarray):
                self.pseudotime = torch.from_numpy(self.pseudotime).to(device).to(torch.float32)
            elif isinstance(self.pseudotime, torch.Tensor):
                self.pseudotime = self.pseudotime.to(device)

        # 主编码器
        p_net, p_mean, p_var, p_h, p_z, p_attention = self.q_encoder(block, motif_adjs,
                                                                pseudotime=pseudotime)

        # 动量编码器
        n_net, _, _, n_h, n_z, n_attention = self.k_encoder(raw_block, motif_adjs, pseudotime=pseudotime)

        return p_net, n_net, p_mean, p_var, p_h, n_h, p_attention, n_attention

    def compute_position_similarity(self, attention_weights):
        """计算位置相似性权重"""
        # 基于注意力权重的相似性
        # 这里简化处理，实际可以根据具体的生物学位置信息计算
        position_sim = torch.mean(attention_weights, dim=1)  # [num_nodes]
        return position_sim

    def training_step(self, batch, batch_idx):
        self.train()
        src_nodes, dst_nodes, batch = batch
        # 确保batch是副本以避免梯度重复计算
        batch = batch.to(batch.device)
        motif_adjs = getattr(self, 'motif_adjs', {})  # 从实例变量获取motif_adjs，如果不存在则使用空字典
        # 确保所有张量在正确的设备上
        device = self.device
        if device is None:
            device = torch.device('cpu')

        batch = self.ensure_graph_device(batch, device)

        # 获取特征
        query_readout = torch.sum(batch.ndata['feature'], dim=0)
        truth_net = batch.adjacency_matrix().to_dense().to(torch.int64)
        truth_net = self.ensure_tensor_device(truth_net, device)
        weight_tensor, norm = compute_loss_para(truth_net)

        # 前向传播 - 使用存储的motif_adjs和伪时间
        # 注意：由于时间特征已拼接在features中，pseudotime参数现在主要用于兼容性
        # 实际上模型不再需要原始的pseudotime数组，因为时间信息已编码在特征中
        pseudotime_batch = None
        # 安全地处理可能的pseudotime（防止numpy数组访问.device属性）
        if self.pseudotime is not None:
            if isinstance(self.pseudotime, np.ndarray):
                pseudotime_batch = torch.from_numpy(self.pseudotime).to(device).to(torch.float32)
            elif isinstance(self.pseudotime, torch.Tensor):
                pseudotime_batch = self.pseudotime.to(device)

        p_net, n_net, p_mean, p_var, p_h, n_h, p_attention, n_attention = self(batch, motif_adjs=motif_adjs,
                                                                               pseudotime=pseudotime_batch)

        # 改进的损失计算
        # 1. 重构损失
        # 使用克隆避免梯度重复计算，但保留梯度链
        truth_net_float = truth_net.view(-1).to(torch.float64)
        p_bce_loss = norm * F.binary_cross_entropy_with_logits(
            p_net.view(-1), truth_net_float, weight=weight_tensor
        )
        n_bce_loss = norm * F.binary_cross_entropy_with_logits(
            n_net.view(-1), truth_net_float, weight=weight_tensor
        )

        # 2. KL散度损失 - 使用更稳定的计算方式
        # 使用标准的VAE KL散度公式
        kl_loss = -0.5 * torch.mean(1 + p_var - p_mean.pow(2) - p_var.exp())

        # 3. 位置加权对比损失
        pos_readout = torch.sum(p_h, dim=0)
        n_readout = torch.sum(n_h, dim=0)

        # 更新负样本池
        if n_readout is not None and n_readout.numel() > 0:
            self.neg_samples.append(n_readout.detach())
        else:
            # 如果n_readout无效，添加一个零向量作为占位符
            try:
                device = n_readout.device if n_readout is not None else torch.device(
                    'cuda:0' if torch.cuda.is_available() else 'cpu')
                zero_sample = torch.zeros(self.hidden_dim, device=device)
                self.neg_samples.append(zero_sample)
            except Exception as e:
                # 安全回退
                zero_sample = torch.zeros(self.hidden_dim)
                self.neg_samples.append(zero_sample)

        if len(self.neg_samples) > self.neg_samples_maxlen:
            self.neg_samples.pop(0)

        # 计算位置相似性
        position_sim = self.compute_position_similarity(p_attention)

        # 位置加权InfoNCE损失
        # 只有在有足够的负样本时才计算InfoNCE损失
        if len(self.neg_samples) >= 5:  # 至少需要5个负样本
            pseudotime_query = None
            pseudotime_pos = None
            pseudotime_neg = None

            info_nce_loss = self.position_weighted_info_nce(
                query_readout, pos_readout, [sample.clone().detach() for sample in self.neg_samples], position_sim,
                pseudotime_query=pseudotime_query, pseudotime_pos=pseudotime_pos, pseudotime_neg=pseudotime_neg
            )
        else:
            # 负样本不足时，InfoNCE损失设为0
            info_nce_loss = torch.tensor(0.0, device=query_readout.device)

        # 改进的总损失计算
        total_loss = (
                self.bce_weight * p_bce_loss +
                self.kl_weight * kl_loss +
                self.info_nce_weight * info_nce_loss +
                self.n_bce_weight * n_bce_loss
        )

        # 记录损失
        self.log('total_loss', total_loss, on_step=True, on_epoch=True)
        self.log('p_bce_loss', p_bce_loss, on_step=True, on_epoch=True)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True)
        self.log('info_nce_loss', info_nce_loss, on_step=True, on_epoch=True)
        self.log('n_bce_loss', n_bce_loss, on_step=True, on_epoch=True)

        # 打印训练信息
        self.print(f'Total Loss: {total_loss:.4f}, P_BCE: {p_bce_loss:.4f}, '
                   f'KL: {kl_loss:.4f}, InfoNCE: {info_nce_loss:.4f}')

        # 计算评估指标
        with torch.no_grad():
            training_step_au_roc = self.au_roc(p_net.detach(), truth_net.detach())
            training_step_au_prc = self.au_prc(p_net.detach(), truth_net.detach())
        self.print(f'AUROC: {training_step_au_roc:.4f}, AUPRC: {training_step_au_prc:.4f}')

        # 重置指标
        self.au_roc.reset()
        self.au_prc.reset()

        # 确保没有保留对中间张量的引用
        del p_net, n_net, p_mean, p_var, p_h, n_h, p_attention, n_attention

        return total_loss

    def configure_optimizers(self):
        # 使用AdamW优化器，通常在深度学习中效果更好
        optimizer = torch.optim.AdamW(
            self.q_encoder.parameters(),
            lr=CONFIG.LEARNING_RATE,
            weight_decay=CONFIG.WEIGHT_DECAY,
            betas=(0.9, 0.999)  # 使用标准的beta值
        )
        return optimizer

    def on_after_backward(self):
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)

class GaussianTemporalKernel(nn.Module):
    """高斯时间核，用于时间局部性加权"""

    def __init__(self, sigma=0.1):
        """
        Args:
            sigma: 高斯核的标准差，控制时间窗口大小
        """
        super(GaussianTemporalKernel, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, pseudotime_i, pseudotime_j):
        """
        计算时间核权重
        Args:
            pseudotime_i: 当前细胞的伪时间 [N] 或标量
            pseudotime_j: 邻居细胞的伪时间 [N, M] 或 [N]
        Returns:
            weights: 时间核权重
        """
        if pseudotime_j.dim() == 1:
            time_diff = (pseudotime_i.unsqueeze(-1) - pseudotime_j.unsqueeze(0)) ** 2
        else:
            time_diff = (pseudotime_i.unsqueeze(-1) - pseudotime_j) ** 2

        weights = torch.exp(-time_diff / (2 * self.sigma ** 2 + 1e-8))
        return weights


class NeuralODELayer(nn.Module):
    """
    神经常微分方程层，用于连续时间建模
    通过求解 dz/dt = f(z, t) 来建模基因表达的动态演化
    """
    
    def __init__(self, input_dim, hidden_dim, method='euler', num_steps=5):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            method: ODE求解方法 ('euler' 或 'rk4')
            num_steps: Euler/RK4方法的步数
        """
        super(NeuralODELayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.method = method
        self.num_steps = num_steps
        
        # ODE函数网络: f(z, t) -> dz/dt
        # 将时间t拼接到状态z上作为输入
        self.ode_func = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 初始化权重
        for m in self.ode_func.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
    
    def ode_func_wrapper(self, t, z):
        """
        ODE函数包装器
        Args:
            t: 时间点 [batch_size] 或标量
            z: 状态 [batch_size, input_dim]
        Returns:
            dz/dt: [batch_size, input_dim]
        """
        # 将时间t广播到所有样本
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.size(0))
        t = t.unsqueeze(-1)  # [batch_size, 1]
        
        # 拼接时间和状态
        zt = torch.cat([z, t], dim=-1)  # [batch_size, input_dim + 1]
        
        # 计算导数
        dzdt = self.ode_func(zt)
        return dzdt
    
    def euler_solve(self, z0, t_start, t_end):
        """
        Euler方法求解ODE
        Args:
            z0: 初始状态 [batch_size, input_dim]
            t_start: 起始时间
            t_end: 结束时间 [batch_size] 或标量
        Returns:
            z_final: 最终状态 [batch_size, input_dim]
        """
        z = z0
        
        # 如果t_end是标量，扩展到batch_size
        if t_end.dim() == 0:
            t_end = t_end.unsqueeze(0).expand(z0.size(0))
        
        # 逐步求解
        for step in range(self.num_steps):
            # 当前时间
            t_current = t_start + (t_end - t_start) * step / self.num_steps
            
            # 计算导数
            dzdt = self.ode_func_wrapper(t_current, z)
            
            # 时间步长
            dt = (t_end - t_start) / self.num_steps
            dt = dt.unsqueeze(-1)  # [batch_size, 1]
            
            # Euler更新
            z = z + dt * dzdt
        
        return z
    
    def rk4_solve(self, z0, t_start, t_end):
        """
        Runge-Kutta 4阶方法求解ODE（更精确但更慢）
        Args:
            z0: 初始状态 [batch_size, input_dim]
            t_start: 起始时间
            t_end: 结束时间 [batch_size] 或标量
        Returns:
            z_final: 最终状态 [batch_size, input_dim]
        """
        z = z0
        
        if t_end.dim() == 0:
            t_end = t_end.unsqueeze(0).expand(z0.size(0))
        
        for step in range(self.num_steps):
            t_current = t_start + (t_end - t_start) * step / self.num_steps
            dt = (t_end - t_start) / self.num_steps
            dt = dt.unsqueeze(-1)  # [batch_size, 1]
            
            # RK4方法
            k1 = self.ode_func_wrapper(t_current, z)
            k2 = self.ode_func_wrapper(t_current + dt/2, z + dt/2 * k1)
            k3 = self.ode_func_wrapper(t_current + dt/2, z + dt/2 * k2)
            k4 = self.ode_func_wrapper(t_current + dt, z + dt * k3)
            
            z = z + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return z
    
    def forward(self, z0, pseudotime):
        """
        前向传播：从初始状态演化到指定伪时间点
        Args:
            z0: 初始状态 [batch_size, input_dim]
            pseudotime: 目标伪时间 [batch_size] 或标量，范围[0, 1]
        Returns:
            z_t: 在伪时间点的状态 [batch_size, input_dim]
        """
        device = z0.device
        t_start = torch.zeros_like(pseudotime)  # 从t=0开始
        
        if self.method == 'rk4':
            z_t = self.rk4_solve(z0, t_start, pseudotime)
        else:
            z_t = self.euler_solve(z0, t_start, pseudotime)
        
        return z_t


class LocalPathEncoder(nn.Module):
    """局部路径编码器 - MotifGCN with Attention + 时间局部性"""

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, motif_types=['triangle', 'edge'],
                 use_temporal_kernel=True, temporal_sigma=0.1):
        super(LocalPathEncoder, self).__init__()
        self.num_layers = num_layers
        self.use_temporal_kernel = use_temporal_kernel

        self.motif_types = motif_types
        self.layers = nn.ModuleList([
            MotifGCNLayerWithAttention(
                input_dim if i == 0 else hidden_dim, hidden_dim, motif_types=motif_types
            ) for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # 时间核（如果使用）
        if self.use_temporal_kernel:
            self.temporal_kernel = GaussianTemporalKernel(sigma=temporal_sigma)

    def forward(self, features, motif_adjs, pseudotime=None):
        # features: [N, input_dim]
        # motif_adjs: dict, 每种motif类型一个[N, N]邻接矩阵
        # pseudotime: [N] 伪时间值（可选）
        h = features

        # 如果启用了时间核且提供了伪时间，应用时间加权
        if self.use_temporal_kernel and pseudotime is not None:
            # 计算时间核权重矩阵 [N, N]
            time_weights = self.temporal_kernel(pseudotime, pseudotime)
            
            # 将时间权重应用到 motif 邻接矩阵上
            weighted_motif_adjs = {}
            for motif_name, adj in motif_adjs.items():
                # 逐元素相乘：拓扑结构 × 时间相似性
                weighted_motif_adjs[motif_name] = adj * time_weights
            
            motif_adjs_to_use = weighted_motif_adjs
        else:
            motif_adjs_to_use = motif_adjs

        attn_list = []
        for layer in self.layers:
            h, attn = layer(h, motif_adjs_to_use)  # 使用加权后的邻接矩阵
            h = func.relu(h)
            attn_list.append(attn)
        h = self.output_layer(h)
        h = self.norm(h)
        return h, attn_list
class MotifGCNLayerWithAttention(nn.Module):
    def __init__(self, in_dim, out_dim, motif_types=['triangle', 'edge']):
        super().__init__()
        self.motif_types = motif_types
        self.linears = nn.ModuleDict({
            motif: nn.Linear(in_dim, out_dim) for motif in motif_types
        })
        self.attn_linear = nn.Linear(len(motif_types) * out_dim, len(motif_types))
        self.out_linear = nn.Linear(len(motif_types) * out_dim, out_dim)

    def forward(self, x, motif_adjs):
        motif_outputs = []
        for motif in self.motif_types:
            adj = motif_adjs[motif]  # [N, N]
            h = torch.matmul(adj, x) / (adj.sum(dim=1, keepdim=True) + 1e-6)
            h = self.linears[motif](h)
            motif_outputs.append(h)
        h_cat = torch.cat(motif_outputs, dim=-1)  # [N, out_dim * motif_num]
        attn_weights = torch.softmax(self.attn_linear(h_cat), dim=-1)  # [N, motif_num]
        h_weighted = torch.cat([
            motif_outputs[i] * attn_weights[:, i:i + 1] for i in range(len(self.motif_types))
        ], dim=-1)
        out = self.out_linear(h_weighted)  # [N, out_dim]
        return out, attn_weights
