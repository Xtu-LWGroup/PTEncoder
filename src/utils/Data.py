import pandas as pd
import numpy as np
import dgl
import torch
import os

from numpy.random import shuffle
from .DeepProfile import apply_deepprofile_reduction
from torch_geometric.data import Data
import networkx as nx
import multiprocessing as mp
import random
from module import build_motif_adjs


def sc_parser(exp_file, typ_file):
    """
    :param exp_file: the file path of an expression file.
    :param typ_file: the file path of a topology file.
    :return:
        features: (numpy.array) It contains a feature matrix (2D).
        src_nodes: (numpy.array) It contains the index of all source nodes.
        dst_nodes: (numpy.array) It contains the index of all destination nodes.
        gene_frame: (pandas.DataFrame) It indices the index of genes.
    """
    exp_frame = pd.read_csv(exp_file, index_col=0, header=0)
    features = exp_frame.to_numpy(dtype=float)
    n_gene = features.shape[0]
    gene_frame = pd.DataFrame(
        {
            'gene_ids': [i for i in range(n_gene)],
            'gene_name': list(exp_frame.index)
        }
    )

    gene_frame.set_index(['gene_name'], inplace=True)
    src_nodes, dst_nodes = [], []
    net_frame = pd.read_csv(typ_file, header=0)

    for r in net_frame.itertuples():
        src_nodes.append(gene_frame.gene_ids[r[1]])
        dst_nodes.append(gene_frame.gene_ids[r[2]])

    src_nodes = np.array(src_nodes, dtype=int)
    dst_nodes = np.array(dst_nodes, dtype=int)

    return features, src_nodes, dst_nodes, gene_frame


class Preprocessor(object):
    def __init__(self, exp_file, tpy_file, ratio=None):
        super(Preprocessor, self).__init__()

        if ratio is None:
            ratio = [0.6, 0.4]
        if not isinstance(ratio, list):
            raise Exception('g must be a list')

        self.ratio = ratio
        self.exp, self.src_nodes, self.dst_nodes = None, None, None
        self.g, self.train_edge, self.test_edge, self.train_graph, self.test_graph = None, None, None, None, None,

        self.exp, self.src_nodes, self.dst_nodes, self.gene_frame = sc_parser(exp_file, tpy_file)
        self.tpy = np.array(list(zip(self.src_nodes, self.dst_nodes)), dtype=int)
        self.num_edge = self.tpy.shape[0]
        self.num_gene = self.exp.shape[0]

        self.exp = torch.from_numpy(self.exp).to(torch.float32)
        self.src_nodes = torch.from_numpy(self.src_nodes).to(torch.int64)
        self.dst_nodes = torch.from_numpy(self.dst_nodes).to(torch.int64)

        self.__init_exp_and_tpy()

    def __init_exp_and_tpy(self):
        self.g = dgl.graph((self.src_nodes, self.dst_nodes), num_nodes=self.num_gene, idtype=torch.int64)
        self.g.ndata['feature'] = self.exp

        num_train = int(np.floor(self.tpy.shape[0] * self.ratio[0]))
        np.random.shuffle(self.tpy)
        self.train_edge = self.tpy[: num_train]
        self.test_edge = self.tpy[num_train:]
        self.train_edge = list(zip(*list(self.train_edge)))
        train_src_node_idx = torch.tensor(self.train_edge[0], dtype=torch.int64)
        train_dst_node_idx = torch.tensor(self.train_edge[1], dtype=torch.int64)
        self.test_edge = list(zip(*list(self.test_edge)))
        test_src_node_idx = torch.tensor(self.test_edge[0], dtype=torch.int64)
        test_dst_node_idx = torch.tensor(self.test_edge[1], dtype=torch.int64)

        self.train_graph = dgl.graph(
            (train_src_node_idx, train_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
        )
        self.train_graph.ndata['feature'] = self.exp
        self.test_graph = dgl.graph(
            (test_src_node_idx, test_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
        )
        self.test_graph.ndata['feature'] = self.exp

    def generate_test_data(self, ratio):
        # obtain the index of false edges
        all_edges = self.g.adjacency_matrix().to_dense().view(-1)
        all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
        # obtain the index of true edges included in test graph
        test_edges = self.test_graph.adjacency_matrix().to_dense().view(-1)
        test_true_edge_idx = torch.nonzero(test_edges).reshape(1, -1).squeeze(dim=0)
        # initial positive and negative samples
        positive_size = test_true_edge_idx.shape[0]
        negative_size = ratio * positive_size
        # reindex the index of false edges
        neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
        shuffle(neg_sample_idx)
        neg_sample_idx = torch.tensor(neg_sample_idx[0: negative_size], dtype=torch.int64)
        # re-obtain label combined true and false edge.
        test_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
        test_false_edge = torch.gather(all_edges, index=test_false_edge_idx, dim=0)
        test_true_edge = torch.gather(test_edges, index=test_true_edge_idx, dim=0)
        label = torch.cat([test_true_edge, test_false_edge], dim=0)

        return test_true_edge_idx, test_false_edge_idx, label

    # def generate_validation_data(self):
    #     # obtain the index of false edges
    #     all_edges = self.g.adjacency_matrix().to_dense().view(-1)
    #     all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
    #
    #     # obtain the index of true edges included in test graph
    #     val_edges = self.train_graph.adjacency_matrix().to_dense().view(-1)
    #     val_true_edge_idx = torch.nonzero(val_edges).reshape(1, -1).squeeze(dim=0)
    #     # initial positive and negative samples
    #     positive_size = val_edges.shape[0]
    #     negative_size = positive_size
    #
    #     #reindex the index of true edges
    #     pos_sample_idx = np.arange(0, val_true_edge_idx.shape[0], 1)
    #     shuffle(pos_sample_idx)
    #     pos_sample_idx = torch.tensor(pos_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
    #
    #     # reindex the index of false edges
    #     neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
    #     shuffle(neg_sample_idx)
    #     neg_sample_idx = torch.tensor(neg_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
    #
    #     # re-obtain label combined true and false edge.
    #     val_pos_edge_idx = torch.gather(val_true_edge_idx, index=pos_sample_idx, dim=0)
    #     val_true_edge = torch.gather(val_edges, index=val_pos_edge_idx, dim=0)
    #     val_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
    #     val_false_edge = torch.gather(all_edges, index=val_false_edge_idx, dim=0)
    #     label = torch.cat([val_true_edge, val_false_edge], dim=0)
    #
    #     return val_true_edge_idx, val_false_edge_idx, label


# 匹配数据 实验pgnn
class Preprocessor_1(object):
    def __init__(self, exp_file, tpy_file, ratio=None):
        super(Preprocessor_1, self).__init__()

        if ratio is None:
            ratio = [0.6, 0.4]
        if not isinstance(ratio, list):
            raise Exception('g must be a list')

        self.ratio = ratio
        self.exp, self.src_nodes, self.dst_nodes = None, None, None
        self.g, self.train_edge, self.test_edge, self.train_graph, self.test_graph = None, None, None, None, None,

        self.exp, self.src_nodes, self.dst_nodes, self.gene_frame = sc_parser(exp_file, tpy_file)
        self.tpy = np.array(list(zip(self.src_nodes, self.dst_nodes)), dtype=int)
        self.num_edge = self.tpy.shape[0]
        self.num_gene = self.exp.shape[0]

        self.exp = torch.from_numpy(self.exp).to(torch.float32)
        self.src_nodes = torch.from_numpy(self.src_nodes).to(torch.int64)
        self.dst_nodes = torch.from_numpy(self.dst_nodes).to(torch.int64)

        self.__init_exp_and_tpy()

    def __init_exp_and_tpy(self):
        self.g = dgl.graph((self.src_nodes, self.dst_nodes), num_nodes=self.num_gene, idtype=torch.int64)
        self.g.ndata['feature'] = self.exp

        num_train = int(np.floor(self.tpy.shape[0] * self.ratio[0]))
        np.random.shuffle(self.tpy)
        self.train_edge = self.tpy[: num_train]
        train_edge = self.train_edge  #
        self.test_edge = self.tpy[num_train:]
        test_edge = self.test_edge
        self.train_edge = list(zip(*list(self.train_edge)))
        train_src_node_idx = torch.tensor(self.train_edge[0], dtype=torch.int64)
        train_dst_node_idx = torch.tensor(self.train_edge[1], dtype=torch.int64)
        self.test_edge = list(zip(*list(self.test_edge)))
        test_src_node_idx = torch.tensor(self.test_edge[0], dtype=torch.int64)
        test_dst_node_idx = torch.tensor(self.test_edge[1], dtype=torch.int64)

        # 为pgnn输入数据
        train_data = Data(x=self.exp, edge_index=train_edge.T)
        train_data.num_nodes = self.num_gene
        dists_removed = precompute_dist_data(train_data.edge_index, train_data.num_nodes,
                                             approximate=-1)  # 按照原论文给定参数 后续你可以自己设置
        train_data.dists = torch.from_numpy(dists_removed).float()
        # train_data.edge_index = torch.from_numpy(train_data.edge_index).long()
        train_data.edge_index = torch.from_numpy(duplicate_edges(train_data.edge_index)).long()
        preselect_anchor(train_data, layer_num=3, anchor_num=64, device='cpu')  # 需要调试

        test_data = Data(x=self.exp, edge_index=test_edge.T)
        test_data.num_nodes = self.num_gene
        dists_removed = precompute_dist_data(test_data.edge_index, test_data.num_nodes,
                                             approximate=-1)  # 按照原论文给定参数 后续你可以自己设置
        test_data.dists = torch.from_numpy(dists_removed).float()
        test_data.edge_index = torch.from_numpy(duplicate_edges(test_data.edge_index)).long()

        preselect_anchor(test_data, layer_num=3, anchor_num=64, device='cpu')  # 需要调试

        self.train_graph = dgl.graph(
            (train_src_node_idx, train_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
        )
        self.train_graph.ndata['feature'] = self.exp
        self.train_graph.ndata['dists_max'] = train_data.dists_max
        self.train_graph.ndata['dists_argmax'] = train_data.dists_argmax
        self.test_graph = dgl.graph(
            (test_src_node_idx, test_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
        )
        self.test_graph.ndata['feature'] = self.exp
        self.train_graph.ndata['dists_max'] = test_data.dists_max
        self.train_graph.ndata['dists_argmax'] = test_data.dists_argmax

    def generate_test_data(self, ratio):
        # obtain the index of false edges
        all_edges = self.g.adjacency_matrix().to_dense().view(-1)
        all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
        # obtain the index of true edges included in test graph
        test_edges = self.test_graph.adjacency_matrix().to_dense().view(-1)
        test_true_edge_idx = torch.nonzero(test_edges).reshape(1, -1).squeeze(dim=0)
        # initial positive and negative samples
        positive_size = test_true_edge_idx.shape[0]
        negative_size = ratio * positive_size
        # reindex the index of false edges
        neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
        shuffle(neg_sample_idx)
        neg_sample_idx = torch.tensor(neg_sample_idx[0: negative_size], dtype=torch.int64)
        # re-obtain label combined true and false edge.
        test_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
        test_false_edge = torch.gather(all_edges, index=test_false_edge_idx, dim=0)
        test_true_edge = torch.gather(test_edges, index=test_true_edge_idx, dim=0)
        label = torch.cat([test_true_edge, test_false_edge], dim=0)

        return test_true_edge_idx, test_false_edge_idx, label

    # def generate_validation_data(self):
    #     # obtain the index of false edges
    #     all_edges = self.g.adjacency_matrix().to_dense().view(-1)
    #     all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
    #
    #     # obtain the index of true edges included in test graph
    #     val_edges = self.train_graph.adjacency_matrix().to_dense().view(-1)
    #     val_true_edge_idx = torch.nonzero(val_edges).reshape(1, -1).squeeze(dim=0)
    #     # initial positive and negative samples
    #     positive_size = val_edges.shape[0]
    #     negative_size = positive_size
    #
    #     #reindex the index of true edges
    #     pos_sample_idx = np.arange(0, val_true_edge_idx.shape[0], 1)
    #     shuffle(pos_sample_idx)
    #     pos_sample_idx = torch.tensor(pos_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
    #
    #     # reindex the index of false edges
    #     neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
    #     shuffle(neg_sample_idx)
    #     neg_sample_idx = torch.tensor(neg_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
    #
    #     # re-obtain label combined true and false edge.
    #     val_pos_edge_idx = torch.gather(val_true_edge_idx, index=pos_sample_idx, dim=0)
    #     val_true_edge = torch.gather(val_edges, index=val_pos_edge_idx, dim=0)
    #     val_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
    #     val_false_edge = torch.gather(all_edges, index=val_false_edge_idx, dim=0)
    #     label = torch.cat([val_true_edge, val_false_edge], dim=0)
    #
    #     return val_true_edge_idx, val_false_edge_idx, label


class Preprocessor_2(object):
    # def __init__(self, exp_file, tpy_file, ratio=None):
    def __init__(self, exp_file, tpy_file, ratio=None, use_deepprofile=False,
                 latent_dim=100, hidden_dims=[512, 256, 128], deepprofile_epochs=100, pseudotime_file=None):
        super(Preprocessor_2, self).__init__()

        if ratio is None:
            ratio = [0.6, 0.4]
        if not isinstance(ratio, list):
            raise Exception('g must be a list')

        self.ratio = ratio
        self.use_deepprofile = use_deepprofile
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.deepprofile_epochs = deepprofile_epochs
        self.pseudotime_file = pseudotime_file
        self.exp, self.src_nodes, self.dst_nodes = None, None, None
        self.g, self.train_edge, self.test_edge, self.train_graph, self.test_graph = None, None, None, None, None,
        self.pseudotime = None  # 伪时间数据

        self.exp, self.src_nodes, self.dst_nodes, self.gene_frame = sc_parser(exp_file, tpy_file)

        # 保存原始表达矩阵（用于时间特征计算）
        self.exp_raw = self.exp.copy()  # [n_genes, n_cells]

        # 加载伪时间数据并计算基因时间特征
        self.temporal_features = None  # 基因的时间特征 [n_genes, temporal_feat_dim]
        if self.pseudotime_file and os.path.exists(self.pseudotime_file):
            self._load_pseudotime()
            # 计算基因的时间特征
            self._compute_gene_temporal_features()

        if self.use_deepprofile:
            print(f"应用DeepProfile降维，原始维度: {self.exp.shape[1]}, 目标维度: {self.latent_dim}")
            self.exp, self.deepprofile_model = apply_deepprofile_reduction(
                self.exp,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                epochs=self.deepprofile_epochs
            )
            print(f"DeepProfile降维完成，新维度: {self.exp.shape}")

        self.tpy = np.array(list(zip(self.src_nodes, self.dst_nodes)), dtype=int)
        self.num_edge = self.tpy.shape[0]
        self.num_gene = self.exp.shape[0]

        self.exp = torch.from_numpy(self.exp).to(torch.float32)
        self.src_nodes = torch.from_numpy(self.src_nodes).to(torch.int64)
        self.dst_nodes = torch.from_numpy(self.dst_nodes).to(torch.int64)

        # 如果有时间特征，将其与表达特征拼接
        if self.temporal_features is not None:
            # 确保时间特征是torch tensor且与exp在同一设备上
            if not isinstance(self.temporal_features, torch.Tensor):
                self.temporal_features = torch.from_numpy(self.temporal_features).to(torch.float32)
            if hasattr(self.temporal_features, 'device') and self.temporal_features.device != self.exp.device:
                self.temporal_features = self.temporal_features.to(self.exp.device)
            # 拼接时间特征到表达特征
            original_feat_dim = self.exp.shape[1]
            self.exp = torch.cat([self.exp, self.temporal_features], dim=1)
            print(
                f"特征拼接完成: 原始维度 {original_feat_dim} + 时间特征维度 {self.temporal_features.shape[1]} = {self.exp.shape[1]}")

        self.__init_exp_and_tpy()
        # 生成motif邻接矩阵
        self.motif_adjs = build_motif_adjs(self.g)  # 加了motif后的
        # 可选：放到GPU
        for k in self.motif_adjs:
            self.motif_adjs[k] = self.motif_adjs[k].to(self.exp.device)  # 加了motif后的

    def _load_pseudotime(self):
        """加载并预处理伪时间数据"""
        try:
            # 读取伪时间CSV文件
            pseudotime_frame = pd.read_csv(self.pseudotime_file, index_col=0, header=0)
            pseudotime_values = pseudotime_frame.iloc[:, 0].values  # 取第一列作为伪时间值

            # Min-Max归一化到[0, 1]区间
            min_val = pseudotime_values.min()
            max_val = pseudotime_values.max()
            if max_val > min_val:
                pseudotime_normalized = (pseudotime_values - min_val) / (max_val - min_val)
            else:
                pseudotime_normalized = np.zeros_like(pseudotime_values)

            # 转换为numpy数组，维度为 [n_cells]，每个细胞对应一个伪时间值
            self.pseudotime = pseudotime_normalized  # 保持为numpy数组，用于时间特征计算

            print(
                f"伪时间数据加载成功: {len(self.pseudotime)} 个细胞，范围: [{self.pseudotime.min():.4f}, {self.pseudotime.max():.4f}]")

        except Exception as e:
            print(f"警告: 伪时间数据加载失败: {e}")
            self.pseudotime = None

    def _compute_gene_temporal_features(self, n_time_bins=10):
        """
        计算基因的时间特征：将细胞伪时间信息转化为基因层面的时间特征

        Args:
            n_time_bins: 伪时间段的数量
        """
        if self.pseudotime is None or self.exp_raw is None:
            print("警告: 无法计算时间特征，伪时间或表达数据缺失")
            return

        n_genes, n_cells = self.exp_raw.shape

        if len(self.pseudotime) != n_cells:
            print(f"警告: 伪时间长度({len(self.pseudotime)})与细胞数({n_cells})不匹配，无法计算时间特征")
            return

        # 将伪时间分成n_time_bins个时间段
        time_bins = np.linspace(0, 1, n_time_bins + 1)
        bin_indices = np.digitize(self.pseudotime, time_bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_time_bins - 1)  # 确保索引在有效范围内

        # 初始化时间特征矩阵 [n_genes, temporal_feat_dim]
        # 特征包括：
        # 1. 每个时间段的平均表达 (n_time_bins)
        # 2. 每个时间段的表达波动/方差 (n_time_bins)
        # 3. 时间梯度（表达随时间的变化率）(1)
        # 4. 峰值时间（表达最高的时间段）(1)
        # 5. 表达动态范围 (max - min) (1)
        temporal_feat_dim = n_time_bins * 2 + 3  # 2*n_time_bins + 3个额外特征
        temporal_features = np.zeros((n_genes, temporal_feat_dim))

        print(f"计算基因时间特征: {n_genes} 个基因, {n_time_bins} 个时间段...")

        for gene_idx in range(n_genes):
            gene_expr = self.exp_raw[gene_idx, :]  # [n_cells]

            # 1. 计算每个时间段的平均表达和方差
            bin_means = np.zeros(n_time_bins)
            bin_vars = np.zeros(n_time_bins)

            for bin_idx in range(n_time_bins):
                mask = (bin_indices == bin_idx)
                if np.sum(mask) > 0:
                    bin_expr = gene_expr[mask]
                    bin_means[bin_idx] = np.mean(bin_expr)
                    bin_vars[bin_idx] = np.var(bin_expr)
                else:
                    bin_means[bin_idx] = 0.0
                    bin_vars[bin_idx] = 0.0

            # 2. 计算时间梯度（使用线性回归斜率）
            # 对表达值按伪时间排序后计算斜率
            sorted_indices = np.argsort(self.pseudotime)
            sorted_expr = gene_expr[sorted_indices]
            sorted_time = self.pseudotime[sorted_indices]

            # 使用最小二乘法计算斜率
            if len(sorted_time) > 1 and np.std(sorted_time) > 1e-8:
                time_gradient = np.polyfit(sorted_time, sorted_expr, deg=1)[0]
            else:
                time_gradient = 0.0

            # 3. 峰值时间（表达最高的时间段的索引）
            peak_bin = np.argmax(bin_means)
            peak_time_normalized = peak_bin / n_time_bins  # 归一化到[0, 1]

            # 4. 表达动态范围
            expr_range = np.max(gene_expr) - np.min(gene_expr)
            if expr_range < 1e-8:
                expr_range = 0.0

            # 组装特征向量
            feat_idx = 0
            temporal_features[gene_idx, feat_idx:feat_idx + n_time_bins] = bin_means
            feat_idx += n_time_bins
            temporal_features[gene_idx, feat_idx:feat_idx + n_time_bins] = bin_vars
            feat_idx += n_time_bins
            temporal_features[gene_idx, feat_idx] = time_gradient
            feat_idx += 1
            temporal_features[gene_idx, feat_idx] = peak_time_normalized
            feat_idx += 1
            temporal_features[gene_idx, feat_idx] = expr_range

        # 归一化时间特征（可选，但通常有助于训练）
        # 对每个特征维度进行标准化
        feature_mean = temporal_features.mean(axis=0, keepdims=True)
        feature_std = temporal_features.std(axis=0, keepdims=True) + 1e-8
        temporal_features = (temporal_features - feature_mean) / feature_std

        # 转换为torch tensor
        self.temporal_features = torch.from_numpy(temporal_features).to(torch.float32)

        print(f"基因时间特征计算完成: 形状 {self.temporal_features.shape}, "
              f"特征维度: {temporal_feat_dim} (包括 {n_time_bins} 个时间段均值, "
              f"{n_time_bins} 个时间段方差, 时间梯度, 峰值时间, 表达范围)")

    def __init_exp_and_tpy(self):
        self.g = dgl.graph((self.src_nodes, self.dst_nodes), num_nodes=self.num_gene, idtype=torch.int64)
        self.g.ndata['feature'] = self.exp

        num_train = int(np.floor(self.tpy.shape[0] * self.ratio[0]))
        np.random.shuffle(self.tpy)
        self.train_edge = self.tpy[: num_train]
        train_edge = self.train_edge  #
        self.test_edge = self.tpy[num_train:]
        test_edge = self.test_edge
        self.train_edge = list(zip(*list(self.train_edge)))
        train_src_node_idx = torch.tensor(self.train_edge[0], dtype=torch.int64)
        train_dst_node_idx = torch.tensor(self.train_edge[1], dtype=torch.int64)
        self.test_edge = list(zip(*list(self.test_edge)))
        test_src_node_idx = torch.tensor(self.test_edge[0], dtype=torch.int64)
        test_dst_node_idx = torch.tensor(self.test_edge[1], dtype=torch.int64)

        # 为pgnn输入数据
        train_data = Data(x=self.exp, edge_index=train_edge.T)
        train_data.num_nodes = self.num_gene
        dists_removed = precompute_dist_data(train_data.edge_index, train_data.num_nodes,
                                             approximate=-1)  # 按照原论文给定参数 后续你可以自己设置
        train_data.dists = torch.from_numpy(dists_removed).float()
        # train_data.edge_index = torch.from_numpy(train_data.edge_index).long()
        train_data.edge_index = torch.from_numpy(duplicate_edges(train_data.edge_index)).long()
        preselect_anchor(train_data, layer_num=3, anchor_num=64, device='cpu')  # 需要调试

        test_data = Data(x=self.exp, edge_index=test_edge.T)
        test_data.num_nodes = self.num_gene
        dists_removed = precompute_dist_data(test_data.edge_index, test_data.num_nodes,
                                             approximate=-1)  # 按照原论文给定参数 后续你可以自己设置
        test_data.dists = torch.from_numpy(dists_removed).float()
        test_data.edge_index = torch.from_numpy(duplicate_edges(test_data.edge_index)).long()

        preselect_anchor(test_data, layer_num=3, anchor_num=64, device='cpu')  # 需要调试

        self.train_graph = dgl.graph(
            (train_src_node_idx, train_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
        )
        self.train_graph.ndata['feature'] = self.exp
        self.train_graph.ndata['dists_max'] = train_data.dists_max
        self.train_graph.ndata['dists_argmax'] = train_data.dists_argmax
        self.test_graph = dgl.graph(
            (test_src_node_idx, test_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
        )
        self.test_graph.ndata['feature'] = self.exp
        self.test_graph.ndata['dists_max'] = test_data.dists_max
        self.test_graph.ndata['dists_argmax'] = test_data.dists_argmax

    def generate_test_data(self, ratio):
        # obtain the index of false edges
        all_edges = self.g.adjacency_matrix().to_dense().view(-1)
        all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
        # obtain the index of true edges included in test graph
        test_edges = self.test_graph.adjacency_matrix().to_dense().view(-1)
        test_true_edge_idx = torch.nonzero(test_edges).reshape(1, -1).squeeze(dim=0)
        # initial positive and negative samples
        positive_size = test_true_edge_idx.shape[0]
        negative_size = ratio * positive_size
        # reindex the index of false edges
        neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
        shuffle(neg_sample_idx)
        neg_sample_idx = torch.tensor(neg_sample_idx[0: negative_size], dtype=torch.int64)
        # re-obtain label combined true and false edge.
        test_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
        test_false_edge = torch.gather(all_edges, index=test_false_edge_idx, dim=0)
        test_true_edge = torch.gather(test_edges, index=test_true_edge_idx, dim=0)
        label = torch.cat([test_true_edge, test_false_edge], dim=0)

        return test_true_edge_idx, test_false_edge_idx, label

    # def generate_validation_data(self):
    #     # obtain the index of false edges
    #     all_edges = self.g.adjacency_matrix().to_dense().view(-1)
    #     all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
    #
    #     # obtain the index of true edges included in test graph
    #     val_edges = self.train_graph.adjacency_matrix().to_dense().view(-1)
    #     val_true_edge_idx = torch.nonzero(val_edges).reshape(1, -1).squeeze(dim=0)
    #     # initial positive and negative samples
    #     positive_size = val_edges.shape[0]
    #     negative_size = positive_size
    #
    #     #reindex the index of true edges
    #     pos_sample_idx = np.arange(0, val_true_edge_idx.shape[0], 1)
    #     shuffle(pos_sample_idx)
    #     pos_sample_idx = torch.tensor(pos_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
    #
    #     # reindex the index of false edges
    #     neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
    #     shuffle(neg_sample_idx)
    #     neg_sample_idx = torch.tensor(neg_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
    #
    #     # re-obtain label combined true and false edge.
    #     val_pos_edge_idx = torch.gather(val_true_edge_idx, index=pos_sample_idx, dim=0)
    #     val_true_edge = torch.gather(val_edges, index=val_pos_edge_idx, dim=0)
    #     val_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
    #     val_false_edge = torch.gather(all_edges, index=val_false_edge_idx, dim=0)
    #     label = torch.cat([val_true_edge, val_false_edge], dim=0)
    #
    #     return val_true_edge_idx, val_false_edge_idx, label


# class Preprocessor_2(object):
#     #def __init__(self, exp_file, tpy_file, ratio=None):
#     def __init__(self, exp_file, tpy_file, ratio=None, use_deepprofile=True,
#                      latent_dim=200, hidden_dims=[512, 256, 128], deepprofile_epochs=100):
#         super(Preprocessor_2, self).__init__()
#
#         if ratio is None:
#             ratio = [0.6, 0.4]
#         if not isinstance(ratio, list):
#             raise Exception('g must be a list')
#
#         self.ratio = ratio
#         self.use_deepprofile = use_deepprofile
#         self.latent_dim = latent_dim
#         self.hidden_dims = hidden_dims
#         self.deepprofile_epochs = deepprofile_epochs
#         self.exp, self.src_nodes, self.dst_nodes = None, None, None
#         self.g, self.train_edge, self.test_edge, self.train_graph, self.test_graph = None, None, None, None, None,
#
#         self.exp, self.src_nodes, self.dst_nodes, self.gene_frame = sc_parser(exp_file, tpy_file)
#         if self.use_deepprofile:
#             print(f"应用DeepProfile降维，原始维度: {self.exp.shape[1]}, 目标维度: {self.latent_dim}")
#             self.exp, self.deepprofile_model = apply_deepprofile_reduction(
#                 self.exp,
#                 latent_dim=self.latent_dim,
#                 hidden_dims=self.hidden_dims,
#                 epochs=self.deepprofile_epochs
#             )
#             print(f"DeepProfile降维完成，新维度: {self.exp.shape}")
#
#         self.tpy = np.array(list(zip(self.src_nodes, self.dst_nodes)), dtype=int)
#         self.num_edge = self.tpy.shape[0]
#         self.num_gene = self.exp.shape[0]
#
#         self.exp = torch.from_numpy(self.exp).to(torch.float32)
#         self.src_nodes = torch.from_numpy(self.src_nodes).to(torch.int64)
#         self.dst_nodes = torch.from_numpy(self.dst_nodes).to(torch.int64)
#
#         self.__init_exp_and_tpy()
#         # 生成motif邻接矩阵
#         self.motif_adjs = build_motif_adjs(self.g)#加了motif后的
#         # 可选：放到GPU
#         for k in self.motif_adjs:
#              self.motif_adjs[k] = self.motif_adjs[k].to(self.exp.device)#加了motif后的
#
#     def __init_exp_and_tpy(self):
#         # 计算实际需要的节点数量（最大节点ID+1）
#         max_node_id = max(self.src_nodes.max().item(), self.dst_nodes.max().item())
#         actual_num_nodes = max(self.num_gene, max_node_id + 1)
#
#         # 如果实际节点数大于当前设置的节点数，需要扩展特征矩阵
#         if actual_num_nodes > self.num_gene:
#             print(f"警告: 扩展节点数量从 {self.num_gene} 到 {actual_num_nodes}")
#             # 创建新的特征矩阵，新增节点的特征设为0
#             new_exp = torch.zeros((actual_num_nodes, self.exp.shape[1]), dtype=self.exp.dtype)
#             new_exp[:self.num_gene] = self.exp
#             self.exp = new_exp
#             self.num_gene = actual_num_nodes
#
#         # 使用正确的节点数量创建图
#         self.g = dgl.graph((self.src_nodes, self.dst_nodes), num_nodes=self.num_gene, idtype=torch.int64)
#         self.g.ndata['feature'] = self.exp
#
#         # 其余代码保持不变...
#         num_train = int(np.floor(self.tpy.shape[0] * self.ratio[0]))
#         np.random.shuffle(self.tpy)
#         self.train_edge = self.tpy[: num_train]
#         train_edge = self.train_edge  #
#         self.test_edge = self.tpy[num_train:]
#         test_edge = self.test_edge
#         self.train_edge = list(zip(*list(self.train_edge)))
#         train_src_node_idx = torch.tensor(self.train_edge[0], dtype=torch.int64)
#         train_dst_node_idx = torch.tensor(self.train_edge[1], dtype=torch.int64)
#         self.test_edge = list(zip(*list(self.test_edge)))
#         test_src_node_idx = torch.tensor(self.test_edge[0], dtype=torch.int64)
#         test_dst_node_idx = torch.tensor(self.test_edge[1], dtype=torch.int64)
#
#         # 为pgnn输入数据
#         train_data = Data(x=self.exp, edge_index=train_edge.T)
#         train_data.num_nodes = self.num_gene
#         dists_removed = precompute_dist_data(train_data.edge_index, train_data.num_nodes,
#                                              approximate=-1)  # 按照原论文给定参数 后续你可以自己设置
#         train_data.dists = torch.from_numpy(dists_removed).float()
#         train_data.edge_index = torch.from_numpy(duplicate_edges(train_data.edge_index)).long()
#         preselect_anchor(train_data, layer_num=3, anchor_num=64, device='cpu')  # 需要调试
#
#         test_data = Data(x=self.exp, edge_index=test_edge.T)
#         test_data.num_nodes = self.num_gene
#         dists_removed = precompute_dist_data(test_data.edge_index, test_data.num_nodes,
#                                              approximate=-1)  # 按照原论文给定参数 后续你可以自己设置
#         test_data.dists = torch.from_numpy(dists_removed).float()
#         test_data.edge_index = torch.from_numpy(duplicate_edges(test_data.edge_index)).long()
#
#         preselect_anchor(test_data, layer_num=3, anchor_num=64, device='cpu')  # 需要调试
#
#         # 使用正确的节点数量创建训练和测试图
#         self.train_graph = dgl.graph(
#             (train_src_node_idx, train_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
#         )
#         self.train_graph.ndata['feature'] = self.exp
#         self.train_graph.ndata['dists_max'] = train_data.dists_max
#         self.train_graph.ndata['dists_argmax'] = train_data.dists_argmax
#
#         self.test_graph = dgl.graph(
#             (test_src_node_idx, test_dst_node_idx), num_nodes=self.num_gene, idtype=torch.int64
#         )
#         self.test_graph.ndata['feature'] = self.exp
#         # 这里应该是test_data而不是train_data
#         self.test_graph.ndata['dists_max'] = test_data.dists_max
#         self.test_graph.ndata['dists_argmax'] = test_data.dists_argmax
#
#     def generate_test_data(self, ratio):
#         # obtain the index of false edges
#         all_edges = self.g.adjacency_matrix().to_dense().view(-1)
#         all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
#         # obtain the index of true edges included in test graph
#         test_edges = self.test_graph.adjacency_matrix().to_dense().view(-1)
#         test_true_edge_idx = torch.nonzero(test_edges).reshape(1, -1).squeeze(dim=0)
#         # initial positive and negative samples
#         positive_size = test_true_edge_idx.shape[0]
#         negative_size = ratio * positive_size
#         # reindex the index of false edges
#         neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
#         shuffle(neg_sample_idx)
#         neg_sample_idx = torch.tensor(neg_sample_idx[0: negative_size], dtype=torch.int64)
#         # re-obtain label combined true and false edge.
#         test_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
#         test_false_edge = torch.gather(all_edges, index=test_false_edge_idx, dim=0)
#         test_true_edge = torch.gather(test_edges, index=test_true_edge_idx, dim=0)
#         label = torch.cat([test_true_edge, test_false_edge], dim=0)
#
#         return test_true_edge_idx, test_false_edge_idx, label
#
#     # def generate_validation_data(self):
#     #     # obtain the index of false edges
#     #     all_edges = self.g.adjacency_matrix().to_dense().view(-1)
#     #     all_false_edge_idx = torch.nonzero(all_edges == 0).reshape(1, -1).squeeze(dim=0)
#     #
#     #     # obtain the index of true edges included in test graph
#     #     val_edges = self.train_graph.adjacency_matrix().to_dense().view(-1)
#     #     val_true_edge_idx = torch.nonzero(val_edges).reshape(1, -1).squeeze(dim=0)
#     #     # initial positive and negative samples
#     #     positive_size = val_edges.shape[0]
#     #     negative_size = positive_size
#     #
#     #     #reindex the index of true edges
#     #     pos_sample_idx = np.arange(0, val_true_edge_idx.shape[0], 1)
#     #     shuffle(pos_sample_idx)
#     #     pos_sample_idx = torch.tensor(pos_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
#     #
#     #     # reindex the index of false edges
#     #     neg_sample_idx = np.arange(0, all_false_edge_idx.shape[0], 1)
#     #     shuffle(neg_sample_idx)
#     #     neg_sample_idx = torch.tensor(neg_sample_idx[0: np.floor(int(0.2 * negative_size))], dtype=torch.int64)
#     #
#     #     # re-obtain label combined true and false edge.
#     #     val_pos_edge_idx = torch.gather(val_true_edge_idx, index=pos_sample_idx, dim=0)
#     #     val_true_edge = torch.gather(val_edges, index=val_pos_edge_idx, dim=0)
#     #     val_false_edge_idx = torch.gather(all_false_edge_idx, index=neg_sample_idx, dim=0)
#     #     val_false_edge = torch.gather(all_edges, index=val_false_edge_idx, dim=0)
#     #     label = torch.cat([val_true_edge, val_false_edge], dim=0)
#     #
#     #     return val_true_edge_idx, val_false_edge_idx, label
def precompute_dist_data(edge_index, num_nodes, approximate=0):
    '''
            Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
            :return:
            '''
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()
    graph.add_edges_from(edge_list)

    n = num_nodes
    dists_array = np.zeros((n, n))
    # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
    # dists_dict = {c[0]: c[1] for c in dists_dict}
    dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                # dists_array[i, j] = 1 / (dist + 1)
                dists_array[node_i, node_j] = 1 / (dist + 1)
    return dists_array


def all_pairs_shortest_path_length_parallel(graph, cutoff=None, num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes) < 50:
        num_workers = int(num_workers / 4)
    elif len(nodes) < 400:
        num_workers = int(num_workers / 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
                                args=(
                                    graph,
                                    nodes[int(len(nodes) / num_workers * i):int(len(nodes) / num_workers * (i + 1))],
                                    cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1, :]), axis=-1)


def preselect_anchor_1(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu', k=0.5):
    # graph = pyg_to_networkx(data)#新增
    graph = pyg_to_nx(data)
    betweenness_centrality = nx.betweenness_centrality(graph)  # 新增
    # 按介数中心性降序排序
    nodes = sorted(graph.nodes(), key=lambda x: betweenness_centrality[x], reverse=True)  # 新增
    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num // anchor_size_num
    # 选择前k个节点随机被选成锚点
    selected_num = int(data.num_nodes * k)
    for i in range(anchor_size_num):
        selected_nodes = []
        anchor_size = 2 ** (i + 1) - 1
        selected_index = np.random.choice(selected_num, size=(layer_num, anchor_num_per_size, anchor_size),
                                          replace=True)
        selected_index = selected_index.flatten()
        for i in selected_index:
            selected_nodes.append(nodes[i])
        anchors = np.array(selected_nodes).reshape((layer_num, anchor_num_per_size, anchor_size))  # 新增
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    # anchorset_id = get_random_anchorset(data.num_nodes,c=1)

    anchorset_id = get_betweenness_anchorset(graph, c=1)  # 利用介质中心性来选取锚点
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):
    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num // anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2 ** (i + 1) - 1
        anchors = np.random.choice(data.num_nodes, size=(layer_num, anchor_num_per_size, anchor_size),
                                   replace=True)  # 改为每个节点只能被选择一次
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)
    anchorset_id = get_random_anchorset(data.num_nodes, c=1)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)


def pyg_to_networkx(data):
    G = nx.DiGraph()
    # 获取边并转换格式
    edge_index = data.edge_index
    edges = edge_index.t().tolist()
    # 添加边到图中
    G.add_edges_from(edges)
    return G


def pyg_to_nx(data):
    G = nx.DiGraph()

    # 添加节点
    for i in range(data.num_nodes):
        G.add_node(i)

    # 添加边
    edges = data.edge_index.t().tolist()
    G.add_edges_from(edges)

    return G


def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    copy = int(c * m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchorset_id


# 利用介质中心性来选择锚点

def get_betweenness_anchorset(G, c=0.5):
    betweenness = nx.betweenness_centrality(G)
    nodes = sorted(G.nodes(), key=lambda x: betweenness[x], reverse=True)
    n = G.number_of_nodes()
    m = int(np.log2(n))
    copy = int(c * m)
    anchorset_id = []

    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for j in range(copy):
            # 选择高介数中心性的节点作为锚点
            selected_anchors = nodes[:anchor_size]
            anchorset_id.append(selected_anchors)

    return anchorset_id


def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax