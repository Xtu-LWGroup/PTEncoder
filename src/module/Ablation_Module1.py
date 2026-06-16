"""
消融实验模块 1：无时间位置编码的增强型 CLVGAE

创新点验证目标：
验证"伪时间感知的位置编码机制"的必要性

设计思路：
完全移除时间位置编码模块，仅使用原始的静态基因表达特征
与完整模型对比，量化时间编码对动态GRN 推理的贡献度

对应论文创新点：
- 核心创新点 1：面向动态GRN 的伪时间感知位置编码机制设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import numpy as np
import dgl
from dgl.nn.pytorch import SAGEConv

# 导入基础组件（使用绝对路径）
try:
    # 尝试作为模块导入
    from module.Module_improve_1 import LocalPathEncoder, GlobalPathEncoder
except ImportError:
    # 回退到直接导入
    from Module_improve_1 import LocalPathEncoder, GlobalPathEncoder

class SimpleFeatureFusion(nn.Module):
    """简单的特征融合模块（无注意力机制）"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, local_feat, global_feat):
        # 简单拼接 + MLP
        fused = torch.cat([local_feat, global_feat], dim=-1)
        return self.fusion_mlp(fused)


class EnhancedCLVGAE_wTE(pl.LightningModule):
    """
    消融变体 1：无时间位置编码的 EnhancedCLVGAE
    
    与完整模型的差异：
    1. ❌ 移除 TemporalPositionalEncoding 模块
    2. ❌ 不使用伪时间信息
    3. ✅ 保留双路径编码器结构
    4. ✅ 保留基础的图卷积操作
    
    预期结果：
    - AUROC/AUPRC显著下降，证明时间编码的必要性
    - 无法捕捉时序依赖的调控关系
    """
    
    def __init__(
        self,
        n_feat: int = 200,
        hidden_dim: int = 128,
        num_anchors: int = 10,
        use_temporal_encoding: bool = False,  # 固定为False
        dropout: float = 0.1,
        kl_weight: float = 0.01,
        contrastive_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_feat = n_feat
        self.hidden_dim = hidden_dim
        self.num_anchors = num_anchors
        
        # ===== 核心区别：不引入任何时间编码 =====
        
        # 局部路径编码器（仅处理静态特征）
        self.local_encoder = LocalPathEncoder(
            input_dim=n_feat,
            hidden_dim=hidden_dim,
            num_layers=2
        )
        
        # 全局路径编码器（无时间信息输入）
        self.global_encoder = GlobalPathEncoder(
            input_dim=n_feat,
            hidden_dim=hidden_dim,
            num_anchors=num_anchors
        )
        
        # 特征融合模块（简单拼接）
        self.feature_fusion = SimpleFeatureFusion(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # 变分自编码器的投影层
        self.z_mean = nn.Linear(hidden_dim, hidden_dim)
        self.z_logvar = nn.Linear(hidden_dim, hidden_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 损失函数组件
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight
        
        # 评估指标
        self.auroc_metric = BinaryAUROC()
        self.auprc_metric = BinaryAveragePrecision()
    
    def encode(self, graph, features, motif_adjs=None, pseudotime=None):
        """
        编码过程（忽略时间信息）
            
        Args:
            graph: DGL 图
            features: 节点特征 [n_nodes, n_feat]
            motif_adjs: Motif 邻接矩阵字典（可选）
            pseudotime: 伪时间序列（此处被忽略）
                
        Returns:
           z_mean: 潜在表示的均值
           z_logvar: 潜在表示的对数方差
        """
        # 确保 features 是 Float 类型
        if features.dtype != torch.float32:
            features = features.float()
        
        # 双路径编码（无时间增强）
        # LocalPathEncoder 需要 motif_adjs，但 GlobalPathEncoder 不需要
        if motif_adjs is None or len(motif_adjs) == 0:
            # 对于 LocalPathEncoder，创建空的 motif_adjs
            device = features.device
            n_nodes = features.shape[0]
            motif_adjs_local = {
                'triangle': torch.eye(n_nodes, device=device, dtype=torch.float32),
                'edge': torch.eye(n_nodes, device=device, dtype=torch.float32)
            }
        else:
            motif_adjs_local = motif_adjs
        
        local_feat, _ = self.local_encoder(features, motif_adjs_local)
        global_feat = self.global_encoder(graph, features)  # GlobalPathEncoder 不需要 motif_adjs
            
        # 特征融合
        fused_feat = self.feature_fusion(local_feat, global_feat)
            
        # 变分投影
        z_mean = self.z_mean(fused_feat)
        z_logvar = self.z_logvar(fused_feat)
            
        return z_mean, z_logvar
    
    def reparameterize(self, mean, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z, graph):
        """
        解码过程：从潜在表示重构邻接矩阵
        
        Args:
            z: 潜在表示 [n_nodes, hidden_dim]
            graph: DGL图
            
        Returns:
            recon_adj: 重构的邻接矩阵评分 [n_edges, 1]
        """
        n_nodes = z.shape[0]
        
        # 获取所有可能的边（或采样负样本）
        src, dst = graph.edges()
        
        # 拼接源节点和目标节点的特征
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        
        # 预测边的存在概率
        recon_adj = self.decoder(edge_features).squeeze(-1)
        
        return recon_adj
    
    def compute_kl_loss(self, mean, logvar):
        """计算 KL 散度损失"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl_loss / mean.size(0)
    
    def compute_contrastive_loss(self, z, graph):
        """
        简化的对比损失（无时间加权）
        
        注意：这里使用基础的 InfoNCE 损失，不包含时间维度
        """
        # 正样本对：相连的节点
        src, dst = graph.edges()
        z_src = z[src]
        z_dst = z[dst]
        
        # 负样本：随机采样
        n_pos = z_src.shape[0]
        all_nodes = torch.arange(z.shape[0], device=z.device)
        neg_indices = torch.randperm(n_nodes)[:n_pos * 2]
        z_neg = z[neg_indices]
        
        # InfoNCE 损失
        pos_score = torch.sum(z_src * z_dst, dim=1)
        neg_score = torch.matmul(z_src, z_neg.T)
        
        # 温度系数
        temperature = 0.2
        pos_score = pos_score / temperature
        neg_score = neg_score / temperature
        
        # 计算损失
        numerator = torch.exp(pos_score)
        denominator = numerator + torch.sum(torch.exp(neg_score), dim=1)
        contrastive_loss = -torch.log(numerator / denominator + 1e-8).mean()
        
        return contrastive_loss
    
    def forward(self, graph, features, motif_adjs=None, pseudotime=None):
        """
        前向传播
            
        Args:
            graph: DGL 图
            features: 节点特征
            motif_adjs: Motif 邻接矩阵字典（可选）
            pseudotime: 伪时间序列（未使用）
                
        Returns:
            recon_adj: 重构的邻接矩阵
            z_mean: 潜在表示均值
            z_logvar: 潜在表示对数方差
        """
        # 编码
        z_mean, z_logvar = self.encode(graph, features, motif_adjs=motif_adjs, pseudotime=pseudotime)
            
        # 重参数化
        z = self.reparameterize(z_mean, z_logvar)
            
        # 解码
        recon_adj = self.decode(z, graph)
            
        return recon_adj, z_mean, z_logvar
    
    def training_step(self, batch, batch_idx):
        """训练步骤 - 添加全图监督和对比学习"""
        # DGL DataLoader 返回 [input_nodes, output_nodes, subgraph]
        input_nodes, output_nodes, subgraph = batch
        graph = subgraph  # 使用子图作为 graph
        features = subgraph.ndata['feature']  # 从子图中提取特征
        
        # 前向传播
        recon_adj, z_mean, z_logvar = self(graph, features, motif_adjs=None)
        
        # 获取潜在表示
        z = self.reparameterize(z_mean, z_logvar)
        
        # ===== 损失 1: BCE 重构损失（使用全图监督） =====
        # 获取全图邻接矩阵用于监督
        n_nodes = graph.num_nodes()
        full_adj = graph.adjacency_matrix().to_dense().view(-1).float().to(z.device)
        
        # 预测所有节点对
        z_src = z.unsqueeze(0).expand(n_nodes, n_nodes, -1)  # [N, N, dim]
        z_dst = z.unsqueeze(1).expand(n_nodes, n_nodes, -1)  # [N, N, dim]
        edge_features = torch.cat([z_src, z_dst], dim=-1)  # [N, N, dim*2]
        pred_full = self.decoder(edge_features).squeeze(-1).view(-1)  # [N*N]
        
        # BCE 损失（同时学习正负样本）
        bce_loss = self.bce_loss(pred_full, full_adj)
        
        # ===== 损失 2: KL 散度 =====
        kl_loss = self.compute_kl_loss(z_mean, z_logvar)
        
        # ===== 损失 3: 简化的对比学习损失 =====
        # 使用节点表示的对比学习
        pos_readout = torch.sum(z, dim=0)  # 全局池化
        
        # 构建负样本（使用 batch 内的其他节点）
        neg_readouts = []
        for i in range(min(10, n_nodes)):  # 最多 10 个负样本
            idx = torch.randperm(n_nodes)[:1]
            neg_z = z[idx]
            neg_readouts.append(neg_z.squeeze(0))
        
        # 计算对比损失（简化版 InfoNCE）
        if len(neg_readouts) > 0:
            neg_stack = torch.stack(neg_readouts, dim=0)  # [k, dim]
            pos_sim = torch.matmul(pos_readout.unsqueeze(0), pos_readout.unsqueeze(1))  # [1, 1]
            neg_sims = torch.matmul(pos_readout.unsqueeze(0), neg_stack.t())  # [1, k]
            logits = torch.cat([pos_sim, neg_sims], dim=1) / 0.1  # temperature=0.1
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
            info_nce_loss = F.cross_entropy(logits, labels)
        else:
            info_nce_loss = torch.tensor(0.0, device=z.device)
        
        # 总损失
        total_loss = bce_loss + self.kl_weight * kl_loss + 0.1 * info_nce_loss
        
        # 记录日志
        self.log('train_bce', bce_loss, prog_bar=True)
        self.log('train_kl', kl_loss, prog_bar=True)
        self.log('train_total', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        graph, features, labels = batch
        
        # 消融实验不使用全局 motif_adjs，在每个 batch 中动态创建
        # motif_adjs = getattr(self, 'motif_adjs', None)
        
        # 前向传播
        with torch.no_grad():
            recon_adj, z_mean, z_logvar = self(graph, features, motif_adjs=None)
        
        # 计算损失
        bce_loss = self.bce_loss(recon_adj, labels.float())
        kl_loss = self.compute_kl_loss(z_mean, z_logvar)
        total_loss = bce_loss + self.kl_weight * kl_loss
        
        # 计算评估指标
        preds = torch.sigmoid(recon_adj)
        auroc = self.auroc_metric(preds, labels.float())
        auprc = self.auprc_metric(preds, labels.float())
        
        # 记录日志
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_auroc', auroc, prog_bar=True)
        self.log('val_auprc', auprc, prog_bar=True)
        
        return {'val_loss': total_loss, 'val_auroc': auroc, 'val_auprc': auprc}
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=0.0
        )
        return optimizer
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """从检查点加载模型"""
        return super().load_from_checkpoint(checkpoint_path, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("测试消融实验模块 1：无时间位置编码")
    
    # 创建模拟数据
    n_nodes = 100
    n_feat = 200
    hidden_dim = 128
    
    features = torch.randn(n_nodes, n_feat)
    
    # 创建随机图
    src = torch.randint(0, n_nodes, (n_nodes * 2,))
    dst = torch.randint(0, n_nodes, (n_nodes * 2,))
    graph = dgl.graph((src, dst))
    
    # 初始化模型
    model = EnhancedCLVGAE_wTE(n_feat=n_feat, hidden_dim=hidden_dim)
    
    # 前向传播测试
    recon_adj, z_mean, z_logvar = model(graph, features)
    
    print(f"✓ 模型初始化成功")
    print(f"  - 参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 输出形状：recon_adj={recon_adj.shape}, z_mean={z_mean.shape}")
    print(f"\n该模型已移除时间位置编码，用于消融实验对照组")
