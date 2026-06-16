import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


class DeepProfileEncoder(nn.Module):
    """DeepProfile编码器，用于学习基因表达数据的低维表示"""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.2):
        super(DeepProfileEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 构建编码器网络
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class DeepProfileDecoder(nn.Module):
    """DeepProfile解码器，用于重构原始基因表达数据"""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.2):
        super(DeepProfileDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # 构建解码器网络（与编码器对称）
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)


class DeepProfile(nn.Module):
    """DeepProfile自编码器，用于基因表达数据降维"""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.2):
        super(DeepProfile, self).__init__()

        self.encoder = DeepProfileEncoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = DeepProfileDecoder(latent_dim, hidden_dims, input_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        """仅进行编码，返回降维后的特征"""
        return self.encoder(x)


class DeepProfileTrainer:
    """DeepProfile训练器"""

    def __init__(self, model: DeepProfile, learning_rate: float = 1e-3, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(self.device)

            # 前向传播
            recon_batch, _ = self.model(batch)
            loss = self.criterion(recon_batch, batch)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def encode_data(self, dataloader):
        """对数据进行编码"""
        self.model.eval()
        encoded_features = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                encoded_batch = self.model.encode(batch)
                encoded_features.append(encoded_batch.cpu())

        return torch.cat(encoded_features, dim=0)


def apply_deepprofile_reduction(expression_data: np.ndarray,
                                latent_dim: int = 100,
                                hidden_dims: list = [512, 256, 128],
                                epochs: int = 100,
                                batch_size: int = 32,
                                learning_rate: float = 1e-3,
                                device: str = 'cuda') -> Tuple[np.ndarray, DeepProfile]:
    """
    使用DeepProfile对基因表达数据进行降维

    Args:
        expression_data: 基因表达数据矩阵 (n_genes, n_samples)
        latent_dim: 降维后的维度
        hidden_dims: 隐藏层维度列表
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        device: 计算设备

    Returns:
        reduced_data: 降维后的数据
        model: 训练好的DeepProfile模型
    """

    # 数据预处理
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expression_data.T)  # 转置以匹配模型输入格式

    # 创建数据加载器
    dataset = torch.FloatTensor(scaled_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    input_dim = scaled_data.shape[1]  # 基因数量
    model = DeepProfile(input_dim, hidden_dims, latent_dim)
    trainer = DeepProfileTrainer(model, learning_rate, device)

    # 训练模型
    print(f"开始训练DeepProfile模型，输入维度: {input_dim}, 输出维度: {latent_dim}")
    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.6f}")

    # 对数据进行编码
    print("对数据进行降维编码...")
    encoded_data = trainer.encode_data(dataloader)

    return encoded_data.numpy(), model


def biological_regularization_loss(encoded_features: torch.Tensor,
                                   gene_annotations: Optional[dict] = None) -> torch.Tensor:
    """
    生物学正则化损失，鼓励相似功能的基因在潜在空间中聚集

    Args:
        encoded_features: 编码后的特征
        gene_annotations: 基因功能注释信息

    Returns:
        regularization_loss: 正则化损失
    """
    # 这里可以实现基于基因功能注释的正则化
    # 例如：鼓励功能相似的基因在潜在空间中距离更近

    if gene_annotations is None:
        return torch.tensor(0.0, device=encoded_features.device)

    # 示例：基于基因功能相似性的正则化
    # 这里需要根据具体的基因注释数据来实现
    return torch.tensor(0.0, device=encoded_features.device)