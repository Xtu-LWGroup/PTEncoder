"""
动态基因调控网络可视化模块
此模块用于生成随时间变化的基因调控网络可视化
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import torch
import os
import random
from typing import List, Dict, Tuple, Optional

# ✅ 设置随机种子，确保可视化可重复
def set_seed(seed=42):
    """设置所有随机种子，确保可视化可重复"""
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


class DynamicGRNVisualizer:
    """
    动态基因调控网络可视化器
    用于生成随时间变化的基因调控网络动画
    """
    
    def __init__(self, gene_names: List[str], save_dir: str = "./visualizations"):
        """
        初始化动态GRN可视化器
        
        Args:
            gene_names: 基因名称列表
            save_dir: 保存目录
        """
        self.gene_names = gene_names
        self.num_genes = len(gene_names)
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def visualize_dynamic_grn(
        self, 
        grn_matrices: List[np.ndarray], 
        pseudotime_points: List[float],
        title: str = "Dynamic Gene Regulatory Network",
        filename: str = "dynamic_grn.gif"
    ) -> str:
        """
        可视化动态基因调控网络
            
        Args:
            grn_matrices: 不同时间点的 GRN 矩阵列表
            pseudotime_points: 伪时间点列表
            title: 图表标题
            filename: 保存文件名
                
        Returns:
            保存文件的路径
        """
        if len(grn_matrices) != len(pseudotime_points):
            raise ValueError("GRN 矩阵数量必须与伪时间点数量相同")
            
        # 创建图形和两个子图区域（左侧网络图，右侧颜色条）
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax_network = fig.add_subplot(gs[0])
        ax_colorbar = fig.add_subplot(gs[1])
        ax_colorbar.axis('off')  # 隐藏颜色条子图的坐标轴
            
        # 预先计算固定的节点位置（只计算一次，保持所有帧的位置不变）
        # 使用第一个 GRN 矩阵计算布局
        first_grn = grn_matrices[0]
        threshold_first = np.percentile(np.abs(first_grn), 80)
        G_template = nx.DiGraph()
        for i in range(self.num_genes):
            G_template.add_node(i, name=self.gene_names[i])
        for i in range(self.num_genes):
            for j in range(self.num_genes):
                if i != j and abs(first_grn[i, j]) > threshold_first:
                    G_template.add_edge(i, j)
            
        # 使用固定种子和布局，确保节点位置不变
        fixed_pos = nx.spring_layout(G_template, seed=42, k=2, iterations=100)
        
        # 计算全局的颜色范围（所有帧共用）
        vmin_global = min(mat.min() for mat in grn_matrices)
        vmax_global = max(mat.max() for mat in grn_matrices)
        
        # 预先创建颜色条的 ScalarMappable（只创建一次）
        # 使用 Reds 配色方案（红色渐变）
        norm = mcolors.Normalize(vmin=vmin_global, vmax=vmax_global)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])
        
        # 在右侧子图上预先创建颜色条（只创建一次）
        cbar = plt.colorbar(sm, ax=ax_colorbar, fraction=0.8, pad=0.1)
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.set_xlabel('Regulatory Strength', fontsize=31, fontweight='normal', fontfamily='Times New Roman', labelpad=20)
        cbar.ax.yaxis.label.set_visible(False)
        ax_colorbar.set_title('Color Bar', fontsize=31, pad=10, fontfamily='Times New Roman')
            
        # 创建动画
        def animate(frame):
            ax_network.clear()
                
            # 获取当前时间点的 GRN 矩阵
            grn_matrix = grn_matrices[frame]
            pseudotime = pseudotime_points[frame]
                
            # 创建网络图
            G = nx.DiGraph()
                
            # 添加节点
            for i, gene in enumerate(self.gene_names):
                G.add_node(i, name=gene)
                
            # 添加边（只显示权重绝对值大于阈值的边）
            threshold = np.percentile(np.abs(grn_matrix), 80)  # 只显示前 20% 的强连接
            for i in range(self.num_genes):
                for j in range(self.num_genes):
                    if i != j and abs(grn_matrix[i, j]) > threshold:
                        weight = grn_matrix[i, j]
                        G.add_edge(i, j, weight=weight)
                
            # 使用预先计算的固定位置
            pos = fixed_pos
                
            # 绘制节点（大小固定）
            node_colors = ['#1f77b4'] * len(G.nodes())
            nx.draw_networkx_nodes(G, pos, ax=ax_network, node_color=node_colors, 
                                 node_size=300, alpha=0.8)
                
            # 绘制边（根据调控强度映射颜色）
            edges = [(u, v) for u, v, d in G.edges(data=True)]
            
            # ✅ 根据边的权重映射颜色（使用全局颜色范围）
            if edges:
                edge_weights = [G[u][v]['weight'] for u, v in edges]
                # 使用 Reds 配色方案映射颜色
                edge_colors = [plt.cm.Reds((w - vmin_global) / (vmax_global - vmin_global + 1e-8)) for w in edge_weights]
                
                nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                     edge_color=edge_colors, arrowsize=20, width=2, 
                                     arrowstyle='-|>', ax=ax_network, alpha=0.7)
                
            # 添加节点标签
            labels = {i: self.gene_names[i] for i in range(len(self.gene_names))}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax_network,
                                  horizontalalignment='center', verticalalignment='center')
                
            # 设置标题
            ax_network.set_title(f'{title}\nPseudotime: {pseudotime:.3f}', fontsize=14, pad=20)
            ax_network.axis('off')
            # 注意：不在这里处理颜色条，颜色条已在动画创建前静态绘制
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=len(grn_matrices), 
                           interval=1000, repeat=True, blit=False)
        
        # 保存动画
        save_path = os.path.join(self.save_dir, filename)
        anim.save(save_path, writer='pillow', fps=2, dpi=150)
        
        plt.close(fig)
        return save_path
    
    def plot_grn_heatmap_series(
        self, 
        grn_matrices: List[np.ndarray], 
        pseudotime_points: List[float],
        filename: str = "grn_heatmaps.png"
    ) -> str:
        """
        绘制一系列GRN热图（优化坐标轴显示）
        
        Args:
            grn_matrices: GRN矩阵列表
            pseudotime_points: 伪时间点列表
            filename: 保存文件名
            
        Returns:
            保存文件的路径
        """
        n_timepoints = len(grn_matrices)
        cols = min(4, n_timepoints)  # 最多4列
        rows = (n_timepoints + cols - 1) // cols
        
        # ✅ 进一步增大子图尺寸，让字体更清晰
        n_genes = len(self.gene_names)
        # 每个子图的基础大小：宽 9，高 8（进一步增大）
        base_figsize = (9 * cols, 8 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=base_figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 计算全局的vmin和vmax用于统一的颜色映射
        all_values = np.concatenate([mat.flatten() for mat in grn_matrices])
        vmin, vmax = all_values.min(), all_values.max()
        
        # ✅ 智能标签显示策略 - 只显示前10个基因，让热图更清晰
        max_labels = min(10, n_genes)  # 最多显示10个基因
        show_labels = self.gene_names[:max_labels]  # 获取前10个基因名
        
        # 同时截取矩阵的前10x10部分
        grn_matrices_filtered = [matrix[:max_labels, :max_labels] for matrix in grn_matrices]
        
        for idx, (matrix, time_point) in enumerate(zip(grn_matrices_filtered, pseudotime_points)):
            # 创建热图（使用 Reds 配色方案）
            im = sns.heatmap(
                matrix, 
                ax=axes[idx], 
                cmap='Reds',  # 红色渐变
                vmin=vmin, 
                vmax=vmax,
                xticklabels=show_labels,
                yticklabels=show_labels,
                cbar_kws={
                    'label': 'Regulatory Strength',
                    'fraction': 0.05,  # 减小颜色条宽度
                    'pad': 0.15  # 增加与热图的间距
                } if idx == 0 else {},
                square=True,  # 保持方格比例
                cbar=(idx == 0)  # 只在第一个子图显示颜色条
            )
            
            # ✅ 只在第一个子图调整颜色条字体大小和字体
            if idx == 0:
                cbar = im.collections[0].colorbar
                if cbar:
                    cbar.ax.yaxis.label.set_fontsize(31)
                    cbar.ax.yaxis.label.set_fontweight('normal')
                    cbar.ax.yaxis.label.set_fontfamily('Times New Roman')
                    for tick in cbar.ax.yaxis.get_ticklabels():
                        tick.set_fontsize(31)
                        tick.set_fontfamily('Times New Roman')
                    cbar.ax.yaxis.label.set_rotation(270)
                    # ✅ 把颜色条向右移动，标签在左侧，标签与颜色条保持适当距离
                    cbar.ax.yaxis.set_label_position('left')
                    cbar.ax.yaxis.set_label_coords(-2.5, 0.5)
                    # ✅ 调整颜色条位置，整体往左移动
                    cbar.ax.set_position([0.75, 0.15, 0.03, 0.7])
            
            # ✅ 优化坐标轴显示 - 修复对齐问题
            axes[idx].set_title(f'Pseudotime: {time_point:.3f}', fontsize=31, pad=20, fontweight='normal', fontfamily='Times New Roman')
            
            # ✅ 去掉 X 轴和 Y 轴的基因名标签
            axes[idx].set_xticklabels([])
            axes[idx].set_yticklabels([])
            axes[idx].set_xlabel('Target Genes', fontsize=31, labelpad=15, fontweight='normal', fontfamily='Times New Roman')
            axes[idx].set_ylabel('Regulator Genes', fontsize=31, labelpad=15, fontweight='normal', fontfamily='Times New Roman')
        
        # 隐藏多余的子图
        for idx in range(n_timepoints, len(axes)):
            axes[idx].axis('off')
        
        # ✅ 优化布局间距 - 减小子图间距，让子图更近
        plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=0.5)
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"热图已生成，坐标轴已优化显示 {len(show_labels)} 个基因标签")
        return save_path
    
    def plot_regulatory_activity_over_time(
        self, 
        grn_matrices: List[np.ndarray], 
        pseudotime_points: List[float],
        filename: str = "regulatory_activity.png"
    ) -> str:
        """
        绘制随时间变化的调控活性
        
        Args:
            grn_matrices: GRN矩阵列表
            pseudotime_points: 伪时间点列表
            filename: 保存文件名
            
        Returns:
            保存文件的路径
        """
        # 计算每个时间点的调控活性
        total_activities = []
        positive_activities = []
        negative_activities = []
        
        for matrix in grn_matrices:
            total_activity = np.abs(matrix).sum() / 2  # 除以2是因为矩阵通常是对称的
            positive_activity = matrix[matrix > 0].sum()
            negative_activity = np.abs(matrix[matrix < 0]).sum()
            
            total_activities.append(total_activity)
            positive_activities.append(positive_activity)
            negative_activities.append(negative_activity)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(pseudotime_points, total_activities, label='Total Regulatory Activity', linewidth=2)
        ax.plot(pseudotime_points, positive_activities, label='Positive Regulation', linewidth=2)
        ax.plot(pseudotime_points, negative_activities, label='Negative Regulation', linewidth=2)
        
        ax.set_xlabel('Pseudotime')
        ax.set_ylabel('Regulatory Activity')
        ax.set_title('Regulatory Activity Over Pseudotime')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


def generate_sample_data():
    """
    生成示例数据用于测试
    """
    # 示例基因名称
    gene_names = [f'Gene_{i}' for i in range(20)]
    
    # 生成示例GRN矩阵（随时间变化）
    pseudotime_points = np.linspace(0, 1, 10)
    grn_matrices = []
    
    for t in pseudotime_points:
        # 生成基础网络矩阵
        base_matrix = np.random.randn(20, 20) * 0.5
        
        # 添加时间相关的动态变化
        time_effect = np.sin(t * 2 * np.pi) * 0.3
        base_matrix += time_effect * np.random.randn(20, 20) * 0.2
        
        # 确保对角线为0（基因不调控自己）
        np.fill_diagonal(base_matrix, 0)
        
        grn_matrices.append(base_matrix)
    
    return gene_names, grn_matrices, pseudotime_points.tolist()


if __name__ == "__main__":
    # 测试可视化功能
    gene_names, grn_matrices, pseudotime_points = generate_sample_data()
    
    visualizer = DynamicGRNVisualizer(gene_names)
    
    print("生成动态GRN可视化...")
    
    # 生成动态网络动画
    gif_path = visualizer.visualize_dynamic_grn(
        grn_matrices, 
        pseudotime_points,
        title="Sample Dynamic GRN",
        filename="sample_dynamic_grn.gif"
    )
    print(f"动态网络动画已保存至: {gif_path}")
    
    # 生成热图系列
    heatmap_path = visualizer.plot_grn_heatmap_series(
        grn_matrices, 
        pseudotime_points,
        filename="sample_grn_heatmaps.png"
    )
    print(f"GRN热图系列已保存至: {heatmap_path}")
    
    # 生成调控活性图
    activity_path = visualizer.plot_regulatory_activity_over_time(
        grn_matrices, 
        pseudotime_points,
        filename="sample_regulatory_activity.png"
    )
    print(f"调控活性图已保存至: {activity_path}")
    
    print("可视化完成！")