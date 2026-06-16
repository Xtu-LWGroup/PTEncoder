"""
综合动态基因调控网络分析和可视化脚本
整合模型性能改进和动态网络可视化功能
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from huatu_visualization import DynamicGRNVisualizer
from extract_dynamic_grn import analyze_grn_dynamics, visualize_dynamic_grn
from improve_model_performance import hyperparameter_tuning_suggestions


def load_experiment_results(results_path: str) -> Dict:
    """
    加载实验结果文件
    
    Args:
        results_path: 结果文件路径
        
    Returns:
        解析后的结果字典
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def analyze_current_performance(results: Dict):
    """
    分析当前模型性能
    
    Args:
        results: 实验结果字典
    """
    print("=== 当前模型性能分析 ===")
    
    if 'enhanced' in results and 'results' in results['enhanced']:
        enhanced_results = results['enhanced']['results']
        
        print("不同比例下的性能指标:")
        for ratio, metrics in enhanced_results.items():
            print(f"\n{ratio}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")
    
    print(f"\n预处理时间: {results.get('preprocessing_time', 'N/A')}s")
    print(f"训练时间: {results.get('training_time', 'N/A')}s")
    print(f"总时间: {results.get('total_time', 'N/A')}s")
    print(f"特征维度: {results.get('feature_dim', 'N/A')}")
    print(f"基因数量: {results.get('gene_count', 'N/A')}")


def generate_performance_improvement_report():
    """
    生成性能改进报告
    """
    print("\n" + "="*60)
    print("           基因调控网络性能改进报告")
    print("="*60)
    
    print("\n【问题诊断】")
    print("1. AUROC和AUPRC值偏低的原因:")
    print("   - 模型架构过于复杂，导致过拟合")
    print("   - 损失函数权重不平衡")
    print("   - 训练数据不足或质量不高")
    print("   - 超参数设置不合理")
    print("   - 模型容量与任务复杂度不匹配")
    
    print("\n【性能改进策略】")
    print("1. 架构优化:")
    print("   - 简化模型复杂度，减少过拟合风险")
    print("   - 使用残差连接改善梯度流动")
    print("   - 添加注意力机制突出重要基因交互")
    
    print("\n2. 训练策略改进:")
    print("   - 使用学习率调度器动态调整学习率")
    print("   - 实施早停机制防止过拟合")
    print("   - 使用梯度裁剪稳定训练过程")
    print("   - 数据增强提升模型泛化能力")
    
    print("\n3. 损失函数优化:")
    print("   - 平衡重构损失和正则化项")
    print("   - 调整对比学习权重")
    print("   - 使用标签平滑减少过拟合")
    
    print("\n4. 超参数调优:")
    hyperparameter_tuning_suggestions()


def create_dynamic_visualization_pipeline():
    """
    创建动态可视化流水线说明
    """
    print("\n" + "="*60)
    print("         动态基因调控网络可视化流水线")
    print("="*60)
    
    print("\n【流水线概述】")
    print("1. 数据准备阶段:")
    print("   - 从训练模型中提取不同时间点的GRN矩阵")
    print("   - 处理伪时间序列数据")
    print("   - 准备基因名称列表")
    
    print("\n2. 可视化生成阶段:")
    print("   - 生成动态网络动画(.gif)")
    print("   - 创建GRN热图系列")
    print("   - 绘制调控活性随时间变化图")
    print("   - 分析网络拓扑特性动态变化")
    
    print("\n3. 结果输出阶段:")
    print("   - 保存所有可视化结果")
    print("   - 生成动态特性分析报告")
    print("   - 提供交互式探索接口")


def simulate_dynamic_grn_analysis():
    """
    模拟动态GRN分析（用于演示）
    """
    print("\n【动态GRN分析演示】")
    
    # 模拟一些GRN矩阵和时间点
    n_genes = 50
    n_timepoints = 10
    time_points = np.linspace(0, 1, n_timepoints)
    
    grn_matrices = []
    for t in time_points:
        # 创建一个随时间变化的模拟GRN
        base = np.random.randn(n_genes, n_genes) * 0.1
        # 添加时间相关的变化
        time_effect = np.sin(t * 4 * np.pi) * 0.05
        base += time_effect * np.random.randn(n_genes, n_genes) * 0.05
        # 确保对角线为0
        np.fill_diagonal(base, 0)
        grn_matrices.append(base)
    
    # 创建基因名称
    gene_names = [f'TF_{i:03d}' for i in range(n_genes)]
    
    print(f"已生成 {n_timepoints} 个时间点的GRN数据")
    print(f"基因数量: {n_genes}")
    print(f"时间范围: {time_points[0]:.2f} 到 {time_points[-1]:.2f}")
    
    # 运行分析
    analyze_grn_dynamics(grn_matrices, time_points.tolist())
    
    # 可视化（使用子集以加快演示速度）
    subset_indices = slice(None, None, 2)  # 每隔一个取一个
    subset_matrices = [grn_matrices[i] for i in range(0, len(grn_matrices), 2)]
    subset_times = [time_points[i] for i in range(0, len(time_points), 2)]
    
    print(f"\n正在生成可视化... (使用 {len(subset_matrices)} 个时间点的子集)")
    # 根据实际基因数量调整 max_nodes，避免索引越界
    visualize_dynamic_grn(subset_matrices, subset_times, gene_names[:20], max_nodes=min(15, len(gene_names[:20])))  # 只使用前 20 个基因


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='动态基因调控网络分析和可视化')
    parser.add_argument('--results-path', type=str, default='../enhanced_clvgae_comparison_results.json',
                       help='实验结果文件路径')
    parser.add_argument('--generate-report', action='store_true',
                       help='生成性能改进报告')
    parser.add_argument('--visualize-dynamic', action='store_true',
                       help='生成动态可视化')
    parser.add_argument('--analyze-performance', action='store_true',
                       help='分析当前性能')
    
    args = parser.parse_args()
    
    print("动态基因调控网络分析工具")
    print("="*50)
    
    # 如果提供了结果文件路径，尝试加载并分析
    if os.path.exists(args.results_path):
        try:
            results = load_experiment_results(args.results_path)
            print(f"成功加载结果文件: {args.results_path}")
            
            if args.analyze_performance or not any([args.generate_report, args.visualize_dynamic]):
                analyze_current_performance(results)
        except Exception as e:
            print(f"加载结果文件失败: {e}")
            results = None
    else:
        print(f"结果文件不存在: {args.results_path}")
        results = None
    
    # 生成性能改进报告
    if args.generate_report or not any([args.analyze_performance, args.visualize_dynamic]):
        generate_performance_improvement_report()
    
    # 创建可视化流水线说明
    create_dynamic_visualization_pipeline()
    
    # 演示动态分析
    if args.visualize_dynamic or not any([args.analyze_performance, args.generate_report]):
        simulate_dynamic_grn_analysis()
    
    print("\n" + "="*60)
    print("分析完成！")
    print("生成的文件将保存在 ./dynamic_grn_visualization/ 目录中")
    print("分析图表将保存在 ./grn_analysis/ 目录中")
    print("="*60)


def run_complete_analysis():
    """
    运行完整的分析流程
    """
    print("运行完整的动态GRN分析流程...")
    
    # 1. 显示当前性能
    print("\n1. 当前性能分析")
    print("-" * 30)
    # 这里会加载实际的结果文件进行分析
    
    # 2. 性能改进建议
    print("\n2. 性能改进建议")
    print("-" * 30)
    generate_performance_improvement_report()
    
    # 3. 可视化流水线
    print("\n3. 动态可视化流水线")
    print("-" * 30)
    create_dynamic_visualization_pipeline()
    
    # 4. 演示动态分析
    print("\n4. 动态GRN分析演示")
    print("-" * 30)
    simulate_dynamic_grn_analysis()
    
    print("\n完成！")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有命令行参数，运行完整分析
        run_complete_analysis()
    else:
        # 否则按参数执行
        main()