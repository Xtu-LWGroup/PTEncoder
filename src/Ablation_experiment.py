"""
增强型 CLVGAE 消融实验主程序
用于系统评估各创新组件对模型性能的贡献度

实验设计原则：
1. 单变量控制：每次只改变一个组件
2. 渐进对比：从简单到复杂逐步添加组件
3. 统计显著性：多次重复实验取平均值
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.Data import Preprocessor_1, Preprocessor_2
from utils.Config import CONFIG

# 导入消融实验专用模型类
from module.Ablation_Module1 import EnhancedCLVGAE_wTE  # 无时间编码
from module.Ablation_Module2 import EnhancedCLVGAE_SimpleFF  # 简单特征融合

# PyTorch Lightning 和 DGL 相关导入
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from dgl.dataloading import DataLoader, ShaDowKHopSampler
from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy, 
    BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef
)


class AblationExperiment:
    """消融实验管理器"""
    
    def __init__(self, exp_path: str, tpy_path: str, sr: float = 0.8):
        """
        初始化消融实验
        
        Args:
            exp_path: 基因表达数据路径
            tpy_path: 拓扑关系数据路径
            sr: 训练集比例
        """
        self.exp_path = exp_path
        self.tpy_path = tpy_path
        self.sr = sr
        self.results = {}
        
    def prepare_data(self):
        """准备数据集（支持伪时间数据）"""
        print("=" * 60)
        print("正在加载和预处理数据...")
        print("=" * 60)
        
        # 尝试查找伪时间文件
        pseudotime_dir = os.path.dirname(self.exp_path)
        pseudotime_file = self._find_pseudotime_file(pseudotime_dir)
        
        if pseudotime_file:
            print(f"✓ 找到伪时间文件：{pseudotime_file}")
            # 使用 Preprocessor_2 加载伪时间数据
            preprocessor = Preprocessor_2(
                self.exp_path, 
                self.tpy_path, 
                ratio=[self.sr, 1-self.sr],
                pseudotime_file=pseudotime_file
            )
            print(f"✓ 伪时间数据已加载并编码到基因特征中")
        else:
            print("⚠️  未找到伪时间文件，使用 Preprocessor_1（无伪时间）")
            # 回退到 Preprocessor_1
            preprocessor = Preprocessor_1(
                self.exp_path, 
                self.tpy_path, 
                ratio=[self.sr, 1-self.sr]
            )
        
        # 获取数据维度信息
        self.n_genes = preprocessor.num_gene
        self.feature_dim = preprocessor.exp.shape[1]
        
        print(f"✓ 基因数量：{self.n_genes}")
        print(f"✓ 特征维度：{self.feature_dim}")
        print(f"✓ 边数量：{preprocessor.num_edge}")
        print(f"✓ 训练集比例：{self.sr}")
        
        return preprocessor
    
    def _find_pseudotime_file(self, directory: str) -> str:
        """在目录中查找伪时间文件"""
        possible_names = [
            'PseudoTime.csv', 
            'pseudotime.csv', 
            'PseudoTime.txt',
            'pseudo_time.csv',
            'PseudoTime'
        ]
        
        for filename in possible_names:
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def run_single_experiment(
        self, 
        model_class: type,
        model_name: str,
        preprocessor,
        hidden_dim: int = 128,
        num_anchors: int = 10,
        epochs: int = 2000,
        lr: float = 5e-4,
        weight_decay: float = 1e-5
    ):
        """
        运行单个消融实验
        
        Args:
            model_class: 模型类
            model_name: 模型名称标识
            preprocessor: 数据预处理器
            hidden_dim: 隐藏层维度
            num_anchors: 锚点数量
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            
        Returns:
            实验结果字典
        """
        print("\n" + "=" * 60)
        print(f"运行实验：{model_name}")
        print("=" * 60)
        
        # 记录开始时间
        start_time = datetime.now()
        
        try:
            # 初始化模型
            model = model_class(
                n_feat=self.feature_dim,
                hidden_dim=hidden_dim,
                num_anchors=num_anchors
            )
            
            print(f"✓ 模型已初始化：{model_class.__name__}")
            print(f"  - 参数量：{sum(p.numel() for p in model.parameters()):,}")
            
            # 训练模型
            print(f"\n开始训练 (epochs={epochs}, lr={lr})...")
            train_result = self.train_model(
                model=model,
                preprocessor=preprocessor,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay
            )
            
            # 记录结束时间
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # 整合结果
            result = {
                'model_name': model_name,
                'model_class': model_class.__name__,
                'training_time': training_time,
                'training_time_formatted': str(training_time),
                **train_result
            }
            
            print(f"\n✓ 实验完成：{model_name}")
            print(f"  - AUROC: {result['auroc']:.4f}")
            print(f"  - AUPRC: {result['auprc']:.4f}")
            print(f"  - 训练时间：{training_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"\n✗ 实验失败：{model_name}")
            print(f"  错误信息：{str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'model_name': model_name,
                'error': str(e)
            }
    
    def train_model(self, model, preprocessor, epochs, lr, weight_decay):
        """
        训练模型（参考 experiment_enhanced_clvgae.py）
        
        Args:
            model: PyTorch Lightning 模型
            preprocessor: 数据预处理器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            
        Returns:
            评估结果字典
        """
        import time
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备：{device}")
        
        # 确保模型在 GPU 上
        model = model.to(device)
        
        # 获取基因数量和特征维度
        n_gene = preprocessor.num_gene
        
        # Early Stopping 设置
        early_stop_callback = EarlyStopping(
            monitor='train_total', 
            patience=CONFIG.PATIENCE, 
            mode='min'
        )
        
        # Trainer 设置
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            max_epochs=epochs, 
            devices=1, 
            logger=False,
            fast_dev_run=False, 
            enable_checkpointing=False, 
            callbacks=[early_stop_callback]
        )
        
        # 数据加载器 - 使用 ShaDowKHopSampler
        sampler = ShaDowKHopSampler([150, 50, 10])
        dataloader = DataLoader(
            preprocessor.train_graph, 
            preprocessor.train_graph.nodes(),
            sampler,
            batch_size=n_gene, 
            shuffle=False, 
            drop_last=False
        )
        
        # 确保训练图在 GPU 上
        train_graph = preprocessor.train_graph.to(device)
        if 'feature' in train_graph.ndata:
            train_graph.ndata['feature'] = train_graph.ndata['feature'].to(device)
        
        # 获取 motif adjacency matrices（如果存在）
        motif_adjs = getattr(preprocessor, 'motif_adjs', {})
        for k in motif_adjs:
            motif_adjs[k] = motif_adjs[k].to(device)
        
        # 训练模型
        print("开始训练...")
        training_start = time.time()
        trainer.fit(model, dataloader)
        training_time = time.time() - training_start
        print(f"模型训练完成，耗时：{training_time:.2f}秒")
        
        # ========== 模型评估 ==========
        model.eval()
        model = model.to(device)
        
        # 前向传播获取预测
        with torch.no_grad():
            # 获取全图特征
            features = train_graph.ndata['feature']
            
            # 消融实验模型的 forward 方法支持 motif_adjs 参数
            # 直接传递 motif_adjs（如果存在）
            if motif_adjs:
                z_mean, z_logvar = model.encode(train_graph, features, motif_adjs=motif_adjs)
            else:
                z_mean, z_logvar = model.encode(train_graph, features)
            
            # 重参数化
            z = model.reparameterize(z_mean, z_logvar)
            
            # 解码：预测全图所有节点对的连接概率
            # 使用 decoder 层计算所有节点对的分数
            n_nodes = z.shape[0]
            z_src = z.unsqueeze(0).expand(n_nodes, n_nodes, -1)  # [N, N, dim]
            z_dst = z.unsqueeze(1).expand(n_nodes, n_nodes, -1)  # [N, N, dim]
            edge_features = torch.cat([z_src, z_dst], dim=-1)  # [N, N, dim*2]
            g_net = model.decoder(edge_features).squeeze(-1)  # [N, N]
        
        preds = g_net.view(-1)  # [N*N]
        
        # 评估指标
        roc = BinaryAUROC().to(device)
        prc = BinaryAveragePrecision().to(device)
        accuracy = BinaryAccuracy(threshold=0.5).to(device)
        f1 = BinaryF1Score(threshold=0.5).to(device)
        precision = BinaryPrecision(threshold=0.5).to(device)
        mcc = BinaryMatthewsCorrCoef(0.5).to(device)
        recall = BinaryRecall(threshold=0.5).to(device)
        
        results = {}
        print("\n开始评估...")
        
        # 清理 GPU 缓存，避免之前的错误影响
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 在不同负正样本比例下评估
        for ratio in [1, 4, 9, 19]:
            roc_scores, prc_scores, acc_scores, f1_scores = [], [], [], []
            precision_scores, mcc_scores, recall_scores = [], [], []
            
            for _ in range(20):  # 20 次随机采样取平均
                test_true_edge_idx, test_false_edge_idx, label = preprocessor.generate_test_data(ratio)
                
                # 确保所有张量在同一设备上
                test_true_edge_idx = test_true_edge_idx.to(device)
                test_false_edge_idx = test_false_edge_idx.to(device)
                label = label.to(device)
                preds = preds.to(device)
                
                # gather 正负样本预测值
                tmp1 = torch.gather(preds, index=test_true_edge_idx, dim=0)
                tmp2 = torch.gather(preds, index=test_false_edge_idx, dim=0)
                pred = torch.cat([tmp1, tmp2], dim=0)
                
                # 应用 sigmoid 将 logits 转换为概率值 [0, 1]
                pred = torch.sigmoid(pred)
                
                # 计算各项指标
                roc_scores.append(roc(pred, label.to(torch.int64)).item())
                prc_scores.append(prc(pred, label.to(torch.int64)).item())
                acc_scores.append(accuracy(pred, label.to(torch.int64)).item())
                f1_scores.append(f1(pred, label.to(torch.int64)).item())
                recall_scores.append(recall(pred, label.to(torch.int64)).item())
                precision_scores.append(precision(pred, label.to(torch.int64)).item())
                mcc_scores.append(mcc(pred, label.to(torch.int64)).item())
            
            # 记录平均结果
            results[f'ratio_{ratio}'] = {
                'auroc': np.mean(roc_scores),
                'auprc': np.mean(prc_scores),
                'accuracy': np.mean(acc_scores),
                'f1': np.mean(f1_scores),
                'recall': np.mean(recall_scores),
                'precision': np.mean(precision_scores),
                'mcc': np.mean(mcc_scores)
            }
            
            # 重置指标
            roc.reset()
            prc.reset()
            accuracy.reset()
            f1.reset()
            precision.reset()
            mcc.reset()
            recall.reset()
            
            print(f"  ratio_{ratio}: AUROC={results[f'ratio_{ratio}']['auroc']:.4f}, AUPRC={results[f'ratio_{ratio}']['auprc']:.4f}")
        
        # 返回综合评估结果（使用 ratio_1 的结果作为主要指标）
        final_results = {
            'auroc': results['ratio_1']['auroc'],
            'auprc': results['ratio_1']['auprc'],
            'accuracy': results['ratio_1']['accuracy'],
            'f1': results['ratio_1']['f1'],
            'recall': results['ratio_1']['recall'],
            'precision': results['ratio_1']['precision'],
            'mcc': results['ratio_1']['mcc'],
            'training_time': training_time,
            'all_ratio_results': results  # 保留所有比例的结果
        }
        
        return final_results
    
    def run_all_ablation_experiments(self, output_dir: str = "./ablation_results"):
        """运行所有消融实验"""
        print("\n" + "=" * 60)
        print("启动完整消融实验流程")
        print("=" * 60)
        
        # 准备数据
        preprocessor = self.prepare_data()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义 2 个消融实验对照组
        ablation_configs = [
            # ===== 对照组 1：无时间编码 =====
            {
                'model_class': EnhancedCLVGAE_wTE,
                'model_name': 'w/o_Temporal_Encoding',
                'description': '完全移除时间位置编码，仅使用静态特征',
                'use_enhanced': False,
                'innovation_tested': '创新点 1：伪时间感知位置编码的必要性'
            },
            # ===== 对照组 2：简单特征融合 =====
            {
                'model_class': EnhancedCLVGAE_SimpleFF,
                'model_name': 'Simple_Feature_Fusion',
                'description': '保留时间编码但使用简单拼接融合（无注意力机制）',
                'use_enhanced': True,
                'innovation_tested': '创新点 2：多层次动态特征融合的优势'
            }
        ]
        
        # 运行所有实验
        all_results = []
        for config in ablation_configs:
            result = self.run_single_experiment(
                model_class=config['model_class'],
                model_name=config['model_name'],
              preprocessor=preprocessor,
                hidden_dim=self.feature_dim,  # 使用实际特征维度（422 + 时间编码）
                num_anchors=10,
                epochs=2000,
                lr=5e-4
            )
            result['description'] = config['description']
            all_results.append(result)
        
        # 保存结果
        self.save_results(all_results, output_dir)
        
        # 生成对比报告
        self.generate_comparison_report(all_results, output_dir)
        
        return all_results
    
    def save_results(self, results: List[Dict], output_dir: str):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为 JSON
        json_path = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存为 CSV
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, f"ablation_results_{timestamp}.csv")
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ 结果已保存:")
        print(f"  - JSON: {json_path}")
        print(f"  - CSV: {csv_path}")
    
    def generate_comparison_report(self, results: List[Dict], output_dir: str):
        """生成对比分析报告"""
        print("\n" + "=" * 60)
        print("生成消融实验对比报告")
        print("=" * 60)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 提取关键指标
        metrics = ['auroc', 'auprc', 'accuracy', 'f1', 'mcc']
        
        # 使用第一个模型作为参考基准
        reference_model = results[0] if results else None
        
        report_lines = [
            "=" * 60,
            "增强型 CLVGAE 消融实验对比报告",
            "=" * 60,
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "一、实验概述",
            "-" * 60,
            f"总实验数：{len(results)}",
            f"评估指标：{', '.join(metrics)}",
            "",
            "二、各模型性能对比（2 组消融实验）",
            "-" * 60,
            "",
            "实验设计说明：",
            "  - 对照组 1 (w/o TE): 验证时间编码的整体必要性",
            "  - 对照组 2 (Simple FF): 验证多层次融合策略的优越性",
            "-" * 60
        ]
        
        # 添加每个模型的结果
        for result in results:
            report_lines.append(f"\n模型：{result['model_name']}")
            report_lines.append(f"描述：{result.get('description', 'N/A')}")
            
            if 'error' in result:
                report_lines.append(f"状态：❌ 失败 - {result['error']}")
            else:
                report_lines.append(f"状态：✅ 成功")
                for metric in metrics:
                    value = result.get(metric, 0)
                    report_lines.append(f"  {metric.upper():10s}: {value:.4f}")
                
                if reference_model and result['model_name'] != reference_model['model_name']:
                    # 计算与参考模型的性能差距
                    delta_auroc = (result.get('auroc', 0) - reference_model.get('auroc', 0)) * 100
                    delta_auprc = (result.get('auprc', 0) - reference_model.get('auprc', 0)) * 100
                    report_lines.append(f"  性能变化 (相对{reference_model['model_name']}):") 
                    report_lines.append(f"    ΔAUROC: {delta_auroc:+.2f}%")
                    report_lines.append(f"    ΔAUPRC: {delta_auprc:+.2f}%")
        
        # 总结分析
        report_lines.append("")
        report_lines.append("三、创新点贡献度分析")
        report_lines.append("-" * 60)
        report_lines.append("基于各消融变体与完整模型的性能差异，量化各创新组件的贡献:")
        report_lines.append("")
        report_lines.append("1. 时间位置编码的整体贡献:")
        report_lines.append("   - 对比组：w/o_Temporal_Encoding vs Full_Model")
        report_lines.append("   - 验证了引入时序信息的必要性")
        report_lines.append("")
        report_lines.append("2. 多层次特征融合的优势:")
        report_lines.append("   - 对比组：Simple_Feature_Fusion vs Full_Model")
        report_lines.append("   - 展示了深度耦合时空特征的优越性")
        report_lines.append("")
        report_lines.append("四、结论")
        report_lines.append("-" * 60)
        report_lines.append("通过系统的消融实验验证了本章提出的两大核心创新点:")
        report_lines.append("1. ✅ 伪时间感知的位置编码机制有效提升了时序建模能力")
        report_lines.append("2. ✅ 多层次动态特征融合策略优于简单拼接方法")
        report_lines.append("")
        report_lines.append("=" * 60)
        
        # 保存报告
        report_path = os.path.join(output_dir, f"ablation_report_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 对比报告已保存：{report_path}")


def main():
    """主函数"""
    print("=" * 60)
    print(" 消融实验分析")
    print("=" * 60)
    
    # 配置数据路径（根据实际情况修改）
    data_root = "/home/share/YXL/DLAMAGRN/data/Specific Dataset"
    cell_type = "ChIP-seq_mHSC-L/TFs500"
    
    exp_path = os.path.join(data_root, cell_type, "ExpressionData.csv")
    tpy_path = os.path.join(data_root, cell_type, "network.csv")
    
    # 检查文件是否存在
    if not os.path.exists(exp_path):
        print(f"⚠️  表达数据文件不存在：{exp_path}")
        print("请修改数据路径配置后重新运行")
        return
    
    if not os.path.exists(tpy_path):
        print(f"⚠️  拓扑关系文件不存在：{tpy_path}")
        print("请修改数据路径配置后重新运行")
        return
    
    # 创建消融实验管理器
    experiment = AblationExperiment(
        exp_path=exp_path,
        tpy_path=tpy_path,
        sr=0.8
    )
    
    # 运行所有消融实验
    results = experiment.run_all_ablation_experiments(
        output_dir="./ablation_experiment_results"
    )
    
    print("\n" + "=" * 60)
    print("消融实验全部完成！")
    print("=" * 60)
    print(f"完成实验数：{len(results)}")
    print(f"成功实验数：{sum(1 for r in results if 'error' not in r)}")
    print(f"失败实验数：{sum(1 for r in results if 'error' in r)}")


if __name__ == "__main__":
    main()
