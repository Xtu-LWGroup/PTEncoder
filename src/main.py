import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import json
import os
import time
import numpy as np
import torch
from pytorch_lightning import Trainer
from dgl.dataloading import DataLoader, ShaDowKHopSampler
from pytorch_lightning.callbacks import EarlyStopping
from utils.Data import Preprocessor_2
from module.Module_improve_1 import EnhancedCLGVAE
from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy, BinaryF1Score, BinaryPrecision,
    BinaryRecall, BinaryMatthewsCorrCoef
)
from utils.Config import CONFIG

torch.manual_seed(29)
torch.cuda.manual_seed_all(29)
np.random.seed(29)


def _find_first_existing(path_dir, candidates):
    for name in candidates:
        p = os.path.join(path_dir, name)
        if os.path.exists(p):
            return p
    return None


def _auto_pick_network_path(dataset_dir):
    """
    ChIP-seq_hESC 这类数据集的网络文件名可能不是 network.csv（例如 500_ChIP-seq_hESC-network.csv）。
    这里优先找 network.csv，否则找包含 network 的任意 .csv。
    """
    preferred = os.path.join(dataset_dir, "network.csv")
    if os.path.exists(preferred):
        return preferred

    for fn in os.listdir(dataset_dir):
        if fn.lower().endswith(".csv") and "network" in fn.lower():
            return os.path.join(dataset_dir, fn)

    raise FileNotFoundError(f"未找到网络文件(network.csv 或 *network*.csv): {dataset_dir}")


# def run_enhanced_experiment(exp_path, tpy_path, sr, use_enhanced=True,
#                             hidden_dim=128, num_anchors=10):
def run_enhanced_experiment(exp_path, tpy_path, sr, use_enhanced=True,
                            hidden_dim=256, num_anchors=10):
    """运行增强版CLVGAE实验"""

    print(f"开始实验 - 增强版CLVGAE: {use_enhanced}")
    start_time = time.time()

    # 数据预处理
    # 尝试加载伪时间数据
    pseudotime_dir = os.path.dirname(exp_path)
    pseudotime_file = _find_first_existing(
        pseudotime_dir,
        ['PseudoTime.csv', 'pseudotime.csv', 'PseudoTime.txt', 'PseudoTime']
    )
    if pseudotime_file is not None:
        print(f"找到伪时间文件: {pseudotime_file}")

    preprocessor = Preprocessor_2(exp_path, tpy_path, sr, pseudotime_file=pseudotime_file)
    # 加了motif后的
    motif_adjs = preprocessor.motif_adjs
    # 伪时间数据（现在是numpy数组，用于时间特征计算，模型不再直接使用）
    pseudotime = preprocessor.pseudotime  # numpy数组，不再直接用于模型
    # motif_adjs建议放到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for k in motif_adjs:
        motif_adjs[k] = motif_adjs[k].to(device)  # 加了motif后的
    preprocessing_time = time.time() - start_time
    print(f"数据预处理完成，耗时: {preprocessing_time:.2f}秒")

    # 模型初始化
    n_gene, n_feat = preprocessor.exp.shape
    # 强制使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if use_enhanced:
        # model = EnhancedCLGVAE(n_gene, n_feat, hidden_dim, num_anchors,device=device)#未加motif时的
        model = EnhancedCLGVAE(n_gene, n_feat, hidden_dim, num_anchors, device=device,
                               motif_adjs=motif_adjs)  # 加了motif后的
        # 注意：伪时间信息已经通过时间特征编码在表达特征中，模型不再需要原始的pseudotime数组
        # model.pseudotime = pseudotime  # 不再需要，因为时间信息已编码在features中
        print(f"使用增强版CLVGAE - 隐藏维度: {hidden_dim}, 锚点数量: {num_anchors}")
        if pseudotime is not None:
            print(f"伪时间数据已加载: {len(pseudotime)} 个细胞，已转化为基因时间特征并拼接在表达特征中")
    else:
        # model = CLGVAE(n_gene, n_feat)
        print(f"使用原始CLVGAE")

    # 确保模型在GPU上
    model = model.to(device)
    # 训练器设置
    early_stop_callback = EarlyStopping(monitor='total_loss', patience=CONFIG.PATIENCE, mode='min')
    trainer = Trainer(
        accelerator='gpu', max_epochs=CONFIG.EPOCH, devices=1, logger=False,
        fast_dev_run=False, enable_checkpointing=False, callbacks=[early_stop_callback]
    )

    # 数据加载器
    sampler = ShaDowKHopSampler([150, 50, 10])
    dataloader = DataLoader(
        preprocessor.train_graph, preprocessor.train_graph.nodes(), sampler,
        batch_size=n_gene, shuffle=False, drop_last=False
    )
    # 确保训练图在GPU上
    preprocessor.train_graph = preprocessor.train_graph.to(device)
    if 'feature' in preprocessor.train_graph.ndata:
        preprocessor.train_graph.ndata['feature'] = preprocessor.train_graph.ndata['feature'].to(device)

    # 训练模型
    training_start = time.time()
    trainer.fit(model, dataloader)
    training_time = time.time() - training_start
    print(f"模型训练完成，耗时: {training_time:.2f}秒")

    # 模型评估
    model.eval()
    model = model.to(device)
    # 确保图在正确的设备上
    # device = next(model.parameters()).device
    print(f"模型设备: {device}")

    # 确保图及其特征在正确设备上
    train_graph = preprocessor.train_graph.to(device)
    if 'feature' in train_graph.ndata:
        train_graph.ndata['feature'] = train_graph.ndata['feature'].to(device)

    print(f"评估图设备: {train_graph.device}")
    if 'feature' in train_graph.ndata:
        print(f"评估特征设备: {train_graph.ndata['feature'].device}")
    if use_enhanced:
        # g_net, _, _, _, _, _, _, _ = model(train_graph)#未加了motif时的
        # 注意：时间信息已编码在features中，不再需要单独传递pseudotime参数
        g_net, _, _, _, _, _, _, _ = model(train_graph, motif_adjs=motif_adjs,
                                            pseudotime=None)  # 加了motif后的，时间信息已在features中
    else:
        g_net, _, _, _, _, _ = model(train_graph)
    preds = g_net.view(-1)

    # 评估指标
    roc = BinaryAUROC().to(device)
    prc = BinaryAveragePrecision().to(device)
    accuracy = BinaryAccuracy(threshold=0.5).to(device)
    f1 = BinaryF1Score(threshold=0.5).to(device)
    precision = BinaryPrecision(threshold=0.5).to(device)
    mcc = BinaryMatthewsCorrCoef(0.5).to(device)
    recall = BinaryRecall(threshold=0.5).to(device)

    results = {}
    for i, r in enumerate([1, 4, 9, 19]):
        roc_scores, prc_scores, acc_scores, f1_scores = [], [], [], []
        precision_scores, mcc_scores, recall_scores = [], [], []

        for _ in range(20):
            test_true_edge_idx, test_false_edge_idx, label = preprocessor.generate_test_data(r)
            # 确保所有张量在同一个设备上
            test_true_edge_idx = test_true_edge_idx.to(device)
            test_false_edge_idx = test_false_edge_idx.to(device)
            label = label.to(device)
            # 确保预测结果也在正确设备上
            # print(f"g_net: ")
            preds = preds.to(device)

            tmp1 = torch.gather(preds, index=test_true_edge_idx, dim=0)
            tmp2 = torch.gather(preds, index=test_false_edge_idx, dim=0)
            pred = torch.cat([tmp1, tmp2], dim=0)

            roc_scores.append(roc(pred, label.to(torch.int64)).item())
            prc_scores.append(prc(pred, label.to(torch.int64)).item())
            acc_scores.append(accuracy(pred, label.to(torch.int64)).item())
            f1_scores.append(f1(pred, label.to(torch.int64)).item())
            recall_scores.append(recall(pred, label.to(torch.int64)).item())
            precision_scores.append(precision(pred, label.to(torch.int64)).item())
            mcc_scores.append(mcc(pred, label.to(torch.int64)).item())
            # print(f"评估图设备: mcc_scores")

        results[f'ratio_{r}'] = {
            'auroc': np.mean(roc_scores),
            'auprc': np.mean(prc_scores),
            'accuracy': np.mean(acc_scores),
            'f1': np.mean(f1_scores),
            'recall': np.mean(recall_scores),
            'precision': np.mean(precision_scores),
            'mcc': np.mean(mcc_scores)
        }

        roc.reset()
        prc.reset()
        accuracy.reset()
        f1.reset()
        precision.reset()
        mcc.reset()
        recall.reset()

    total_time = time.time() - start_time

    return {
        'results': results,
        'preprocessing_time': preprocessing_time,
        'training_time': training_time,
        'total_time': total_time,
        'feature_dim': n_feat,
        'gene_count': n_gene,
        'model_type': 'EnhancedCLVGAE' if use_enhanced else 'OriginalCLVGAE'
    }


def compare_enhanced_vs_original(exp_path, tpy_path, test_dataset, sr=[0.8, 0.2]):
    """对比增强版和原始CLVGAE的性能"""

    # # 实验参数
    # exp_file = 'ExpressionData.csv'
    # tpy_file = 'network.csv'
    # root_dir = '/home/share/YXL/CLVGAE/data'
    # save_dir = '/home/share/YXL/CLVGAE/processed'
    # 选择一个数据集进行测试
    # test_dataset = 'Non-Specific Dataset/hESC/TFs500'
    # exp_path = os.path.join(root_dir, test_dataset, exp_file)
    # tpy_path = os.path.join(root_dir, test_dataset, tpy_file)

    # sr = [0.6, 0.4]  # 训练/测试分割比例

    # print("=" * 60)
    # print("增强版CLVGAE vs 原始CLVGAE 性能对比实验")
    # print("=" * 60)
    #
    # # 运行原始CLVGAE实验
    # print("\n1. 运行原始CLVGAE...")
    # original_results = run_enhanced_experiment(
    #     exp_path, tpy_path, sr, use_enhanced=False
    # )

    # 运行增强版CLVGAE实验
    print("\n2. 运行增强版CLVGAE...")
    enhanced_results = run_enhanced_experiment(
        exp_path, tpy_path, sr, use_enhanced=True,
        hidden_dim=406, num_anchors=10
    )

    # 结果对比
    print("\n" + "=" * 60)
    print("实验结果对比")
    print("=" * 60)

    print(f"\n模型信息:")
    # print(f"原始CLVGAE: {original_results['model_type']}")
    print(f"增强版CLVGAE: {enhanced_results['model_type']}")
    # print(f"特征维度: {original_results['feature_dim']} -> {enhanced_results['feature_dim']}")

    print(f"\n时间对比:")
    # print(
    #   f"原始CLVGAE - 预处理: {original_results['preprocessing_time']:.2f}s, 训练: {original_results['training_time']:.2f}s, 总计: {original_results['total_time']:.2f}s")
    print(
        f"增强版CLVGAE - 预处理: {enhanced_results['preprocessing_time']:.2f}s, 训练: {enhanced_results['training_time']:.2f}s, 总计: {enhanced_results['total_time']:.2f}s")

    print(f"\n性能指标对比:")
    ratios = [9]
    for ratio in ratios:
        print(f"\n比例 {ratio}:")
        # orig = original_results['results'][f'ratio_{ratio}']
        enh = enhanced_results['results'][f'ratio_{ratio}']

        # print(
        #     f"  AUROC: 原始 {orig['auroc']:.3f} vs 增强 {enh['auroc']:.3f} ({'↑' if enh['auroc'] > orig['auroc'] else '↓'})")
        # print(
        #     f"  AUPRC: 原始 {orig['auprc']:.3f} vs 增强 {enh['auprc']:.3f} ({'↑' if enh['auprc'] > orig['auprc'] else '↓'})")
        # print(f"  F1:    原始 {orig['f1']:.3f} vs 增强 {enh['f1']:.3f} ({'↑' if enh['f1'] > orig['f1'] else '↓'})")
        # print(f"  MCC:   原始 {orig['mcc']:.3f} vs 增强 {enh['mcc']:.3f} ({'↑' if enh['mcc'] > orig['mcc'] else '↓'})")

    # # 保存结果
    comparison_results = {
        # 'original': original_results,
        'enhanced': enhanced_results,
        'dataset': test_dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    #
    # with open('enhanced_clvgae_comparison_results.json', 'w', encoding='UTF-8') as f:
    #     json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    #
    # print(f"\n结果已保存到: enhanced_clvgae_comparison_results.json")
    return comparison_results


def parameter_sensitivity_analysis():
    """参数敏感性分析"""

    # 实验参数
    exp_file = 'ExpressionData.csv'
    tpy_file = 'network.csv'
    root_dir = '/home/share/YXL/CLVGAE/data'
    test_dataset = 'Non-Specific Dataset/hESC/TFs500'

    exp_path = os.path.join(root_dir, test_dataset, exp_file)
    tpy_path = os.path.join(root_dir, test_dataset, tpy_file)
    sr = [0.6, 0.4]

    print("=" * 60)
    print("增强版CLVGAE参数敏感性分析")
    print("=" * 60)

    # 测试不同的隐藏维度
    hidden_dims = [64, 128, 256]
    anchor_nums = [5, 10, 15]

    results = {}

    for hidden_dim in hidden_dims:
        for num_anchors in anchor_nums:
            print(f"\n测试参数: hidden_dim={hidden_dim}, num_anchors={num_anchors}")

            try:
                result = run_enhanced_experiment(
                    exp_path, tpy_path, sr, use_enhanced=True,
                    hidden_dim=hidden_dim, num_anchors=num_anchors
                )

                # 计算平均性能
                avg_auroc = np.mean([result['results'][f'ratio_{r}']['auroc'] for r in [1, 4, 9, 19]])
                avg_auprc = np.mean([result['results'][f'ratio_{r}']['auprc'] for r in [1, 4, 9, 19]])

                results[f'h{hidden_dim}_a{num_anchors}'] = {
                    'hidden_dim': hidden_dim,
                    'num_anchors': num_anchors,
                    'avg_auroc': avg_auroc,
                    'avg_auprc': avg_auprc,
                    'training_time': result['training_time']
                }

                print(f"  平均AUROC: {avg_auroc:.3f}, 平均AUPRC: {avg_auprc:.3f}")

            except Exception as e:
                print(f"  参数组合失败: {e}")
                continue

    # 保存参数敏感性分析结果
    with open('parameter_sensitivity_results.json', 'w', encoding='UTF-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n参数敏感性分析结果已保存到: parameter_sensitivity_results.json")

    # 找到最佳参数组合
    if results:
        best_config = max(results.items(), key=lambda x: x[1]['avg_auroc'])
        print(f"\n最佳参数组合: {best_config[0]}")
        print(f"隐藏维度: {best_config[1]['hidden_dim']}, 锚点数量: {best_config[1]['num_anchors']}")
        print(f"最佳平均AUROC: {best_config[1]['avg_auroc']:.3f}")


if __name__ == '__main__':

    datasets = ['Specific Dataset']
    cell_type = ['ChIP-seq_mDC']

    #cell_type = ['ChIP-seq_mHSC-L']

    dataset_names = ['Specific']

    n_tf = ['TFs500']
    # n_tf = [ 'TFs1000']
    tf = ['500']
    split_ratios = [
        [0.8, 0.2]
    ]
    # init params
    exp_file = 'ExpressionData.csv'
    # 不再写死 network.csv（ChIP-seq_hESC/TFs500 下是 500_ChIP-seq_hESC-network.csv）
    tpy_file = 'network.csv'

    root_dir = '/home/share/YXL/DLAMAGRN/data'
    save_dir = '/home/share/YXL/DLAMAGRN/processed'
    threshold = 0.5

    raw_dirs = [os.path.join(root_dir, d, c, t) for d in datasets for c in cell_type for t in n_tf]
    file_names = ['{}-{}{}'.format(d, c, t) for d in dataset_names for c in cell_type for t in tf]

    sampler = ShaDowKHopSampler([150, 50, 10])

    for idx, f in enumerate(file_names):
        if not os.path.exists(os.path.join(save_dir, f)):
            os.makedirs(os.path.join(save_dir, f))
        exp_path = os.path.join(raw_dirs[idx], exp_file)
        # 自动选择网络文件
        try:
            tpy_path = os.path.join(raw_dirs[idx], tpy_file)
            if not os.path.exists(tpy_path):
                tpy_path = _auto_pick_network_path(raw_dirs[idx])
        except Exception:
            tpy_path = _auto_pick_network_path(raw_dirs[idx])

        for sr in split_ratios:
            mf = os.path.join(save_dir, f, 'metric{}{}.json'.format(int(sr[0] * 10), int(sr[1] * 10)))
            # 运行主要对比实验
            rs = compare_enhanced_vs_original(exp_path, tpy_path, raw_dirs[idx], sr)
            with open(mf, mode='w', encoding='UTF-8') as jf:
                json.dump(rs, jf)

        # 运行参数敏感性分析
    # parameter_sensitivity_analysis()