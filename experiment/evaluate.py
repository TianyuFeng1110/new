import argparse
import os
import torch
from ruamel.yaml import YAML
import sys

from models.cc_model import Model
import numpy as np
import utils as utils

from cc import get_loader

def load_model_and_evaluate(args, config):
    device = torch.device(args.device)
    
    datasets_path = args.path
    namespace = args.namespace
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    num_classes = config['num_classes']
    features_path = '/archive/hot5/fty/TALE/'
    
    # 加载模型初始化所需的特征与原型数据
    protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'), 
        weights_only=True, map_location='cpu'
    )
    test_protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'), 
        weights_only=True, map_location='cpu'
    )
    
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go_freq = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))
    
    # 实例化模型并完成原型特征初始化
    stacked_feats = torch.stack([protein_feats[k] for k in sorted(protein_feats.keys(), key=int)], dim=0).to(device)
    model = Model(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes, 
        prototype_index=prototype_index, proto_idx_mask=proto_idx_mask, go_freq=go_freq
    ).to(device)
    model.init_prototype_feats(num_classes, hidden_dim, stacked_feats)
    
    # 加载训练好的模型权重
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Successfully loaded checkpoint from: {args.checkpoint_path}")
    
    # 获取测试集的数据加载器 (这里 mode 传入 'test')
    _, test_loader = get_loader(
        datasets_path, namespace, args.batch_size, 
        protein_feats, test_protein_feats, num_classes, mode='test'
    )

    # =========================================================
    #                           评估
    # =========================================================
    print("Starting evaluation...")

    # 1. 收集所有预测概率和真实标签
    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_feats, labels, indices in test_loader:
            batch_feats = batch_feats.to(device)
            labels = labels.to(device)
            indices = indices.to(device)

            final_probs, _, _, _, _, _, _, _ = model(batch_feats, indices, labels)

            all_probs.append(final_probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()   # (N, num_classes)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N, num_classes)

    # 2. 计算每个蛋白质的 ALF (Average Label Frequency)
    go_freq_np = go_freq.cpu().numpy()  # (num_classes,)

    alfs = []
    for i in range(all_labels.shape[0]):
        label_indices = np.where(all_labels[i] > 0)[0]
        alf = utils.compute_protein_alf(label_indices.tolist(), go_freq_np)
        alfs.append(alf)
    alfs = np.array(alfs)

    # 3. 按 ALF 区间划分，计算各区间的 Fmax
    #    区间定义: [0, 0.2], (0.2, 0.3], (0.3, 0.4], (0.4, 1]
    #    为避免边界重叠，第一个区间闭区间，后续区间左开右闭
    bins = [
        (0.0, 0.2, "[0, 0.2]"),
        (0.2, 0.3, "(0.2, 0.3]"),
        (0.3, 0.4, "(0.3, 0.4]"),
        (0.4, 1.0, "(0.4, 1]"),
    ]

    print("\n===== ALF-based Fmax Evaluation =====")
    for low, high, bin_label in bins:
        if low == 0.0:
            mask = (alfs >= low) & (alfs <= high)
        else:
            mask = (alfs > low) & (alfs <= high)

        n_proteins = mask.sum()
        if n_proteins == 0:
            print(f"  ALF {bin_label}: No proteins in this bin")
            continue

        bin_probs = all_probs[mask]
        bin_labels = all_labels[mask]
        metrics = utils.calculate_metrics_new(bin_labels, bin_probs)
        print(f"  ALF {bin_label}: n={n_proteins}, "
              f"Fmax={metrics['Fmax']:.4f}, micro_AUPRC={metrics['micro_AUPRC']:.4f}")

    # 4. 整体评估
    metrics_all = utils.calculate_metrics_new(all_labels, all_probs)
    print(f"\n  Overall: Fmax={metrics_all['Fmax']:.4f}, "
          f"micro_AUPRC={metrics_all['micro_AUPRC']:.4f}")

    # =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CC Model')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/cc.yml', help='config yml')
    parser.add_argument('--path', type=str, default="./data_tale/TALE/", help='datasets path')
    parser.add_argument('--namespace', default='CC', type=str, help='[BP/CC/MF]')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--checkpoint_path', type=str, default='/archive/hot5/fty/checkpoints/TALE/checkpoint_49.pth', help='path to trained checkpoint (.pth)')

    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
        
    load_model_and_evaluate(args, config)