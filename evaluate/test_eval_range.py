import argparse
import os
import sys
import torch
import numpy as np
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, SequentialSampler

# 将项目根目录加入搜索路径，确保从任意位置运行都能找到 utils / datasets / models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils as utils
from datasets.dataset import Dataset
from datasets.collator import collator


def get_test_loader(datasets_path, namespace, batch_size, test_protein_feats, num_classes, valid_mask):
    """构建测试集 DataLoader。"""
    dataset = Dataset(datasets_path, namespace, train_mode='test', dataset_mode='test', valid_mask=valid_mask)
    sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, drop_last=False, num_workers=2,
        collate_fn=collator(num_classes, test_protein_feats),
        worker_init_fn=utils.seed_worker,
    )


def main(args, config):
    device = torch.device(args.device)

    namespace = config['namespace']
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    model_type = config['model_type']
    dataset_name = config['dataset']
    is_zero_shot = config['zero_shot']
    features_path = os.path.join(config['features_path'], dataset_name)
    checkpoint_path = os.path.join(config['checkpoint_path'], namespace, dataset_name)
    datasets_path = os.path.join(config['datasets_path'], dataset_name)

    # ---- 1. 加载基础数据（所有模型共用） ----
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    test_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"test_seq_{namespace.lower()}"))

    # 构建 (N_protein × num_classes) 的二值标注矩阵并过滤无效类
    num_classes_raw = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_matrix_1_sparse.npy')).shape[0]
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes_raw)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1          # 确保根 GO term 至少有一条标注
    valid_mask = prototype_index.sum(dim=0) > 0
    if not is_zero_shot: prototype_index = prototype_index[:, valid_mask]             # (N_protein, num_classes)
    num_classes = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_matrix_1_sparse.npy')).shape[0]

    # 训练/测试蛋白质特征
    protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')

    # GO term 频率 & 样本计数
    go_freq, class_counts = utils.compute_go_term_frequency(prototype_index.T, len(train_seq_data))

    # ---- 2. 测试 DataLoader ----
    test_loader = get_test_loader(datasets_path, namespace, args.batch_size,
                                  test_protein_feats, num_classes, (None if is_zero_shot else valid_mask))

    # 预计算原型所需的训练集特征（model_type=1/2 使用）
    indices = sorted(protein_feats.keys(), key=int)
    all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)

    # msi_cache_path = os.path.join(datasets_path, f'msi_values_{namespace.lower()}.npy')

    epochs = range(args.epoch_start, args.epoch_end + 1)
    print(f"Will evaluate checkpoints for epochs {args.epoch_start} .. {args.epoch_end} "
          f"(model_type={model_type})")

    if model_type == 2:
        # ---- 纯 PrototypeNet：proto checkpoint 逐 epoch 评估 ----
        from models.test_model import PrototypeNet

        go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
        parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
        if not is_zero_shot: parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
        hop_counts = utils.get_ancestor_hop_matrix(_edges)
        if not is_zero_shot: hop_counts = hop_counts[valid_mask][:, valid_mask]
        obo_path = os.path.join(datasets_path, 'go-basic.obo')
        ic = utils.compute_ic(go2id, train_seq_data, obo_path)
        if not is_zero_shot: ic = ic[valid_mask]
        # lambda/beta/smooth_tau 为模型内可学习参数，checkpoint 中已含学习后的值；
        # 此处传入的数据驱动初始值仅决定加载前的结构，不影响评估结果
        lambda_init, beta_init, tau_init = utils.compute_smooth_param_inits(hop_counts, ic, class_counts)
        tau_min = float(config.get('tau_min', 0.0))

        # 模型只构建一次并复用：
        # PrototypeNet 的 parents/hop buffer 是 (num_classes, num_classes) 的大张量（BP 下每个 ~1.6GB），
        # 若每个 epoch 都新建模型会反复在 GPU 上分配并跨 epoch 累积/碎片化，导致 OOM。
        model = PrototypeNet(
            input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
            frequency=go_freq, parents_matrix=parents_matrix,
            class_counts=class_counts, hop_counts=hop_counts, ic=ic,
            smooth_tau=tau_init, lambda_init=lambda_init, beta_init=beta_init, tau_min=tau_min,
        ).to(device)

        prototype_index_gpu = prototype_index.to(device)

        for epoch in epochs:
            ckpt_path = os.path.join(checkpoint_path, 'prototype', f'checkpoint_{epoch}.pth')
            if not os.path.exists(ckpt_path):
                print(f"[skip] checkpoint not found: {ckpt_path}")
                continue
            # checkpoint 在 CPU 上加载，再原地拷贝进复用模型的参数/buffer，
            # 避免 GPU 上同时存在多份大 buffer（省去整份 state_dict 的 GPU 副本）
            ckpt = torch.load(ckpt_path, map_location='cpu')
            proto_state = utils.extract_state_dict(ckpt)
            if 'tau_min' not in proto_state:  # 兼容旧 checkpoint（无 tau_min buffer）
                proto_state['tau_min'] = torch.tensor(0.0)
            model.load_state_dict(proto_state)
            del ckpt, proto_state
            model.eval()

            # 预计算原型（依赖当前模型权重，逐 epoch 重算）
            prototypes = model._get_prototypes(all_feats, prototype_index_gpu)

            print(f"\n===== Evaluating epoch {epoch} (prototype) =====")
            utils.eval_func_generalizability(model, test_loader, device, go_freq, prototypes, model_type)
            # utils.eval_msi_generalizability(model, test_loader, device, test_seq_data, prototypes, model_type, cache_path=msi_cache_path)

            # 释放本 epoch 的临时张量并回收缓存，避免跨 epoch 显存累积/碎片化
            del prototypes
            torch.cuda.empty_cache()

    else:
        raise ValueError(f"Only model_type=2 (PrototypeNet) is supported in this script, got model_type={model_type}")

    print("\nDone. All epochs evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CC Model (checkpoint range)')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/eval.yml', help='config yml')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--epoch_start', type=int, default=99, help='first epoch to evaluate (inclusive)')
    parser.add_argument('--epoch_end', type=int, default=120, help='last epoch to evaluate (inclusive)')

    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    main(args, config)
