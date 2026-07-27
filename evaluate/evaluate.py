import argparse
import os
import torch
import numpy as np
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, SequentialSampler

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
    dataset_name = config['dataset']
    datasets_path = os.path.join(config['datasets_path'], dataset_name)

    # ---- 1. 加载基础数据（所有模型共用） ----
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))

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

    # ---- 3. 按 model_type 构建模型、加载权重 ----
    prototypes = None

    if model_type == 0:
        # ---- 纯 MLP 基线 ----
        from models.custom_model import Model
        model = Model(input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

        ckpt = torch.load(os.path.join(checkpoint_path, 'custom', config['checkpoint_custom']), map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()

    elif model_type == 1:
        # ---- 门控融合: MLP + PrototypeNet ----
        from models.model import Model
        from models.custom_model import Model as CustomModel
        from models.prototype_model import PrototypeNet

        # 原型网络所需的层次结构数据
        go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
        parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
        if not is_zero_shot: parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
        hop_counts = utils.get_ancestor_hop_matrix(_edges)
        if not is_zero_shot: hop_counts = hop_counts[valid_mask][:, valid_mask]
        obo_path = os.path.join(datasets_path, 'go-basic.obo')
        ic = utils.compute_ic(go2id, train_seq_data, obo_path)
        if not is_zero_shot: ic = ic[valid_mask]
        proto_w = utils.compute_ancestor_weights_ic(
            hop_counts, ic, class_counts, config.get('lambda', 30), config.get('beta', 2))
        alpha_ = config.get('alpha', 50)

        # 加载两个预训练子模型
        mlp_model = CustomModel(input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
        proto_model = PrototypeNet(
            input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
            frequency=go_freq, parents_matrix=parents_matrix,
            class_counts=class_counts, proto_w=proto_w, smooth_tau=alpha_,
        ).to(device)

        mlp_ckpt = torch.load(os.path.join(checkpoint_path, 'custom', config['checkpoint_custom']), map_location=device)
        proto_ckpt = torch.load(os.path.join(checkpoint_path, 'prototype', config['checkpoint_proto']), map_location=device)
        mlp_model.load_state_dict(mlp_ckpt['model'])
        proto_model.load_state_dict(proto_ckpt['model'])

        model = Model(mlp_model, proto_model, go_freq).to(device)

        # 预计算原型
        indices = sorted(protein_feats.keys(), key=int)
        all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
        prototypes = proto_model._get_prototypes(all_feats, prototype_index.to(device))

    elif model_type == 2:
        # ---- 纯 PrototypeNet ----
        from models.prototype_model import PrototypeNet

        go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
        parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
        if not is_zero_shot: parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
        hop_counts = utils.get_ancestor_hop_matrix(_edges)
        if not is_zero_shot: hop_counts = hop_counts[valid_mask][:, valid_mask]
        obo_path = os.path.join(datasets_path, 'go-basic.obo')
        ic = utils.compute_ic(go2id, train_seq_data, obo_path)
        if not is_zero_shot: ic = ic[valid_mask]
        proto_w = utils.compute_ancestor_weights_ic(
            hop_counts, ic, class_counts, config.get('lambda', 30), config.get('beta', 2))
        alpha_ = config.get('alpha', 50)

        model = PrototypeNet(
            input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
            frequency=go_freq, parents_matrix=parents_matrix,
            class_counts=class_counts, proto_w=proto_w, smooth_tau=alpha_,
        ).to(device)

        ckpt = torch.load(os.path.join(checkpoint_path, 'prototype', config['checkpoint_proto']), map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        # 预计算原型
        indices = sorted(protein_feats.keys(), key=int)
        all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
        prototypes = model._get_prototypes(all_feats, prototype_index.to(device))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Successfully loaded model_type={model_type}")

    # =========================================================
    #                           评估
    # =========================================================
    utils.eval_func_generalizability(model, test_loader, device, go_freq, prototypes, model_type)
    # utils.eval_term_freq_generalizability(model, test_loader, device, go_freq, prototypes)
    # =========================================================
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CC Model')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/eval.yml', help='config yml')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')

    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
        
    main(args, config)