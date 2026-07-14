import argparse
import os
import utils as utils
import torch
import numpy as np

from ruamel.yaml import YAML
from models.model import Model
from models.custom_model import Model as CustomModel
from models.test_model import PrototypeNet
from datasets.cc_dataset import Dataset
from datasets.cc_collator import collator
from torch.utils.data import DataLoader, SequentialSampler


def get_loader(datasets_path, namespace, batch_size, test_protein_feats, num_classes, mode, valid_mask):
    dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test', valid_mask=valid_mask)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, drop_last=False, num_workers=2,
        collate_fn=collator(num_classes, test_protein_feats),
        worker_init_fn=utils.seed_worker
    )
    return loader


@torch.no_grad()
def evaluate(model, prototypes, loader, device, go_freq):
    model.eval()

    all_probs, all_labels = [], []
    for batch_feats, labels, _indices in loader:
        final_probs, loss, sigma, _, _ = model(
            batch_feats.to(device), prototypes, labels.to(device))
        all_probs.append(final_probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    metric = utils.calculate_metrics(all_labels, all_probs)
    print(f"Fmax: {metric['Fmax']:.4f}, micro AUPRC: {metric['micro_AUPRC']:.4f}")

    return metric


def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    datasets_path = args.path
    batch_size = args.batch_size
    namespace = args.namespace
    mode = args.mode
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    num_classes = config['num_classes']
    features_path = '/archive/hot5/fty/TALE/'

    # ---- 数据加载 ----
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1
    valid_mask = prototype_index.sum(dim=0) > 0
    prototype_index = prototype_index[:, valid_mask]
    num_classes = valid_mask.sum().item()

    test_protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')
    protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')

    _, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go_freq, class_counts = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))
    go_freq, class_counts = go_freq[valid_mask], class_counts[valid_mask]

    # 原型网络所需的层次结构相关矩阵
    go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
    parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
    parents_matrix = parents_matrix[valid_mask][:, valid_mask]
    _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
    hop_counts = utils.get_ancestor_hop_matrix(_edges)
    hop_counts = hop_counts[valid_mask][:, valid_mask]
    obo_path = os.path.join(datasets_path, 'go-basic.obo')
    ic = utils.compute_ic(go2id, train_seq_data, obo_path)[valid_mask]
    lambda_ = config.get('lambda', 30)
    beta_ = config.get('beta', 2)
    proto_w = utils.compute_ancestor_weights_ic(hop_counts, ic, class_counts, lambda_, beta_)
    alpha_ = config.get('alpha', 50)

    utils.set_random_seed(seed)

    loader = get_loader(
        datasets_path, namespace, batch_size, test_protein_feats, num_classes, mode, valid_mask)

    # ---- 构建两个预训练子模型并加载检查点 ----
    mlp_model = CustomModel(input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    proto_model = PrototypeNet(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
        frequency=go_freq, parents_matrix=parents_matrix, class_counts=class_counts,
        proto_w=proto_w, smooth_tau=alpha_).to(device)

    mlp_ckpt = torch.load("/archive/hot5/fty/checkpoints/TALE/custom/checkpoint_24.pth", map_location=device)
    mlp_model.load_state_dict(mlp_ckpt['model'])
    proto_ckpt = torch.load("/archive/hot5/fty/checkpoints/TALE/prototype/checkpoint_75.pth", map_location=device)
    proto_model.load_state_dict(proto_ckpt['model'])

    mlp_model.eval()
    proto_model.eval()

    # ---- 预计算原型 ----
    indices = sorted(protein_feats.keys(), key=int)
    all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
    prototypes = proto_model._get_prototypes(all_feats, prototype_index.to(device))

    # ---- 门控融合模型（sigma 在 __init__ 中由 go_freq 纯数学计算） ----
    model = Model(mlp_model, proto_model, go_freq).to(device)

    print('Evaluating with frequency-derived sigma...')
    evaluate(model, prototypes, loader, device, go_freq)
    utils.eval_func_generalizability(model, loader, device, go_freq, prototypes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gate fusion: MLP + PrototypeNet')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/cc.yml', help='config yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--path', type=str, default="./data_tale/TALE/", help='datasets path')
    parser.add_argument('--mode', type=str, default='test', help='[train/test]')
    parser.add_argument('--namespace', default='CC', type=str, help='[BP/CC/MF]')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    main(args, config)