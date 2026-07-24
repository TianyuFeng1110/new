import argparse
import os
import utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from models.prototype_model import PrototypeNet
from datasets.balanced_dataset import Dataset  # 保持索引对齐（不过滤空标签）
from datasets.prototype_valid_collator import collator
from datasets.prototype_sampler import PrototypeSampler
from datasets.prototype_collator import PrototypeCollator
from torch.utils.data import DataLoader, SequentialSampler

def train(model, optimizer, loader, epoch, device):
    """原型网络训练：每 batch 采样 n_way 个 GO term，support 计算原型，query 计算距离 → BCE loss。"""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('temperature', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    for inputs_feats, labels, support_feats, data2classes, support_indices in metric_logger.log_every(loader, print_freq=1, header=header):
        optimizer.zero_grad(set_to_none=True)

        logits, loss, temp = model(query=inputs_feats.to(device), labels=labels.to(device), support=support_feats.to(device), support_indices=support_indices.to(device))
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(temperature=temp.item())

    print("Averaged stats: {}".format(metric_logger.global_avg()))


@torch.no_grad()
def valid(model, prototypes, valid_loader, epoch, device):
    """验证：用全部训练蛋白质计算的原型，对测试集蛋白质预测全类 logits。"""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('temperature', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_labels = [], []

    for inputs_feats, labels, _ in metric_logger.log_every(valid_loader, print_freq=1, header=header):

        logits, loss, temp = model(query=inputs_feats.to(device), labels=labels.to(device), prototypes=prototypes.to(device))

        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        metric_logger.update(loss=loss.item())
        metric_logger.update(temperature=temp.item())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    metric = utils.calculate_metrics(all_labels, all_probs)
    macro_aupr = utils.macro_auprc(all_labels, all_probs)
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}".format(
        metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC']))
    # print("Averaged stats: {}, macro aupr: {:.4f}".format(metric_logger.global_avg(), macro_aupr))

    return all_probs, all_labels


def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    epochs = args.epochs
    dataset_name = config['dataset']
    datasets_path = os.path.join(config['datasets_path'], dataset_name)
    namespace = config['namespace']
    mode = config['mode']
    base_lr = float(config['base_lr'])
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    num_classes = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_matrix_1_sparse.npy')).shape[0]
    n_way = config.get('n_way', 32)
    n_query = config.get('n_query', 4)
    n_support = config.get('n_support', 50)
    lambda_ = config.get('lambda')
    alpha_ = config.get('alpha')
    beta_ = config.get('beta')
    print('数据集:', dataset_name, '超参数alpha:', alpha_, 'lambda:', lambda_, 'beta:', beta_)
    features_path = os.path.join('/archive/hot5/fty/', dataset_name)
    protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1 # 确保所有数据都注释了根go term的功能
    valid_mask = prototype_index.sum(dim=0) > 0  # 形状为 (标签类别数,) 的布尔 Tensor
    prototype_index = prototype_index[:, valid_mask] # 过滤没有正样本的标签
    pos_indices = prototype_index.T  # (num_classes, num_proteins)
    num_classes = pos_indices.shape[0]  # 更新 num_classes 为有效标签数
    indices = sorted(protein_feats.keys(), key=int)
    all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
    _, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go_freq, class_counts = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))
    go_freq, class_counts = go_freq[valid_mask], class_counts[valid_mask]
    adj_matrix = utils.load_npy_file(os.path.join(datasets_path, f"{namespace.lower()}_adj_matrix.npy"))
    adj_matrix = adj_matrix[valid_mask][:, valid_mask]
    _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
    hop_counts = utils.get_ancestor_hop_matrix(_edges) # 每个节点到任意祖先节点的跳数
    hop_counts = hop_counts[valid_mask][:, valid_mask]
    # proto_w = utils.compute_ancestor_weights(hop_counts, class_counts, lambda_)
    if dataset_name == 'TALE':
        # TALE 数据集：通过 go2id pickle 构建 parents_matrix，并用 goatools 计算 IC
        go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
        parents_matrix, childs_matrix = utils.get_go_adjacency_matrices(go2id)
        parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        obo_path = os.path.join(datasets_path, 'go-basic.obo')
        ic = utils.compute_ic(go2id, train_seq_data, obo_path)[valid_mask]
    else:
        # CAFA3 数据集：缺少可靠的 *_go_1.pickle，直接从 label_regular_1.npy 和 train_seq_data 计算
        # 1) parents_matrix：label_regular_1.npy 中每行为 [parent, child]，构建 child->parent 邻接矩阵
        num_nodes = int(_edges.max()) + 1
        parents_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for parent, child in _edges:
            parents_matrix[int(child)][int(parent)] = 1
        parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        # 2) ic：train_seq_data 的 label 字段已沿 DAG 传播祖先标注，
        #    故直接统计每个类别索引的蛋白质数，除以根节点（最大计数）得到频率，
        #    IC = -log(freq)。与 goatools TermCounts 结果一致（已在 TALE 上验证）。
        from collections import Counter
        import math
        label_counts = Counter()
        for item in train_seq_data:
            for idx in item.get('label', []):
                label_counts[idx] += 1
        total_count = max(label_counts.values()) if label_counts else 1  # 根节点计数
        ic = np.zeros(num_nodes, dtype=np.float64)
        for idx in range(num_nodes):
            cnt = label_counts.get(idx, 0)
            freq = cnt / total_count if total_count > 0 else 0.0
            ic[idx] = -math.log(freq) if freq > 0 else 0.0
        ic = torch.from_numpy(ic).float()[valid_mask]
    proto_w = utils.compute_ancestor_weights_ic(hop_counts, ic, class_counts, lambda_, beta_)

    g = utils.set_random_seed(seed)

    # ---- 训练 DataLoader：原型网络采样 ----
    # 注意：train_mode='test' 避免 shuffle 导致索引错位
    raw_dataset = Dataset(datasets_path, namespace, train_mode='test', dataset_mode='train', valid_mask=valid_mask)
    _pos_pools = [torch.nonzero(pos_indices[c]).squeeze(1).tolist() for c in range(pos_indices.shape[0])]
    support, query = utils.get_support_query_indices(_pos_pools)
    train_sampler = PrototypeSampler(query, n_way=n_way, n_query=n_query)
    train_collator = PrototypeCollator(
        protein_feats=protein_feats, n_way=n_way, support=support, sampler=train_sampler,
        n_query=n_query, n_support=n_support, num_classes=num_classes)
    train_loader = DataLoader(
        raw_dataset, batch_sampler=train_sampler, collate_fn=train_collator, num_workers=0,
        generator=g, worker_init_fn=utils.seed_worker)

    # ---- 验证 DataLoader：全类预测 ----
    valid_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test', valid_mask=valid_mask)
    test_collator = collator(num_classes, test_protein_feats)
    valid_loader = DataLoader(
        valid_dataset, batch_size=n_query*n_way, shuffle=False,
        sampler=SequentialSampler(valid_dataset), drop_last=False, num_workers=0,
        collate_fn=test_collator,
        generator=g, worker_init_fn=utils.seed_worker)

    # ---- 模型 ----
    model = PrototypeNet(input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes, frequency=go_freq, parents_matrix=parents_matrix, class_counts=class_counts, proto_w=proto_w, smooth_tau=alpha_).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

    print('start training......')

    for epoch in range(epochs):
        # 训练
        train(model, optimizer, train_loader, epoch, device)
        # scheduler.step()

        # 验证
        prototypes = model._get_prototypes(all_feats, prototype_index.to(device))
        all_probs, all_labels = valid(model, prototypes, valid_loader, epoch, device)

        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        os.makedirs(os.path.join("/archive/hot5/fty/checkpoints/", dataset_name, "prototype"), exist_ok=True)
        torch.save(save_obj, os.path.join("/archive/hot5/fty/checkpoints/", dataset_name, "prototype", 'checkpoint_%02d.pth' % epoch))
        utils.eval_func_generalizability(model, valid_loader, device, go_freq, prototypes, model_type=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prototypical Network for Protein Function Prediction')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--config', type=str, default='./config/prototype.yml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    main(args, config)