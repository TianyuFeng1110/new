import argparse
import os
import utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from models.test1_model import Model
from models.custom_model import Model as CustomModel
from models.test_model import PrototypeNet
from datasets.cc_dataset import Dataset
from datasets.cc_collator import collator
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import average_precision_score


@torch.no_grad()
def precompute_quantiles(mlp_model, proto_model, all_feats, prototypes, prototype_index, num_quantiles=100):
    """在全训练集上预计算两个子模型的 per-class 百分位边界表。

    两个子模型冻结，分布固定，只需计算一次。
    """
    # MLP 概率: (N, C)
    _, custom_logits, _ = mlp_model(all_feats, None, prototype_index.to(all_feats.device))
    mlp_probs = torch.sigmoid(custom_logits)

    # 原型网络概率: (N, C)
    q_feats = proto_model.mlp(all_feats)
    proto_logits = proto_model.predict(q_feats, prototypes)
    proto_probs = torch.sigmoid(proto_logits)

    grid = torch.linspace(0.0, 1.0, num_quantiles + 1, device=mlp_probs.device)
    mlp_quantiles   = torch.quantile(mlp_probs,   grid, dim=0).T   # (C, Q+1)
    proto_quantiles = torch.quantile(proto_probs, grid, dim=0).T   # (C, Q+1)

    return mlp_quantiles, proto_quantiles


@torch.no_grad()
def diagnose_gate_behavior(model, loader, device, go_freq, prototypes):
    """诊断门控梯度行为：对每个频率桶，报告 sigma 趋势和模块分歧方向。

    关键诊断指标：
    - d_pos: 正样本的 (mlp_n - proto_n)，负值表示 proto 排更高
    - d_neg: 负样本的 (mlp_n - proto_n)，正值表示 proto 排更低
    - 若 proto 排序更好：d_pos<0 且 d_neg>0，但两者梯度同号 → sigma↑（倒向MLP）
    """
    model.eval()

    custom_probs_list, proto_probs_list = [], []
    all_labels_list = []

    with torch.no_grad():
        for batch_feats, labels, _ in loader:
            batch_feats = batch_feats.to(device)
            # MLP
            _, custom_logits, _ = model.mlp_model(batch_feats, None, labels.to(device))
            custom_probs_list.append(torch.sigmoid(custom_logits).cpu())
            # Proto
            q_feats = model.proto_model.mlp(batch_feats)
            proto_logits = model.proto_model.predict(q_feats, prototypes)
            proto_probs_list.append(torch.sigmoid(proto_logits).cpu())
            all_labels_list.append(labels.cpu())

    custom_probs = torch.cat(custom_probs_list, dim=0).to(device)  # (N, C)
    proto_probs  = torch.cat(proto_probs_list,  dim=0).to(device)  # (N, C)
    all_labels   = torch.cat(all_labels_list,   dim=0).to(device)  # (N, C)

    # --- 百分位归一化 ---
    custom_n = model._quantile_normalize(custom_probs, model.mlp_quantiles)
    proto_n  = model._quantile_normalize(proto_probs,  model.proto_quantiles)

    # --- 计算 sigma ---
    gate_k = torch.nn.functional.softplus(model.raw_gate_k)
    sigma = torch.sigmoid(gate_k * (model.log_freq - model.gate_c))  # (C,)

    # --- per-class AUPRC ---
    labels_np = all_labels.cpu().numpy()
    mlp_probs_np = custom_probs.cpu().numpy()
    proto_probs_np = proto_probs.cpu().numpy()
    num_classes = labels_np.shape[1]

    mlp_auprcs = np.zeros(num_classes)
    proto_auprcs = np.zeros(num_classes)
    for c in range(num_classes):
        if labels_np[:, c].sum() == 0:
            mlp_auprcs[c] = 0.0
            proto_auprcs[c] = 0.0
        else:
            mlp_auprcs[c] = average_precision_score(labels_np[:, c], mlp_probs_np[:, c])
            proto_auprcs[c] = average_precision_score(labels_np[:, c], proto_probs_np[:, c])

    proto_auprcs = torch.from_numpy(proto_auprcs)
    mlp_auprcs = torch.from_numpy(mlp_auprcs)

    # --- 分类: proto-better / mlp-better (margin > 0.02 才算) ---
    margin = 0.02
    proto_better = (proto_auprcs - mlp_auprcs) > margin
    mlp_better   = (mlp_auprcs - proto_auprcs) > margin
    tie = ~(proto_better | mlp_better)

    # --- 百分位分歧: d = mlp_n - proto_n ---
    d = custom_n - proto_n  # (N, C)

    # per-class 的正/负样本分歧
    pos_mask = all_labels.bool()
    neg_mask = ~pos_mask

    d_pos = torch.zeros(num_classes)
    d_neg = torch.zeros(num_classes)
    for c in range(num_classes):
        if pos_mask[:, c].sum() > 0:
            d_pos[c] = d[pos_mask[:, c], c].mean()
        if neg_mask[:, c].sum() > 0:
            d_neg[c] = d[neg_mask[:, c], c].mean()

    # --- 频率分桶 ---
    bins = [
        (0.0, 0.0002, "U-Low"),
        (0.0002, 0.001, "Low"),
        (0.001, 0.002, "Mid"),
        (0.002, 0.02, "High"),
        (0.02, 1.0, "U-High"),
    ]

    print("\n===== Gate Behavior Diagnostics =====")
    print(f"{'Bin':>7s} | {'#cls':>5s} | {'sigma':>7s} | "
          f"{'proto>mlp':>10s} | {'d_pos':>7s} | {'d_neg':>7s} | {'grad→':>6s}")
    print("-" * 72)

    for low, high, label in bins:
        mask = (go_freq >= low) & (go_freq < high)
        if mask.sum() == 0:
            continue

        # 该桶中 proto 更好的类
        pb = proto_better & mask
        mb = mlp_better & mask
        tb = tie & mask

        def _report(sub_mask, tag):
            if sub_mask.sum() == 0:
                return
            n = sub_mask.sum().item()
            s = sigma[sub_mask].mean().item()
            dp = d_pos[sub_mask].mean().item()
            dn = d_neg[sub_mask].mean().item()
            # 梯度符号: pos: (f-1)*d, neg: f*d. 近似看 d_neg 主导
            grad_dir = "MLP↑" if dn > 0.01 else ("PROTO↑" if dn < -0.01 else "~")
            print(f"{label:>7s} | {n:>5d} | {s:>7.4f} | "
                  f"{tag:>10s} | {dp:>7.4f} | {dn:>7.4f} | {grad_dir:>6s}")

        _report(pb, "proto>mlp")
        _report(mb, "mlp>proto")
        _report(tb, "tie")

    print("=" * 72)
    print("  d_pos<0 & d_neg>0 → proto 排序好，但梯度推 σ→MLP（结构陷阱）")
    print("  d_pos>0 & d_neg<0 → mlp 排序好，梯度推 σ→PROTO（反直觉）")
    return sigma


def get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode, valid_mask):
    dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='train', valid_mask=valid_mask)

    if mode == 'train':
        valid_dataset = dataset.val_dataset
    else:
        valid_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test', valid_mask=valid_mask)

    train_sampler = RandomSampler(dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=train_sampler, drop_last=True, num_workers=2,
        collate_fn=collator(num_classes, protein_feats),
        worker_init_fn=utils.seed_worker
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        sampler=valid_sampler, drop_last=False, num_workers=2,
        collate_fn=collator(num_classes, test_protein_feats),
        worker_init_fn=utils.seed_worker
    )
    return train_loader, valid_loader


@torch.no_grad()
def valid(model, prototypes, loader, epoch, device, go_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('max_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('min_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('mean_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('k', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('c', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_labels = [], []
    for batch_feats, labels, indices in metric_logger.log_every(loader, print_freq=1, header=header):
        final_probs, loss, sigma, k, c = model(
            batch_feats.to(device), prototypes, labels.to(device))
        all_probs.append(final_probs.cpu())
        all_labels.append(labels.cpu())
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(max_sigma=sigma.max().item())
        metric_logger.update(min_sigma=sigma.min().item())
        metric_logger.update(mean_sigma=sigma.mean().item())
        metric_logger.update(k=k.item())
        metric_logger.update(c=c.item())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    metric = utils.calculate_metrics(all_labels, all_probs)
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}".format(
        metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC']))
    

    # 样本量分别为 0-10,10-50,50-100，100-1000，1000-all
    bins = [
        (0.0, 0.0002, "[0, 0.0002]"),
        (0.0002, 0.001, "(0.0002, 0.001]"),
        (0.001, 0.002, "(0.001, 0.002]"),
        (0.002, 0.02, "(0.002, 0.02]"),
        (0.02, 1.0, "(0.02, 1]"),
    ]
    # 按频率分桶统计 sigma
    ultra_low_freq_mask = go_freq < bins[0][1]   # 根据实际数据调整阈值
    low_freq_mask = (go_freq >= bins[1][0]) & (go_freq < bins[1][1])
    mid_freq_mask = (go_freq >= bins[2][0]) & (go_freq < bins[2][1])
    high_freq_mask = (go_freq >= bins[3][0]) & (go_freq < bins[3][1])
    ultra_high_freq_mask = go_freq >= bins[4][0]

    print(f"  Ultra-low-freq  sigma: {sigma[ultra_low_freq_mask].mean():.4f}")
    print(f"  Low-freq  sigma: {sigma[low_freq_mask].mean():.4f}")
    print(f"  Mid-freq  sigma: {sigma[mid_freq_mask].mean():.4f}")
    print(f"  High-freq sigma: {sigma[high_freq_mask].mean():.4f}")
    print(f"  Ultra-high-freq  sigma: {sigma[ultra_high_freq_mask].mean():.4f}")

    return sigma


def train(model, prototypes, optimizer, loader, epoch, device):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('max_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('min_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('mean_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('k', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('c', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    for batch_feats, labels, indices in metric_logger.log_every(loader, print_freq=1, header=header):
        optimizer.zero_grad(set_to_none=True)

        final_probs, loss, sigma, k, c = model(
            batch_feats.to(device), prototypes, labels.to(device))
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(max_sigma=sigma.max().item())
        metric_logger.update(min_sigma=sigma.min().item())
        metric_logger.update(mean_sigma=sigma.mean().item())
        metric_logger.update(k=k.item())
        metric_logger.update(c=c.item())

    print("Averaged stats: {}".format(metric_logger.global_avg()))


def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    epochs = args.epochs
    datasets_path = args.path
    batch_size = args.batch_size
    namespace = args.namespace
    mode = args.mode
    base_lr = float(config['base_lr'])
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

    protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'),
                               weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'),
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

    train_loader, valid_loader = get_loader(
        datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode, valid_mask)

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

    # ---- 预计算原型（原型网络冻结，原型不变） ----
    indices = sorted(protein_feats.keys(), key=int)
    all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
    prototypes = proto_model._get_prototypes(all_feats, prototype_index.to(device))

    # ---- 门控融合模型 ----
    lambda_sigma = config.get('lambda_sigma', 0.1)
    model = Model(mlp_model, proto_model, go_freq, lambda_sigma=lambda_sigma).to(device)

    # ---- 预计算逐类百分位归一化表（两子模型冻结，仅需一次） ----
    mlp_quantiles, proto_quantiles = precompute_quantiles(
        mlp_model, proto_model, all_feats, prototypes, prototype_index)
    model.set_quantiles(mlp_quantiles, proto_quantiles)

    # 仅训练门控参数
    gate_params = [model.raw_gate_k, model.gate_c]
    optimizer = optim.AdamW(gate_params, lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

    print('start training......')
    for epoch in range(epochs):
        train(model, prototypes, optimizer, train_loader, epoch, device)
        scheduler.step()
        sigma = valid(model, prototypes, valid_loader, epoch, device, go_freq)

        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join("/archive/hot5/fty/checkpoints/TALE/gate", 'checkpoint_%02d.pth' % epoch))

        utils.eval_func_generalizability(model, valid_loader, device, go_freq, prototypes)
        # diagnose_gate_behavior(model, valid_loader, device, go_freq, prototypes)
        # utils.eval_term_freq_generalizability(model, valid_loader, device, go_freq, prototypes)


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