"""效率基准测试：Params / FLOPs per sample / 训练时间与峰值显存。

对 TALE 与 CAFA3 两个数据集、MF/CC/BP 三个 namespace，分别按各自训练采样方式
实测 MLP（经典随机采样）与原型网络（情景采样）两个分支：

  - Params (M):        分支参数量
  - FLOPs/sample (M):  单蛋白推理前向计算量（与 batch/硬件无关）
  - 训练 GPU·h:        实测 N 步均耗时 × 每epoch步数 × epochs 折算
  - 峰值显存 (GB):     torch.cuda.max_memory_allocated（含训练所需激活/梯度）
  - proteins/step:     每步实际流过编码器的蛋白数（两种采样方式的等效换算单位）

显存与时间对比使用两种管线各自的"标准训练步"（MLP: bs=1024 随机采样；
Proto: n_way=32 × (n_query=10+n_support=50) 情景采样），并报告 proteins/step
以便换算为"每万蛋白的秒数"这一公平单位。

用法:
  python benchmark_cost.py --device cuda:0
  python benchmark_cost.py --device cuda:0 --datasets TALE --namespaces CC
  python benchmark_cost.py --device cuda:0 --n_steps 50 --mlp_epochs 50 --proto_epochs 150
"""

import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils as utils
from models.custom_model import Model as MLPModel
from models.test_model import PrototypeNet
from datasets.dataset import Dataset
from datasets.balanced_dataset import Dataset as BalancedDataset
from datasets.collator import collator
from datasets.prototype_valid_collator import collator as proto_valid_collator
from datasets.prototype_sampler import PrototypeSampler
from datasets.prototype_collator import PrototypeCollator

DATASETS = ['TALE', 'CAFA3']
NAMESPACES = ['MF', 'CC', 'BP']
CONFIG_MAP = {'MLP': './config/custom.yml', 'Proto': './config/prototype.yml'}


# ============================================================
#  Params
# ============================================================

def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ============================================================
#  FLOPs per sample（推理：单蛋白一次前向）
# ============================================================

def measure_flops_per_sample(model, branch, esm_dim, num_classes, hidden_dim, device):
    """单蛋白一次前向的 FLOPs。

    MLP 分支:  encoder + classifier 复杂度 O(esm*hidden + hidden^2 + hidden*C)
    Proto 分支: encoder 复杂度 O(esm*hidden + hidden^2) + predict 复杂度 O(hidden*C)
    均可用 thop 直接 profile；thop 缺失时用解析式估算（数量级一致）。
    """
    try:
        from thop import profile
        if branch == 'MLP':
            dummy = torch.randn(1, esm_dim).to(device)
            model.eval()
            flops, _ = profile(model, inputs=(dummy,
                                              torch.zeros(1, dtype=torch.long).to(device),
                                              torch.zeros(1, num_classes).to(device)),
                               verbose=False)
        else:
            dummy = torch.randn(1, esm_dim).to(device)
            # _get_prototypes 返回 (num_classes, hidden_dim)，predict 中做 q @ p.T
            protos = torch.randn(num_classes, hidden_dim).to(device)
            model.eval()
            flops, _ = profile(model, inputs=(dummy,
                                              torch.zeros(1, num_classes).to(device),
                                              None, None, protos),
                               verbose=False)
        return flops / 1e6
    except ImportError:
        # 解析估算: encoder(2层Linear+GELU) + head
        enc = 2 * (esm_dim * hidden_dim + hidden_dim * hidden_dim)
        head = hidden_dim * num_classes * (2 if branch == 'MLP' else 1)
        return (enc + head) / 1e6


# ============================================================
#  DataLoader 构建（完全复刻各自训练脚本的采样方式）
# ============================================================

def build_mlp_train_loader(datasets_path, namespace, protein_feats, num_classes,
                           valid_mask, is_zero_shot, bs, g):
    """经典随机采样，复刻 custom.py get_loader()。"""
    dataset = Dataset(datasets_path, namespace, train_mode='train',
                      dataset_mode='train', valid_mask=valid_mask)
    return DataLoader(dataset, batch_size=bs, shuffle=False,
                      sampler=RandomSampler(dataset, generator=g), drop_last=True,
                      num_workers=0, collate_fn=collator(num_classes, protein_feats),
                      generator=g, worker_init_fn=utils.seed_worker)


def build_proto_train_loader(datasets_path, namespace, protein_feats, prototype_index,
                             num_classes, is_zero_shot, valid_mask,
                             n_way, n_query, n_support, g):
    """情景采样，复刻 prototype.py 的 PrototypeSampler + PrototypeCollator。"""
    raw_dataset = BalancedDataset(datasets_path, namespace, train_mode='test',
                                  dataset_mode='train',
                                  valid_mask=(None if is_zero_shot else valid_mask))
    pos_indices = prototype_index.T
    _pos_pools = [torch.nonzero(pos_indices[c]).squeeze(1).tolist()
                  for c in range(pos_indices.shape[0])]
    support, query = utils.get_support_query_indices(_pos_pools)
    sampler = PrototypeSampler(query, n_way=n_way, n_query=n_query)
    collate = PrototypeCollator(protein_feats=protein_feats, n_way=n_way,
                                support=support, sampler=sampler,
                                n_query=n_query, n_support=n_support,
                                num_classes=num_classes)
    return DataLoader(raw_dataset, batch_sampler=sampler, collate_fn=collate,
                      num_workers=0, generator=g, worker_init_fn=utils.seed_worker)


# ============================================================
#  训练步计时 + 峰值显存（严格走各自真实 forward + backward + step）
# ============================================================

def bench_mlp_train_step(model, loader, optimizer, device, n_steps):
    model.train()
    it = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    done = 0
    proteins = 0
    for i in range(n_steps):
        try:
            feats, labels, indices = next(it)
        except StopIteration:
            it = iter(loader)
            feats, labels, indices = next(it)
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = model(feats.to(device), indices.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
        proteins += feats.shape[0]
        done += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / max(done, 1), proteins / max(done, 1), torch.cuda.max_memory_allocated()


def bench_proto_train_step(model, loader, optimizer, device, n_steps):
    model.train()
    it = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    done = 0
    proteins = 0
    for i in range(n_steps):
        try:
            q_feats, labels, s_feats, data2classes, s_idx = next(it)
        except StopIteration:
            it = iter(loader)
            q_feats, labels, s_feats, data2classes, s_idx = next(it)
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(query=q_feats.to(device), labels=labels.to(device),
                           support=s_feats.to(device), support_indices=s_idx.to(device))
        loss.backward()
        optimizer.step()
        proteins += q_feats.shape[0] + s_feats.shape[0]
        done += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / max(done, 1), proteins / max(done, 1), torch.cuda.max_memory_allocated()


# ============================================================
#  单个 (dataset, namespace, branch) 的完整基准
# ============================================================

def load_cfg(path):
    yaml = YAML(typ='safe')
    with open(path) as f:
        return yaml.load(f)


def run_branch_benchmark(dataset_name, namespace, branch, args, device, rows):
    cfg = load_cfg(CONFIG_MAP[branch])
    cfg['dataset'], cfg['namespace'] = dataset_name, namespace
    datasets_path = os.path.join(cfg['datasets_path'], dataset_name)
    # 与 custom.py / prototype.py 保持一致：训练脚本将特征根路径硬编码
    features_path = os.path.join(cfg.get('features_path', '/archive/hot5/fty/'), dataset_name)
    is_zero_shot = cfg.get('zero_shot', True)
    esm_dim, hidden_dim = cfg['esm_dim'], cfg['hidden_dim']

    # ---- 数据 ----
    num_classes_raw = np.load(os.path.join(
        datasets_path, f'{namespace.lower()}_label_matrix_1_sparse.npy')).shape[0]
    train_seq_data = utils.load_data_from_pkl(
        os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes_raw)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1
    valid_mask = prototype_index.sum(dim=0) > 0
    if not is_zero_shot:
        prototype_index = prototype_index[:, valid_mask]
    num_classes = prototype_index.shape[1]

    protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')
    _, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go_freq, class_counts = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))
    if not is_zero_shot:
        go_freq, class_counts = go_freq[valid_mask], class_counts[valid_mask]

    g = torch.Generator()
    g.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- 模型与优化器 ----
    if branch == 'MLP':
        model = MLPModel(input_dim=esm_dim, hidden_dim=hidden_dim,
                         num_classes=num_classes).to(device)
    else:
        _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
        hop_counts = utils.get_ancestor_hop_matrix(_edges)
        if not is_zero_shot:
            hop_counts = hop_counts[valid_mask][:, valid_mask]
        go2id = utils.load_data_from_pkl(
            os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
        parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
        if not is_zero_shot:
            parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        ic = utils.compute_ic(go2id, train_seq_data,
                              os.path.join(datasets_path, 'go-basic.obo'))
        if not is_zero_shot:
            ic = ic[valid_mask]
        lam_i, beta_i, tau_i = utils.compute_smooth_param_inits(hop_counts, ic, class_counts)
        model = PrototypeNet(
            input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
            frequency=go_freq, parents_matrix=parents_matrix, class_counts=class_counts,
            hop_counts=hop_counts, ic=ic,
            smooth_tau=tau_i, lambda_init=lam_i, beta_init=beta_i,
            tau_min=float(cfg.get('tau_min', 0.0))).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['base_lr']))

    # ---- Params / FLOPs ----
    params_m = count_params_m(model)
    flops_m = measure_flops_per_sample(model, branch, esm_dim, num_classes, hidden_dim, device)

    # ---- DataLoader（各自采样方式） ----
    if branch == 'MLP':
        bs = args.batch_size
        loader = build_mlp_train_loader(datasets_path, namespace, protein_feats,
                                        num_classes,
                                        (None if is_zero_shot else valid_mask),
                                        is_zero_shot, bs, g)
        steps_per_epoch = len(loader)
        sec_per_step, proteins_per_step, peak_mem = bench_mlp_train_step(
            model, loader, optimizer, device, args.n_steps)
    else:
        n_way = cfg.get('n_way', 32)
        n_query = cfg.get('n_query', 10)
        n_support = cfg.get('n_support', 50)
        loader = build_proto_train_loader(datasets_path, namespace, protein_feats,
                                          prototype_index, num_classes, is_zero_shot,
                                          (None if is_zero_shot else valid_mask),
                                          n_way, n_query, n_support, g)
        steps_per_epoch = len(loader)
        sec_per_step, proteins_per_step, peak_mem = bench_proto_train_step(
            model, loader, optimizer, device, args.n_steps)

    # ---- 折算训练成本 ----
    epochs = args.mlp_epochs if branch == 'MLP' else args.proto_epochs
    gpu_hours = sec_per_step * steps_per_epoch * epochs / 3600.0
    sec_per_10k = sec_per_step / proteins_per_step * 1e4

    rows.append({
        'dataset': dataset_name, 'namespace': namespace, 'branch': branch,
        'num_classes': num_classes,
        'params_M': round(params_m, 3),
        'flops_per_sample_M': round(flops_m, 2),
        'steps_per_epoch': steps_per_epoch,
        'proteins_per_step': round(proteins_per_step, 1),
        'sec_per_step': round(sec_per_step, 4),
        'sec_per_10k_proteins': round(sec_per_10k, 3),
        'epochs': epochs,
        'train_GPU_h': round(gpu_hours, 3),
        'peak_mem_GB': round(peak_mem / 1024**3, 3),
    })

    del model, optimizer, loader
    torch.cuda.empty_cache()


# ============================================================
#  汇总
# ============================================================

def print_and_save(rows, args):
    cols = ['dataset', 'namespace', 'branch', 'num_classes', 'params_M',
            'flops_per_sample_M', 'steps_per_epoch', 'proteins_per_step',
            'sec_per_step', 'sec_per_10k_proteins', 'epochs', 'train_GPU_h',
            'peak_mem_GB']
    w = max(len(c) for c in cols)
    print('\n' + '=' * (w * len(cols) // 2))
    print('效率基准测试结果')
    print('=' * (w * len(cols) // 2))
    header = ' | '.join(f'{c:>{w}}' for c in cols)
    print(header)
    print('-' * len(header))
    for r in rows:
        print(' | '.join(f'{str(r[c]):>{w}}' for c in cols))

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, 'cost_benchmark.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n结果已保存至 {out_csv}')

    # ---- 总表：两分支之和（Params / FLOPs / GPU·h 相加，墙钟取 max） ----
    print('\n---- 分支汇总（MLP + Proto 总成本） ----')
    key_set = {}
    for r in rows:
        key_set.setdefault((r['dataset'], r['namespace']), {})[r['branch']] = r
    for (ds, ns), d in sorted(key_set.items()):
        if 'MLP' in d and 'Proto' in d:
            m, p = d['MLP'], d['Proto']
            total_params = m['params_M'] + p['params_M']
            total_flops = m['flops_per_sample_M'] + p['flops_per_sample_M']
            total_gpu_h = m['train_GPU_h'] + p['train_GPU_h']
            wall = max(m['train_GPU_h'], p['train_GPU_h'])
            peak = max(m['peak_mem_GB'], p['peak_mem_GB'])
            print(f"{ds}/{ns}: Params={total_params:.3f}M, "
                  f"FLOPs/sample={total_flops:.2f}M, "
                  f"GPU·h total={total_gpu_h:.2f} (wall≈{wall:.2f}), "
                  f"peak_mem={peak:.2f}GB")


def main():
    parser = argparse.ArgumentParser(description='Efficiency benchmark: Params/FLOPs/time/memory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS)
    parser.add_argument('--namespaces', type=str, nargs='+', default=NAMESPACES)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='MLP 分支训练 batch size（与 custom.py 一致）')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='计时步数（更多更稳定，耗时更长）')
    parser.add_argument('--mlp_epochs', type=int, default=50)
    parser.add_argument('--proto_epochs', type=int, default=150)
    parser.add_argument('--out_dir', type=str, default='./tmp')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)   # 显式初始化 CUDA 上下文，避免显存统计报 Invalid device argument
    rows = []
    for ds in args.datasets:
        for ns in args.namespaces:
            for branch in ['MLP', 'Proto']:
                print(f'\n>>> benchmark {ds}/{ns}/{branch} ...')
                try:
                    run_branch_benchmark(ds, ns, branch, args, device, rows)
                except Exception as e:
                    print(f'  失败 {ds}/{ns}/{branch}: {type(e).__name__}: {e}')
                    traceback.print_exc()

    print_and_save(rows, args)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
