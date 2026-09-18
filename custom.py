import argparse
import os
import utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from models.custom_model import Model
from datasets.dataset import Dataset
from datasets.collator import collator
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

def get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode, valid_mask, g):
    dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='train', valid_mask=valid_mask)

    if mode == 'train': valid_dataset = dataset.val_dataset
    else: valid_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test', valid_mask=valid_mask)

    train_sampler = RandomSampler(dataset, generator=g)
    valid_sampler = SequentialSampler(valid_dataset)
    train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            sampler=train_sampler, drop_last=True, num_workers=0,
            collate_fn=collator(num_classes, protein_feats),
            generator=g,
            worker_init_fn=utils.seed_worker
        )
    valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            sampler=valid_sampler, drop_last=False, num_workers=0,
            collate_fn=collator(num_classes, test_protein_feats),
            generator=g,
            worker_init_fn=utils.seed_worker
        )
    return train_loader, valid_loader

@torch.no_grad()
def valid(model, loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('hier_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_labels = [],[]
    for batch_idx, (protein_feats, labels, indices) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        final_probs, custom_logits, custom_loss = model(protein_feats.to(device), indices.to(device), labels.to(device))
        # hier_loss = utils.compute_hier_loss(custom_logits, parent_indices, child_indices)
        # loss = (1 - hier_reg_lambda) * (scale['custom_scale'] * custom_loss) + hier_reg_lambda * (scale['hier_scale'] * hier_loss)

        all_probs.append(final_probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        metric_logger.update(total_loss=custom_loss.item())
        metric_logger.update(custom_loss=custom_loss.item())
        metric_logger.update(hier_loss=0)

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy())
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC']))
    # all_probs = torch.cat(all_probs, dim=0).numpy()
    # all_labels = torch.cat(all_labels, dim=0).numpy()
    # macro_aupr = utils.macro_auprc(all_labels, all_probs)
    # print("Averaged stats: {}, macro aupr: {:.4f}".format(metric_logger.global_avg(), macro_aupr))

    return all_probs, all_labels

def train(model, optimizer, loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('hier_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    for batch_idx, (protein_feats, labels, indices) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):
        optimizer.zero_grad(set_to_none=True)

        final_probs, custom_logits, custom_loss = model(protein_feats.to(device), indices.to(device), labels.to(device))
        # hier_loss = utils.compute_hier_loss(custom_logits, parent_indices, child_indices)
        # loss = (1 - hier_reg_lambda) * (scale['custom_scale'] * custom_loss) + hier_reg_lambda * (scale['hier_scale'] * hier_loss)

        custom_loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])  
        metric_logger.update(total_loss=custom_loss.item())
        metric_logger.update(custom_loss=custom_loss.item())
        metric_logger.update(hier_loss=0)
        
    print("Averaged stats: {}".format(metric_logger.global_avg()))

def build_ancestor_prior(datasets_path, namespace, dataset_name, edges,
                         train_seq_data, class_counts, valid_mask, is_zero_shot):
    """构建祖先加权分类头所需的层级先验，复用 prototype.py 中的 hop_counts / ic 计算过程。

    Returns:
        (ancestor_mask, ic, lambda_init, gamma_init):
        - ancestor_mask: (C, C) BoolTensor，mask[i][j]=True 表示 j 是 i 的祖先
        - ic: (C,) 各 GO term 的信息量
        - lambda_init / gamma_init: α 权重中样本量敏感度 λ 与 IC 衰减速率 γ 的数据驱动初始值
    """
    hop_counts = utils.get_ancestor_hop_matrix(edges)  # 每个节点到任意祖先节点的跳数
    if not is_zero_shot: hop_counts = hop_counts[valid_mask][:, valid_mask]
    ancestor_mask = hop_counts > 0
    if dataset_name == 'TALE':
        # TALE 数据集：通过 go2id pickle 与 goatools 计算 IC
        # （祖先掩码直接由 hop_counts 二值化得到，无需再构建 parents_matrix）
        go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
        obo_path = os.path.join(datasets_path, 'go-basic.obo')
        ic = utils.compute_ic(go2id, train_seq_data, obo_path)
        if not is_zero_shot: ic = ic[valid_mask]
    else:
        # CAFA3 数据集：缺少可靠的 *_go_1.pickle，直接从 label_regular_1.npy 和 train_seq_data 计算
        # train_seq_data 的 label 字段已沿 DAG 传播祖先标注，故直接统计每个类别索引的蛋白质数，
        # 除以根节点（最大计数）得到频率，IC = -log(freq)。与 goatools TermCounts 结果一致（已在 TALE 上验证）。
        from collections import Counter
        import math
        num_nodes = int(edges.max()) + 1
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
        ic = torch.from_numpy(ic).float()
        if not is_zero_shot: ic = ic[valid_mask]
    # α 的可学习超参初始值由数据分布自动估计（与 prototype.py 一致，无需手工设置）
    lambda_init, gamma_init, _ = utils.compute_smooth_param_inits(hop_counts, ic, class_counts)
    print(f'祖先加权分类头可学习参数初始值(数据驱动): lambda={lambda_init:.4f}, gamma={gamma_init:.4f}')
    return ancestor_mask, ic, lambda_init, gamma_init

def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    dataset_name = config['dataset']
    datasets_path = os.path.join(config['datasets_path'], dataset_name)
    namespace = config['namespace']
    mode = config['mode']
    base_lr = float(config['base_lr'])
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    is_zero_shot = config['zero_shot']
    # 分类头配置: True 为"自由残差 + 祖先加权构造"（默认），False 为原始经典 MLP 分类头（消融基准）
    use_ancestor_weighting = config.get('use_ancestor_weighting', True)
    beta_init = float(config.get('beta_init', 0.5))
    print('数据集:', dataset_name, 'namespace: ', namespace)
    num_classes =  np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_matrix_1_sparse.npy')).shape[0]
    features_path = os.path.join('/archive/hot5/fty/', dataset_name)
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1 # 确保所有数据都注释了根go term的功能
    valid_mask = prototype_index.sum(dim=0) > 0  # 形状为 (标签类别数,) 的布尔 Tensor
    if not is_zero_shot: num_classes = valid_mask.sum().item()  # 更新 num_classes 为有效标签数
    # residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'train_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    # test_residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'test_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    hier_reg_lambda = config.get('hier_reg_lambda', 0.1)
    _, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go_freq, class_counts = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))
    if not is_zero_shot: go_freq, class_counts = go_freq[valid_mask], class_counts[valid_mask]

    # 设置随机种子
    g = utils.set_random_seed(seed) 

    if not is_zero_shot:
        train_loader, valid_loader = get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode, valid_mask, g)
    else:
        train_loader, valid_loader = get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode, None, g)

    _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
    parent_indices = torch.from_numpy(_edges[:, 0].copy()).long().to(device)
    child_indices = torch.from_numpy(_edges[:, 1].copy()).long().to(device)

    # ---- 祖先加权分类头：复用 prototype.py 的 hop_counts / ic 计算过程构建层级先验 ----
    ancestor_mask, ic, alpha_lambda_init, ic_gamma_init = None, None, 10.0, 2.0
    if use_ancestor_weighting:
        ancestor_mask, ic, alpha_lambda_init, ic_gamma_init = build_ancestor_prior(
            datasets_path, namespace, dataset_name, _edges, train_seq_data,
            class_counts, valid_mask, is_zero_shot)
    del _edges  # 立即释放 numpy 内存

    model = Model(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
        use_ancestor_weighting=use_ancestor_weighting,
        ancestor_mask=ancestor_mask, ic=ic, class_counts=class_counts,
        alpha_lambda_init=alpha_lambda_init, ic_gamma_init=ic_gamma_init,
        beta_init=beta_init
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

    # utils.print_loss_gradients(model, train_loader, parent_indices, child_indices, device)
    # hier_scale = utils.get_hier_scale(model, train_loader, parent_indices, child_indices, device) # 静态标定法缩放梯度
    scale = {'custom_scale': 1, 'hier_scale': 0}

    # ---- 检查点目录与断点恢复（仅恢复模型权重，不恢复 optimizer/scheduler 状态）----
    ckpt_dir = os.path.join("/archive/hot5/fty/checkpoints/", namespace, dataset_name, "custom")
    start_epoch = 0
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == 'auto':  # 自动选择目录下修改时间最新的 checkpoint
            ckpt_path = utils.find_latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                print('未找到可恢复的 checkpoint，从头开始训练')
        if ckpt_path:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f'恢复训练的 checkpoint 不存在: {ckpt_path}')
            print(f'从 checkpoint 恢复: {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(utils.extract_state_dict(ckpt))
            del ckpt
            resumed_epoch = utils.parse_checkpoint_epoch(ckpt_path)
            start_epoch = (resumed_epoch + 1) if resumed_epoch is not None else 0
            print(f'恢复成功，从 epoch {start_epoch} 继续训练')
    if start_epoch >= epochs:
        print(f'start_epoch={start_epoch} >= epochs={epochs}，无需训练。如需继续训练请增大 --epochs')
        return

    print('start traing......')
    for epoch in range(start_epoch, epochs):

        # 训练
        train(model, optimizer, train_loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices)
        scheduler.step()

        # 验证
        all_probs, all_labels = valid(model, valid_loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices)

        # 打印祖先加权分类头可学习参数的当前实际值，观察学习进度
        if use_ancestor_weighting:
            lam, gam, bet = model.get_ancestor_params()
            print("ancestor head params: lambda: {:.4f}, gamma: {:.4f}, beta: {:.4f}".format(lam, gam, bet))

    
    # utils.draw_frequencies_AUPRC(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_probs_averge, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), valid_freq, save_path=result_path, namespace=namespace) # 训练结束绘制结果曲线图
        # 仅保存模型权重（每个 epoch 一个文件），节省磁盘；不再保存 optimizer/scheduler/config
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_%02d.pth' % epoch))
        # utils.eval_func_generalizability(model, valid_loader, device, go_freq, None, model_type=0)

        # ---- 每个 epoch 结束后的额外评估 ----
        # 零样本 top-k 注释评估（TALE 协议）：零样本类内部 top-k，扫描 k=1..10
        utils.eval_zero_shot_topk(model, valid_loader, device, class_counts,
                                  prototypes=None, model_type=0, max_k=10)
        # 门控分桶对比实验（实验 1）：纯 MLP（model_type=0），
        # 预测逻辑与上方 valid() 完全一致（feats + indices + labels → final_probs）
        utils.eval_count_bucket_comparison(model, valid_loader, device, None, class_counts, model_type=0)

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/custom.yml', help='config yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--resume', type=str, nargs='?', const='auto', default=None,
                        help='断点恢复：指定 checkpoint 路径（仅保存模型权重）；或仅写 --resume（不带值）自动选择目录下最新 checkpoint')
    
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)
"""
只有MLP，去掉原型网络，作为最开始的比较基准
"""
