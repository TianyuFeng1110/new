import argparse
import os
import utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from models.model import Model
from datasets.dataset import Dataset
from datasets.collator import collator
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

def get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode):
    dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='train')

    if mode == 'train': valid_dataset = dataset.val_dataset
    else: valid_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test')

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
def valid(model, loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('hier_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('max_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('min_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('mean_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_labels = [],[]
    diag_accum = {}  # 累积所有 batch 的诊断统计
    for batch_idx, (protein_feats, labels, indices) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        final_probs, custom_logits, custom_loss, proto_loss, gate_loss, sigma = model(protein_feats.to(device), indices.to(device), labels.to(device))
        hier_loss = utils.compute_hier_loss(custom_logits, parent_indices, child_indices)
        loss = (1 - hier_reg_lambda) * (scale['custom_scale'] * custom_loss + scale['proto_scale'] * proto_loss + scale['gate_scale'] * gate_loss) + hier_reg_lambda * (scale['hier_scale'] * hier_loss)

        all_probs.append(final_probs.detach().cpu())  # final_probs 已经是概率值，不需要再 sigmoid
        all_labels.append(labels.detach().cpu())
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(custom_loss=custom_loss.item())
        metric_logger.update(proto_loss=proto_loss.item())
        metric_logger.update(gate_loss=gate_loss.item())
        metric_logger.update(hier_loss=hier_loss.item())
        metric_logger.update(max_sigma=sigma.max().item())
        metric_logger.update(min_sigma=sigma.min().item())
        metric_logger.update(mean_sigma=sigma.mean().item())

        # 诊断：累积每个频率桶的 loss=====================================
        batch_diag = model.diagnose_proto_vs_custom(
            protein_feats.to(device), indices.to(device), labels.to(device))
        for k, v in batch_diag.items():
            if k not in diag_accum:
                diag_accum[k] = {"custom_loss": 0.0, "proto_loss": 0.0, "num_terms": v["num_terms"], "sigma_mean": 0.0, "count": 0}
            diag_accum[k]["custom_loss"] += v["custom_loss"]
            diag_accum[k]["proto_loss"] += v["proto_loss"]
            diag_accum[k]["sigma_mean"] += v["sigma_mean"]
            diag_accum[k]["count"] += 1

    # 打印频率桶诊断（按 batch 平均）
    print("\n--- GO term frequency bucket diagnosis ---")
    for k in sorted(diag_accum.keys()):
        v = diag_accum[k]
        avg_custom = v["custom_loss"] / v["count"]
        avg_proto = v["proto_loss"] / v["count"]
        winner = "Proto" if avg_proto < avg_custom else "Custom"
        print(f"  {k}: custom={avg_custom:.4f}, proto={avg_proto:.4f}, "
              f"terms={v['num_terms']}, σ={v['sigma_mean']/v['count']:.4f} [{winner}]")
    print("--- end ---\n")
    # ======================================================================

    target_log = model.gate_c.item() * (model.freq_log_std + 1e-8) + model.freq_log_mean # 逆 Z-Score 标准化
    target_freq = torch.exp(target_log) - 1e-6 # 逆对数变换（解出原始频率）
    target_freq = torch.clamp(target_freq, min=0.0) # sigma=0.5时对应的go term频率是多少

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy())
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}, target frequency: {:.6f}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC'], target_freq))

    return all_probs, all_labels

def train(model, optimizer, loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('max_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('min_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('mean_sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    for batch_idx, (protein_feats, labels, indices) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):
        optimizer.zero_grad(set_to_none=True)

        final_probs, custom_logits, custom_loss, proto_loss, gate_loss, sigma = model(protein_feats.to(device), indices.to(device), labels.to(device))
        hier_loss = utils.compute_hier_loss(custom_logits, parent_indices, child_indices)
        loss = (1 - hier_reg_lambda) * (scale['custom_scale'] * custom_loss + scale['proto_scale'] * proto_loss + scale['gate_scale'] * gate_loss) + hier_reg_lambda * (scale['hier_scale'] * hier_loss)

        loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])  
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(custom_loss=custom_loss.item())
        metric_logger.update(proto_loss=proto_loss.item())
        metric_logger.update(gate_loss=gate_loss.item())
        metric_logger.update(max_sigma=sigma.max().item())
        metric_logger.update(min_sigma=sigma.min().item())
        metric_logger.update(mean_sigma=sigma.mean().item())
        
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
    features_path = '/archive/hot5/fty/'
    # residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'train_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    # test_residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'test_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go_freq = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))
    hier_reg_lambda = config.get('hier_reg_lambda', 0.1)

    # 设置随机种子
    utils.set_random_seed(seed) 

    train_loader, valid_loader = get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode)

    stacked_feats = torch.stack([protein_feats[k] for k in sorted(protein_feats.keys(), key=int)], dim=0).to(device) 
    model = Model(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes, prototype_index=prototype_index,
          proto_idx_mask=proto_idx_mask, go_freq=go_freq
    ).to(device)
    model.init_protein_feats(stacked_feats)
    model.init_prototype_feats(num_classes, hidden_dim)

    prototype_params = list(model.proto_module.parameters())
    # gate_params = list(model.gate_k.parameters(), model.gate_c.parameters())
    special_param_ids = set(id(p) for p in prototype_params)
    default_params = [p for p in model.parameters() if id(p) not in special_param_ids]
    optimizer = optim.AdamW([
        {'params': prototype_params, 'lr': base_lr * 10}, # 原型模块
        {'params': default_params} # 其余所有模块 — 动态打包，继承最外层的默认学习率
    ], lr=base_lr)

    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

    _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
    parent_indices = torch.from_numpy(_edges[:, 0].copy()).long().to(device)
    child_indices = torch.from_numpy(_edges[:, 1].copy()).long().to(device)
    del _edges  # 立即释放 numpy 内存

    # utils.print_loss_gradients(model, train_loader, parent_indices, child_indices, device)
    # hier_scale = utils.get_hier_scale(model, train_loader, parent_indices, child_indices, device) # 静态标定法缩放梯度
    scale = {'proto_scale': 0.185, 'gate_scale': 0.291, 'custom_scale': 1, 'hier_scale': 7003.65}
    
    print('start traing......')
    for epoch in range(epochs):

        # 训练
        train(model, optimizer, train_loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices)
        scheduler.step()

        # 验证
        all_probs, all_labels = valid(model, valid_loader, epoch, device, scale, hier_reg_lambda, parent_indices, child_indices)
    
    # utils.draw_frequencies_AUPRC(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_probs_averge, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), valid_freq, save_path=result_path, namespace=namespace) # 训练结束绘制结果曲线图
    save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 
                'config': config,
                'epoch': epoch,
            }
    torch.save(save_obj, os.path.join("/archive/hot3/fty/checkpoints/", 'checkpoint_%02d.pth'%epoch))  

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/cc.yml', help='config yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size') # CC:32, BP:4, MF:14
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--path', type=str, default="./data_tale/CAFA3/", help='datasets path')
    parser.add_argument('--result_path', type=str, default="/archive/hot3/fty/result/", help='result save path')
    parser.add_argument('--mode', type=str, default='test', help='[train/test]')
    parser.add_argument('--namespace', default='CC', type=str, help='[BP/CC/MF]')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)