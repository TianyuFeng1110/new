import argparse
import os
import utils as utils
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from models.prototype_model import Model
from datasets.balanced_dataset import Dataset
from datasets.collator import collator
from datasets.balanced_collator import BalancedCollator
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from datasets.balanced_sampler import ClassBalanceSampler

@torch.no_grad()
def valid(model, loader, epoch, device):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_labels = [],[]
    diag_accum = {}  # 累积所有 batch 的诊断统计
    for batch_idx, (inputs, labels, mask) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        logits, tau, bias = model(inputs.to(device), labels.to(device))
        final_probs = torch.sigmoid(logits)
        total_loss = F.binary_cross_entropy_with_logits(logits, labels.to(device))

        all_probs.append(final_probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(proto_loss=total_loss.item())
        metric_logger.update(tau_max=tau.max())
        metric_logger.update(tau_min=tau.min())
        metric_logger.update(tau_mean=tau.mean())
        metric_logger.update(bias_max=bias.max())
        metric_logger.update(bias_min=bias.min())
        metric_logger.update(bias_mean=bias.mean())

    # target_freq = torch.exp(torch.tensor(model.gate_c.item())) - 1e-6 # sigma=0.5时 log_freq=gate_c, 逆对数得原始频率
    # target_freq = torch.clamp(target_freq, min=0.0)
    # utils.eval_func_generalizability(model, loader, device, go_freq)

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy())
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}, target frequency: {:.6f}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC'], 0.0))

    return all_probs, all_labels

def train(model, optimizer, loader, epoch, device, balanced_N, balanced_k):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('cb_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    for batch_idx, (inputs, labels, mask) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):
        optimizer.zero_grad(set_to_none=True)

        logits, tau, bias = model(inputs.to(device), labels.to(device))
        total_loss = utils.balanced_bce_loss(logits, labels.to(device), mask, balanced_N, balanced_k)
        # total_loss = utils.balanced_asl_loss(logits, labels.to(device), mask, balanced_N, balanced_k)

        total_loss.backward()
        optimizer.step()
        model._momentum_update()
        
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])  
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(proto_loss=total_loss.item())
        metric_logger.update(cb_loss=total_loss.item())
        metric_logger.update(tau_max=tau.max())
        metric_logger.update(tau_min=tau.min())
        metric_logger.update(tau_mean=tau.mean())
        metric_logger.update(bias_max=bias.max())
        metric_logger.update(bias_min=bias.min())
        metric_logger.update(bias_mean=bias.mean())
        
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
    queue_size = config['queue_size']
    hier_reg_lambda = config.get('hier_reg_lambda', 0.1)
    features_path = '/archive/hot5/fty/TALE/'
    # residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'train_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    # test_residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'test_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    # prototype_index, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1 # 蛋白质数据集中有几条数据未标注namespace根节点功能(CC中是3条)
    stacked_feats = torch.stack([protein_feats[k] for k in sorted(protein_feats.keys(), key=int)], dim=0).to(device) 
    # go_freq = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))
    go_embeddings = torch.load(os.path.join(datasets_path, f'{namespace.lower()}_go_feats.pt'), weights_only=True, map_location='cpu')
    queue_indices = utils.get_queue_index(stacked_feats, prototype_index, namespace, datasets_path, queue_size)
    pos_indices = prototype_index.T
    hard_neg_indices = utils.get_hard_neg_indices(pos_indices, go2id)  # 困难负样本
    easy_neg_indices = (1 - pos_indices) - hard_neg_indices  # 简单负样本 

    # 设置随机种子
    utils.set_random_seed(seed)

    # ---- 构建类平衡索引 -------
    raw_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='train') 

    # ---- 类平衡参数 ----
    balanced_N = config.get('balanced_N', 32)
    balanced_k = config.get('balanced_k', 4)

    # ---- DataLoader：训练集用类平衡采样，验证集保持原始分布 ----
    train_sampler = ClassBalanceSampler(pos_indices, easy_neg_indices, hard_neg_indices, num_classes, N=32, k=4)
    train_collator = BalancedCollator(protein_feats=protein_feats, data=raw_dataset, num_classes=num_classes)
    train_loader = DataLoader(
        raw_dataset, batch_sampler=train_sampler,
        collate_fn=train_collator, num_workers=2,
    )

    # 验证集：使用原始分布
    valid_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test')
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        sampler=valid_sampler, drop_last=False, num_workers=2,
        collate_fn=collator(num_classes, test_protein_feats),
        worker_init_fn=utils.seed_worker,
    )
    
    model = Model(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
           raw_feats=stacked_feats, queue_size=queue_size, queue_indices=queue_indices
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)
    
    print('start training......')
    for epoch in range(epochs):

        # 训练
        train(model, optimizer, train_loader, epoch, device, balanced_N, balanced_k)
        scheduler.step()

        # 验证
        all_probs, all_labels = valid(model, valid_loader, epoch, device)
    
    # utils.draw_frequencies_AUPRC(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_probs_averge, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), valid_freq, save_path=result_path, namespace=namespace) # 训练结束绘制结果曲线图
        save_obj = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 
                    'config': config,
                    'epoch': epoch,
                }
        torch.save(save_obj, os.path.join("/archive/hot5/fty/checkpoints/TALE/prototype", 'checkpoint_%02d.pth'%epoch))  

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/cc.yml', help='config yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size') # CC:32, BP:4, MF:14
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--path', type=str, default="./data_tale/TALE/", help='datasets path')
    parser.add_argument('--result_path', type=str, default="/archive/hot3/fty/result/", help='result save path')
    parser.add_argument('--mode', type=str, default='test', help='[train/test]')
    parser.add_argument('--namespace', default='CC', type=str, help='[BP/CC/MF]')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)