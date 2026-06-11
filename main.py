import argparse
import os
import new.utils as utils
import torch
import torch.optim as optim
import numpy as np
import sys

from ruamel.yaml import YAML
from new.models.model import Model
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from new.datasets.dataset import Dataset
from new.datasets.collator import collator
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


def get_loader(datasets_path, namespace, batch_size, residue_feats, test_residue_feats, num_classes, mode):
    dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='train')

    if mode == 'train': valid_dataset = dataset.val_dataset
    else: valid_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test')

    train_sampler = RandomSampler(dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            sampler=train_sampler, drop_last=True, num_workers=2,
            collate_fn=collator(num_classes, residue_feats),
            worker_init_fn=utils.seed_worker
        )
    valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            sampler=valid_sampler, drop_last=False, num_workers=2,
            collate_fn=collator(num_classes, test_residue_feats),
            worker_init_fn=utils.seed_worker
        )
    return train_loader, valid_loader

@torch.no_grad()
def valid(model, loader, epoch, device, ic,
          hier_reg_lambda, parent_indices, child_indices, scale):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('hier_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('freq', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_labels = [],[]
    for batch_idx, (batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        logits, custom_logits, custom_loss, proto_loss, gate_loss, sigma, freq = model(batch_residue_feats.to(device), mask.to(device), labels.to(device))
        hier_loss_raw = utils.compute_hier_loss(custom_logits, parent_indices, child_indices)
        hier_loss = scale['hier_scale'] * hier_loss_raw
        loss = (1 - hier_reg_lambda) * (custom_loss +  scale['proto_scale'] * proto_loss + scale['gate_scale'] * gate_loss) + hier_reg_lambda * hier_loss

        all_probs.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(labels.detach().cpu())
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(hier_loss=hier_loss.item())
        metric_logger.update(custom_loss=custom_loss.item())
        metric_logger.update(proto_loss=proto_loss.item())
        metric_logger.update(gate_loss=gate_loss.item())
        metric_logger.update(freq=freq.item())
        metric_logger.update(sigma=sigma.item())

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy())
    # metric = utils.calculate_metrics_new(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_custom_probs, dim=0).numpy())
    # metric = utils.calculate_metrics_old(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_custom_probs, dim=0).numpy())
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC']))

    return all_probs, all_labels

def train(model, optimizer, loader, epoch, device,
          hier_reg_lambda, parent_indices, child_indices, scale):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('hier_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('freq', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    for batch_idx, (batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):
        optimizer.zero_grad(set_to_none=True)

        logits, custom_logits, custom_loss, proto_loss, gate_loss, sigma, freq = model(batch_residue_feats.to(device), mask.to(device), labels.to(device))
        hier_loss_raw = utils.compute_hier_loss(custom_logits, parent_indices, child_indices)
        hier_loss = scale['hier_scale'] * hier_loss_raw

        loss = (1 - hier_reg_lambda) * (custom_loss +  scale['proto_scale'] * proto_loss + scale['gate_scale'] * gate_loss) + hier_reg_lambda * hier_loss
        loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])  
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(hier_loss=hier_loss.item())
        metric_logger.update(custom_loss=custom_loss.item())
        metric_logger.update(proto_loss=proto_loss.item())
        metric_logger.update(gate_loss=gate_loss.item())
        metric_logger.update(freq=freq.item())
        metric_logger.update(sigma=sigma.item())
        
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
    warmup_epochs = config['warmup_epochs']
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    num_classes = config['num_classes']
    hier_reg_lambda = config.get('hier_reg_lambda', 0.1)
    result_path = args.result_path
    # ======================================= NOTE 此处开始 =======================================
    obo_path = os.path.join(datasets_path, 'go-basic.obo')
    features_path = '/archive/hot5/fty/'
    residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'train_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu') # 加载残基特征
    # protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu') # 加载蛋白质池化特征
    if mode == 'test': 
        test_residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'test_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
        # test_protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')

    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index = utils.get_prototype_index(train_seq_data) # 获取每个go term注释到的蛋白质索引
    go_freq = torch.tensor(utils.compute_go_term_frequency(prototype_index, len(train_seq_data)))

    if namespace == 'CC':
        kernel_size = 5
        dilations = [1, 2, 4]
        proto_scale, gate_scale =  2, 3.5
    elif namespace == 'BP':
        kernel_size = 7
        dilations = [1, 2, 4]
        proto_scale, gate_scale =  2, 3.5
    elif namespace == 'MF':
        kernel_size = 3
        dilations = [1, 2, 4]
        proto_scale, gate_scale =  2, 3.5

    # 设置随机种子
    utils.set_random_seed(seed) 

    train_loader, valid_loader = get_loader(datasets_path, namespace, batch_size, residue_feats, test_residue_feats, num_classes, mode) # mode为test时valid_loader是测试集(test datasets)的loader

    model = Model(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes, go_freq=go_freq.to(device),
        kernel_size=kernel_size, dilations=dilations,
    ).to(device)

    _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
    parent_indices = torch.from_numpy(_edges[:, 0].copy()).long().to(device)
    child_indices = torch.from_numpy(_edges[:, 1].copy()).long().to(device)
    del _edges  # 立即释放 numpy 内存

    # 优化器参数设置：不同的模块设置不同学习率 TODO 目前只有常规模块
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

    hier_scale = utils.get_hier_scale(model, train_loader, parent_indices, child_indices, device) # 静态标定法缩放梯度
    scale = {'proto_scale': proto_scale, 'gate_scale':gate_scale, 'hier_scale':hier_scale}

    print('start traing......')
    for epoch in range(epochs):

        # 训练
        # train(model, optimizer, train_loader, epoch, device,
        #       hier_reg_lambda, parent_indices, child_indices, scale)
        # scheduler.step()

        # 验证
        all_probs, all_labels = valid(
            model, valid_loader, epoch, device, ic=None,
            hier_reg_lambda=hier_reg_lambda, parent_indices=parent_indices, child_indices=child_indices,
            scale=scale)
    # TODO 下面这块还需要修改，把不同概率的评估方法重写一遍。
    # utils.draw_frequencies_AUPRC(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_probs_averge, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), valid_freq, save_path=result_path, namespace=namespace) # 训练结束绘制结果曲线图

'''
baseline: 平均池化路线
'''
if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./new/config/cc.yml', help='config yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size') # CC:32, BP:4, MF:14
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    parser.add_argument('--path', type=str, default="./new/data_tale/CAFA3/", help='datasets path')
    parser.add_argument('--result_path', type=str, default="/archive/hot3/fty/result/", help='result save path')
    parser.add_argument('--mode', type=str, default='test', help='[train/test]')
    parser.add_argument('--namespace', default='CC', type=str, help='[BP/CC/MF]')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)