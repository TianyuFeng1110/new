import argparse
import os
import new.utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from new.models.model import Model
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from new.datasets.dataset import Dataset
from new.datasets.collator import collator
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

@torch.no_grad()
def valid(model, loader, epoch, device, batch_size):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('mean_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('max_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('attn_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_std', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('k', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('c', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_probs_mean, all_probs_max, all_probs_attn, all_probs_proto, all_labels = [],[],[],[],[],[]
    alpha_mean ,alpha_max, alpha_attn, alpha_proto, alpha_gate, scale = 0.1, 0.1, 0.2, 1, 0.7, (1 / batch_size) # 比例系数
    for batch_idx, (batch_protein_feats, batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        mean_probs, max_probs, cross_attn_probs, proto_probs, final_probs, mean_loss, max_loss, cross_attn_loss, proto_loss, gate_loss, tau, bias, k, c, weights, sigma = model(batch_protein_feats.to(device), batch_residue_feats.to(device), mask.to(device), labels.to(device))
        loss = (alpha_mean * mean_loss + alpha_max * max_loss + alpha_attn * cross_attn_loss + alpha_proto * proto_loss + alpha_gate * gate_loss) * scale

        all_probs.append(final_probs.detach().cpu())
        all_probs_mean.append(mean_probs.detach().cpu())
        all_probs_max.append(max_probs.detach().cpu())
        all_probs_attn.append(cross_attn_probs.detach().cpu())
        all_probs_proto.append(proto_probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

        metric_logger.update(total_loss=loss.item())
        metric_logger.update(mean_loss=mean_loss.item() * alpha_mean * scale)
        metric_logger.update(max_loss=max_loss.item() * alpha_max * scale)
        metric_logger.update(attn_loss=cross_attn_loss.item() * alpha_attn * scale)
        metric_logger.update(proto_loss=proto_loss.item() * alpha_proto * scale)
        metric_logger.update(gate_loss=gate_loss.item() * alpha_gate * scale)
        metric_logger.update(tau=tau.item())
        metric_logger.update(bias_mean=bias.mean().item())
        metric_logger.update(bias_std=bias.std().item())
        metric_logger.update(bias_max=bias.max().item())
        metric_logger.update(bias_min=bias.min().item())
        metric_logger.update(k=k.item())
        metric_logger.update(c=c.item())
        metric_logger.update(sigma=sigma.item())

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy(), is_logits = False)
    w = utils.get_mean_weights(weights.detach().cpu()) # 权重顺序为平均、最大、注意力
    metric_logger.synchronize_between_processes()
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}, macro AUPRC: {:.4f}, mean negative score: {:.4f}, weights: {}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC'], metric['macro_AUPRC'], metric['mean_neg_score'], w))

    return all_probs, all_probs_mean, all_probs_max, all_probs_attn, all_probs_proto, all_labels

def train(model, optimizer, loader, epoch, device, batch_size):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('mean_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('max_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('attn_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_std', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('k', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('c', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('sigma', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    alpha_mean ,alpha_max, alpha_attn, alpha_proto, alpha_gate, scale = 0.1, 0.1, 0.2, 1, 0.7, (1 / batch_size) # 比例系数
    for batch_idx, (batch_protein_feats, batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):
        optimizer.zero_grad()

        _, _, _, _, _ , mean_loss, max_loss, cross_attn_loss, proto_loss, gate_loss, tau, bias, k, c, weights, sigma = model(batch_protein_feats.to(device), batch_residue_feats.to(device), mask.to(device), labels.to(device))
        loss = (alpha_mean * mean_loss + alpha_max * max_loss + alpha_attn * cross_attn_loss + alpha_proto * proto_loss + alpha_gate * gate_loss) * scale
        loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])  
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(mean_loss=mean_loss.item() * alpha_mean * scale)
        metric_logger.update(max_loss=max_loss.item() * alpha_max * scale)
        metric_logger.update(attn_loss=cross_attn_loss.item() * alpha_attn * scale)
        metric_logger.update(proto_loss=proto_loss.item() * alpha_proto * scale)
        metric_logger.update(gate_loss=gate_loss.item() * alpha_gate * scale)
        metric_logger.update(tau=tau.item())
        metric_logger.update(bias_mean=bias.mean().item())
        metric_logger.update(bias_std=bias.std().item())
        metric_logger.update(bias_max=bias.max().item())
        metric_logger.update(bias_min=bias.min().item())
        metric_logger.update(k=k.item())
        metric_logger.update(c=c.item())
        metric_logger.update(sigma=sigma.item())
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats: {}".format(metric_logger.global_avg()))

def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    epochs = args.epochs
    datasets_path = args.path
    batch_size = args.batch_size
    namespace = args.namespace
    base_lr = 5e-5
    warmup_epochs = config['warmup_epochs']
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    rgcn_dim = config['rgcn_dim']
    result_path = args.result_path
    go2id = utils.go2id(os.path.join(datasets_path,namespace,'term_list')) # term_list为训练集中出现过的go term，直接根据顺序转为id。直接作为待预测的类别
    train_protein_feats = utils.load_dict_from_pkl(os.path.join(datasets_path,'train_protein_feats.pkl'))
    valid_protein_feats = utils.load_dict_from_pkl(os.path.join(datasets_path,'evaluate_protein_feats.pkl'))
    train_protein_residue_feats = utils.load_dict_from_safetensors(os.path.join("/archive/hot3/fty",namespace + '_train_residue_feats.safetensors')) # TODO 存储空间不够了， 地址先暂时换到其他位置
    valid_protein_residue_feats = utils.load_dict_from_safetensors(os.path.join("/archive/hot3/fty",namespace + '_evaluate_residue_feats.safetensors'))
    prototype_feats = utils.process_prototype(utils.load_dict_from_pkl(os.path.join(datasets_path, namespace + '_prototype.pkl')), go2id).to(device)
    train_freq = utils.frequencies_projector(utils.process_frequencies(utils.count_occurrences(os.path.join(datasets_path,namespace,'train_gene_label'), os.path.join(datasets_path,namespace,'term_list')), go2id).to(device))
    valid_freq = utils.frequencies_projector(utils.process_frequencies(utils.count_occurrences(os.path.join(datasets_path,namespace,'evaluate_gene_label'), os.path.join(datasets_path,namespace,'term_list')), go2id)) # 用于绘制频率AUPRC曲线图
    labels_index = utils.get_labels_index(os.path.join(datasets_path,'graph_node_id.txt'), os.path.join(datasets_path, namespace,'term_list'))

    # 设置随机种子
    utils.set_random_seed(seed) 

    # 加载模型
    model = Model(prototype_feats, train_freq, rgcn_dim, labels_index, input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=len(go2id)).to(device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.device])

    # 优化器 — 分模块设置不同学习率
    optimizer = optim.AdamW([
        # 门控模块 — 最低学习率，防止过早过拟合
        {'params': model.term_gate_weight.parameters(), 'lr': base_lr * 0.1},
        {'params': [model.gate_k, model.gate_c_raw], 'lr': base_lr * 0.1},
        # 原型模块 — 最高学习率，加速收敛
        {'params': model.prototype_Module.parameters(), 'lr': base_lr * 3.0},
        # 其余模块 — 默认学习率
        {'params': model.meanFeature_module.parameters()},
        {'params': model.maxFeature_module.parameters()},
        {'params': model.cross_attention_module.parameters()},
        {'params': model.LayerNorm.parameters()},
        {'params': model.residue_LayerNorm.parameters()},
        {'params': model.go_LayerNorm.parameters()},
    ], lr=base_lr)
    # 设置学习率策略
    lr_lambda = partial(utils.lr_scheduler, 
                    warmup_epochs=warmup_epochs, 
                    max_epochs=epochs)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_dataset = Dataset(datasets_path, namespace, split='train')
    valid_dataset = Dataset(datasets_path, namespace, split='valid')
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, rank=utils.get_rank(), num_replicas=utils.get_world_size())
        valid_sampler = DistributedSampler(valid_dataset, shuffle=True, rank=utils.get_rank(), num_replicas=utils.get_world_size())
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset) 
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True, num_workers=0, collate_fn=collator(go2id, train_protein_feats, train_protein_residue_feats), worker_init_fn=utils.seed_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, drop_last=False, num_workers=0, collate_fn=collator(go2id, valid_protein_feats, valid_protein_residue_feats),  worker_init_fn=utils.seed_worker)

    if utils.is_main_process():      
        print('start traing......')
    for epoch in range(epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # 训练
        train(model, optimizer, train_loader, epoch, device, batch_size)
        scheduler.step()

        # 验证
        all_probs, all_probs_mean, all_probs_max, all_probs_attn, all_probs_proto, all_labels = valid(model, valid_loader, epoch, device, batch_size)
    # TODO 下面这块还需要修改，把不同概率的评估方法重写一遍。
    # utils.draw_frequencies_AUPRC(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_probs_averge, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), valid_freq, save_path=result_path, namespace=namespace) # 训练结束绘制结果曲线图

'''
baseline: 平均池化路线
'''
if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./new/config/cc.yml', help='config/.yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size') # CC:32, BP:4, MF:16
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--path', type=str, default="./new/data/", help='datasets path')
    parser.add_argument('--result_path', type=str, default="/archive/hot3/fty/result/", help='result save path')
    parser.add_argument('--namespace', default='CC', type=str, help='[BP/CC/MF]')
    # parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    utils.init_distributed_mode(args)
    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)