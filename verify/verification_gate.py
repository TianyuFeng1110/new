import argparse
import os
import new.utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from new.models.verification_gate_model import Model
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
    metric_logger.add_meter('mlp_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_std', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('k', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('c', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_probs_mlp, all_labels = [],[],[]
    alpha_mlp, alpha_proto, alpha_gate, scale = 1.0, 10, 50, (1 / batch_size)  # 比例系数
    for batch_idx, (data, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        probs_mlp, _, probs, mlp_loss, proto_loss, gate_loss, tau, bias, k, c = model(data.to(device), labels.to(device))
        
        loss = (alpha_mlp * mlp_loss + alpha_proto * proto_loss + alpha_gate * gate_loss) * scale
        all_probs.append(probs.detach().cpu())
        all_probs_mlp.append(probs_mlp.detach().cpu())
        all_labels.append(labels.detach().cpu())
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(mlp_loss=mlp_loss.item() * alpha_mlp * scale)
        metric_logger.update(proto_loss=proto_loss.item() * alpha_proto * scale)
        metric_logger.update(gate_loss=gate_loss.item() * alpha_gate * scale)
        metric_logger.update(tau=tau.item())
        metric_logger.update(bias_mean=bias.mean().item())
        metric_logger.update(bias_std=bias.std().item())
        metric_logger.update(bias_max=bias.max().item())
        metric_logger.update(bias_min=bias.min().item())
        metric_logger.update(k=k.item())
        metric_logger.update(c=c.item())

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy(), is_logits = False)

    metric_logger.synchronize_between_processes()
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}, macro AUPRC: {:.4f}, mean negative score: {:.4f}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC'], metric['macro_AUPRC'], metric['mean_neg_score']))

    return all_probs, all_probs_mlp, all_labels

def train(model, optimizer, loader, epoch, device, batch_size):

    model.train()
    optimizer.zero_grad()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('mlp_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('proto_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('gate_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('tau', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_mean', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_std', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_max', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('bias_min', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('k', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('c', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    alpha_mlp, alpha_proto, alpha_gate, scale = 1.0, 10, 50, (1 / batch_size) # 比例系数
    # all_logits, all_labels = [],[] # 指标计算有点慢，先不在训练时计算了
    for batch_idx, (data, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        _, _, _, mlp_loss, proto_loss, gate_loss, tau, bias, k, c  = model(data.to(device), labels.to(device))
        
        loss = (alpha_mlp * mlp_loss + alpha_proto * proto_loss + alpha_gate * gate_loss) * scale
        loss.backward()
        optimizer.step()
        
        # all_logits.append(logits.detach().cpu())
        # all_labels.append(labels.detach().cpu())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(mlp_loss=mlp_loss.item() * alpha_mlp * scale)
        metric_logger.update(proto_loss=proto_loss.item() * alpha_proto * scale)
        metric_logger.update(gate_loss=gate_loss.item() * alpha_gate * scale)
        metric_logger.update(tau=tau.item())
        metric_logger.update(bias_mean=bias.mean().item())
        metric_logger.update(bias_std=bias.std().item())
        metric_logger.update(bias_max=bias.max().item())
        metric_logger.update(bias_min=bias.min().item())
        metric_logger.update(k=k.item())
        metric_logger.update(c=c.item())

    # metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_logits, dim=0).numpy())

    metric_logger.synchronize_between_processes()
    # print("Averaged stats: {}, Fmax: {:.4f}, AUPRC: {:.4f}".format(metric_logger.global_avg(), metric['Fmax'], metric['AUPRC']))
    print("Averaged stats: {}".format(metric_logger.global_avg()))

def main(args, config):
    device = torch.device(args.device)
    seed = args.seed
    epochs = args.epochs
    datasets_path = args.path
    batch_size = args.batch_size
    namespace = args.namespace
    warmup_epochs = config['warmup_epochs']
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    result_path = args.result_path
    go2id = utils.go2id(os.path.join(datasets_path,namespace,'term_list')) # term_list为训练集中出现过的go term，直接根据顺序转为id。直接作为待预测的类别
    train_protein_feats = utils.load_dict_from_pkl(os.path.join(datasets_path,'train_protein_feats.pkl'))
    valid_protein_feats = utils.load_dict_from_pkl(os.path.join(datasets_path,'evaluate_protein_feats.pkl'))
    prototype_feats = utils.process_prototype(utils.load_dict_from_pkl(os.path.join(datasets_path, namespace + '_prototype.pkl')), go2id).to(device)
    train_freq = utils.frequencies_projector(utils.process_frequencies(utils.count_occurrences(os.path.join(datasets_path,namespace,'train_gene_label'), os.path.join(datasets_path,namespace,'term_list')), go2id).to(device))
    valid_freq = utils.frequencies_projector(utils.process_frequencies(utils.count_occurrences(os.path.join(datasets_path,namespace,'evaluate_gene_label'), os.path.join(datasets_path,namespace,'term_list')), go2id)) # 用于绘制频率AUPRC曲线图
    
    # 设置随机种子
    utils.set_random_seed(seed) 

    # 加载模型
    model = Model(prototype_feats, train_freq, input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=len(go2id)).to(device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.device])

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
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
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True, num_workers=2, collate_fn=collator(go2id, train_protein_feats), worker_init_fn=utils.seed_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, drop_last=False, num_workers=2, collate_fn=collator(go2id, valid_protein_feats),  worker_init_fn=utils.seed_worker)

    if utils.is_main_process():      
        print('start traing......')
    for epoch in range(epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # 训练
        train(model, optimizer, train_loader, epoch, device, batch_size)
        scheduler.step()

        # 验证
        all_probs, all_probs_mlp, all_labels = valid(model, valid_loader, epoch, device, batch_size)
    utils.draw_frequencies_AUPRC(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_probs_mlp, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), valid_freq, save_path=result_path, namespace=namespace) # 训练结束绘制结果曲线图

'''
Quick Verificaiton:
baseline: 验证门控(ESM + 原型)预测蛋白质功能。特征提取模型为6层ESM2
'''
if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./new/config/config.yml', help='config/.yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
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