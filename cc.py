import argparse
import os
import new.utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from new.models.cc_model import Model
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from new.datasets.cc_dataset import Dataset
from new.datasets.cc_collator import collator
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

@torch.no_grad()
def valid(model, loader, epoch, device, alpha_custom, alpha_proto, alpha_gate, scale, ic):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
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

    all_probs, all_custom_probs, all_proto_probs, all_labels, all_weights = [],[],[],[],[]
    for batch_idx, (batch_protein_feats, batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        custom_probs, proto_probs, final_probs, custom_loss, proto_loss, gate_loss, tau, bias, k, c, weights, sigma = model(batch_protein_feats.to(device), batch_residue_feats.to(device), mask.to(device), labels.to(device))
        loss = (alpha_custom * custom_loss + alpha_proto * proto_loss + alpha_gate * gate_loss) * scale

        all_probs.append(final_probs.detach().cpu())
        all_custom_probs.append(custom_probs.detach().cpu())
        all_proto_probs.append(proto_probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_weights.append(weights.detach().cpu().squeeze(0))
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(custom_loss=custom_loss.item() * alpha_custom * scale)
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

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy(), is_logits = False, ic=ic)
    w = utils.get_mean_weights(torch.stack(all_weights)) # 权重顺序为平均、最大、注意力
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}, macro AUPRC: {:.4f}, WFmax: {:.4f}, Smin: {:.4f}, weights: {}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC'], metric['macro_AUPRC'], metric['WFmax'], metric['Smin'], w))

    return all_probs, all_custom_probs, all_proto_probs, all_labels

def train(model, optimizer, loader, epoch, device, alpha_custom, alpha_proto, alpha_gate, scale):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('custom_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
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

    for batch_idx, (batch_protein_feats, batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):
        optimizer.zero_grad(set_to_none=True)

        _, _, _, custom_loss, proto_loss, gate_loss, tau, bias, k, c, weights, sigma = model(batch_protein_feats.to(device), batch_residue_feats.to(device), mask.to(device), labels.to(device))
        loss = (alpha_custom * custom_loss + alpha_proto * proto_loss + alpha_gate * gate_loss) * scale
        loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])  
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(custom_loss=custom_loss.item() * alpha_custom * scale)
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
    gate_freeze_epochs = config.get('gate_freeze_epochs', 0)  # 门控冻结轮数，默认0表示不冻结
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
    ic = utils.get_ic(os.path.join(datasets_path, 'go-basic.obo'), os.path.join(datasets_path, namespace, 'term_list'), os.path.join(datasets_path, namespace, 'train_gene_label'))
    # 局部特征提取模块的参数，根据各个namespace的特点进行设置
    # dilations: 三层膨胀卷积的膨胀率，[1,2,4] 可指数级扩大感受野（参数量不变），padding 自动计算
    if namespace == 'CC': 
        kernel_size = 5
        dilations = [1, 2, 4]
        alpha_custom, alpha_proto, alpha_gate, scale = 0.417, 1, 0.87, (1 / batch_size) # 比例系数1:1:1，学习情况速率通过学习率调节
    elif namespace == 'BP':
        kernel_size = 7
        dilations = [1, 2, 4]
        alpha_custom, alpha_proto, alpha_gate, scale = 0.039, 1, 0.23, (1 / batch_size)
    elif namespace == 'MF':
        kernel_size = 3
        dilations = [1, 2, 4]
        alpha_custom, alpha_proto, alpha_gate, scale = 0.042, 1, 0.245, (1 / batch_size)

    # 设置随机种子
    utils.set_random_seed(seed) 

    # 加载模型
    model = Model(prototype_feats, train_freq, rgcn_dim, labels_index, input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=len(go2id), kernel_size=kernel_size, dilations=dilations).to(device)

    # 优化器参数设置：不同的模块设置不同学习率
    gate_params = [model.gate_k, model.gate_c_raw]
    prototype_params = list(model.prototype_Module.parameters())
    special_param_ids = set(id(p) for p in gate_params + prototype_params)
    default_params = [p for p in model.parameters() if id(p) not in special_param_ids]
    optimizer = optim.AdamW([
        {'params': gate_params, 'lr': base_lr * 10.0}, # 门控模块
        {'params': prototype_params, 'lr': base_lr * 3.0}, # 原型模块
        {'params': default_params} # 其余所有模块 — 动态打包，继承最外层的默认学习率
    ], lr=base_lr)

    # 设置学习率策略
    lr_lambda = partial(utils.lr_scheduler, 
                    warmup_epochs=warmup_epochs, 
                    max_epochs=epochs)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_dataset = Dataset(datasets_path, namespace, split='train')
    valid_dataset = Dataset(datasets_path, namespace, split='valid')
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset) 
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True, num_workers=2, collate_fn=collator(go2id, train_protein_feats, train_protein_residue_feats), worker_init_fn=utils.seed_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler, drop_last=False, num_workers=2, collate_fn=collator(go2id, valid_protein_feats, valid_protein_residue_feats),  worker_init_fn=utils.seed_worker)

    print('start traing......')
    for epoch in range(epochs):

        # --- 门控渐进式解冻逻辑 ---
        if gate_freeze_epochs > 0:
            if epoch < gate_freeze_epochs:
                # 冻结阶段：门控参数不参与梯度更新
                model.gate_k.requires_grad_(False)
                model.gate_k.grad = None
                model.gate_c_raw.requires_grad_(False)
                model.gate_c_raw.grad = None
            elif epoch == gate_freeze_epochs:
                # 解冻：恢复门控参数的梯度，并重置优化器状态（避免冻结期累积的动量干扰）
                model.gate_k.requires_grad_(True)
                model.gate_c_raw.requires_grad_(True)
                gate_opt_state = optimizer.state.get(model.gate_k, None)
                if gate_opt_state is not None:
                    del optimizer.state[model.gate_k]
                gate_opt_state = optimizer.state.get(model.gate_c_raw, None)
                if gate_opt_state is not None:
                    del optimizer.state[model.gate_c_raw]
                print(f'>>> Epoch {epoch}: Gate parameters unfrozen, LR={optimizer.param_groups[0]["lr"]:.2e}')

        # 训练
        train(model, optimizer, train_loader, epoch, device, alpha_custom, alpha_proto, alpha_gate, scale)
        scheduler.step()

        # 验证
        all_probs, all_custom_probs, all_proto_probs, all_labels = valid(model, valid_loader, epoch, device, alpha_custom, alpha_proto, alpha_gate, scale, ic)
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
    parser.add_argument('--batch_size', type=int, default=10, help='batch size') # CC:32, BP:4, MF:14
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--path', type=str, default="./new/data/", help='datasets path')
    parser.add_argument('--result_path', type=str, default="/archive/hot3/fty/result/", help='result save path')
    parser.add_argument('--namespace', default='MF', type=str, help='[BP/CC/MF]')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    
    main(args, config)