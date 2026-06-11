import argparse
import os
import new.utils as utils
import torch
import torch.optim as optim

from ruamel.yaml import YAML
from new.models.cc_model import Model
from new.datasets.cc_dataset import Dataset
from new.datasets.cc_collator import collator
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
def valid(model, loader, epoch, device):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Valid Epoch: [{}]'.format(epoch)

    all_probs, all_labels = [],[]
    for batch_idx, (batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):

        logits, loss = model(batch_residue_feats.to(device), mask.to(device), labels.to(device))

        all_probs.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(labels.detach().cpu())
        metric_logger.update(total_loss=loss.item())

    metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy())
    print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC']))

    return all_probs, all_labels

def train(model, optimizer, loader, epoch, device):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('total_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    for batch_idx, (batch_residue_feats, mask, labels) in enumerate(metric_logger.log_every(loader, print_freq=1, header=header)):
        optimizer.zero_grad(set_to_none=True)

        logits, loss = model(batch_residue_feats.to(device), mask.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
        
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])  
        metric_logger.update(total_loss=loss.item())
        
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
    residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'train_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    test_residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'test_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')

    # 设置随机种子
    utils.set_random_seed(seed) 

    train_loader, valid_loader = get_loader(datasets_path, namespace, batch_size, residue_feats, test_residue_feats, num_classes, mode)

    model = Model(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

    print('start traing......')
    for epoch in range(epochs):

        # 训练
        train(model, optimizer, train_loader, epoch, device)
        scheduler.step()

        # 验证
        all_probs, all_labels = valid(model, valid_loader, epoch, device)
    # TODO 下面这块还需要修改，把不同概率的评估方法重写一遍。
    # utils.draw_frequencies_AUPRC(torch.cat(all_probs, dim=0).numpy(), torch.cat(all_probs_averge, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), valid_freq, save_path=result_path, namespace=namespace) # 训练结束绘制结果曲线图

'''
baseline: 平均池化路线 NOTE all_cc是mlp的
'''
if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./new/config/cc.yml', help='config yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size') # CC:32, BP:4, MF:14
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

    # for lr in [1e-5, 2e-5, 5e-5, 1e-4]:
    #     config['base_lr'] = lr
    #     config.update(vars(args))
    #     print('------------------> lr:', lr)

    #     main(args, config)