import argparse
import os
import utils as utils
import torch
import torch.optim as optim
import numpy as np

from ruamel.yaml import YAML
from models.custom_model import Model
from datasets.cc_dataset import Dataset
from datasets.cc_collator import collator
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

def get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode, valid_mask):
    dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='train', valid_mask=valid_mask)

    if mode == 'train': valid_dataset = dataset.val_dataset
    else: valid_dataset = Dataset(datasets_path, namespace, train_mode=mode, dataset_mode='test', valid_mask=valid_mask)

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

    # metric = utils.calculate_metrics(torch.cat(all_labels, dim=0).numpy(), torch.cat(all_probs, dim=0).numpy())
    # print("Averaged stats(Valid): {}, Fmax: {:.4f}, micro AUPRC: {:.4f}".format(metric_logger.global_avg(), metric['Fmax'], metric['micro_AUPRC']))
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    macro_aupr = utils.macro_auprc(all_labels, all_probs)
    print("Averaged stats: {}, macro aupr: {:.4f}".format(metric_logger.global_avg(), macro_aupr))

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
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1 # 确保所有数据都注释了根go term的功能
    valid_mask = prototype_index.sum(dim=0) > 0  # 形状为 (标签类别数,) 的布尔 Tensor
    num_classes = valid_mask.sum().item()  # 更新 num_classes 为有效标签数
    # residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'train_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    # test_residue_feats = torch.load(os.path.join(features_path, 'residue_feats', f'test_{namespace.lower()}_residue_feats.pt'), weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'), weights_only=True, map_location='cpu')
    hier_reg_lambda = config.get('hier_reg_lambda', 0.1)
    
    _, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    go_freq = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))[valid_mask]

    # 设置随机种子
    utils.set_random_seed(seed) 

    train_loader, valid_loader = get_loader(datasets_path, namespace, batch_size, protein_feats, test_protein_feats, num_classes, mode, valid_mask)

    model = Model(
        input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=base_lr * 0.01)

    _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
    parent_indices = torch.from_numpy(_edges[:, 0].copy()).long().to(device)
    child_indices = torch.from_numpy(_edges[:, 1].copy()).long().to(device)
    del _edges  # 立即释放 numpy 内存

    # utils.print_loss_gradients(model, train_loader, parent_indices, child_indices, device)
    # hier_scale = utils.get_hier_scale(model, train_loader, parent_indices, child_indices, device) # 静态标定法缩放梯度
    scale = {'custom_scale': 1, 'hier_scale': 0}
    
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
        torch.save(save_obj, os.path.join("/archive/hot5/fty/checkpoints/TALE/custom/", 'checkpoint_%02d.pth'%epoch))  
        utils.eval_func_generalizability(model, valid_loader, device, go_freq, None)
        if epoch % 9 == 0:
            utils.eval_term_freq_generalizability(model, valid_loader, device, go_freq, prototypes=None)

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/custom.yml', help='config yml')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size') # CC:32, BP:4, MF:14
    parser.add_argument('--epochs', type=int, default=50, help='epoch')
    # parser.add_argument('--path', type=str, default="./data_tale/CAFA3/", help='datasets path')
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
"""
只有MLP，去掉原型网络，作为最开始的比较基准
"""