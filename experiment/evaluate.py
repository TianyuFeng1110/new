import argparse
import os
import torch
from ruamel.yaml import YAML
import sys

import numpy as np
import utils as utils

from custom import get_loader

def get_model(model_type, **kwargs):
    if model_type == 0:
        from models.custom_model import Model
        model = Model(
            input_dim=kwargs['esm_dim'], hidden_dim=kwargs['hidden_dim'], num_classes=kwargs['num_classes']
        ).to(kwargs['device'])

    elif model_type == 1:
        # from models.model import Model
        from models.verify_gate_model import Model
        model = Model(
            input_dim=kwargs['esm_dim'], hidden_dim=kwargs['hidden_dim'], num_classes=kwargs['num_classes'], go_freq=kwargs['go_freq'], 
            raw_feats=kwargs['stacked_feats'], go_embeddings=kwargs['go_embeddings'], queue_size=kwargs['queue_size'], queue_indices=kwargs['queue_indices']
        ).to(kwargs['device'])

    elif model_type == 2:
        # from models.prototype_model import Model
        # model = Model(
        #     input_dim=kwargs['esm_dim'], hidden_dim=kwargs['hidden_dim'], num_classes=kwargs['num_classes'],
        #     # raw_feats=kwargs['stacked_feats'],  queue_size=kwargs['queue_size'], queue_indices=kwargs['queue_indices']#, go_embeddings=kwargs['go_embeddings'],
        # ).to(kwargs['device'])
        from models.test_model import PrototypeNet
        model = PrototypeNet(input_dim=kwargs['esm_dim'], hidden_dim=kwargs['hidden_dim'], num_classes=kwargs['num_classes'], go_freq=kwargs['go_freq']).to(kwargs['device'])

    return model

def main(args, config):
    device = torch.device(args.device)
    
    datasets_path = args.path
    namespace = args.namespace
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    num_classes = config['num_classes']
    features_path = config['features_path']
    model_type = config['model_type']
    checkpoint = config['checkpoint']
    queue_size = config['queue_size']
    match model_type:
        case 0:
            checkpoint_path = os.path.join(config['checkpoint_path'], 'custom', checkpoint)
        case 1:
            checkpoint_path = os.path.join(config['checkpoint_path'], 'custom+prototype', checkpoint)
        case 2:
            checkpoint_path = os.path.join(config['checkpoint_path'], 'prototype1', checkpoint)
        case _:
            raise ValueError
    
    # 加载模型初始化所需的特征与原型数据
    protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'), 
        weights_only=True, map_location='cpu'
    )
    test_protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'), 
        weights_only=True, map_location='cpu'
    )
    
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    prototype_index, proto_idx_mask = utils.get_prototype_index_tensor(train_seq_data)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1
    valid_mask = prototype_index.sum(dim=0) > 0
    go_freq = utils.compute_go_term_frequency(proto_idx_mask, len(train_seq_data))

    
    # 实例化模型并完成原型特征初始化
    stacked_feats = torch.stack([protein_feats[k] for k in sorted(protein_feats.keys(), key=int)], dim=0).to(device)
    go_embeddings = torch.load(os.path.join(datasets_path, f'{namespace.lower()}_go_feats.pt'), weights_only=True, map_location='cpu')
    queue_indices = utils.get_queue_index(stacked_feats, prototype_index, namespace, datasets_path, queue_size)

    model = get_model(model_type=model_type, esm_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes, prototype_index=prototype_index, 
                      proto_idx_mask=proto_idx_mask, go_freq=go_freq, stacked_feats=stacked_feats, go_embeddings=go_embeddings, 
                      queue_indices=queue_indices, queue_size=queue_size, device=device)
    
    # 加载训练好的模型权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Successfully loaded checkpoint from: {checkpoint_path}")
    
    # 获取测试集的数据加载器 (这里 mode 传入 'test')
    _, test_loader = get_loader(
        datasets_path, namespace, args.batch_size, 
        protein_feats, test_protein_feats, num_classes, mode='test', valid_mask=valid_mask
    )

    # =========================================================
    #                           评估
    # =========================================================
    utils.eval_func_generalizability(model, test_loader, device, go_freq)
    utils.eval_term_freq_generalizability(model, test_loader, device, go_freq)
    # utils.analyze_queue_coverage(model, go_freq)
    # print('对每个类分别统计：正样本与原型、负样本与原型的余弦相似度分布: ')
    # utils.analyze_proto_quality(model, test_loader, device, go_freq, num_classes)
    # print('计算原型之间的两两余弦相似度，检查是否坍缩(prototypes = F.normalize(model._compute_prototypes(), p=2, dim=-1))')
    # utils.analyze_proto_collapse(model, go_freq)
    # print('计算go feats的分离度(和余弦相似度)')
    # utils.analyze_go_separation(model, go_freq)
    # utils.diagnose_prototype_failure(model, test_loader, device, go_freq, thresholds=(0.01, 0.05))
    # # utils.eval_term_freq_generalizability(model, test_loader, device, go_freq)
    # =========================================================
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CC Model')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/eval.yml', help='config yml')
    parser.add_argument('--path', type=str, default="./data_tale/TALE/", help='datasets path')
    parser.add_argument('--namespace', default='CC', type=str, help='[BP/CC/MF]')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')

    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
        
    main(args, config)