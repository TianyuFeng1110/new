import argparse
import os
import sys
import torch
import numpy as np
import json
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, SequentialSampler

# 将项目根目录加入搜索路径，确保从任意位置运行都能找到 utils / datasets / models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils as utils
from datasets.dataset import Dataset
from datasets.collator import collator


def get_test_loader(datasets_path, namespace, batch_size, test_protein_feats, num_classes, valid_mask):
    """构建测试集 DataLoader。"""
    dataset = Dataset(datasets_path, namespace, train_mode='test', dataset_mode='test', valid_mask=valid_mask)
    sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, drop_last=False, num_workers=2,
        collate_fn=collator(num_classes, test_protein_feats),
        worker_init_fn=utils.seed_worker,
    )


def main(args, config):
    device = torch.device(args.device)

    namespace = config['namespace']
    esm_dim = config['esm_dim']
    hidden_dim = config['hidden_dim']
    model_type = config['model_type']
    dataset_name = config['dataset']
    is_zero_shot = config['zero_shot']
    features_path = os.path.join(config['features_path'], dataset_name)
    checkpoint_path = os.path.join(config['checkpoint_path'], namespace, dataset_name)
    dataset_name = config['dataset']
    datasets_path = os.path.join(config['datasets_path'], dataset_name)

    # ---- 1. 加载基础数据（所有模型共用） ----
    train_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"train_seq_{namespace.lower()}"))
    test_seq_data = utils.load_data_from_pkl(os.path.join(datasets_path, f"test_seq_{namespace.lower()}"))

    # 构建 (N_protein × num_classes) 的二值标注矩阵并过滤无效类
    num_classes_raw = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_matrix_1_sparse.npy')).shape[0]
    prototype_index = utils.get_prototype_index(train_seq_data, num_classes_raw)
    prototype_index[prototype_index[:, 0] == 0, 0] = 1          # 确保根 GO term 至少有一条标注
    valid_mask = prototype_index.sum(dim=0) > 0
    if not is_zero_shot: prototype_index = prototype_index[:, valid_mask]             # (N_protein, num_classes)
    num_classes = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_matrix_1_sparse.npy')).shape[0]

    # 训练/测试蛋白质特征
    protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'train_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')
    test_protein_feats = torch.load(
        os.path.join(features_path, 'protein_feats', f'test_{namespace.lower()}_protein_feats.pt'),
        weights_only=True, map_location='cpu')

    # GO term 频率 & 样本计数
    go_freq, class_counts = utils.compute_go_term_frequency(prototype_index.T, len(train_seq_data))

    # ---- 2. 测试 DataLoader ----
    test_loader = get_test_loader(datasets_path, namespace, args.batch_size,
                                  test_protein_feats, num_classes, (None if is_zero_shot else valid_mask))

    # ---- 3. 按 model_type 构建模型、加载权重 ----
    prototypes = None

    if model_type == 0:
        # ---- 纯 MLP 基线 ----
        from models.custom_model import Model
        ckpt = torch.load(os.path.join(checkpoint_path, 'custom', config['checkpoint_custom']), map_location=device)
        custom_state = utils.extract_state_dict(ckpt)
        # 按 checkpoint 键自动识别分类头版本：新版（自由残差 + 祖先加权构造）含 w_free，
        # 旧版经典 MLP 消融头含 classifier.0.weight
        use_ancestor_weighting = 'w_free' in custom_state
        if use_ancestor_weighting:
            # 复用 custom.py 的新初始化流程构建层级先验（与训练入口保持一致）；
            # ancestor_mask/ic/class_counts 为 buffer，加载 checkpoint 后以训练时保存的值为准，
            # 此处计算仅决定加载前的模型结构与可学习参数初始值，不影响评估结果
            from custom import build_ancestor_prior
            _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
            ancestor_mask, ic, alpha_lambda_init, ic_gamma_init = build_ancestor_prior(
                datasets_path, namespace, dataset_name, _edges, train_seq_data,
                class_counts, valid_mask, is_zero_shot)
            del _edges  # 立即释放 numpy 内存
            model = Model(
                input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
                use_ancestor_weighting=True, ancestor_mask=ancestor_mask, ic=ic,
                class_counts=class_counts, alpha_lambda_init=alpha_lambda_init,
                ic_gamma_init=ic_gamma_init,
            ).to(device)
        else:
            model = Model(input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
                          use_ancestor_weighting=False).to(device)
        model.load_state_dict(custom_state)
        model.eval()

    elif model_type == 1:
        # ---- 门控融合: MLP + PrototypeNet ----
        from models.model import Model
        from models.custom_model import Model as CustomModel
        from models.prototype_model import PrototypeNet

        # 原型网络所需的层次结构数据
        go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))
        parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
        if not is_zero_shot: parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
        hop_counts = utils.get_ancestor_hop_matrix(_edges)
        if not is_zero_shot: hop_counts = hop_counts[valid_mask][:, valid_mask]
        obo_path = os.path.join(datasets_path, 'go-basic.obo')
        ic = utils.compute_ic(go2id, train_seq_data, obo_path)
        if not is_zero_shot: ic = ic[valid_mask]
        # lambda/beta/smooth_tau 为模型内可学习参数，checkpoint 中已含学习后的值；
        # 此处传入的数据驱动初始值仅决定加载前的结构，不影响评估结果
        lambda_init, beta_init, tau_init = utils.compute_smooth_param_inits(hop_counts, ic, class_counts)
        tau_min = float(config.get('tau_min', 0.0))
        tau_g = float(config.get('tau_g'))   # 门控半饱和超参数

        # 加载两个预训练子模型
        mlp_ckpt = torch.load(os.path.join(checkpoint_path, 'custom', config['checkpoint_custom']), map_location=device)
        mlp_state = utils.extract_state_dict(mlp_ckpt)
        # 按 checkpoint 键自动识别 MLP 分类头版本（新版含 w_free，旧版含 classifier.0.weight）；
        # 新版复用 custom.py 的 build_ancestor_prior 初始化（同 model_type 0，
        # ancestor_mask/ic/class_counts 为 buffer，加载 checkpoint 后以训练时保存的值为准；
        # 变量带 _mlp 后缀，避免覆盖本分支后续原型分析所需的 ic / hop_counts 等）
        if 'w_free' in mlp_state:
            from custom import build_ancestor_prior
            ancestor_mask_mlp, ic_mlp, alpha_lambda_init_mlp, ic_gamma_init_mlp = build_ancestor_prior(
                datasets_path, namespace, dataset_name, _edges, train_seq_data,
                class_counts, valid_mask, is_zero_shot)
            mlp_model = CustomModel(
                input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
                use_ancestor_weighting=True, ancestor_mask=ancestor_mask_mlp, ic=ic_mlp,
                class_counts=class_counts, alpha_lambda_init=alpha_lambda_init_mlp,
                ic_gamma_init=ic_gamma_init_mlp,
            ).to(device)
        else:
            mlp_model = CustomModel(input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
                                    use_ancestor_weighting=False).to(device)
        proto_model = PrototypeNet(
            input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
            frequency=go_freq, parents_matrix=parents_matrix,
            class_counts=class_counts, hop_counts=hop_counts, ic=ic,
            smooth_tau=tau_init, lambda_init=lambda_init, beta_init=beta_init, tau_min=tau_min,
        ).to(device)

        proto_ckpt = torch.load(os.path.join(checkpoint_path, 'prototype', config['checkpoint_proto']), map_location=device)
        mlp_model.load_state_dict(mlp_state)
        proto_state = utils.extract_state_dict(proto_ckpt)
        if 'tau_min' not in proto_state:  # 兼容旧 checkpoint（无 tau_min buffer）
            proto_state['tau_min'] = torch.tensor(0.0)
        proto_model.load_state_dict(proto_state)

        model = Model(mlp_model, proto_model, class_counts, tau_g=tau_g).to(device)

        # 预计算原型
        indices = sorted(protein_feats.keys(), key=int)
        all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
        prototypes = proto_model._get_prototypes(all_feats, prototype_index.to(device))

    elif model_type == 2:
        # ---- 纯 PrototypeNet ----
        from models.prototype_model import PrototypeNet

        go2id = utils.load_data_from_pkl(os.path.join(datasets_path, f"{namespace.lower()}_go_1.pickle"))   
        parents_matrix, _ = utils.get_go_adjacency_matrices(go2id)
        if not is_zero_shot: parents_matrix = parents_matrix[valid_mask][:, valid_mask]
        _edges = np.load(os.path.join(datasets_path, f'{namespace.lower()}_label_regular_1.npy'))
        hop_counts = utils.get_ancestor_hop_matrix(_edges)
        if not is_zero_shot: hop_counts = hop_counts[valid_mask][:, valid_mask]
        obo_path = os.path.join(datasets_path, 'go-basic.obo')
        ic = utils.compute_ic(go2id, train_seq_data, obo_path)
        if not is_zero_shot: ic = ic[valid_mask]
        # lambda/beta/smooth_tau 为模型内可学习参数，checkpoint 中已含学习后的值；
        # 此处传入的数据驱动初始值仅决定加载前的结构，不影响评估结果
        lambda_init, beta_init, tau_init = utils.compute_smooth_param_inits(hop_counts, ic, class_counts)
        tau_min = float(config.get('tau_min', 0.0))

        model = PrototypeNet(
            input_dim=esm_dim, hidden_dim=hidden_dim, num_classes=num_classes,
            frequency=go_freq, parents_matrix=parents_matrix,
            class_counts=class_counts, hop_counts=hop_counts, ic=ic,
            smooth_tau=tau_init, lambda_init=lambda_init, beta_init=beta_init, tau_min=tau_min,
        ).to(device)

        ckpt = torch.load(os.path.join(checkpoint_path, 'prototype', config['checkpoint_proto']), map_location=device)
        proto_state = utils.extract_state_dict(ckpt)
        if 'tau_min' not in proto_state:  # 兼容旧 checkpoint（无 tau_min buffer）
            proto_state['tau_min'] = torch.tensor(0.0)
        model.load_state_dict(proto_state)
        model.eval()

        # 预计算原型
        indices = sorted(protein_feats.keys(), key=int)
        all_feats = torch.stack([protein_feats[k] for k in indices], dim=0).to(device)
        prototypes = model._get_prototypes(all_feats, prototype_index.to(device))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Successfully loaded model_type={model_type}")

    # =========================================================
    #                           评估
    # =========================================================

    # 零样本 top-k 注释评估（TALE 原文 Supp. Tables S18/S23/S28 协议，零样本类内 top-k）
    utils.eval_zero_shot_topk(model, test_loader, device, class_counts, prototypes, model_type, max_k=10)

    # 按训练正样本数分桶
    utils.eval_count_bucket_comparison(model, test_loader, device, prototypes, class_counts, model_type=model_type)

    if model_type == 1 and prototypes is not None:

        # τ_g 敏感性扫描
        # utils.eval_tau_g_sensitivity(model, test_loader, device, prototypes, class_counts)

        # 零样本类的祖先溯源（机制解释：原型如何从祖先借力）
        # utils.eval_zero_shot_ancestry(model.proto_model, hop_counts, class_counts, go2id)

        # 可解释性: 层级违反率（MLP vs Proto vs 融合 的 true-path 违反占比）
        utils.eval_hier_violation_rate(model, test_loader, device, prototypes,
                                       class_counts, hop_counts, model_type=1, eps=1e-4)
        
        # 可解释性: 零样本案例研究（Proto 从祖先借力 → 命中 MLP 盲区的完整证据链）
        case_records = utils.eval_zero_shot_case_study(model, test_loader, device, prototypes, class_counts, hop_counts, go2id, n_cases=3, top_anc=3, test_seq_data=test_seq_data, protein_id_key='ac')
        # 保存案例记录供绘图脚本使用（仅用于 plot_case_figure.py 读取，避免绘图时重跑推理）
        json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, f'case_records_{dataset_name}_{namespace}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(case_records, f, ensure_ascii=False, indent=2)
        print(f"  案例记录已保存至 {json_path}")

    # 原型嵌入层级一致性分析，仅原型网络相关模型（model_type 1 / 2）有原型与层次结构数据
    if model_type in (1, 2) and prototypes is not None:
        # 取原型网络本体（model_type 1 时原型网络是融合模型的子模块）
        proto_net = model.proto_model if model_type == 1 else model

        # 计算未平滑的纯均值原型，作为"平滑前"对照
        with torch.no_grad():
            proto_net.eval()
            s_feats = proto_net.mlp(all_feats)                       # (N, D)
            proto_mask = prototype_index.to(device).float()          # (N, C)
            mean_feats = proto_mask.T @ s_feats                      # (C, D)
            counts = proto_mask.sum(dim=0).clamp(min=1)              # (C,)
            prototypes_mean = mean_feats / counts.unsqueeze(1)       # (C, D)

        # ---- 协议分流：BP 类数巨大（~2万）走祖先闭包子图近似；CC/MF 走全图精确 ----
        if namespace == 'BP':
            # BP: 分层采样目标节点 + 祖先闭包，构成祖先闭合子图
            S, target_mask = utils.sample_ancestor_closed_subgraph(
                hop_counts, class_counts, n_target=800, seed=42)
            S_cpu = torch.from_numpy(S)                    # CPU 索引（hop/ic/counts 在 CPU）
            S_dev = S_cpu.to(device)                       # GPU 索引（prototypes 在 device）

            # 将全部相关量切片到子图（祖先闭合保证 DAG 距离/语义相似度精确）
            hop_sub = hop_counts[S_cpu][:, S_cpu]
            ic_sub = ic[S_cpu]
            counts_sub = class_counts[S_cpu]
            proto_stable_sub = prototypes[S_dev]
            proto_mean_sub = prototypes_mean[S_dev]
            target_mask_t = torch.from_numpy(target_mask)

            print(f"[BP 子图协议] |S|={len(S)}, 目标节点 {int(target_mask.sum())}")

            # Mantel 检验（子图全上三角，秩聚合加速）
            utils.eval_prototype_mantel(
                prototypes_stable=proto_stable_sub, prototypes_mean=proto_mean_sub,
                hop_counts=hop_sub, permutations=999)

            # 语义相似度相关（分箱仅统计目标节点）
            utils.eval_prototype_semantic_similarity(
                prototypes_stable=proto_stable_sub, prototypes_mean=proto_mean_sub,
                hop_counts=hop_sub, ic=ic_sub, class_counts=counts_sub,
                target_mask=target_mask_t)

            # kNN 谱系纯度（分箱仅统计目标节点）
            utils.eval_prototype_knn_purity(
                prototypes_stable=proto_stable_sub, prototypes_mean=proto_mean_sub,
                hop_counts=hop_sub, class_counts=counts_sub, k=5, h=2,
                target_mask=target_mask_t)
        else:
            # CC / MF: 全图精确计算
            # Mantel 检验（原型余弦距离 vs DAG 距离）
            utils.eval_prototype_mantel(prototypes_stable=prototypes, prototypes_mean=prototypes_mean, hop_counts=hop_counts, permutations=999,)

            # 语义相似度相关（原型余弦相似度 vs Resnik / Lin）
            utils.eval_prototype_semantic_similarity(prototypes_stable=prototypes,prototypes_mean=prototypes_mean,hop_counts=hop_counts,ic=ic, class_counts=class_counts,)

            # kNN 谱系纯度
            utils.eval_prototype_knn_purity(prototypes_stable=prototypes, prototypes_mean=prototypes_mean, hop_counts=hop_counts, class_counts=class_counts, k=5, h=2,)

    # =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CC Model')
    parser.add_argument('--device', type=str, default='cuda', help='device id')
    parser.add_argument('--config', type=str, default='./config/eval.yml', help='config yml')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')

    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open(args.config, 'r') as f:
        config = yaml.load(f)
        
    main(args, config)