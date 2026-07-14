"""
验证实验：定位罕见类原型失效的根因

假设链：
  H1: 罕见类的原型代表蛋白数量不足 → 原型质量差 (高方差)
  H2: 原型质量差 → GO 特征学习不稳定 (低信噪比梯度)
  H3: 罕见类的 GO 特征未学到有意义的方向 → pos/neg 无法区分

实验设计：
  E1: 原型Bootstrap稳定性分析 (验证H1)
  E2: 代表蛋白数量 vs 类级别AUPRC (验证H1)
  E3: 常见类子采样消融实验 (验证H1→H2的因果链)
  E4: GO 特征梯度信噪比分析 (验证H2)
  E5: 原型-正样本对齐度分析 (验证H3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score

# ============================================================
# E1: 原型 Bootstrap 稳定性分析
# ============================================================
@torch.no_grad()
def experiment1_prototype_stability(model, go_freq, num_bootstrap=20):
    """
    对每个类：多次从代表蛋白中随机子采样，计算原型，测量原型间的余弦相似度。
    
    直觉：如果罕见类的原型在子采样后方向变化大 (低相似度)，说明原型不稳定。
    
    输出：
      - 每类的 bootstrap 原型间平均成对余弦相似度
      - 按频率分桶汇总
    """
    print("\n" + "=" * 60)
    print("E1: 原型 Bootstrap 稳定性分析")
    print("=" * 60)
    print("直觉: 罕见类代表蛋白少 → Bootstrap子采样波动大 → 原型稳定性低")
    
    raw_feats = model.raw_feats  # (num_proteins, dim)
    queue_indices = model.queue_indices  # (num_classes, queue_size)
    momentum_protein_mlp = model.momentum_protein_mlp
    momentum_go_feats = model.momentum_go_feats
    num_classes = queue_indices.shape[0]
    
    freq_np = go_freq.cpu().numpy()
    
    stability_scores = np.zeros(num_classes)
    
    for c in range(num_classes):
        # 获取当前类的有效代表蛋白索引
        valid_mask = queue_indices[c] >= 0
        valid_indices = queue_indices[c][valid_mask]  # 实际代表蛋白索引
        # 安全裁剪: 确保索引不超出 raw_feats 范围 (queue_indices 可能来自旧缓存)
        valid_indices = valid_indices.clamp(max=raw_feats.shape[0] - 1)
        n_valid = valid_indices.numel()
        
        if n_valid < 2:
            stability_scores[c] = 1.0  # 只有1个代表，无法bootstrap
            continue
        
        prototypes = []
        for _ in range(num_bootstrap):
            # Bootstrap: 有放回采样 n_valid 个
            boot_idx = valid_indices[torch.randint(0, n_valid, (n_valid,))]
            boot_feats = raw_feats[boot_idx]
            boot_feats = momentum_protein_mlp(boot_feats)
            
            # 哈达玛积 + 平均 → bootstrap 原型
            func_all = boot_feats * momentum_go_feats[c].unsqueeze(0)
            proto = func_all.mean(dim=0)
            proto = F.normalize(proto, p=2, dim=-1)
            prototypes.append(proto)
        
        # 计算 bootstrap 原型间的成对余弦相似度
        proto_stack = torch.stack(prototypes)  # (B, D)
        sim_matrix = proto_stack @ proto_stack.T  # (B, B)
        # 排除对角线
        diag_mask = ~torch.eye(num_bootstrap, dtype=torch.bool)
        stability_scores[c] = sim_matrix[diag_mask].mean().item()
    
    # 按频率分桶
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1%-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        if mask.sum() > 0:
            mean_stab = stability_scores[mask].mean()
            std_stab = stability_scores[mask].std()
            # 也统计平均代表蛋白数
            classes_in = np.where(mask)[0]
            avg_reps = np.mean([(queue_indices[i] >= 0).sum().item() for i in classes_in])
            print(f"  {name} (n={mask.sum()}): stability={mean_stab:.4f}±{std_stab:.4f}, "
                  f"avg_reps={avg_reps:.1f}")
    
    return stability_scores


# ============================================================
# E2: 代表蛋白数量 vs 类级别 AUPRC
# ============================================================
@torch.no_grad()
def experiment2_reps_vs_performance(model, loader, device, go_freq, num_classes):
    """
    统计每类的有效代表蛋白数 (非padding)，与该类的 AUPRC 做相关性分析。
    
    输出：
      - 每个类的代表蛋白数、GO频率、per-class AUPRC
      - 按代表蛋白数分桶的 AUPRC 分布
      - Spearman 相关系数
    """
    from scipy.stats import spearmanr
    
    print("\n" + "=" * 60)
    print("E2: 代表蛋白数量 vs 类级别性能")
    print("=" * 60)
    print("直觉: 代表蛋白越少 → 原型越不可靠 → per-class AUPRC 越低")
    
    # 统计每类的有效代表蛋白数
    queue_indices = model.queue_indices
    n_reps = np.array([(queue_indices[c] >= 0).sum().item() for c in range(num_classes)])
    
    # 收集 per-class 预测
    all_probs = []
    all_labels = []
    
    model.eval()
    for protein_feats, labels, indices in loader:
        protein_feats = protein_feats.to(device)
        logits, _, _ = model(protein_feats, labels.to(device))
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    freq_np = go_freq.cpu().numpy()
    
    # 计算 per-class AUPRC
    per_class_auprc = []
    for c in range(num_classes):
        if all_labels[:, c].sum() == 0:
            per_class_auprc.append(np.nan)
        else:
            auprc = average_precision_score(all_labels[:, c], all_probs[:, c])
            per_class_auprc.append(auprc)
    per_class_auprc = np.array(per_class_auprc)
    
    # 过滤掉无正样本的类
    valid = ~np.isnan(per_class_auprc)
    
    # Spearman 相关系数
    r_nreps, p_nreps = spearmanr(n_reps[valid], per_class_auprc[valid])
    r_freq, p_freq = spearmanr(freq_np[valid], per_class_auprc[valid])
    
    print(f"  Spearman r (n_reps vs AUPRC): {r_nreps:.4f} (p={p_nreps:.2e})")
    print(f"  Spearman r (go_freq vs AUPRC): {r_freq:.4f} (p={p_freq:.2e})")
    
    # 按代表蛋白数分桶
    for lo, hi, label in [(1, 4, "[1,4]"), (5, 8, "[5,8]"), (9, 12, "[9,12]"), (13, 16, "[13,16]")]:
        mask = (n_reps >= lo) & (n_reps <= hi)
        mask &= valid
        if mask.sum() > 0:
            print(f"  n_reps {label}: n={mask.sum()}, "
                  f"mean_AUPRC={per_class_auprc[mask].mean():.4f}, "
                  f"mean_freq={freq_np[mask].mean():.6f}")
    
    return n_reps, per_class_auprc


# ============================================================
# E3: 常见类代表蛋白子采样消融实验
# ============================================================
@torch.no_grad()
def experiment3_subsample_ablation(model, loader, device, go_freq, num_classes):
    """
    对常见类，人为将其代表蛋白数子采样到与罕见类相同的数量，
    重新计算原型，测量性能变化。
    
    如果常见类在子采样后性能下降到罕见类水平，
    则证明 "代表蛋白数量不足 → 原型质量差 → 性能差" 是因果链。
    """
    print("\n" + "=" * 60)
    print("E3: 常见类代表蛋白子采样消融实验")
    print("=" * 60)
    print("直觉: 如果常见类子采样代表蛋白后性能骤降，证明代表蛋白数=根因")
    
    queue_indices = model.queue_indices
    raw_feats = model.raw_feats
    momentum_protein_mlp = model.momentum_protein_mlp
    momentum_go_feats = model.momentum_go_feats
    freq_np = go_freq.cpu().numpy()
    
    # 定义罕见类代表蛋白数的典型值 (取罕见类的中位数)
    rare_mask = freq_np < 0.01
    rare_n_reps = int(np.median([(queue_indices[c] >= 0).sum().item() 
                                  for c in np.where(rare_mask)[0]]))
    print(f"  罕见类代表蛋白数中位数: {rare_n_reps}")
    
    # 选常见类 (频率 > 10%)
    freq_mask = freq_np >= 0.1
    freq_classes = np.where(freq_mask)[0]
    
    if len(freq_classes) == 0:
        print("  没有足够的高频类进行消融实验")
        return
    
    # 原始原型
    orig_prototypes = model._compute_prototypes()
    orig_prototypes_norm = F.normalize(orig_prototypes, p=2, dim=-1)
    
    # 收集原始预测 (仅对常见类)
    all_orig_cos_pos = []
    all_subsampled_cos_pos = []
    
    # 遍历几个 batch 收集统计
    n_batches = 0
    for protein_feats, labels, indices in loader:
        if n_batches >= 10:
            break
        n_batches += 1
        
        protein_feats = protein_feats.to(device)
        p_feats = model.protein_mlp(protein_feats)
        go_feats = model.go_feats
        
        func_specific = p_feats.unsqueeze(1) * go_feats.unsqueeze(0)
        func_norm = F.normalize(func_specific, p=2, dim=-1)
        
        for c in freq_classes[:5]:  # 取前5个常见类测试
            # 原始余弦相似度 (正样本)
            mask_pos = labels[:, c] == 1
            if mask_pos.sum() == 0:
                continue
            
            orig_cos = (func_norm[mask_pos, c] * orig_prototypes_norm[c]).sum(dim=-1)
            all_orig_cos_pos.extend(orig_cos.cpu().tolist())
            
            # 子采样原型: 只用 rare_n_reps 个代表蛋白
            valid_indices = queue_indices[c][queue_indices[c] >= 0]
            valid_indices = valid_indices.clamp(max=raw_feats.shape[0] - 1)  # 安全裁剪
            if len(valid_indices) > rare_n_reps:
                sub_idx = valid_indices[torch.randperm(len(valid_indices))[:rare_n_reps]]
            else:
                sub_idx = valid_indices
            
            sub_feats = raw_feats[sub_idx]
            sub_feats = momentum_protein_mlp(sub_feats)
            func_sub = sub_feats * momentum_go_feats[c].unsqueeze(0)
            sub_proto = F.normalize(func_sub.mean(dim=0), p=2, dim=-1)
            
            sub_cos = (func_norm[mask_pos, c] * sub_proto).sum(dim=-1)
            all_subsampled_cos_pos.extend(sub_cos.cpu().tolist())
    
    if len(all_orig_cos_pos) > 0:
        orig_mean = np.mean(all_orig_cos_pos)
        sub_mean = np.mean(all_subsampled_cos_pos)
        print(f"  常见类正样本 cos (原始原型): {orig_mean:.4f}")
        print(f"  常见类正样本 cos (子采样原型): {sub_mean:.4f}")
        print(f"  退化幅度: {orig_mean - sub_mean:.4f}")
        print(f"  → 如果子采样后cos接近罕见类水平(-0.27)，则证明根因确为代表蛋白数不足")


# ============================================================
# E4: GO 特征梯度信噪比分析
# ============================================================
# 注意: 此函数内部调用 torch.autograd.grad()，不能加 @torch.no_grad()
def experiment4_gradient_snr(model, loader, device, go_freq, num_classes, num_batches=20):
    """
    在训练模式下，对每个 GO term 测量其 GO 特征梯度的方向一致性 (SNR)。
    
    直觉: 
      - 如果同一类的不同 batch 给出的 GO 特征梯度方向高度一致 (高SNR)，
        说明学习信号清晰。
      - 如果方向分散 (低SNR)，说明学习信号嘈杂，GO 特征无法收敛到好方向。
    
    方法: 对每个类收集多个 batch 的 go_feats[c] 梯度，计算梯度间余弦相似度。
    
    注意: 此实验需要临时将模型设为 train 模式以获取梯度。
    """
    print("\n" + "=" * 60)
    print("E4: GO 特征梯度信噪比分析")
    print("=" * 60)
    print("直觉: 罕见类 GO 特征梯度方向不一致 (低 SNR) → 学习信号嘈杂")
    
    model.train()  # 需要梯度
    go_feats = model.go_feats
    
    freq_np = go_freq.cpu().numpy()
    
    # 收集每个类的梯度方向
    grad_directions = defaultdict(list)
    
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        # 兼容两种 loader 格式: (feats, labels, indices) 或 (feats, labels, mask)
        if len(batch_data) == 3:
            protein_feats, labels, _ = batch_data  # test loader: (feats, labels, indices)
        else:
            protein_feats, labels = batch_data
        
        protein_feats = protein_feats.to(device)
        labels = labels.to(device)
        
        model.zero_grad()
        logits, _, _ = model(protein_feats, labels)
        
        # 一次性计算所有类的 go_feats 梯度 (避免 retain_graph 循环)
        # go_feats[c] 只影响 logits[:, c]，所以全量 sum 等价于逐类计算
        grads_all = torch.autograd.grad(
            logits.sum(), go_feats, retain_graph=False, allow_unused=True
        )[0]  # (num_classes, hidden_dim)
        
        for c in range(num_classes):
            if labels[:, c].sum() == 0:
                continue
            grad_c_norm = F.normalize(grads_all[c:c+1], p=2, dim=-1).flatten()
            grad_directions[c].append(grad_c_norm.cpu())
        
        model.zero_grad(set_to_none=True)
    
    # 计算每个类的梯度方向一致性 (类内梯度余弦相似度均值)
    snr_by_class = np.zeros(num_classes)
    for c in range(num_classes):
        if len(grad_directions[c]) < 2:
            snr_by_class[c] = np.nan
        else:
            grads = torch.stack(grad_directions[c])  # (n_batches, D)
            sim_matrix = grads @ grads.T
            mask_triu = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
            snr_by_class[c] = sim_matrix[mask_triu].mean().item()
    
    # 按频率分桶
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        valid_mask = mask & ~np.isnan(snr_by_class)
        if valid_mask.sum() > 0:
            print(f"  {name} (valid n={valid_mask.sum()}): "
                  f"grad_SNR={snr_by_class[valid_mask].mean():.4f}±{snr_by_class[valid_mask].std():.4f}")
    
    model.eval()
    return snr_by_class


# ============================================================
# E5: 原型-正样本对齐度分析 (按类)
# ============================================================
@torch.no_grad()
def experiment5_prototype_positive_alignment(model, loader, device, go_freq, num_classes):
    """
    对每个类，计算 prototype 与 "正样本功能特异特征均值" 的余弦相似度。
    
    直觉:
      - 如果 prototype ≈ mean(positive func_specific features)，
        说明原型很好地代表了该类。
      - 如果相似度低，说明原型方向偏移了 (代表蛋白不代表真正的正样本分布)。
    
    关键区分：
      - 此实验测量 "原型是否代表了该类正样本的真实分布"
      - 不同于 analyze_proto_quality 的 per-sample cos，
        这里用均值聚合后计算整体对齐
    """
    print("\n" + "=" * 60)
    print("E5: 原型-正样本对齐度分析")
    print("=" * 60)
    print("直觉: 罕见类原型与正样本分布对齐度低 → 原型不代表该类")
    
    freq_np = go_freq.cpu().numpy()
    prototypes = model._compute_prototypes()
    prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
    
    # 收集每个类的正样本功能特异特征 (累积求和)
    hidden_dim = prototypes.shape[1]
    pos_func_sums = torch.zeros(num_classes, hidden_dim, device=device)
    pos_counts = torch.zeros(num_classes, device=device)
    
    for protein_feats, labels, indices in loader:
        protein_feats = protein_feats.to(device)
        p_feats = model.protein_mlp(protein_feats)
        go_feats = model.go_feats
        
        func_specific = p_feats.unsqueeze(1) * go_feats.unsqueeze(0)  # (B, C, D)
        
        for c in range(num_classes):
            mask_pos = labels[:, c] == 1
            if mask_pos.sum() > 0:
                pos_func_sums[c] += func_specific[mask_pos, c].sum(dim=0)
                pos_counts[c] += mask_pos.sum()
    
    # 计算对齐度
    alignment_scores = np.zeros(num_classes)
    for c in range(num_classes):
        if pos_counts[c] > 0:
            mean_pos_func = F.normalize(pos_func_sums[c] / pos_counts[c], p=2, dim=-1)
            alignment_scores[c] = (mean_pos_func * prototypes_norm[c]).sum().item()
        else:
            alignment_scores[c] = np.nan
    
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        valid = mask & ~np.isnan(alignment_scores)
        if valid.sum() > 0:
            print(f"  {name} (valid n={valid.sum()}): "
                  f"proto-pos_alignment={alignment_scores[valid].mean():.4f}±{alignment_scores[valid].std():.4f}")
    
    return alignment_scores


# ============================================================
# E6: 梯度范数对比 — 原型模型 go_feats vs 等效线性分类器权重
# ============================================================
# 注意: 此函数内部调用 torch.autograd.grad()，不能加 @torch.no_grad()
def experiment6_gradient_norm_comparison(proto_model, loader, device, go_freq, num_classes, num_batches=20):
    """
    对比实验 (核心): 
      在原型模型中，测量 go_feats[c] 的梯度范数，与等价的线性层权重梯度范数对比。
    
    关键洞察:
      Custom model 的 classifier 是 Linear(hidden_dim, num_classes)，
      权重 w_c 的梯度 = ∂L/∂logit[c] · protein_feat → 强信号。
    
      原型模型中 go_feats[c] 的梯度需要穿过:
        sigmoid → tau → cos_sim → L2-normalize → Hadamard → go_feats[c]
      L2 归一化的梯度包含投影算子 (I - x̂x̂^T)/||x||，
      这会削弱梯度 → 弱信号。

    如果罕见类的 go_feats 梯度范数显著小于等效线性权重梯度范数，
    则证明梯度瓶颈是根因。
    """
    print("\n" + "=" * 60)
    print("E6: 梯度范数对比 — go_feats vs 等效线性分类器")
    print("=" * 60)
    print("直觉: L2归一化削弱梯度 → 罕见类GO特征无法有效学习")
    
    proto_model.train()
    go_feats_param = proto_model.go_feats  # nn.Parameter
    hidden_dim = go_feats_param.shape[1]
    freq_np = go_freq.cpu().numpy()
    
    # 创建一个等效的线性分类器 (模拟 custom model 的最后层)
    # 设为 Parameter 以便计算梯度
    dummy_linear_weight = nn.Parameter(torch.randn(num_classes, hidden_dim, device=device) * 0.02)
    
    go_grad_norms = defaultdict(list)
    linear_grad_norms = defaultdict(list)
    
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= num_batches:
            break
        
        if len(batch_data) == 3:
            protein_feats, labels, _ = batch_data
        else:
            protein_feats, labels = batch_data
        
        protein_feats = protein_feats.to(device)
        labels = labels.to(device)
        
        # --- 原型模型梯度 (一次性计算所有类，避免 retain_graph 循环) ---
        # 关键: go_feats[c] 只影响 logits[:, c]，所以 ∂logits.sum()/∂go_feats[c] = ∂logits[:,c].sum()/∂go_feats[c]
        proto_model.zero_grad()
        logits, _, _ = proto_model(protein_feats, labels)
        grads_all = torch.autograd.grad(
            logits.sum(), go_feats_param, retain_graph=False, allow_unused=True
        )[0]  # (num_classes, hidden_dim)
        
        for c in range(num_classes):
            if labels[:, c].sum() == 0:
                continue
            go_grad_norms[c].append(grads_all[c].norm().item())
        
        proto_model.zero_grad(set_to_none=True)
        
        # --- 等效线性分类器梯度 (同样一次性计算) ---
        h = proto_model.protein_mlp(protein_feats)  # 复用同一个 protein MLP 的输出
        linear_logits = h @ dummy_linear_weight.T   # (B, num_classes)
        
        grads_lin = torch.autograd.grad(
            linear_logits.sum(), dummy_linear_weight, retain_graph=False, allow_unused=True
        )[0]  # (num_classes, hidden_dim)
        
        for c in range(num_classes):
            if labels[:, c].sum() == 0:
                continue
            linear_grad_norms[c].append(grads_lin[c].norm().item())
        
        proto_model.zero_grad(set_to_none=True)
    
    # 按频率分组统计
    print(f"\n  {'Bucket':<20} {'go_feats grad':>14} {'linear grad':>14} {'ratio':>10}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*10}")
    
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        classes_in = np.where(mask)[0]
        
        go_vals = []
        lin_vals = []
        for c in classes_in:
            if len(go_grad_norms[c]) > 0:
                go_vals.append(np.mean(go_grad_norms[c]))
            if len(linear_grad_norms[c]) > 0:
                lin_vals.append(np.mean(linear_grad_norms[c]))
        
        if go_vals and lin_vals:
            go_mean = np.mean(go_vals)
            lin_mean = np.mean(lin_vals)
            ratio = go_mean / max(lin_mean, 1e-8)
            print(f"  {name:<20} {go_mean:>14.6f} {lin_mean:>14.6f} {ratio:>10.4f}")
    
    proto_model.eval()
    print(f"\n  → 如果 go_feats 梯度范数远小于 linear 梯度范数 (ratio≪1)，")
    print(f"    则证明 L2 归一化 + 余弦相似度的梯度瓶颈是罕见类学习失败的根本原因。")
    
    return go_grad_norms, linear_grad_norms


# ============================================================
# E7: GO特征有效秩分析 — 罕见类 GO 特征是否仍然接近随机？
# ============================================================
@torch.no_grad()
def experiment7_go_feature_effective_rank(proto_model, go_freq):
    """
    检查 GO 特征矩阵的有效秩，判断罕见类的 GO 特征是否学到了结构。
    
    方法:
      对 go_feats 做 SVD，计算有效秩 (累积解释方差 90% 所需的奇异值数量)。
      然后分别对罕见类和常见类的 GO 特征子矩阵做 SVD。
      
    直觉:
      - 如果所有 GO 特征都学到了结构 → 有效秩 ≈ 特征维度
      - 如果罕见类 GO 特征接近随机 → 罕见类子矩阵有效秩更低
    """
    print("\n" + "=" * 60)
    print("E7: GO 特征有效秩分析")
    print("=" * 60)
    print("直觉: 罕见类 GO 特征如果仍然接近随机初始化分布，则未学到有意义方向")
    
    go_feats = proto_model.go_feats.detach().cpu()  # (C, D)
    freq_np = go_freq.cpu().numpy()
    
    # 全量 GO 特征 SVD
    U, S, V = torch.svd(go_feats.float())
    S = S.numpy()
    total_var = (S ** 2).sum()
    cum_var_ratio = np.cumsum(S ** 2) / total_var
    
    eff_rank_all = np.searchsorted(cum_var_ratio, 0.9) + 1
    print(f"  全部 GO 特征: shape={list(go_feats.shape)}, "
          f"有效秩(90%)={eff_rank_all}, 总奇异值数={len(S)}")
    
    # 按频率分组 SVD
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        
        sub_feats = go_feats[idx]  # (n, D)
        U_sub, S_sub, V_sub = torch.svd(sub_feats.float())
        S_sub = S_sub.numpy()
        total_var_sub = (S_sub ** 2).sum()
        cum_var_sub = np.cumsum(S_sub ** 2) / max(total_var_sub, 1e-10)
        eff_rank_sub = np.searchsorted(cum_var_sub, 0.9) + 1
        
        # 与随机矩阵对比
        random_mat = torch.randn_like(sub_feats)
        _, S_rand, _ = torch.svd(random_mat.float())
        S_rand = S_rand.numpy()
        total_var_rand = (S_rand ** 2).sum()
        cum_var_rand = np.cumsum(S_rand ** 2) / total_var_rand
        eff_rank_rand = np.searchsorted(cum_var_rand, 0.9) + 1
        
        print(f"  {name} (n={len(idx)}): eff_rank={eff_rank_sub}, "
              f"random_baseline={eff_rank_rand}, "
              f"ratio={eff_rank_sub/max(eff_rank_rand,1):.3f}")
    
    print(f"\n  → 如果罕见类的 eff_rank ≈ random_baseline，说明罕见类 GO 特征本质上仍是随机噪声。")


# ============================================================
# E8: 移除归一化瓶颈 — 用点积替代余弦相似度
# ============================================================
@torch.no_grad()
def experiment8_dot_product_vs_cosine(model, loader, device, go_freq, num_classes):
    """
    消融实验: 比较 "余弦相似度" vs "原始点积" 对 pos/neg 区分度的影响。
    
    直觉:
      L2 归一化会削弱梯度 (如 E6 所示)。如果移除归一化后，
      罕见类的 pos/neg 分离度显著改善，则证明归一化是瓶颈。
    
    方法:
      对每个 batch，同时计算:
        cos_sim = normalized(func) · normalized(proto)
        dot_prod = func · proto  (不做归一化)
      对比两者的 pos/neg separation。
    """
    print("\n" + "=" * 60)
    print("E8: 余弦相似度 vs 点积 — 消融归一化瓶颈")
    print("=" * 60)
    print("直觉: 移除L2归一化后罕见类区分度改善 → 归一化是梯度瓶颈根因")
    
    freq_np = go_freq.cpu().numpy()
    prototypes = model._compute_prototypes()
    prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
    
    all_cos_pos = defaultdict(list)
    all_cos_neg = defaultdict(list)
    all_dot_pos = defaultdict(list)
    all_dot_neg = defaultdict(list)
    
    for protein_feats, labels, indices in loader:
        protein_feats = protein_feats.to(device)
        p_feats = model.protein_mlp(protein_feats)
        go_feats = model.go_feats
        
        func_specific = p_feats.unsqueeze(1) * go_feats.unsqueeze(0)  # (B, C, D)
        
        # 余弦相似度
        func_norm = F.normalize(func_specific, p=2, dim=-1)
        cos_sim = (func_norm * prototypes_norm.unsqueeze(0)).sum(dim=-1)  # (B, C)
        
        # 点积 (无归一化)
        dot_prod = (func_specific * prototypes.unsqueeze(0)).sum(dim=-1)  # (B, C)
        
        for c in range(num_classes):
            mask_pos = labels[:, c] == 1
            mask_neg = labels[:, c] == 0
            
            if mask_pos.sum() > 0:
                all_cos_pos[c].extend(cos_sim[mask_pos, c].cpu().tolist())
                all_dot_pos[c].extend(dot_prod[mask_pos, c].cpu().tolist())
            if mask_neg.sum() > 0:
                all_cos_neg[c].extend(cos_sim[mask_neg, c].cpu().tolist())
                all_dot_neg[c].extend(dot_prod[mask_neg, c].cpu().tolist())
    
    print(f"\n  {'Bucket':<20} {'cos_sep':>10} {'dot_sep':>10} {'improvement':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")
    
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        classes_in = np.where(mask)[0]
        
        cos_pos_vals, cos_neg_vals = [], []
        dot_pos_vals, dot_neg_vals = [], []
        
        for c in classes_in:
            if len(all_cos_pos[c]) > 0 and len(all_cos_neg[c]) > 0:
                cos_pos_vals.extend(all_cos_pos[c])
                cos_neg_vals.extend(all_cos_neg[c])
                dot_pos_vals.extend(all_dot_pos[c])
                dot_neg_vals.extend(all_dot_neg[c])
        
        if cos_pos_vals:
            cos_sep = np.mean(cos_pos_vals) - np.mean(cos_neg_vals)
            dot_sep = np.mean(dot_pos_vals) - np.mean(dot_neg_vals)
            improvement = dot_sep - cos_sep
            print(f"  {name:<20} {cos_sep:>10.4f} {dot_sep:>10.4f} {improvement:>+12.4f}")
    
    print(f"\n  → 如果罕见类 dot_sep >> cos_sep (显著改善)，则 L2 归一化是核心瓶颈。")
    print(f"    修复方向: 用点积(或可学习温度缩放的版本)替代余弦相似度。")


# ============================================================
# E9: tau/bias 分析 — 模型是否退化为按类频率设偏置？
# ============================================================
@torch.no_grad()
def experiment9_tau_bias_analysis(model, go_freq):
    """
    检查 tau[c] 和 bias[c] 是否与类频率高度相关。

    直觉:
      如果 GO 特征是随机的 (E7 已证实)，则余弦相似度 ≈ 0，
      logits[c] ≈ bias[c]。模型退化为 "按类频率设默认概率"。
      
      常见类: bias 大 → sigmoid 输出高 → "预测为有功能"
      罕见类: bias 小/负 → sigmoid 输出低 → "预测为无功能"
      
      这种情况下 Fmax 来自 bias 的校准，而非真正的特征学习。
    """
    print("\n" + "=" * 60)
    print("E9: tau/bias 分析 — 模型是否退化为频率偏置？")
    print("=" * 60)
    print("直觉: GO特征随机 → cos≈0 → logits≈bias → 模型只学了类频率")
    
    tau = model.tau.detach().cpu().numpy()    # (num_classes,)
    bias = model.bias.detach().cpu().numpy()   # (num_classes,)
    freq_np = go_freq.cpu().numpy()
    
    from scipy.stats import spearmanr
    
    # tau 和 bias 与频率的相关性
    r_tau, p_tau = spearmanr(freq_np, tau)
    r_bias, p_bias = spearmanr(freq_np, bias)
    
    print(f"  Spearman r (go_freq vs tau):  {r_tau:.4f} (p={p_tau:.2e})")
    print(f"  Spearman r (go_freq vs bias): {r_bias:.4f} (p={p_bias:.2e})")
    
    # 按频率分桶统计
    print(f"\n  {'Bucket':<20} {'bias_mean':>12} {'bias_std':>12} {'tau_mean':>12} {'tau_std':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        if mask.sum() > 0:
            print(f"  {name:<20} {bias[mask].mean():>12.4f} {bias[mask].std():>12.4f} "
                  f"{tau[mask].mean():>12.4f} {tau[mask].std():>12.4f}")
    
    # 关键诊断: 如果 logits[c] = exp(tau[c]) * cos_sim + bias[c]，
    # 且 cos_sim ≈ 0 (GO特征随机)，则预测概率 = sigmoid(bias[c])
    # 检查 sigmoid(bias) 是否接近类频率
    pred_prob_from_bias = 1.0 / (1.0 + np.exp(-bias))
    
    print(f"\n  sigmoid(bias) vs go_freq:")
    for name, lo, hi in [("rare (<1%)", 0, 0.01), ("mid (1-10%)", 0.01, 0.1), ("freq (>10%)", 0.1, 1.0)]:
        mask = (freq_np >= lo) & (freq_np < hi)
        if mask.sum() > 0:
            print(f"  {name:<20}: mean_sigmoid(bias)={pred_prob_from_bias[mask].mean():.6f}, "
                  f"mean_freq={freq_np[mask].mean():.6f}")
    
    print(f"\n  → 如果 sigmoid(bias) 与类频率高度相关，则模型退化为频率偏置模型。")
    print(f"  → 常见类靠高 bias 获得 Fmax，罕见类因低 bias 无法预测。")


# ============================================================
# E10: GO 特征消融 — 设 go_feats=1 vs 随机 vs 当前值
# ============================================================
@torch.no_grad()
def experiment10_go_feature_ablation(model, loader, device, go_freq, num_classes):
    """
    消融实验: 比较三种 GO 特征配置下的 pos/neg 分离度。
    
    (a) ones:   go_feats[c] = 全1向量 → 功能特异特征 = 蛋白特征本身
    (b) random: go_feats[c] = 重新随机初始化
    (c) current: 当前训练后的 go_feats
    
    直觉:
      - 如果 (a) ones 的分离度 ≈ (c) current → GO 特征完全无用
      - 如果 (b) random 的分离度 ≈ (c) current → GO 特征仍处于随机状态
    """
    print("\n" + "=" * 60)
    print("E10: GO 特征消融 — ones vs random vs current")
    print("=" * 60)
    print("直觉: 如果 ones/random 与 current 性能相同 → GO 特征未学到任何东西")
    
    freq_np = go_freq.cpu().numpy()
    prototypes = model._compute_prototypes()
    hidden_dim = prototypes.shape[1]
    orig_go = model.go_feats.clone()
    
    # 三种 GO 配置
    go_ones = torch.ones_like(orig_go)
    go_random = torch.randn_like(orig_go) * 0.02  # 模拟初始分布
    
    go_configs = [
        ("current", orig_go),
        ("ones", go_ones),
        ("random", go_random),
    ]
    
    results = {}
    for config_name, go_feats in go_configs:
        all_pos_cos = defaultdict(list)
        all_neg_cos = defaultdict(list)
        
        for protein_feats, labels, indices in loader:
            protein_feats = protein_feats.to(device)
            p_feats = model.protein_mlp(protein_feats)
            
            func_specific = p_feats.unsqueeze(1) * go_feats.unsqueeze(0)
            func_norm = F.normalize(func_specific, p=2, dim=-1)
            proto_norm = F.normalize(prototypes, p=2, dim=-1)
            cos_sim = (func_norm * proto_norm.unsqueeze(0)).sum(dim=-1)
            
            for c in range(num_classes):
                mask_pos = labels[:, c] == 1
                mask_neg = labels[:, c] == 0
                if mask_pos.sum() > 0:
                    all_pos_cos[c].extend(cos_sim[mask_pos, c].cpu().tolist())
                if mask_neg.sum() > 0:
                    all_neg_cos[c].extend(cos_sim[mask_neg, c].cpu().tolist())
        
        # 按频率汇总
        bucket_results = {}
        for name, lo, hi in [("rare", 0, 0.01), ("mid", 0.01, 0.1), ("freq", 0.1, 1.0)]:
            mask = (freq_np >= lo) & (freq_np < hi)
            classes_in = np.where(mask)[0]
            pos_vals, neg_vals = [], []
            for c in classes_in:
                if len(all_pos_cos[c]) > 0 and len(all_neg_cos[c]) > 0:
                    pos_vals.extend(all_pos_cos[c])
                    neg_vals.extend(all_neg_cos[c])
            if pos_vals:
                sep = np.mean(pos_vals) - np.mean(neg_vals)
                bucket_results[name] = sep
        
        results[config_name] = bucket_results
    
    # 对比输出
    print(f"\n  {'Bucket':<10} {'current_sep':>12} {'ones_sep':>12} {'random_sep':>12} {'ones=current?':>14}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*14}")
    for name in ["rare", "mid", "freq"]:
        cur = results["current"].get(name, float('nan'))
        ones = results["ones"].get(name, float('nan'))
        rand = results["random"].get(name, float('nan'))
        match = "YES" if abs(cur - ones) < 0.01 else f"diff={cur-ones:.4f}"
        print(f"  {name:<10} {cur:>12.4f} {ones:>12.4f} {rand:>12.4f} {match:>14}")
    
    print(f"\n  → 如果 current ≈ ones 或 current ≈ random，证明 GO 特征没有学到有用的东西。")
    print(f"  → 这是 E7 (有效秩) 的行为验证。")


# 综合诊断入口
# ============================================================
def run_all_experiments(model, loader, device, go_freq, num_classes):
    """运行所有验证实验"""
    
    print("\n" + "#" * 60)
    print("#  罕见功能原型失效 - 根因定位实验套件")
    print("#" * 60)
    
    # E1: 原型稳定性
    stability = experiment1_prototype_stability(model, go_freq)
    
    # E2: 代表蛋白数 vs 性能
    n_reps, per_class_auprc = experiment2_reps_vs_performance(model, loader, device, go_freq, num_classes)
    
    # E3: 子采样消融
    experiment3_subsample_ablation(model, loader, device, go_freq, num_classes)
    
    # E4: 梯度 SNR (需要在训练模式下运行，当前用 test_loader 可能梯度信号不足)
    # 建议单独用 train_loader 调用:
    #   experiment4_gradient_snr(model, train_loader, device, go_freq, num_classes)
    print("\n[跳过] E4 梯度SNR需要train_loader和训练模式，请在训练脚本中单独调用")
    # experiment4_gradient_snr(model, loader, device, go_freq, num_classes)
    
    # E5: 原型-正样本对齐
    alignment = experiment5_prototype_positive_alignment(model, loader, device, go_freq, num_classes)
    
    # E6: 梯度范数对比 (核心) — 需要训练模式
    print("\n[跳过] E6 梯度范数对比需要train_loader，请在训练脚本中调用:")
    print("  from experiment.verify_prototype import experiment6_gradient_norm_comparison")
    print("  experiment6_gradient_norm_comparison(model, train_loader, device, go_freq, num_classes)")
    
    # E7: GO特征有效秩
    experiment7_go_feature_effective_rank(model, go_freq)
    
    # E8: 余弦相似度 vs 点积消融
    experiment8_dot_product_vs_cosine(model, loader, device, go_freq, num_classes)
    
    # E9: tau/bias 分析 — 模型是否退化为频率偏置
    experiment9_tau_bias_analysis(model, go_freq)
    
    # E10: GO 特征消融 — ones vs random vs current
    experiment10_go_feature_ablation(model, loader, device, go_freq, num_classes)
    
    # ---- 综合结论 ----
    print("\n" + "=" * 60)
    print("综合诊断结论 (基于实验数据修订)")
    print("=" * 60)
    print("""
    关键发现: 
      custom model (简单MLP) 在 ALF[0,0.2] 上 Fmax=0.48，
      原型模型仅 0.21。问题出在原型架构的 GO 特征学习失败。
    
    ❌ 已排除的假设:
      E1: 原型不稳定 → 排除 (stability>0.98, 非常稳定)
      E3: 代表蛋白不足 → 排除 (子采样无影响)
      E8: L2归一化瓶颈 → 排除 (移除后无改善)
    
    ✅ 确认为根因:
      E7: 所有频率段的 GO 特征有效秩 ≈ 随机基线 (ratio≈0.95)
           → GO 特征对所有类都是随机噪声，未学到任何结构
      
      E5: 罕见类 proto-pos_alignment = -0.40
           → 原型与正样本方向相反，模型学反了
    
    推断 (待 E9/E10 验证):
      E9: tau/bias 承担了全部预测能力
           → 常见类靠高 bias 获得 Fmax，罕见类 bias 低无法预测
      
      E10: ones/random GO 特征与当前 GO 特征效果相同
           → 确证 GO 特征完全没有贡献
    
    根因总结:
      GO 特征梯度路径过长 (Hadamard → L2-norm → cos → sigmoid → BCE)，
      100 epoch 的类平衡训练不足以驱动 2574×1024 个参数的 GO 特征
      脱离随机初始化状态。模型退化为纯 bias 预测。
    
    修复方向 (按可行性排序):
    1. GO 特征初始化为 custom model 的分类器权重 (迁移学习)
       → 从 "已学会分类" 的起点出发，而非随机噪声
    2. 添加残差连接: logits = cos_proto + direct_linear(protein_feat)
       → 给 GO 特征一个 "退路"，类平衡 BCE 可直接驱动线性头
    3. 对 GO 特征单独使用更大的学习率 (×10~×100)
       → 补偿梯度路径过长导致的信号衰减
    """)
