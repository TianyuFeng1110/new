import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel

class AsymmetricLossWithProbs(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLossWithProbs, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x_prob, y):
        # 直接使用传入的混合概率，无需计算 sigmoid
        xs_pos = x_prob
        xs_neg = 1.0 - x_prob

        # 不对称裁剪 (Asymmetric Clipping)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # 基础交叉熵计算，通过 eps 避免 log(0)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # 不对称聚焦 (Asymmetric Focusing)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
            
        return -loss.sum()

class LocalMotifFeatureExtractionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dilations=None, dropout=0.4):
        """
        Args:
            dilations: 3层卷积各自的膨胀率，默认 [1, 1, 1]（等价于普通卷积）。
                       膨胀卷积可在不增加参数量的前提下指数级扩大感受野。
                       
            第i层的卷积核为 (k-1) * dilations_i + 1。第i层感受野为i-1层的感受野加上当前层卷积核大小再减去1。
        """
        super(LocalMotifFeatureExtractionModule, self).__init__()
        if dilations is None:
            dilations = [1, 1, 1]
        d1, d2, d3 = dilations

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),    
            nn.LayerNorm(input_dim)
        )

        # padding 根据 dilation 自动计算以保持序列长度不变: pad = (k-1) * d / 2
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size,
                               padding=(kernel_size - 1) * d1 // 2, dilation=d1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               padding=(kernel_size - 1) * d2 // 2, dilation=d2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                               padding=(kernel_size - 1) * d3 // 2, dilation=d3)
        # GN代替LN，以避免频繁的维度顺序切换造成的开销
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.norm2 = nn.GroupNorm(1, hidden_dim)
        self.norm3 = nn.GroupNorm(1, hidden_dim)
        
        self.LayerNorm = nn.LayerNorm(hidden_dim * 3)

    def forward(self, residue_feats, mask):

        x = self.mlp(residue_feats)
        x = x.transpose(1, 2)
        mask_expanded = mask.unsqueeze(1)
        
        # 三层卷积堆叠处理, 卷积后乘以 mask 来屏蔽 padding 的影响
        x_conv1 = self.conv1(x)
        conv1_out = F.gelu(self.norm1(x_conv1)) * mask_expanded

        x_conv2 = self.conv2(conv1_out)
        conv2_out = F.gelu(self.norm2(x_conv2)) * mask_expanded
        
        x_conv3 = self.conv3(conv2_out)
        conv3_out = F.gelu(self.norm3(x_conv3)) * mask_expanded

        # 在 hidden_dim 维度直接拼接
        merged = torch.cat([conv1_out, conv2_out, conv3_out], dim=1).transpose(1, 2)
        merged = self.LayerNorm(merged)
        
        return merged

class CrossAttentionLayer(nn.Module):
    """单层交叉注意力：GO(query) 交叉注意力到 residue(key/value)，再接 FFN。
       不包含 query 端自注意力，因为 GO 术语之间相互独立。"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.2):
        super(CrossAttentionLayer, self).__init__()

        # 交叉注意力
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                         num_heads=num_heads, 
                                         dropout=dropout, 
                                         batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, go_features, residue_feats_data, key_padding_mask):
        # 交叉注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.mha(
            query=go_features,
            key=residue_feats_data,
            value=residue_feats_data,
            key_padding_mask=key_padding_mask
        )
        go_features = self.norm1(go_features + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(go_features)
        output = self.norm2(go_features + ffn_output)

        return output

class CrossAttentionModule(nn.Module):
    def __init__(self, go_dim, input_dim, hidden_dim, num_heads=8, num_layers=2, dropout=0.2):
        super(CrossAttentionModule, self).__init__()

        self.residue_proj = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim)
                            )
        self.go_proj = nn.Sequential(
                                nn.Linear(go_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim)
                            )

        self.layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, go_features, residue_feats_data, mask):
        # 投影到统一维度
        go_features = self.go_proj(go_features)
        residue_feats_data = self.residue_proj(residue_feats_data)

        batch_size = residue_feats_data.size(0)
        go_features = go_features.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = (mask == 0)  # True 表示忽略该位置

        # 逐层通过交叉注意力
        for layer in self.layers:
            go_features = layer(go_features, residue_feats_data, key_padding_mask)

        return go_features

class PrototypeModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, eps=1e-8):
        super(PrototypeModule, self).__init__()

        # 原型概率可学习的参数τ和bias
        self.base_tau = 5.0 # tau的缩放因子
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.zeros(num_classes))

        self.proto_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),           
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, protein_features, prototype_feats, labels):
        # 原型网络
        protein_features = F.normalize(self.proto_proj(protein_features), p=2, dim=1) # 先投影至同一空间，再归一化：消除模长干扰 
        prototype_feats = F.normalize(self.proto_proj(prototype_feats), p=2, dim=1)
        # distances = torch.cdist(protein_features, prototype_feats, p=2.0) ** 2 # 计算欧氏距离的平方, distances[i, j] 表示第 i 个蛋白质到第 j 个原型的距离
        distances = 2.0 * (1.0 - torch.matmul(protein_features, prototype_feats.T)) # 上面公式的简化版本，说是会减少计算开销。
        distances = torch.clamp(distances, min=0.0)
        tau = -torch.exp(self.log_tau) * self.base_tau # tau 一定为负
        logits = tau * distances + self.b

        return logits, tau

class Model(nn.Module):
    def __init__(self, prototype_feats, frequencies, rgcn_dim, labels_index, input_dim, hidden_dim, num_classes, kernel_size, dilations=None, dropout=0.2, node_feat_path="./new/data/go_embeddings.pt"):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.frequencies = frequencies
        rgcn_node_features = torch.load(node_feat_path, weights_only=True, map_location='cpu')
        self.register_buffer('go_features', rgcn_node_features[labels_index])
        self.register_buffer('prototype_feats', prototype_feats)
        self.LayerNorm = nn.LayerNorm(input_dim)
        self.residue_LayerNorm = nn.LayerNorm(input_dim)
        self.go_LayerNorm = nn.LayerNorm(rgcn_dim)

        self.local_feats_extraction_module = LocalMotifFeatureExtractionModule(input_dim, hidden_dim, kernel_size=kernel_size, dilations=dilations, dropout=dropout)
        self.cross_attention_module = CrossAttentionModule(rgcn_dim, input_dim, hidden_dim, dropout=dropout) # 跨模态注意力模块
        self.prototype_Module = PrototypeModule(input_dim, hidden_dim, num_classes) # 原型模块

        self.mean_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.max_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.LayerNorm(hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.branch_gate_temperature = 2.5
        self.grad_scale_factor = 1.0 / (num_classes ** 0.5) 
        self.branch_gate_weight = nn.Sequential(
            nn.Linear(rgcn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        self.norm_fused = nn.LayerNorm(hidden_dim)

        # 概率结合可学习的参数: 中心阈值c和平滑系数k
        self.gate_k = nn.Parameter(torch.tensor([2.0])) 
        self.gate_c_raw = nn.Parameter(torch.logit(self.frequencies.mean())) # 初始化 gate_c_raw，使得 gate_c 接近 frequencies 的平均值

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # self.loss_fn = AsymmetricLossMultiLabel(
        #     gamma_neg=4,
        #     gamma_pos=1,
        #     clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
        #     eps=1e-8 
        # )
        self.loss_fn_with_probs = AsymmetricLossWithProbs(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=1e-8 
        )

    def forward(self, protein_feats_data, residue_feats_data, mask, labels):

        protein_features = self.LayerNorm(protein_feats_data)
        prototype_feats = self.LayerNorm(self.prototype_feats)
        residue_feats_data = self.residue_LayerNorm(residue_feats_data)
        go_LayerNorm = self.go_LayerNorm(self.go_features)
        
        merged_feats = self.local_feats_extraction_module(residue_feats_data, mask)

        # custom 模块
        mean_feats = self._get_mean_pooled_features(merged_feats, mask)
        max_feats = self._get_max_pooled_features(merged_feats, mask)
        cross_attn_feats = self.cross_attention_module(go_LayerNorm, residue_feats_data, mask)

        mean_part = self.mean_proj(mean_feats).unsqueeze(1)
        max_part = self.max_proj(max_feats).unsqueeze(1)
        self._register_hook(mean_part, max_part) # 注册梯度缩放hook, 因为广播机制，乘以权重后梯度会自动传播到 mean_part 和 max_part 上，反向传递时会导致梯度放大，所以注册 hook 来缩放梯度，保持训练稳定。
        weights = torch.softmax(self.branch_gate_weight(go_LayerNorm) / self.branch_gate_temperature, dim=-1).unsqueeze(0) # 形状: [1, num_classes, 3]
        feats = (weights[:, :, 0:1] * mean_part) + \
                (weights[:, :, 1:2] * max_part) + \
                (weights[:, :, 2:3] * cross_attn_feats)
        feats = self.norm_fused(feats)

        custom_logits = self.classifier(feats).squeeze(-1)
        custom_probs = torch.sigmoid(custom_logits)
        custom_loss = self.loss_fn_with_probs(custom_probs, labels)

        # 原型模块
        proto_logits, tau = self.prototype_Module(protein_features, prototype_feats, labels)
        proto_probs = torch.sigmoid(proto_logits)
        proto_loss = self.loss_fn_with_probs(proto_probs, labels)

        # 联合概率
        gate_c = torch.sigmoid(self.gate_c_raw) # 范围在0-1之间
        sigma = torch.sigmoid(self.gate_k * (self.frequencies - gate_c))
        final_probs = sigma * custom_probs.detach() + (1.0 - sigma) * proto_probs.detach()

        # 通过概率倒推逻辑上的logits
        # final_logits = torch.logit(final_probs, eps=1e-7)
        # gate_loss = self.loss_fn(final_logits, labels)
        gate_loss = self.loss_fn_with_probs(final_probs, labels)
        
        return custom_probs, proto_probs, final_probs, custom_loss, proto_loss, gate_loss, tau, self.prototype_Module.b, self.gate_k, gate_c, weights, sigma.mean()
    
    def _get_mean_pooled_features(self, merged, mask):
        # 平均池化
        mask_expanded = mask.unsqueeze(-1)
        masked_feats = merged * mask_expanded
        pooled = masked_feats.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return pooled
    
    def _get_max_pooled_features(self, merged, mask):
        # 处理 Mask：将 Padding 部分设为极小值以忽略其对 Max Pooling 的影响
        mask_expanded = mask.unsqueeze(-1)
        merged = merged.masked_fill(mask_expanded == 0, -1e9)
        
        # 全局最大池化
        pooled = torch.max(merged, dim=1)[0]
        return pooled
    
    def _register_hook(self, mean_part, max_part):
        '''
        注册反向梯度缩放hook, 反向传播到 mean_proj 和 max_proj 的梯度会被缩放
        '''
        if self.training:
            if mean_part.requires_grad:
                mean_part.register_hook(lambda grad: grad * self.grad_scale_factor)
            if max_part.requires_grad:
                max_part.register_hook(lambda grad: grad * self.grad_scale_factor)



