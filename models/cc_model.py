import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.loss import AsymmetricLossMultiLabel

class similarityModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, rgcn_dim, eps=1e-6):
        super(similarityModule, self).__init__()

        self.go_proj = nn.Sequential(
            nn.Linear(rgcn_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.alpha = nn.Parameter(torch.tensor([10.0]))

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )

    def forward(self, protein_features, go_features, labels):
        # MLP预测路线
        go_features = self.go_proj(go_features)
        prot_features = self.prot_proj(protein_features)
        # 计算余弦相似度
        go_features = F.normalize(go_features, p=2, dim=1)  # 形状: (batch, dim)
        prot_features = F.normalize(prot_features, p=2, dim=1)  # 形状: (node_num, dim)
        similarity = torch.mm(prot_features, go_features.t())

        logits = similarity * self.alpha
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        return probs, loss

class prototypeModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, eps=1e-6):
        super(prototypeModule, self).__init__()

        # 原型概率可学习的参数τ和bias
        self.base_tau = 5.0 # tau的缩放因子
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.zeros(num_classes))

        self.proto_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )

    def forward(self, protein_features, prototype_feats, labels):
        # 原型网络
        protein_features = F.normalize(self.proto_proj(protein_features), p=2, dim=1) # 先投影至同一空间，再归一化：消除模长干扰 
        prototype_feats = F.normalize(self.proto_proj(prototype_feats), p=2, dim=1)
        distances = torch.cdist(protein_features, prototype_feats, p=2.0) ** 2 # 计算欧氏距离的平方, distances[i, j] 表示第 i 个蛋白质到第 j 个原型的距离
        tau = -torch.exp(self.log_tau) * self.base_tau # tau 一定为负
        logits = tau * distances + self.b
        proto = torch.sigmoid(logits)
        loss = self.loss_fn(logits, labels)

        return proto, loss, tau

class Model(nn.Module):
    def __init__(self, prototype_feats, frequencies, rgcn_dim, labels_index, input_dim, hidden_dim, num_classes, eps=1e-6, node_feat_path="./new/data/go_embeddings.pt"):
        super(Model, self).__init__()

        rgcn_node_features = torch.load(node_feat_path, weights_only=True, map_location='cpu')
        self.register_buffer('go_features', rgcn_node_features[labels_index])
        self.LayerNorm = nn.LayerNorm(input_dim)

        self.prototype_feats = prototype_feats
        self.frequencies = frequencies
        self.eps = eps # 用于稳定数值

        self.similarity_module = similarityModule(input_dim, hidden_dim, rgcn_dim, eps=eps) # 相似度预测模块
        self.prototype_Module = prototypeModule(input_dim, hidden_dim, num_classes, eps=eps) # 相似度预测模块

        # 概率结合可学习的参数: 中心阈值c和平滑系数k
        self.gate_k = nn.Parameter(torch.tensor([2.0])) 
        self.gate_c = nn.Parameter(torch.tensor([0.5]))

        self.loss_fn = AsymmetricLossMultiLabel(
            gamma_neg=4,
            gamma_pos=1,
            clip=0.05,  # 概率裁剪阈值，用于剔除简单的负样本
            eps=eps 
        )


    def forward(self, data, labels):

        protein_features = self.LayerNorm(data)
        prototype_feats = self.LayerNorm(self.prototype_feats)
        
        sim_probs, sim_loss = self.similarity_module(protein_features, self.go_features, labels)
        proto_probs, proto_loss, tau = self.prototype_Module(protein_features, prototype_feats, labels)

        # 联合概率
        sigma = 1.0 / (1.0 + torch.exp(-self.gate_k * (self.frequencies - self.gate_c)))
        final_probs = sigma * sim_probs.detach() + (1.0 - sigma) * proto_probs.detach() # 截断两部分概率的计算图，要求final_loss只负责优化门控参数k和c
        # final_probs = sigma * sim_probs + (1.0 - sigma) * proto_probs

        # 通过概率倒推逻辑上的logits
        final_logits = torch.logit(final_probs, eps=self.eps)
        gate_loss = self.loss_fn(final_logits, labels)
        
        return sim_probs, proto_probs, final_probs, sim_loss, proto_loss, gate_loss, tau, self.prototype_Module.b, self.gate_k, self.gate_c