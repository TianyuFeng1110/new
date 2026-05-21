import torch

class collator:
    def __init__(self, go2id, protein_feats, residue_feats):
        self.go2id = go2id
        self.protein_feats = protein_feats
        self.residue_feats = residue_feats
        
    def __call__(self, data):
        batch_protein_feats = torch.stack([self.protein_feats[elem[0]] for elem in data])
        batch_residue_feats, mask = self._process_residue_feats([self.residue_feats[elem[0]] for elem in data])
        labels = self._get_labels(data)

        return batch_protein_feats, batch_residue_feats, mask, labels

    def _process_residue_feats(self, residue_feats):
        """将不同长度的残基特征统一为固定长度，并与 Transformer padding 方案保持一致。

        Args:
            residue_feats: List[tensor[L_i, d]]，每个蛋白质的残基特征，长度 L_i 各不相同

        Returns:
            padded_feats: tensor[B, max_len, d]，padding 部分填充 0
            mask: tensor[B, max_len]，bool 类型，True=有效残基，False=padding
        """
        batch_size = len(residue_feats)
        target_len = max(residue_feats[i].size(0) for i in range(batch_size)) # 当前 batch 中最长的残基序列长度
        feat_dim = residue_feats[0].size(-1)  # 特征维度 d

        # 初始化全零张量和 mask
        padded_feats = torch.zeros(batch_size, target_len, feat_dim)
        mask = torch.zeros(batch_size, target_len, dtype=torch.bool)

        for i, feat in enumerate(residue_feats):
            cur_len = min(feat.size(0), target_len)  # 防止超过目标长度
            padded_feats[i, :cur_len] = feat[:cur_len]
            mask[i, :cur_len] = True

        return padded_feats, mask

    def _get_labels(self, data):
        num_classes = len(self.go2id)

        # 初始化全 0 tensor
        multi_hot = torch.zeros((len(data), num_classes))

        for i, (_, index_set) in enumerate(data):
            if index_set:  # 确保集合不为空
                # 将 set 转为 list，一次性将该行对应的多个位置设为 1
                multi_hot[i, [self.go2id[go] for go in index_set if go in self.go2id]] = 1.0
                
        return multi_hot