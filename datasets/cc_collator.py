import torch

class collator:

    def __init__(self, num_classes, residue_feats):
        self.num_classes = num_classes
        self.residue_feats = residue_feats

    def __call__(self, data):
        # data 中每个元素: (idx, mode, seq, label_indices, go_ids)
        batch_residue_feats, mask = self._process_residue_feats(
            [self.residue_feats[elem[0]] for elem in data]
        )
        labels = self._get_labels(data)

        return batch_residue_feats, mask, labels

    def _process_residue_feats(self, residue_feats):
        """将不同长度的残基特征统一为固定长度，padding 部分填充 0。

        Args:
            residue_feats: List[tensor[L_i, d]]，各蛋白质的残基特征，长度 L_i 各不相同

        Returns:
            padded_feats: tensor[B, max_len, d]
            mask:         tensor[B, max_len]，bool，True=有效残基，False=padding
        """
        batch_size = len(residue_feats)
        target_len = max(f.size(0) for f in residue_feats)
        feat_dim = residue_feats[0].size(-1)

        padded_feats = torch.zeros(batch_size, target_len, feat_dim)
        mask = torch.zeros(batch_size, target_len, dtype=torch.bool)

        for i, feat in enumerate(residue_feats):
            cur_len = min(feat.size(0), target_len)
            padded_feats[i, :cur_len] = feat[:cur_len]
            mask[i, :cur_len] = True

        return padded_feats, mask

    def _get_labels(self, data):
        """将整数标签索引列表转为 multi-hot 向量。

        Args:
            data: list of (idx, mode, seq, label_indices, go_ids)

        Returns:
            multi_hot: tensor[B, num_classes]，值为 0.0 或 1.0
        """
        multi_hot = torch.zeros((len(data), self.num_classes))

        for i, (_, _, label_indices) in enumerate(data):
            if label_indices:
                multi_hot[i, label_indices] = 1.0

        return multi_hot