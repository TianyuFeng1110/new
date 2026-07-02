import torch

class collator:

    def __init__(self, num_classes, feats):
        self.num_classes = num_classes
        self.protein_feats = feats

    def __call__(self, data):
        # data 中每个元素: (idx, seq, label_indices)
        indices = torch.tensor([elem[0] for elem in data])
        batch_feats = torch.stack([self.protein_feats[elem[0]] for elem in data], dim=0)
        labels = self._get_labels(data)

        return batch_feats, labels, indices

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