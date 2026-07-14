"""
诊断脚本：检查数据管道中的索引对齐问题
"""
import os
import sys
import torch
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DATASETS_PATH = "./data_tale/TALE/"
FEATURES_PATH = "/archive/hot5/fty/TALE/"
NAMESPACE = "CC"  # 改成 BP/MF 测试其他本体

print("=" * 70)
print(f"诊断命名空间: {NAMESPACE}")
print("=" * 70)

# ============================================================
# 1. 加载 train_seq 数据
# ============================================================
print("\n[1] 加载 train_seq 数据...")
train_seq_path = os.path.join(DATASETS_PATH, f"train_seq_{NAMESPACE.lower()}")
with open(train_seq_path, 'rb') as f:
    train_seq_data = pickle.load(f)
print(f"  train_seq_data 长度: {len(train_seq_data)}")
print(f"  train_seq_data[0] 类型: {type(train_seq_data[0])}")
if isinstance(train_seq_data[0], dict):
    print(f"  train_seq_data[0] keys: {train_seq_data[0].keys()}")
elif isinstance(train_seq_data, list):
    print(f"  train_seq_data[0] 示例: {train_seq_data[0][:3] if isinstance(train_seq_data[0], (list, str)) else train_seq_data[0]}")

# ============================================================
# 2. 加载 train_label 数据
# ============================================================
print("\n[2] 加载 train_label 数据...")
train_label_path = os.path.join(DATASETS_PATH, f"train_label_{NAMESPACE.lower()}")
with open(train_label_path, 'rb') as f:
    train_label_data = pickle.load(f)
print(f"  train_label_data 长度: {len(train_label_data)}")
print(f"  len(train_seq) vs len(train_label): {len(train_seq_data)} vs {len(train_label_data)}")
assert len(train_seq_data) == len(train_label_data), "序列和标签长度不匹配！"

# ============================================================
# 3. 加载 protein_feats
# ============================================================
print("\n[3] 加载 protein_feats...")
protein_feats_path = os.path.join(FEATURES_PATH, 'protein_feats', f'train_{NAMESPACE.lower()}_protein_feats.pt')
protein_feats = torch.load(protein_feats_path, weights_only=True, map_location='cpu')
print(f"  protein_feats 类型: {type(protein_feats)}")
print(f"  protein_feats 条目数: {len(protein_feats)}")
keys = sorted(protein_feats.keys(), key=int)
print(f"  protein_feats key 范围: [{keys[0]}, {keys[-1]}]")
print(f"  protein_feats key 是否从0连续: {keys == list(range(len(keys)))}")
print(f"  是否有 key 缺失: {len(keys) != len(set(range(keys[0], keys[-1]+1)))}")

# ============================================================
# 4. 检查 protein_feats vs train_seq_data 对齐
# ============================================================
print("\n[4] 检查 protein_feats vs train_seq_data 对齐...")
print(f"  train_seq_data 长度: {len(train_seq_data)}")
print(f"  protein_feats 条目数: {len(protein_feats)}")
if len(train_seq_data) != len(protein_feats):
    print(f"  ⚠️  长度不匹配！差异: {abs(len(train_seq_data) - len(protein_feats))}")
else:
    print(f"  ✓ 长度匹配")

# ============================================================
# 5. 检查 stacked_feats 构造
# ============================================================
print("\n[5] 检查 stacked_feats 构造...")
stacked_feats = torch.stack([protein_feats[k] for k in sorted(protein_feats.keys(), key=int)], dim=0)
print(f"  stacked_feats shape: {stacked_feats.shape}")
print(f"  stacked_feats[0] == protein_feats[0]: {torch.equal(stacked_feats[0], protein_feats[0])}")
if len(protein_feats) > 1:
    print(f"  stacked_feats[1] == protein_feats[1]: {torch.equal(stacked_feats[1], protein_feats[1])}")

# ============================================================
# 6. 检查 prototype_index 生成
# ============================================================
print("\n[6] 检查 prototype_index...")
num_classes_from_data = max(
    (label for item in train_seq_data for label in item.get('label', [])),
    default=-1
) + 1
print(f"  推断 num_classes: {num_classes_from_data}")

prototype_index = torch.zeros(len(train_seq_data), num_classes_from_data)
for i, item in enumerate(train_seq_data):
    labels = item.get('label', [])
    prototype_index[i, labels] = 1.0

print(f"  prototype_index shape: {prototype_index.shape}")
print(f"  prototype_index 每一行的蛋白质索引 == train_seq 位置索引 ? ✓ (构造一致)")

# 验证第0个蛋白质的标签
row0_labels = prototype_index[0].nonzero(as_tuple=True)[0].tolist()
print(f"  蛋白质0的标签: {row0_labels[:10]}{'...' if len(row0_labels) > 10 else ''} (共{len(row0_labels)}个)")
print(f"  train_seq_data[0]['label'] == row0_labels: {sorted(train_seq_data[0].get('label', [])) == sorted(row0_labels)}")

# ============================================================
# 7. 检查 pos_indices / easy_neg_indices / hard_neg_indices
# ============================================================
print("\n[7] 检查正/负样本索引...")
pos_indices = prototype_index.T  # (num_classes, num_proteins)
print(f"  pos_indices shape: {pos_indices.shape}")
print(f"  正样本总数: {pos_indices.sum().item()}")

# 检查正样本索引是否在有效范围内
max_pos_idx = pos_indices.nonzero()[:, 1].max().item()
print(f"  pos_indices 最大蛋白质索引: {max_pos_idx}")
print(f"  train_seq_data 最大有效索引: {len(train_seq_data) - 1}")
print(f"  ✓ 索引在范围内" if max_pos_idx < len(train_seq_data) else "  ⚠️  索引越界！")

# ============================================================
# 8. 检查 Dataset 创建的索引
# ============================================================
print("\n[8] 检查 Dataset...")
from datasets.balanced_dataset import Dataset

raw_dataset = Dataset(DATASETS_PATH, NAMESPACE, train_mode='test', dataset_mode='train')
print(f"  raw_dataset 长度: {len(raw_dataset)}")
print(f"  raw_dataset[0]: idx={raw_dataset[0][0]}, labels_count={len(raw_dataset[0][2])}")
print(f"  raw_dataset[-1]: idx={raw_dataset[-1][0]}, labels_count={len(raw_dataset[-1][2])}")

# 检查 Dataset 索引是否与 train_seq_data 索引对齐
for test_idx in [0, 10, 100, len(raw_dataset)-1]:
    ds_item = raw_dataset[test_idx]
    ds_label = ds_item[2]  # label_indices
    ts_label = train_label_data[test_idx]
    match = sorted(ds_label) == sorted(ts_label)
    if not match and test_idx < 10:
        print(f"  raw_dataset[{test_idx}] labels: {ds_label}")
        print(f"  train_label[{test_idx}] labels: {ts_label}")
    if not match:
        print(f"  ⚠️  raw_dataset[{test_idx}] 标签与 train_label[{test_idx}] 不匹配！")

# ============================================================
# 9. 检查 collator 访问 protein_feats 的索引
# ============================================================
print("\n[9] 检查 collator 的 protein_feats 索引访问...")
from datasets.balanced_collator import BalancedCollator

collator = BalancedCollator(protein_feats=protein_feats, data=raw_dataset, num_classes=num_classes_from_data)

# 模拟一个小 batch
sample_batch = [raw_dataset[i] for i in range(min(5, len(raw_dataset)))]
try:
    inputs, labels, mask = collator(sample_batch)
    print(f"  collator 输出: inputs shape={inputs.shape}, labels shape={labels.shape}, mask={mask}")
    print(f"  ✓ collator 正常工作")
except Exception as e:
    print(f"  ⚠️  collator 出错: {e}")

# ============================================================
# 10. 检查 test_protein_feats
# ============================================================
print("\n[10] 检查 test 数据...")
test_protein_feats_path = os.path.join(FEATURES_PATH, 'protein_feats', f'test_{NAMESPACE.lower()}_protein_feats.pt')
test_protein_feats = torch.load(test_protein_feats_path, weights_only=True, map_location='cpu')
print(f"  test_protein_feats 条目数: {len(test_protein_feats)}")
test_keys = sorted(test_protein_feats.keys(), key=int)
print(f"  test key 范围: [{test_keys[0]}, {test_keys[-1]}]")
print(f"  test key 是否从0连续: {test_keys == list(range(len(test_keys)))}")

# 加载 test dataset
test_dataset = Dataset(DATASETS_PATH, NAMESPACE, train_mode='test', dataset_mode='test')
print(f"  test_dataset 长度: {len(test_dataset)}")
print(f"  test_dataset == test_protein_feats 条目: {len(test_dataset) == len(test_protein_feats)}")

# ============================================================
# 11. 检查 go_embeddings 是否被正确使用
# ============================================================
print("\n[11] 检查 go_embeddings...")
go_emb_path = os.path.join(DATASETS_PATH, f'{NAMESPACE.lower()}_go_feats.pt')
if os.path.exists(go_emb_path):
    go_embeddings = torch.load(go_emb_path, weights_only=True, map_location='cpu')
    print(f"  go_embeddings shape: {go_embeddings.shape}")
    print(f"  num_classes 一致性: go_emb[{go_embeddings.shape[0]}] vs proto_idx[{num_classes_from_data}]")
    if go_embeddings.shape[0] != num_classes_from_data:
        print(f"  ⚠️  go_embeddings 类别数 ({go_embeddings.shape[0]}) 与 num_classes ({num_classes_from_data}) 不匹配！")
else:
    print(f"  ⚠️  未找到 {go_emb_path}")

# ============================================================
# 12. 检查 queue_index
# ============================================================
print("\n[12] 检查 queue_index...")
queue_idx_path = os.path.join(DATASETS_PATH, f'{NAMESPACE.lower()}_queue_index.pt')
if os.path.exists(queue_idx_path):
    queue_indices = torch.load(queue_idx_path)
    print(f"  queue_indices shape: {queue_indices.shape}")
    print(f"  queue_indices 最大值: {queue_indices.max().item()}")
    print(f"  stacked_feats 有效索引范围: [0, {stacked_feats.shape[0]-1}]")
    if queue_indices[queue_indices >= 0].max().item() >= stacked_feats.shape[0]:
        print(f"  ⚠️  queue_indices 中存在越界索引！")
    else:
        print(f"  ✓ queue_indices 索引在有效范围内")
else:
    print(f"  未找到 queue_index，将在运行时计算")

# ============================================================
# 13. 总结
# ============================================================
print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)
print("""
请重点关注以下检查项:
1. protein_feats key 是否从 0 连续
2. train_seq_data 长度是否等于 protein_feats 条目数
3. raw_dataset[idx].label 是否等于 train_label[idx]
4. go_embeddings 类别数是否与 num_classes 一致
5. queue_indices 是否有越界

如果所有项都通过，问题可能出在:
- 模型架构本身（如随机初始化的 go_embeddings 导致退化为先验概率）
- 损失函数设计
- 学习率/优化器设置
""")
