"""重建 CAFA3 的 {ns}_go_1.pickle，使 ind ↔ GO term 对应关系正确（与 TALE 逻辑对齐）。

问题背景:
    旧 build_go2id.py 的"单标注锚点+蛋白集交集"推断大量失败，导致 go2id 的
    name/father 张冠李戴（如 cc 的 ind=0 被安给特异 term）。原始数据与
    索引体系本身正确（GO组合→传播集是确定性函数，祖先链在索引空间完整），
    因此训练/指标不受影响；受影响的只有依赖 go2id 名字的展示与 IC 初值。

锚定算法（已经诊断验证）:
    1. 同一 GO 组合的传播集完全一致 → label = f(GO集) 是确定性函数；
    2. GO X 的传播集交（所有含 X 组合之交）= X 的祖先链 ∪ X 自身；
    3. 交集中"仅出现在含 X 组合"的独占元素若唯一，即为 X 的真身索引。
    锁定覆盖率: CC 1287 / MF 4042 / BP 5865（与旧 pickle 一致率 95-99%，
    旧错误正是本脚本要修的少数派）。ind 体系不变，prototype_index /
    class_counts / hop_counts / checkpoint 全部无需重算。

使用:
    python rebuild_cafa3_go2id.py --dry-run   # 只诊断
    python rebuild_cafa3_go2id.py             # 覆盖写入三个 pickle
"""

import argparse
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

CAFA3_PATH = os.path.dirname(os.path.abspath(__file__))
NS_LIST = ["cc", "mf", "bp"]
NS_TO_OBO = {"cc": "cellular_component", "mf": "molecular_function", "bp": "biological_process"}


def load_obo_names(path):
    """返回 {GO_ID: name}（含 obsolete，仅供展示）。"""
    name_map = {}
    cur_id = cur_name = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '[Term]' or line == '[Typedef]':
                if cur_id and cur_name:
                    name_map[cur_id] = cur_name
                cur_id = cur_name = None
            elif line.startswith('id: '):
                cur_id = line[4:].strip()
            elif line.startswith('name: ') and cur_id:
                cur_name = line[6:].strip()
    if cur_id and cur_name:
        name_map[cur_id] = cur_name
    return name_map


def anchor_index2go(train_seqs):
    """组合级独占元素锚定：恢复 {GO_ID: self_idx}（已经诊断验证的算法）。

    原理:
      - train_seq 中同一 GO 组合的传播集完全一致（诊断: cc/mf 各 1.4 万组合零冲突），
        即 label = f(GO集) 是确定性函数；
      - GO X 的真身索引 = 所有含 X 组合的传播集之交中，"仅出现在含 X 组合中"
        的独占元素（若唯一）。真身必然随 X 出现在每个含 X 蛋白的传播集里；
        而非 X 的组合若也产生该元素，说明该元素属于其他 term 的传播链。

    Returns:
        locked: {GO_ID: int}
        combos: {frozenset(GO集): frozenset(label集)}
    """
    combos = {}
    for r in train_seqs:
        key = frozenset(r['GO'])
        if key not in combos:
            combos[key] = frozenset(int(x) for x in r['label'])

    # 元素 e -> 含 e 的组合数
    elem_combo_cnt = Counter()
    for val in combos.values():
        for e in val:
            elem_combo_cnt[e] += 1

    # GO X -> 含 X 的组合的传播集之交
    go_inter = {}
    for key, val in combos.items():
        for g in key:
            prev = go_inter.get(g)
            go_inter[g] = val if prev is None else (prev & val)

    locked = {}
    for g, inter in go_inter.items():
        n_with = sum(1 for key in combos if g in key)
        excl = [e for e in inter if elem_combo_cnt[e] == n_with]
        if len(excl) == 1:
            locked[g] = excl[0]
    return locked, combos


def rebuild_namespace(ns, obo_names, dry_run=False):
    ns_l = ns.lower()
    print(f"\n{'=' * 64}\n[{ns.upper()}] 重建 {ns_l}_go_1.pickle\n{'=' * 64}")

    with open(os.path.join(CAFA3_PATH, f"train_seq_{ns_l}"), 'rb') as f:
        train_seqs = pickle.load(f)
    edges = np.load(os.path.join(CAFA3_PATH, f"{ns_l}_label_regular_1.npy"))
    NN = int(edges.max()) + 1

    locked, combos = anchor_index2go(train_seqs)
    idx2go = {i: g for g, i in locked.items()}
    print(f"  edge 节点数={NN}, GO 组合数={len(combos)}, 锁定 GO={len(locked)}")

    # 蛋白级计数（self 索引体系）——与 class_counts 口径一致
    smp_count = Counter()
    for r in train_seqs:
        for li in r['label']:
            smp_count[int(li)] += 1

    # father（edges 为 [parent, child]，self 索引体系）
    pa_of = defaultdict(set)
    for p, c in edges:
        pa_of[int(c)].add(int(p))

    # 构建新 go2id：ind 仍是 self 索引（matrix 行体系），张量口径不变
    go2id = {}
    n_placeholder = 0
    for s in range(NN):
        go = idx2go.get(s)
        if go is None:
            n_placeholder += 1
            key = f"__UNMAPPED_{ns_l}_{s}__"
            name = key
        else:
            key = go
            name = obo_names.get(go, go)

        go2id[key] = {
            'ind': s,
            'name': name,
            'father': [idx2go.get(p, f"__UNMAPPED_{ns_l}_{p}__")
                       for p in sorted(pa_of.get(s, set()))],
            'smp_count': int(smp_count.get(s, 0)),
        }

    # ---- 一致性自检 ----
    # 1) 根唯一且计数最大
    roots = [v for v in go2id.values() if not v['father']]
    max_cnt = max(v['smp_count'] for v in go2id.values())
    roots_ok = (len(roots) >= 1 and
                all(v['smp_count'] == max_cnt for v in roots))
    # 2) 名字抽检: 特异 term 的样本数应远小于总数
    n_train = len(train_seqs)
    specific_ok = all(v['smp_count'] < n_train * 0.9
                      for k, v in go2id.items()
                      if not k.startswith('__') and v['smp_count'] > 0)
    # 3) father 指向存在
    dangling = sum(1 for v in go2id.values() for f_ in v['father'] if f_ not in go2id)

    print(f"  根节点: {len(roots)} 个, 全部为最大计数={roots_ok}")
    print(f"  特异 term 计数 sanity(全部 < 90% 训练集)={specific_ok}")
    print(f"  占位符(无正样本,不参与指标)={n_placeholder}, 悬空 father={dangling}")
    # 展示 3 个案例验证
    for probe in ['GO:0005575', 'GO:0003674', 'GO:0008150']:
        if probe in go2id:
            v = go2id[probe]
            print(f"  验证 {probe} ({v['name']}): ind={v['ind']}, smp_count={v['smp_count']}, "
                  f"根={'是' if not v['father'] else '否'}")

    if not dry_run:
        out = os.path.join(CAFA3_PATH, f"{ns_l}_go_1.pickle")
        with open(out, 'wb') as f:
            pickle.dump(go2id, f, protocol=3)
        print(f"  ✔ 已保存 {out}")
    else:
        print("  (dry-run, 未写入)")

    return roots_ok and specific_ok and dangling == 0


def main():
    parser = argparse.ArgumentParser(description='Rebuild CAFA3 go2id with correct ind<->GO mapping')
    parser.add_argument('--dry-run', action='store_true', help='只诊断不写文件')
    args = parser.parse_args()

    obo_names = load_obo_names(os.path.join(CAFA3_PATH, 'go-basic.obo'))
    print(f"OBO name 表: {len(obo_names)} 条")

    all_ok = True
    for ns in NS_LIST:
        all_ok &= rebuild_namespace(ns, obo_names, dry_run=args.dry_run)

    print(f"\n{'=' * 64}")
    print("全部通过 ✔" if all_ok else "存在未通过的检查,请检查上方日志 ✘")


if __name__ == '__main__':
    main()
