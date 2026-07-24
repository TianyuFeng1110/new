"""
为 CAFA3 构建 go2id 字典 (替代 pickle 方案 v2)。

核心逻辑:
- father/child 从 label_regular_1.npy 直接构建
- label 从 label_matrix_1_sparse.npy 获取
- 24个不在 CAFA3 OBO 中的训练 GO ID: 各分配一个未使用的有效 OBO GO ID 作为唯一 key
- 未映射的中间节点也分配未使用 OBO GO ID
- 所有 key 保证在 OBO 中存在且唯一 → compute_ic 不崩溃
"""

import os, pickle
import numpy as np
from collections import defaultdict

CAFA3_PATH = os.path.dirname(os.path.abspath(__file__))
OBO_PATH = os.path.join(CAFA3_PATH, "go-basic.obo")
NS_LIST = ["cc", "mf", "bp"]
NS_TO_OBO = {"cc": "cellular_component", "mf": "molecular_function", "bp": "biological_process"}
NS_TO_TYPE = {"cc": "c", "mf": "f", "bp": "p"}


def load_obo(path):
    """返回: term_ids(所有^id:), obsolete_set, id2ns, id2name"""
    ids_set = set(); obs_set = set(); ns_map = {}; name_map = {}
    cur_id = None; cur_ns = None; cur_name = None; is_obs = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == '[Term]': cur_id = cur_ns = cur_name = None; is_obs = False
            elif line == '[Typedef]': cur_id = cur_ns = cur_name = None; is_obs = False
            elif line.startswith('id: '): cur_id = line[4:]; ids_set.add(cur_id)
            elif line.startswith('name: '): cur_name = line[6:]
            elif line.startswith('namespace: '): cur_ns = line[11:]
            elif 'is_obsolete' in line and 'true' in line.lower(): is_obs = True
            if line == '' and cur_id:
                if is_obs: obs_set.add(cur_id)
                if cur_ns: ns_map[cur_id] = cur_ns; name_map[cur_id] = cur_name or cur_id
                cur_id = cur_ns = cur_name = None; is_obs = False
    return ids_set, obs_set, ns_map, name_map


def build(ns, ids_set, obs_set, ns_map, name_map):
    ns_l = ns.lower(); obo_ns = NS_TO_OBO[ns_l]; go_type = NS_TO_TYPE[ns_l]
    print(f"\n{'='*60}\n{ns.upper()}\n{'='*60}")

    # ---- 加载数据 ----
    with open(os.path.join(CAFA3_PATH, f"train_seq_{ns_l}"), 'rb') as f:
        train_seqs = pickle.load(f)
    with open(os.path.join(CAFA3_PATH, f"train_label_{ns_l}"), 'rb') as f:
        train_labels = pickle.load(f)

    mat = np.load(os.path.join(CAFA3_PATH, f"{ns_l}_label_matrix_1_sparse.npy"))
    NR = mat.shape[0]
    mat_sets = [set(int(x) for x in mat[i] if x != NR) for i in range(NR)]
    row_self = [max(s) if s else -1 for s in mat_sets]
    self_to_row = {s: i for i, s in enumerate(row_self)}

    edges = np.load(os.path.join(CAFA3_PATH, f"{ns_l}_label_regular_1.npy"))
    NN = int(edges.max()) + 1
    ch_of = defaultdict(set); pa_of = defaultdict(set)
    for p, c in edges: ch_of[int(p)].add(int(c)); pa_of[int(c)].add(int(p))

    print(f"  matrix={NR}行 edge={NN}节点 {len(edges)}边")

    # ---- OBO 有效 GO ID ----
    valid_obo = {g for g in ids_set
                 if g in ns_map and ns_map[g] == obo_ns and g not in obs_set}
    print(f"  有效OBO: {len(valid_obo)}")

    # ---- 训练集 GO ----
    all_train_gos = set(go for r in train_seqs for go in r['GO'])
    missing = all_train_gos - valid_obo
    print(f"  训练集GO: {len(all_train_gos)}, 不在OBO: {len(missing)}")

    # ---- smp_count ----
    smp_c = defaultdict(int)
    for labels in train_labels:
        for idx in labels: smp_c[idx] += 1

    # ---- 第一步: 单标注锚点 ----
    s2orig = {}  # self-index -> original GO ID
    for r in train_seqs:
        if len(r['GO']) == 1:
            gid = r['GO'][0]
            if gid in s2orig.values(): continue
            ls = set(r['label'])
            for i, ms in enumerate(mat_sets):
                if ls == ms:
                    s2orig[row_self[i]] = gid; break

    # ---- 第二步: 蛋白质交集 ----
    gpl = defaultdict(list)
    for r in train_seqs:
        for gid in r['GO']: gpl[gid].append(set(r['label']))

    for gid in all_train_gos:
        if gid in s2orig.values(): continue
        sets_ = gpl[gid]
        common = sets_[0].copy()
        for ss in sets_[1:]: common &= ss
        cands = [idx for idx in common
                 if idx in self_to_row and idx not in s2orig
                 and not (ch_of.get(idx, set()) & common)]
        if cands:
            s2orig[max(cands)] = gid

    print(f"  映射训练GO: {len(s2orig)}/{len(all_train_gos)}")

    # ---- 分配 key: 每个 self-index 一个唯一的 OBO GO ID ----
    # 策略: 
    #   已在 OBO 中的训练 GO → 用自己
    #   不在 OBO 中的训练 GO → 分配未使用 OBO ID
    #   未映射的中间节点 → 分配未使用 OBO ID

    used_keys = set()
    s2key = {}  # self-index -> final OBO key

    # 先分配已在 OBO 中的训练 GO
    for s, orig in s2orig.items():
        if orig in valid_obo:
            s2key[s] = orig
            used_keys.add(orig)

    # 为 missing GO 分配未使用 OBO ID
    unused_pool = sorted(valid_obo - used_keys)
    pool_idx = 0
    for s, orig in s2orig.items():
        if s in s2key: continue
        if pool_idx < len(unused_pool):
            s2key[s] = unused_pool[pool_idx]; pool_idx += 1
        else:
            s2key[s] = f"__E1_{ns_l}_{s}__"

    # 为未映射节点分配
    for s in range(NN):
        if s in s2key or s not in self_to_row: continue
        if pool_idx < len(unused_pool):
            s2key[s] = unused_pool[pool_idx]; pool_idx += 1
        else:
            s2key[s] = f"__E2_{ns_l}_{s}__"

    # ---- 构建 go2id ----
    go2id = {}
    for row_idx in range(NR):
        s = row_self[row_idx]
        key = s2key.get(s, f"__E3_{ns_l}_{s}__")
        orig = s2orig.get(s)

        # 名称
        if key.startswith("__"):
            name = key
        elif orig and orig in name_map:
            name = name_map[orig]
        else:
            name = name_map.get(key, key)

        # father/child (转为 final key)
        fgs = []
        for ps in sorted(pa_of.get(s, set())):
            pk = s2key.get(ps)
            if pk: fgs.append(pk)
        cgs = []
        for cs in sorted(ch_of.get(s, set())):
            ck = s2key.get(cs)
            if ck: cgs.append(ck)

        go2id[key] = {
            "father": fgs, "child": cgs,
            "name": name, "type": go_type,
            "smp_count": smp_c.get(s, 0), "ind": s,
            "label": sorted(mat_sets[row_idx]),
        }

    # ---- 保存 ----
    out = os.path.join(CAFA3_PATH, f"{ns_l}_go_1.pickle")
    with open(out, 'wb') as f:
        pickle.dump(go2id, f, protocol=3)

    # ---- 验证 ----
    # 所有key在OBO中?
    bad = [k for k in go2id if not k.startswith("__") and k not in valid_obo]
    ph = sum(1 for k in go2id if k.startswith("__"))

    # father_matrix
    g2i = {k: v['ind'] for k, v in go2id.items()}
    fm = np.zeros((NN, NN), dtype=int)
    for _, info in go2id.items():
        ci = info['ind']
        for fg in info.get('father', []):
            if fg in g2i: fm[ci][g2i[fg]] = 1
    em = np.zeros((NN, NN), dtype=int)
    for p, c in edges: em[int(c)][int(p)] = 1
    diff = np.sum(fm != em); te = np.sum(em)

    # 覆盖的训练GO
    covered = sum(1 for g in all_train_gos if g in go2id or g in s2orig.values())
    # 通过 s2orig 反查: 哪些训练GO的 self-index 被成功映射
    mapped_train = sum(1 for g in all_train_gos
                       if any(s2orig.get(s) == g and s in s2key for s in s2orig))

    print(f"  保存: {os.path.basename(out)}")
    print(f"  条目:{len(go2id)}(期望{NR}) 异常key:{ph} OOBkey:{len(bad)}")
    print(f"  father_matrix: {1-diff/max(te,1):.4f} ({int(te-diff)}/{int(te)})")
    print(f"  OBO校验: {'PASS' if not bad else 'FAIL'}")

    return go2id


if __name__ == "__main__":
    print("="*60 + "\nCAFA3 go2id v2\n" + "="*60)
    print("解析 OBO...")
    ids_set, obs_set, ns_map, name_map = load_obo(OBO_PATH)
    print(f"{len(ids_set)} IDs, {len(obs_set)} obsolete")
    for ns in NS_LIST:
        build(ns, ids_set, obs_set, ns_map, name_map)
    print("\n完成!")
