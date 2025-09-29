import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

# 🎯 修正后的目标分布（根据实际 ethnicity_num 映射顺序）
TARGET_ETH_PERCENT = {
    "men": {
        0: 64.82,  # White
        1: 8.33,   # Indian
        2: 2.05,   # Pakistani
        3: 1.35,   # Bangladeshi
        4: 4.4,    # Other Asian
        5: 5.32,   # Black Caribbean
        6: 6.73,   # Black African
        7: 0.96,   # Chinese
        8: 6.04,   # Other
    },
    "women": {
        0: 65.65,
        1: 8.32,
        2: 1.62,
        3: 1.18,
        4: 4.28,
        5: 6.57,
        6: 6.1,
        7: 1.04,
        8: 5.24,
    }
}


def reassign_ethnicity_to_match_target():
    base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT/data/original data"
    genders = ["men", "women"]
    features_for_similarity = ["age", "SBP", "DBP", "BMI", "Total/HDL_ratio"]

    for gender in genders:
        print(f"\n🎯 Processing: {gender.upper()}")
        file_path = os.path.join(base_path, f"{gender}_London_imputed.csv")
        df = pd.read_csv(file_path)

        df['original_index'] = df.index  # 记录原始顺序

        # 统计当前分布
        original_dist = df['ethnicity_num'].value_counts(normalize=True) * 100
        print("\n📊 Original Ethnicity Distribution (%):")
        for eth, pct in original_dist.round(2).items():
            print(f"  {eth}: {pct:.2f}%")

        # 获取目标分布
        target = TARGET_ETH_PERCENT[gender]
        total = len(df)
        target_counts = {k: int(round(v * total / 100)) for k, v in target.items()}

        # Step 1: 当前白人太多，先拿出多余的白人
        white_df = df[df['ethnicity_num'] == 0].copy()
        other_df = df[df['ethnicity_num'] != 0].copy()

        white_keep_n = target_counts[0]
        excess_white_n = len(white_df) - white_keep_n

        print(f"\n👕 White count: {len(white_df):,}, keep: {white_keep_n:,}, excess: {excess_white_n:,}")

        if excess_white_n <= 0:
            print("✅ No need to reassign, white proportion already meets target.")
            continue

        # Step 2: 计算特征中心
        ethnicity_centers = {}
        for eth in target:
            if eth == 0:
                continue
            group = df[df['ethnicity_num'] == eth]
            if len(group) >= 5:
                ethnicity_centers[eth] = group[features_for_similarity].mean().values

        # Step 3: 从白人中选出 excess_n 个最适合 reassignment 的
        white_pool = white_df[features_for_similarity].values
        center_matrix = np.vstack(list(ethnicity_centers.values()))
        distances = pairwise_distances(white_pool, center_matrix)
        closest_eth_idx = np.argmin(distances, axis=1)
        closest_eth = [list(ethnicity_centers.keys())[i] for i in closest_eth_idx]

        white_df = white_df.reset_index(drop=True)
        white_df['reassign_to'] = closest_eth

        # Step 4: 精准控制每个种族的最终目标人数
        current_counts = df['ethnicity_num'].value_counts().to_dict()
        needed = {}
        for eth, tgt in target_counts.items():
            current = current_counts.get(eth, 0)
            needed[eth] = max(0, tgt - current)

        reassigned_rows = []
        used_idx = set()
        for eth, n in needed.items():
            if eth == 0 or n == 0:
                continue
            candidates = white_df[white_df['reassign_to'] == eth].drop(index=list(used_idx))
            selected = candidates.iloc[:n].copy()
            used_idx.update(selected.index)
            selected['ethnicity_num'] = eth
            reassigned_rows.append(selected.drop(columns=['reassign_to']))

        # 保留未被替换的白人
        white_keep = white_df.drop(index=list(used_idx)).drop(columns=['reassign_to'])

        # 合并所有
        new_df = pd.concat([white_keep, other_df] + reassigned_rows, ignore_index=True)

        # 按 original_index 排序，保证顺序不变
        new_df = new_df.sort_values('original_index').drop(columns=['original_index'])

        # 打印新分布
        new_dist = new_df['ethnicity_num'].value_counts(normalize=True) * 100
        print("\n📊 ✅ Final Ethnicity Distribution (%):")
        for eth, pct in new_dist.round(2).items():
            print(f"  {eth}: {pct:.2f}%")

        # 保存
        save_path = os.path.join(base_path, f"{gender}_London_adjusted.csv")
        new_df.to_csv(save_path, index=False)
        print(f"💾 Saved adjusted file to: {save_path}")


# 🚀 Run it
reassign_ethnicity_to_match_target()
