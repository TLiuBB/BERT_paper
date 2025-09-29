import os
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

# ✅ 目标分布（ethnicity_num）
TARGET_ETH_PERCENT_NUM = {
    "men": {
        0: 64.82,
        1: 8.33,
        2: 2.05,
        3: 1.35,
        7: 0.96,
        4: 4.4,
        5: 5.32,
        6: 6.73,
        8: 6.04,
    },
    "women": {
        0: 65.65,
        1: 8.32,
        2: 1.62,
        3: 1.18,
        7: 1.04,
        4: 4.28,
        5: 6.57,
        6: 6.10,
        8: 5.24,
    }
}

# 🎯 特征列
FEATURES = ["age", "SBP", "DBP", "BMI", "Total/HDL_ratio"]

def reassign_ethnicity_to_match_target():
    base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT/data/original data"
    genders = ["men", "women"]

    for gender in genders:
        print(f"\n🎯 Processing: {gender.upper()}")
        file_path = os.path.join(base_path, f"{gender}_London_imputed.csv")
        df = pd.read_csv(file_path)
        df = df.reset_index(drop=False)  # 保留原始顺序

        target = TARGET_ETH_PERCENT_NUM[gender]
        total = len(df)
        target_counts = {k: int(round(v * total / 100)) for k, v in target.items()}

        # 当前分布
        eth_counts = df["ethnicity_num"].value_counts(normalize=True) * 100
        print("\n📊 Original Ethnicity Distribution (%):")
        for k in range(9):
            print(f"  {k}: {eth_counts.get(k, 0):.2f}%")

        # 处理白人
        white_df = df[df["ethnicity_num"] == 0].copy()
        other_df = df[df["ethnicity_num"] != 0].copy()
        white_keep_n = target_counts[0]
        white_excess_n = len(white_df) - white_keep_n

        print(f"\n👕 White count: {len(white_df):,}, keep: {white_keep_n:,}, excess: {white_excess_n:,}")

        if white_excess_n <= 0:
            print("✅ No reassignment needed.")
            final_df = df.sort_values("index").drop(columns="index")
            final_df.to_csv(os.path.join(base_path, f"{gender}_London_adjusted.csv"), index=False)
            continue

        white_excess_pool = white_df.sample(n=white_excess_n, random_state=42).copy()
        white_keep = white_df.drop(white_excess_pool.index)
        white_features = white_excess_pool[FEATURES].fillna(white_excess_pool[FEATURES].mean())

        used_idx = set()
        reassigned_rows = []

        for eth, tgt_n in target_counts.items():
            if eth == 0:
                continue
            curr_n = (df["ethnicity_num"] == eth).sum()
            needed = tgt_n - curr_n
            if needed <= 0:
                continue

            remaining = white_excess_pool.drop(index=list(used_idx), errors="ignore")
            if len(remaining) < needed:
                print(f"⚠️ Not enough white samples to assign to {eth}. Needed {needed}, available {len(remaining)}.")
                needed = len(remaining)  # 截断防崩

            # 代表样本
            example = other_df[other_df["ethnicity_num"] == eth][FEATURES].dropna()
            if len(example) == 0:
                example = white_features.sample(n=1, random_state=eth)
            else:
                example = example.mean().to_frame().T

            distances = pairwise_distances(remaining[FEATURES].fillna(0), example)
            top_idx = np.argsort(distances.ravel())[:needed]
            chosen = remaining.iloc[top_idx].copy()
            chosen["ethnicity_num"] = eth
            reassigned_rows.append(chosen)
            used_idx.update(chosen.index.tolist())

        # ✅ 未被分配的白人回补回来
        unassigned_white=white_excess_pool.drop(index=list(used_idx), errors="ignore")
        if len(unassigned_white) > 0:
            print(f"♻️ Re-adding {len(unassigned_white)} unassigned whites to preserve row count.")
            reassigned_rows.append(unassigned_white)  # 保持原样

        # 合并并恢复顺序
        final_df=pd.concat([white_keep, other_df] + reassigned_rows, ignore_index=False)
        final_df=final_df.sort_values("index").drop(columns="index")

        # ✅ Check
        if len(final_df) != len(df):
            raise ValueError(f"❌ Row mismatch after reassignment! Original = {len(df)}, Final = {len(final_df)}")

        final_counts = final_df["ethnicity_num"].value_counts(normalize=True) * 100
        print("\n📊 ✅ Final Ethnicity Distribution (%):")
        for k in range(9):
            print(f"  {k}: {final_counts.get(k, 0):.2f}%")

        save_path = os.path.join(base_path, f"{gender}_London_adjusted.csv")
        final_df.to_csv(save_path, index=False)
        print(f"💾 Saved adjusted file to: {save_path}")


# 🚀 Run it
reassign_ethnicity_to_match_target()
