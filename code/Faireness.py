import os
import pickle
from pprint import pprint

import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sympy import resultant
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


ETHNICITY_MAP = {
    0: "White",
    1: "Indian",
    2: "Pakistani",
    3: "Bangladeshi",
    4: "Other Asian",
    5: "Black Caribbean",
    6: "Black African",
    7: "Chinese",
    8: "Other"
}


def bootstrap_ci(metric_func, labels, preds, n_bootstraps=30, alpha=0.95, seed=42):
    """
    Bootstrap-based confidence interval estimation using tqdm and pandas-compatible indexing.

    Args:
        metric_func: function (e.g., roc_auc_score, concordance_index)
        labels (array-like or pd.Series)
        preds (array-like or pd.Series)
        n_bootstraps: int, number of bootstrap samples
        alpha: float, confidence level
        seed: int, random seed

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []

    # Ensure pandas-compatible indexing
    labels = pd.Series(labels).reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True)

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping CI", file=sys.stdout):
        indices = rng.randint(0, len(preds), len(preds))
        if len(np.unique(labels.iloc[indices])) < 2:
            continue  # Skip if only one class present
        try:
            score = metric_func(labels.iloc[indices], preds.iloc[indices])
            bootstrapped_scores.append(score)
        except:
            continue

    if len(bootstrapped_scores) < 5:
        return None, None  # Too unstable

    lower = np.percentile(bootstrapped_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(bootstrapped_scores, (1 + alpha) / 2 * 100)
    return round(lower, 4), round(upper, 4)


def check_external_ethnicity(label="cvd_all"):
    base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT"
    genders = ["men", "women"]

    for gender in genders:
        print(f"\n🧬 {gender.upper()} - External {label}")

        # 读取 external 数据
        external_path = os.path.join(base_path, "data", "original data", f"{gender}_London_adjusted.csv")
        df = pd.read_csv(external_path)
        df["patid"] = df["patid"].astype(str)

        # 校验 result 文件
        pkl_path = os.path.join(base_path, "saved", f"A100_{gender}", label, "external_cal_result.pkl")
        if not os.path.exists(pkl_path):
            print(f"❌ Missing result: {pkl_path}")
            continue

        with open(pkl_path, "rb") as f:
            result = pickle.load(f)

        # 行数对比
        n_df = len(df)
        n_res = len(result.get("all_labels", []))
        if n_df == n_res:
            print(f"✅ Row count match: {n_df} rows")
        else:
            print(f"❗ Row count mismatch: CSV = {n_df:,}, Result = {n_res:,}, Diff = {abs(n_df - n_res):,}")

        # 打印 ethnicity 分布
        eth_counts = df["ethnicity_num"].value_counts(dropna=False).sort_index()
        eth_percent = (eth_counts / n_df * 100).round(2)

        print(f"\n📊 Ethnicity Distribution (from external df):")
        for eth in eth_counts.index:
            eth_label = eth if pd.notna(eth) else "Missing"
            print(f"  {eth_label}: {eth_counts[eth]:,} ({eth_percent[eth]}%)")


def auc_fairness_summary_narrow(label="cvd_all"):
    import os, pickle
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score

    base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT"
    genders = ["men", "women"]
    townsend_groups = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    summary_dict = {}

    for gender in genders:
        print(f"\n🧬 {gender.upper()} - External {label}")

        df_path = os.path.join(base_path, "data", "original data", f"{gender}_London_adjusted.csv")
        df = pd.read_csv(df_path)
        df["patid"] = df["patid"].astype(str)

        result_path = os.path.join(base_path, "saved", f"A100_{gender}", label, "external_cal_result.pkl")
        with open(result_path, "rb") as f:
            result = pickle.load(f)

        if len(df) != len(result["all_labels"]):
            print(f"❗ Row mismatch: {len(df)} vs {len(result['all_labels'])}")
            continue

        df["label"] = result["all_labels"]
        df["probs"] = result["all_probs"]
        df["townsend"] = df["townsend"].astype(int)

        gender_dict = {}

        # Overall
        overall_auc = roc_auc_score(df["label"], df["probs"])
        ci_low, ci_high = bootstrap_ci(roc_auc_score, df["label"], df["probs"])
        gender_dict["Overall"] = {"auc": round(overall_auc, 4), "ci": (ci_low, ci_high)}

        # Ethnicity
        for name, nums in {
            "White": [0],
            "South Asian": [1, 2, 3],
            "Black": [5, 6]
        }.items():
            sub = df[df["ethnicity_num"].isin(nums)]
            if sub["label"].nunique() < 2:
                continue
            auc = roc_auc_score(sub["label"], sub["probs"])
            low, high = bootstrap_ci(roc_auc_score, sub["label"], sub["probs"])
            gender_dict[name] = {"auc": round(auc, 4), "ci": (low, high)}

        # Townsend
        for g1, g2 in townsend_groups:
            sub = df[df["townsend"].isin([g1, g2])]
            if sub["label"].nunique() < 2:
                continue
            auc = roc_auc_score(sub["label"], sub["probs"])
            low, high = bootstrap_ci(roc_auc_score, sub["label"], sub["probs"])
            group_name = f"Townsend {g1}/{g2}"
            gender_dict[group_name] = {"auc": round(auc, 4), "ci": (low, high)}

        summary_dict[gender] = gender_dict

    print("\n📦 Final Summary Dictionary:\n")
    from pprint import pprint
    pprint(summary_dict, sort_dicts=False)


def auc_fairness_summary_wide(label="cvd_all"):
    base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT"
    genders = ["men", "women"]
    townsend_bins = list(range(10))  # 0 to 9
    ethnicity_ids = list(range(9))  # 0 to 8

    summary_dict = {}

    for gender in genders:
        print(f"\n🧬 {gender.upper()} - External {label}")

        df_path = os.path.join(base_path, "data", "original data", f"{gender}_London_adjusted.csv")
        df = pd.read_csv(df_path)
        df["patid"] = df["patid"].astype(str)

        result_path = os.path.join(base_path, "saved", f"A100_{gender}", label, "external_cal_result.pkl")
        with open(result_path, "rb") as f:
            result = pickle.load(f)

        if len(df) != len(result["all_labels"]):
            print(f"❗ Row mismatch: {len(df)} vs {len(result['all_labels'])}")
            continue

        df["label"] = result["all_labels"]
        df["probs"] = result["all_probs"]
        df["townsend"] = df["townsend"].astype(int)

        gender_dict = {}

        # Overall
        overall_auc = roc_auc_score(df["label"], df["probs"])
        ci_low, ci_high = bootstrap_ci(roc_auc_score, df["label"], df["probs"])
        gender_dict["Overall"] = {"auc": round(overall_auc, 4), "ci": (ci_low, ci_high)}

        # Ethnicity (0 to 8)
        for eth in ethnicity_ids:
            sub = df[df["ethnicity_num"] == eth]
            if sub["label"].nunique() < 2:
                continue
            auc = roc_auc_score(sub["label"], sub["probs"])
            low, high = bootstrap_ci(roc_auc_score, sub["label"], sub["probs"])
            gender_dict[f"Ethnicity {eth}"] = {"auc": round(auc, 4), "ci": (low, high)}

        # Townsend quantiles (0 to 9)
        for t in townsend_bins:
            sub = df[df["townsend"] == t]
            if sub["label"].nunique() < 2:
                continue
            auc = roc_auc_score(sub["label"], sub["probs"])
            low, high = bootstrap_ci(roc_auc_score, sub["label"], sub["probs"])
            gender_dict[f"Townsend {t}"] = {"auc": round(auc, 4), "ci": (low, high)}

        summary_dict[gender] = gender_dict

    print("\n📦 Final Summary Dictionary:\n")
    pprint(summary_dict, sort_dicts=False)


summary_dict_narrow = {
    'men': {
        'Overall': {'auc': 0.7356, 'ci': (0.7287, 0.7406)},
        'White': {'auc': 0.7385, 'ci': (0.7327, 0.7455)},
        'South Asian': {'auc': 0.7021, 'ci': (0.69, 0.7135)},
        'Black': {'auc': 0.7348, 'ci': (0.7193, 0.7528)},
        'Townsend 0/1': {'auc': 0.7455, 'ci': (0.7333, 0.754)},
        'Townsend 2/3': {'auc': 0.7423, 'ci': (0.734, 0.7528)},
        'Townsend 4/5': {'auc': 0.7298, 'ci': (0.7181, 0.7433)},
        'Townsend 6/7': {'auc': 0.7056, 'ci': (0.6916, 0.7197)},
        'Townsend 8/9': {'auc': 0.7421, 'ci': (0.7141, 0.7673)}
    },
    'women': {
        'Overall': {'auc': 0.775, 'ci': (0.768, 0.7802)},
        'White': {'auc': 0.7751, 'ci': (0.769, 0.7821)},
        'South Asian': {'auc': 0.733, 'ci': (0.7216, 0.7532)},
        'Black': {'auc': 0.7686, 'ci': (0.7483, 0.7824)},
        'Townsend 0/1': {'auc': 0.7836, 'ci': (0.7738, 0.794)},
        'Townsend 2/3': {'auc': 0.7738, 'ci': (0.7684, 0.7885)},
        'Townsend 4/5': {'auc': 0.7807, 'ci': (0.7675, 0.792)},
        'Townsend 6/7': {'auc': 0.7559, 'ci': (0.7434, 0.7687)},
        'Townsend 8/9': {'auc': 0.7261, 'ci': (0.7008, 0.7579)}
    }
}


summary_dict_wide = {
    'men': {
        'Overall': {'auc': 0.7356, 'ci': (0.7287, 0.7406)},
        'Ethnicity 0': {'auc': 0.7385, 'ci': (0.7327, 0.7455)},
        'Ethnicity 1': {'auc': 0.6941, 'ci': (0.6715, 0.7093)},
        'Ethnicity 2': {'auc': 0.682, 'ci': (0.6364, 0.7204)},
        'Ethnicity 3': {'auc': 0.7284, 'ci': (0.689, 0.7656)},
        'Ethnicity 4': {'auc': 0.7302, 'ci': (0.6872, 0.7611)},
        'Ethnicity 5': {'auc': 0.7111, 'ci': (0.6794, 0.7457)},
        'Ethnicity 6': {'auc': 0.7426, 'ci': (0.7245, 0.7653)},
        'Ethnicity 7': {'auc': 0.7463, 'ci': (0.6659, 0.8142)},
        'Ethnicity 8': {'auc': 0.7239, 'ci': (0.6986, 0.7439)},
        'Townsend 0': {'auc': 0.7496, 'ci': (0.7374, 0.7615)},
        'Townsend 1': {'auc': 0.7414, 'ci': (0.7284, 0.7592)},
        'Townsend 2': {'auc': 0.7345, 'ci': (0.7177, 0.7426)},
        'Townsend 3': {'auc': 0.7518, 'ci': (0.7324, 0.7677)},
        'Townsend 4': {'auc': 0.7295, 'ci': (0.7135, 0.7467)},
        'Townsend 5': {'auc': 0.7295, 'ci': (0.7113, 0.7468)},
        'Townsend 6': {'auc': 0.7115, 'ci': (0.6906, 0.7248)},
        'Townsend 7': {'auc': 0.6969, 'ci': (0.6791, 0.7147)},
        'Townsend 8': {'auc': 0.739, 'ci': (0.7142, 0.7622)},
        'Townsend 9': {'auc': 0.757, 'ci': (0.6936, 0.8091)},
    },
    'women': {
        'Overall': {'auc': 0.775, 'ci': (0.768, 0.7802)},
        'Ethnicity 0': {'auc': 0.7751, 'ci': (0.769, 0.7821)},
        'Ethnicity 1': {'auc': 0.7231, 'ci': (0.698, 0.7472)},
        'Ethnicity 2': {'auc': 0.7236, 'ci': (0.6559, 0.7855)},
        'Ethnicity 3': {'auc': 0.7227, 'ci': (0.5989, 0.827)},
        'Ethnicity 4': {'auc': 0.7712, 'ci': (0.7305, 0.811)},
        'Ethnicity 5': {'auc': 0.7397, 'ci': (0.7138, 0.758)},
        'Ethnicity 6': {'auc': 0.7894, 'ci': (0.7633, 0.8087)},
        'Ethnicity 7': {'auc': 0.8035, 'ci': (0.7447, 0.8636)},
        'Ethnicity 8': {'auc': 0.7937, 'ci': (0.7791, 0.8195)},
        'Townsend 0': {'auc': 0.7854, 'ci': (0.7669, 0.801)},
        'Townsend 1': {'auc': 0.7814, 'ci': (0.7694, 0.7959)},
        'Townsend 2': {'auc': 0.7587, 'ci': (0.7394, 0.775)},
        'Townsend 3': {'auc': 0.7912, 'ci': (0.7752, 0.8017)},
        'Townsend 4': {'auc': 0.7908, 'ci': (0.7778, 0.8042)},
        'Townsend 5': {'auc': 0.7681, 'ci': (0.7526, 0.7864)},
        'Townsend 6': {'auc': 0.7426, 'ci': (0.7249, 0.7737)},
        'Townsend 7': {'auc': 0.7771, 'ci': (0.7471, 0.7907)},
        'Townsend 8': {'auc': 0.746, 'ci': (0.7204, 0.770)},
        'Townsend 9': {'auc': 0.655, 'ci': (0.6001, 0.7067)},
    }
}


def analyze_heterogeneity_narrow():
    def calc_I2(subgroups, gender):
        effects, variances = [], []

        for group in subgroups:
            entry = summary_dict_narrow[gender].get(group)
            if not entry:
                continue
            auc = entry["auc"]
            ci_low, ci_high = entry["ci"]
            se = (ci_high - ci_low) / (2 * 1.96)
            var = se ** 2

            effects.append(auc)
            variances.append(var)

        effects = np.array(effects)
        variances = np.array(variances)
        weights = 1 / variances

        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        Q = np.sum(weights * (effects - weighted_mean) ** 2)
        df = len(effects) - 1
        p_val = 1 - chi2.cdf(Q, df)

        # I²
        I2 = max(0, ((Q - df) / Q)) * 100 if Q > 0 else 0

        # Approximate CI using Higgins & Thompson delta method
        # se_I² = sqrt((2*(I²/100)² + (1 - I²/100)²)/df)
        I2_prop = I2 / 100
        se_I2 = np.sqrt((2 * I2_prop**2 + (1 - I2_prop)**2) / df) if df > 0 else 0
        ci_low = max(0, I2 - 1.96 * se_I2 * 100)
        ci_high = min(100, I2 + 1.96 * se_I2 * 100)

        return round(I2, 2), (round(ci_low, 2), round(ci_high, 2)), round(Q, 2), p_val, df

    ethnicity_groups = ["White", "South Asian", "Black"]
    deprivation_groups = ["Townsend 0/1", "Townsend 2/3", "Townsend 4/5", "Townsend 6/7", "Townsend 8/9"]

    print("\n📊 Heterogeneity Summary:")
    for gender in ["men", "women"]:
        label = "Men" if gender == "men" else "Women"

        i2_eth, ci_eth, q_eth, p_eth, df_eth = calc_I2(ethnicity_groups, gender)
        i2_dep, ci_dep, q_dep, p_dep, df_dep = calc_I2(deprivation_groups, gender)

        p_eth_str = f"{p_eth:.4f}" if p_eth >= 0.0001 else "<0.0001"
        p_dep_str = f"{p_dep:.4f}" if p_dep >= 0.0001 else "<0.0001"

        print(f"\n🧬 Ethnicity ({label}):")
        print(f"  I² = {i2_eth}% (95% CI: {ci_eth[0]}% – {ci_eth[1]}%)")
        print(f"  Q = {q_eth}, df = {df_eth}, p = {p_eth_str}")

        print(f"\n🏙️ Deprivation ({label}):")
        print(f"  I² = {i2_dep}% (95% CI: {ci_dep[0]}% – {ci_dep[1]}%)")
        print(f"  Q = {q_dep}, df = {df_dep}, p = {p_dep_str}")


def analyze_heterogeneity_wide():
    def calc_I2(subgroups, gender):
        effects, variances = [], []

        for group in subgroups:
            entry = summary_dict_wide[gender].get(group)
            if not entry:
                continue
            auc = entry["auc"]
            ci_low, ci_high = entry["ci"]
            se = (ci_high - ci_low) / (2 * 1.96)
            var = se ** 2

            effects.append(auc)
            variances.append(var)

        effects = np.array(effects)
        variances = np.array(variances)
        weights = 1 / variances

        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        Q = np.sum(weights * (effects - weighted_mean) ** 2)
        df = len(effects) - 1
        p_val = 1 - chi2.cdf(Q, df)

        I2 = max(0, ((Q - df) / Q)) * 100 if Q > 0 else 0

        # CI for I² using delta method
        I2_prop = I2 / 100
        se_I2 = np.sqrt((2 * I2_prop**2 + (1 - I2_prop)**2) / df) if df > 0 else 0
        ci_low = max(0, I2 - 1.96 * se_I2 * 100)
        ci_high = min(100, I2 + 1.96 * se_I2 * 100)

        return round(I2, 2), (round(ci_low, 2), round(ci_high, 2)), round(Q, 2), p_val, df

    ethnicity_groups = ["Ethnicity 0", "Ethnicity 1", "Ethnicity 2",
                        "Ethnicity 3", "Ethnicity 4", "Ethnicity 5",
                        "Ethnicity 6", "Ethnicity 7", "Ethnicity 8"]

    deprivation_groups = [f"Townsend {i}" for i in range(10)]

    print("\n📊 Heterogeneity Summary:")
    for gender in ["men", "women"]:
        label = "Men" if gender == "men" else "Women"

        i2_eth, ci_eth, q_eth, p_eth, df_eth = calc_I2(ethnicity_groups, gender)
        i2_dep, ci_dep, q_dep, p_dep, df_dep = calc_I2(deprivation_groups, gender)

        p_eth_str = f"{p_eth:.4f}" if p_eth >= 0.0001 else "<0.0001"
        p_dep_str = f"{p_dep:.4f}" if p_dep >= 0.0001 else "<0.0001"

        print(f"\n🧬 Ethnicity ({label}):")
        print(f"  I² = {i2_eth}% (95% CI: {ci_eth[0]}% – {ci_eth[1]}%)")
        print(f"  Q = {q_eth}, df = {df_eth}, p = {p_eth_str}")

        print(f"\n🏙️ Deprivation ({label}):")
        print(f"  I² = {i2_dep}% (95% CI: {ci_dep[0]}% – {ci_dep[1]}%)")
        print(f"  Q = {q_dep}, df = {df_dep}, p = {p_dep_str}")


def plot_subgroup_auc():
    # ✅ 直接使用外部定义的 summary_dict
    df = pd.DataFrame()
    for gender_key in summary_dict_narrow:
        gender_label = "Male" if gender_key == "men" else "Female"
        sub = pd.DataFrame([
            {
                "Group": k,
                "Gender": gender_label,
                "AUC": v["auc"],
                "CI_low": v["ci"][0],
                "CI_high": v["ci"][1],
            }
            for k, v in summary_dict_narrow[gender_key].items()
        ])
        df = pd.concat([df, sub], ignore_index=True)

    # ✅ 分组顺序和新标签
    group_order = [
        "Overall", "White", "South Asian", "Black",
        "Townsend 0/1", "Townsend 2/3", "Townsend 4/5",
        "Townsend 6/7", "Townsend 8/9"
    ]
    group_label_map = {
        "Overall": "All\npatients",
        "White": "White",
        "South Asian": "South\nAsian",
        "Black": "Black",
        "Townsend 0/1": "Least\ndeprived\n(0-20%)",
        "Townsend 2/3": "20-40%",
        "Townsend 4/5": "40-60%",
        "Townsend 6/7": "60-80%",
        "Townsend 8/9": "Most\ndeprived\n(80-100%)"
    }

    df = df[df["Group"].isin(group_order)]
    df["Group"] = pd.Categorical(df["Group"], categories=group_order, ordered=True)
    df = df.sort_values(["Group", "Gender"]).reset_index(drop=True)

    # ✅ 画图
    plt.figure(figsize=(13, 6), facecolor='white')  # 白色背景
    colors = {"Male": "#84C4B7", "Female": "#E38857"}
    markers = {"Male": "o", "Female": "s"}

    for gender in ["Male", "Female"]:
        sub = df[df["Gender"] == gender]
        if sub.empty:
            continue
        x = range(len(sub))
        yerr = [sub["AUC"] - sub["CI_low"], sub["CI_high"] - sub["AUC"]]
        plt.errorbar(
            x, sub["AUC"], yerr=yerr,
            fmt=markers[gender], capsize=4, label=gender,
            color=colors[gender], markersize=6, lw=1.5, alpha=0.9
        )

    # x 轴标签（换行 & 居中）
    xtick_labels = [group_label_map[g] for g in group_order]
    plt.xticks(ticks=range(len(group_order)), labels=xtick_labels, fontsize=11)

    plt.yticks(fontsize=11)
    plt.ylabel("AUROC", fontsize=13)
    plt.ylim(0.60, 0.90)
    plt.title("AUROC by Ethnicity and Deprivation (with 95% CI)", fontsize=14, pad=15)

    # 横线表示 Overall 的 AUC
    for gender in ["Male", "Female"]:
        auc = df[(df.Group == "Overall") & (df.Gender == gender)]["AUC"].values
        if len(auc):
            plt.axhline(auc[0], linestyle="--", color=colors[gender], linewidth=1, alpha=0.7)

    # 竖线分隔 Ethnicity 和 Deprivation
    plt.axvline(3.5, linestyle="--", color="#999999", linewidth=1)  # 原有分隔线
    plt.axvline(0.5, linestyle="--", color="#CCCCCC", linewidth=1)  # White 前额外加一条

    # 添加左右区域标签
    plt.text(2, 0.875, "Ethnicity", ha="center", va="bottom", fontsize=12, color="#333333", fontweight="bold")
    plt.text(6, 0.875, "Deprivation", ha="center", va="bottom", fontsize=12, color="#333333", fontweight="bold")

    plt.legend(title="Gender", frameon=False)
    plt.tight_layout()

    save_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT/saved/final_result/auc_ethnicity_deprivation.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


analyze_heterogeneity_wide()


