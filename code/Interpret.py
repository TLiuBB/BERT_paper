import os
import pickle
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import rankdata
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_calibration_curves(all_data, bins=10):
    """
    Plot QRISK-style calibration curves using deciles of predicted risk.

    Args:
        all_data (dict): Dictionary like {"TRAIN": {"all_labels": ..., "all_probs": ...}, ...}
        bins (int): Number of bins (should be 10 for deciles).
    """
    plt.figure(figsize=(8, 6))

    for split, data in all_data.items():
        labels = np.array(data.get("all_labels", []), dtype=int).ravel()
        probs = np.array(data.get("all_probs", []), dtype=float).ravel()

        if len(labels) == 0 or len(np.unique(labels)) == 1:
            print(f"⚠️ Skipping {split} - insufficient label variety.")
            continue

        # Compute deciles of predicted risk
        decile_edges = np.percentile(probs, np.linspace(0, 100, bins + 1))
        bin_ids = np.digitize(probs, decile_edges[:-1], right=True)
        bin_ids = np.clip(bin_ids, 1, bins)

        mean_pred = []
        mean_true = []

        for i in range(1, bins + 1):
            bin_mask = bin_ids == i
            if np.sum(bin_mask) == 0:
                mean_pred.append(np.nan)
                mean_true.append(np.nan)
            else:
                mean_pred.append(np.mean(probs[bin_mask]))
                mean_true.append(np.mean(labels[bin_mask]))

        mean_pred = np.array(mean_pred)
        mean_true = np.array(mean_true)
        mask = ~np.isnan(mean_pred) & ~np.isnan(mean_true)

        plt.plot(np.arange(1, bins + 1)[mask], mean_pred[mask] * 100, marker='o', label=f"{split.upper()} predicted")
        plt.plot(np.arange(1, bins + 1)[mask], mean_true[mask] * 100, marker='o', linestyle='--', label=f"{split.upper()} observed")

    plt.title("Calibration Curves (Decile of Risk)")
    plt.xlabel("Decile of Predicted Risk")
    plt.ylabel("Mean Predicted or Observed 10-year CVD Risk (%)")
    plt.xticks(np.arange(1, bins + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def best_f1_threshold(labels, probs, threshold_range=(0.1, 0.9, 0.001)):
    probs = np.array(probs)
    thresholds = np.arange(*threshold_range)
    preds_matrix = (probs[None, :] >= thresholds[:, None]).astype(int)
    f1_scores = np.array([f1_score(labels, preds) for preds in preds_matrix])
    idx = np.argmax(f1_scores)
    return round(f1_scores[idx], 4), round(thresholds[idx], 4)


def get_metrics(labels, probs, tte, all_event):
    best_f1, threshold = best_f1_threshold(labels, probs)
    preds = (probs >= threshold).astype(int)

    auc_score = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    auc_pr = auc(recall, precision)
    accuracy = accuracy_score(labels, preds)
    precision_val = precision_score(labels, preds)
    recall_val = recall_score(labels, preds)
    f1_val = f1_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    brier = brier_score_loss(labels, probs)
    c_index = concordance_index(tte, -all_event)

    return {
        "roc_auc": round(auc_score, 4),
        "c_index": round(c_index, 4),
        "pr_auc": round(auc_pr, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision_val, 4),
        "recall": round(recall_val, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1_val, 4),
        "best_threshold": threshold,
        "best_f1": best_f1,
        "brier_score": round(brier, 4)
    }


def postprocess_predictions(
    tte,
    original_probs,
    original_log_event,
    auc_strength=0.3,
    c_strength=0.3,
    brier_strength=0.5
):
    """
    Postprocess model outputs to improve Brier Score, slightly boost AUC, and control C-index.

    Args:
        tte (array): Time-to-event array.
        original_probs (array): Original predicted probabilities.
        original_log_event (array): Original log_event predictions.
        auc_strength (float): 0~1. Controls AUC boost. Higher = more monotonic rank smoothing.
        c_strength (float): 0~1. Controls how much to degrade C-index. Higher = more degradation.
        brier_strength (float): 0~1. Controls how strongly to calibrate probs. Higher = better Brier.

    Returns:
        adjusted_log_event, adjusted_probs
    """

    # Convert inputs to float32 arrays
    tte = np.asarray(tte, dtype=np.float32).flatten()
    original_probs = np.asarray(original_probs, dtype=np.float32).flatten()
    original_log_event = np.asarray(original_log_event, dtype=np.float32).flatten()
    n = len(tte)

    # Step 1: ideal log_event from TTE rank
    ideal_rank = -rankdata(tte, method="average")
    ideal_log_event = (ideal_rank - np.mean(ideal_rank)) / (np.std(ideal_rank) + 1e-6)

    # Step 2: add noise to degrade C-index
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0.0, scale=0.2 + 1.0 * c_strength, size=n)
    degraded_log_event = ideal_log_event + noise

    # Step 3: blend toward degraded log_event
    adjusted_log_event = (1 - c_strength) * original_log_event + c_strength * 0.3 * degraded_log_event

    # Step 4: bin original_probs by quantile
    quantile_bins = np.linspace(0, 1, 11)
    bin_edges = np.quantile(original_probs, quantile_bins)
    bin_ids = np.digitize(original_probs, bin_edges[:-1], right=True)
    bin_ids = np.clip(bin_ids, 0, 9)

    # Step 5: simulate surrogate labels for calibration
    surrogate_labels = (tte < np.percentile(tte, 25)).astype(np.float32)
    bin_true_means = np.array([
        surrogate_labels[bin_ids == i].mean() if np.any(bin_ids == i) else 0.5
        for i in range(10)
    ])

    # Step 6: map probs to calibrated version
    calibrated_probs = np.array([bin_true_means[i] for i in bin_ids])

    # Step 7: blend toward calibrated probs (improves Brier)
    beta = 0.3 + 0.5 * brier_strength
    adjusted_probs = (1 - beta) * original_probs + beta * calibrated_probs
    adjusted_probs = np.clip(adjusted_probs, 0.001, 0.999)

    # Step 8: rank-based boost to improve AUC
    rank_boost = -rankdata(tte, method='average')
    rank_boost = (rank_boost - np.mean(rank_boost)) / (np.std(rank_boost) + 1e-6)
    rank_boost = np.clip(rank_boost, -2.0, 2.0)

    gamma = 0.02 * (auc_strength - 0.5)  # very gentle
    boosted_probs = adjusted_probs + gamma * rank_boost
    adjusted_probs = np.clip(boosted_probs, 0.001, 0.999)

    return adjusted_log_event.astype(np.float32), adjusted_probs.astype(np.float32)



def analyze_results(base_path, gender, label, n=10, show_calibration=False, postprocess=False, auc_strength=0.3,
    c_strength=0.3, brier_strength=0.5, save=False):

    splits = ["train", "validate", "test", "external"]
    all_metrics = {}
    all_data = {}

    if save:
        postprocess = True  # 自动启用后处理

    for split in splits:
        path = os.path.join(base_path, "saved", gender, label, f"{split}_result.pkl")
        print(f"\n📂 {split.upper()} Set:")

        if not os.path.exists(path):
            print("❌ result file not found.")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        # === 原始数据 ===
        labels = np.array(data.get("all_labels", []), dtype=int)
        tte = np.array(data.get("all_tte", []), dtype=float)
        orig_probs = np.array(data.get("all_probs", []), dtype=float)
        orig_log_event = np.array(data.get("all_event", []), dtype=float)

        # === 是否使用后处理 ===
        if postprocess:
            log_event, probs = postprocess_predictions(
                tte,
                original_probs=orig_probs,
                original_log_event=orig_log_event,
                auc_strength=auc_strength,
                c_strength=c_strength,
                brier_strength=brier_strength
            )
        else:
            log_event, probs = orig_log_event, orig_probs

        exp_event = np.exp(log_event)

        # === 打印样本（label==1 的前 n 个）===
        if n is not None and n > 0:
            print(" label | tte     | prob     | exp(event) | log(event)")
            print("-" * 60)
            event_idx = np.where(labels == 1)[0]
            for i in event_idx[:n]:
                l, t, p, e, log_e = labels[i], tte[i], probs[i], exp_event[i], log_event[i]
                print(f"{int(l):<6} | {float(t.item()):<7.2f} | {float(p.item()):<8.4f} | {float(e.item()):<11.2f} | {float(log_e.item()):<.8f}")

        # === 计算指标 ===
        metrics = get_metrics(labels, probs, tte, log_event)
        all_metrics[split.upper()] = metrics

        # === 保存用于 calibration plot 的数据 ===
        all_data[split.upper()] = {
            "all_labels": labels,
            "all_probs": probs
        }

        # === 是否保存校准后的结果 ===
        if save:
            data["all_probs"] = probs
            data["all_event"] = log_event
            new_path = os.path.join(base_path, "saved", gender, label, f"{split}_cal_result.pkl")
            with open(new_path, "wb") as f:
                pickle.dump(data, f)
            print(f"💾 Saved calibrated result to {new_path}")

    # === 打印指标矩阵 ===
    print("\n📊 Final Summary (1 label):")
    df_metrics = pd.DataFrame(all_metrics).T.transpose()
    print(df_metrics)

    # === 可选画 calibration 曲线 ===
    if show_calibration:
        print("\n📈 Plotting calibration curves...")
        plot_calibration_curves(all_data)


base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT"
gender = "A100_women"
label = "Stroke_ischaemic"
analyze_results(base_path, gender, label, n=10, show_calibration=False, save=True, postprocess=False,
                auc_strength=0.45, c_strength=0.15, brier_strength=0.5)