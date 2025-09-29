import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_logs(base_dir, labels, saved_path):
    """
    Parse training logs and plot training/validation metrics for multiple labels and genders.

    Args:
        base_dir (str): Base directory containing gender/label subdirectories.
        labels (list): List of label names.
        saved_path (str): Path to save plots.
    """
    genders=['A100_men', 'A100_women']

    for gender in genders:
        for label in labels:
            log_path=os.path.join(base_dir, f"{gender}/{label}/training_log.txt")
            fig_title=f"{gender}_{label}"
            save_file=os.path.join(saved_path, f"{fig_title}.png")

            if not os.path.exists(log_path):
                print(f"Log file not found: {log_path}")
                continue

            train_data=[]
            val_data=[]
            epoch=None

            with open(log_path, "r") as f:
                for line in f:
                    line=line.strip()

                    if "Epoch [" in line and "/" in line:
                        try:
                            epoch=int(line.split("Epoch [")[1].split("/")[0])
                        except ValueError:
                            continue

                    if "✅ Epoch" in line:
                        try:
                            train_loss=float(line.split("Training Loss: ")[1].split(",")[0])
                            train_auc=float(line.split("Train AUC: ")[1].split(",")[0])
                            train_cindex=float(line.split("Train C-Index: ")[1].split(",")[0])
                            train_f1=float(line.split("Train F1: ")[1].split(" ")[0])
                            train_data.append([epoch, train_loss, train_auc, train_cindex, train_f1])
                        except (ValueError, IndexError):
                            continue

                    if "🔹 Validation - Loss" in line:
                        try:
                            val_loss=float(line.split("Loss: ")[1].split(",")[0])
                            val_auc=float(line.split("Validate AUC: ")[1].split(",")[0])
                            val_cindex=float(line.split("Validate C-Index: ")[1].split(",")[0])
                            val_f1=float(line.split("Validate F1: ")[1].split(" ")[0])
                            val_data.append([epoch, val_loss, val_auc, val_cindex, val_f1])
                        except (ValueError, IndexError):
                            continue

            if not train_data or not val_data:
                print(f"No data found in: {log_path}")
                continue

            train_df=pd.DataFrame(train_data, columns=["Epoch", "Train Loss", "Train AUC", "Train C-Index", "Train F1"])
            val_df=pd.DataFrame(val_data, columns=["Epoch", "Val Loss", "Val AUC", "Val C-Index", "Val F1"])
            full_df=train_df.merge(val_df, on="Epoch", how="outer")

            fig, axes=plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(fig_title, fontsize=16)

            # AUC
            axes[0, 0].plot(full_df["Epoch"], full_df["Train AUC"], label="Train AUC", marker="o")
            axes[0, 0].plot(full_df["Epoch"], full_df["Val AUC"], label="Val AUC", marker="s")
            axes[0, 0].set_title("AUC Over Epochs")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("AUC")
            axes[0, 0].legend()

            # Loss
            axes[0, 1].plot(full_df["Epoch"], full_df["Train Loss"], label="Train Loss", marker="o", color="r")
            axes[0, 1].plot(full_df["Epoch"], full_df["Val Loss"], label="Val Loss", marker="s", color="g")
            axes[0, 1].set_title("Loss Over Epochs")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()

            # C-Index
            axes[1, 0].plot(full_df["Epoch"], full_df["Train C-Index"], label="Train C-Index", marker="o", color="b")
            axes[1, 0].plot(full_df["Epoch"], full_df["Val C-Index"], label="Val C-Index", marker="s", color="c")
            axes[1, 0].set_title("C-Index Over Epochs")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("C-Index")
            axes[1, 0].legend()

            # F1
            axes[1, 1].plot(full_df["Epoch"], full_df["Train F1"], label="Train F1", marker="o", color="m")
            axes[1, 1].plot(full_df["Epoch"], full_df["Val F1"], label="Val F1", marker="s", color="y")
            axes[1, 1].set_title("F1 Score Over Epochs")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("F1 Score")
            axes[1, 1].legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_file)
            plt.close()


# Run the function with your paths
base_dir="/Users/tonyliubb/not in iCloud/CPRD/BERT/saved"
saved_path="/Users/tonyliubb/not in iCloud/CPRD/BERT/plot/train_plot"
labels=['cvd_q', 'cvd_all', 'CHD', 'Stroke_ischaemic', 'MI', 'HF', 'Angina_stable', 'Dementia']

plot_training_logs(base_dir, labels, saved_path)
