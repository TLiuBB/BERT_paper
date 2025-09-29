import os
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertModel
from Bert import evaluate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from transformers import DistilBertTokenizer

# 环境设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = torch.Generator()
g.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

# ✅ External preprocessing
def preprocess_external(df, features, label_column, batch_size=128):
    tte_column = f'{label_column}_tte'
    core_columns=['text', label_column, f"{label_column}_tte"]
    df=df.dropna(subset=core_columns)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_lengths = df['text'].apply(lambda x: len(tokenizer.tokenize(x))).values
    max_length = int(np.percentile(tokenized_lengths, 99))

    tokens = tokenizer(df['text'].tolist(), padding="longest", truncation=True, max_length=max_length, return_tensors="pt")

    if features:
        scaler = MinMaxScaler()
        continuous_tensor = torch.tensor(scaler.fit_transform(df[features].astype(np.float32))).float()
    else:
        continuous_tensor = torch.empty(len(df), 0)

    label_tensor = torch.tensor(df[label_column].values, dtype=torch.long)
    tte_tensor = torch.tensor(df[tte_column].values, dtype=torch.float32)

    dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], continuous_tensor, label_tensor, tte_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)

    return loader

# ✅ Main function for evaluating all external datasets
def evaluate_all_external_models(base_path, genders, labels, features, threshold_range=(0.0, 1, 0.001), batch_size=128):
    for gender in genders:
        external_path = os.path.join(base_path, "data", f"london_{gender}.csv")
        df = pd.read_csv(external_path)

        for label in labels:
            print(f"\n🔍 Evaluating {gender.upper()} - {label}...")
            loader = preprocess_external(df.copy(), features, label, batch_size=batch_size)

            # Model path
            save_dir = os.path.join(base_path, "saved", f"A100_{gender}", label)
            model_path = os.path.join(save_dir, "best_model.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(save_dir, "temp_model.pt")

            # Load model
            bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            from Bert import BERTMLPClassifier  # Local import inside function
            model = BERTMLPClassifier(bert_model=bert_model, num_continuous_features=len(features))
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            # Evaluate and save results
            result_path = os.path.join(save_dir, "external_result.pkl")
            evaluate(model, loader, threshold_range=threshold_range, path=result_path)

    print("\n✅ All external evaluations completed.")

# Example usage:
base_path = "/Users/tonyliubb/not in iCloud/CPRD/BERT"
genders = ["men", "women"]
labels = ['cvd_q', 'cvd_all', 'CHD', 'Stroke_ischaemic', 'MI', 'HF', 'Angina_stable', 'Dementia']
features = ['age', 'QRISK3_2017', 'SBP', 'SBP_sd', 'DBP', 'Total/HDL_ratio', 'townsend', 'BMI']

# 调用函数
evaluate_all_external_models(base_path, genders, labels, features)
