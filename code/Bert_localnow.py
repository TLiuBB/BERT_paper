# Standard library imports
import math
import os
import sys
import random
import time
from contextlib import redirect_stdout

# Third-party library imports
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, auc)
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from transformers import (BertModel, BertConfig, BertTokenizer,
                          DistilBertTokenizer, DistilBertModel)

from lifelines.utils import concordance_index

import optuna
from optuna.trial import TrialState

def set_seed(seed=2023):
    """Set the seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2023)

def seed_worker(worker_id):
    np.random.seed(2023 + worker_id)
    random.seed(2023 + worker_id)

g = torch.Generator()
g.manual_seed(2023)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "M4 PRO (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"
print(f"Using device: {device} ({device_name})")


def preprocess(df, feature, label_column, test_size=0.15, validation_size=0.15, batch_size=128):
    """
    Optimized preprocessing function for a hybrid BERT-based model with labels and TTE values.

    Args:
        df (DataFrame): Input data containing text, structured features, and TTE values.
        feature (list): List of column names for continuous features.
        label_column (str): Name of the label column.
        test_size (float): Proportion of data for the test set.
        validation_size (float): Proportion of data for the validation set.
        batch_size (int): Batch size for DataLoader.

    Returns:
        tuple: train_loader, val_loader, test_loader, class_weight_tensor
    """
    tte_column = f'{label_column}_tte'

    # Step 1: Drop missing values efficiently
    required_columns = ['text', label_column, tte_column] + feature
    print(f"Dataset before dropping missing values: {df.shape[0]}")
    df.dropna(subset=required_columns, inplace=True)
    print(f"Dataset reduced to {df.shape[0]} rows after dropping missing values.")

    # Step 2: Initialize BERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Step 3: Automatically determine max_length (99th percentile)
    tokenized_lengths = df['text'].apply(lambda x: len(tokenizer.tokenize(x))).values
    max_length = int(np.percentile(tokenized_lengths, 99))
    print(f"Automatically calculated max_length={max_length} based on the dataset.")

    # Step 4: Stratified split
    df_train, df_temp = train_test_split(df, test_size=(test_size + validation_size), stratify=df[label_column], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=test_size / (test_size + validation_size), stratify=df_temp[label_column], random_state=42)

    # Step 5: Tokenization
    def tokenize_text(texts):
        return tokenizer(texts.tolist(), padding="longest", truncation=True, max_length=max_length, return_tensors="pt")

    train_tokens, val_tokens, test_tokens = map(tokenize_text, [df_train['text'], df_val['text'], df_test['text']])

    # Step 6: Feature Scaling
    scaler = MinMaxScaler()
    train_features, val_features, test_features = None, None, None
    if feature:
        train_features = torch.tensor(scaler.fit_transform(df_train[feature]).astype(np.float32)).float()
        val_features = torch.tensor(scaler.transform(df_val[feature]).astype(np.float32)).float()
        test_features = torch.tensor(scaler.transform(df_test[feature]).astype(np.float32)).float()

    # Step 7: Prepare labels & TTE tensors
    def tensorize(series, dtype):
        return torch.tensor(series.values, dtype=dtype)

    y_train, y_val, y_test = map(lambda x: tensorize(x, torch.long), [df_train[label_column], df_val[label_column], df_test[label_column]])
    tte_train, tte_val, tte_test = map(lambda x: tensorize(x, torch.float32), [df_train[tte_column], df_val[tte_column], df_test[tte_column]])

    # Step 8: Compute Class Weights with Continuous Scaling
    class_weight_tensor = None
    if df_train[label_column].nunique() > 1:
        class_weight = compute_class_weight("balanced", classes=np.unique(df_train[label_column]), y=df_train[label_column])
        class_weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device)

        # Calculate event=1 ratio
        event_rate = (df_train[label_column] == 1).mean()

        # Continuous scaling factor
        c = 0.5
        epsilon = 1e-4
        scale_factor = c / (event_rate + epsilon)
        # Apply scaling
        class_weight_tensor *= scale_factor

        print(f"Event Rate: {event_rate:.4f}, Scaling Factor: {scale_factor:.2f}, Adjusted Class Weights: {class_weight_tensor.cpu().numpy()}")

    # Step 9: Create TensorDatasets
    train_dataset = TensorDataset(train_tokens['input_ids'],
                                  train_tokens['attention_mask'],
                                  train_features,
                                  y_train,
                                  tte_train)

    val_dataset = TensorDataset(val_tokens['input_ids'],
                                val_tokens['attention_mask'],
                                val_features,
                                y_val,
                                tte_val)

    test_dataset = TensorDataset(test_tokens['input_ids'],
                                 test_tokens['attention_mask'],
                                 test_features,
                                 y_test,
                                 tte_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)

    return train_loader, val_loader, test_loader, class_weight_tensor


class FocalCoxLoss(nn.Module):
    def __init__(self, class_weight, gamma=2, eps=1e-8, tte_scaling="softplus"):
        """
        Combined Focal Loss and Cox Loss with uncertainty weighting.

        Args:
            class_weight (Tensor): class weights for Focal loss.
            gamma (float): focusing parameter.
            eps (float): small constant for stability.
            tte_scaling (str): one of ["softplus", "tanh", "none"].
        """
        super(FocalCoxLoss, self).__init__()
        self.class_weight = class_weight
        self.gamma = gamma
        self.eps = eps
        self.tte_scaling = tte_scaling
        self.auto_initialized = False

        # Placeholder log_sigma (will auto-init)
        self.log_sigma_focal = nn.Parameter(torch.tensor(-1.0), requires_grad=True)
        self.log_sigma_cox = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, outputs, labels, tte_scores, tte, max_tte=3650):
        # ----- Focal Loss -----
        BCE_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.class_weight.device != labels.device:
            self.class_weight = self.class_weight.to(labels.device)
        alpha_t = self.class_weight[labels.long()]
        focal_loss = alpha_t * ((1 - pt).clamp(min=self.eps) ** self.gamma) * BCE_loss
        focal_loss = focal_loss.mean()

        # ----- TTE scaling -----
        if self.tte_scaling == "softplus":
            tte_scores = F.softplus(tte_scores)
        elif self.tte_scaling == "tanh":
            tte_scores = torch.tanh(tte_scores)
        elif self.tte_scaling == "none":
            pass
        else:
            raise ValueError(f"Unsupported tte_scaling: {self.tte_scaling}")

        tte_scores = torch.nan_to_num(tte_scores, nan=0.0, posinf=10.0, neginf=-10.0)

        # ----- Cox Loss -----
        tte = torch.where(labels == 0, max_tte, tte)
        tte_sorted, order = torch.sort(tte, descending=False)
        tte_scores = tte_scores[order]
        labels = labels[order]

        risk_scores = tte_scores.exp()
        cum_risk_scores = torch.cumsum(risk_scores, dim=0)
        log_hazard_ratio = tte_scores - torch.log(cum_risk_scores + self.eps)
        cox_loss = -torch.sum(log_hazard_ratio * labels) / (labels.sum() + self.eps)

        # ----- Auto-initialize sigma if not done -----
        if not self.auto_initialized:
            ratio = focal_loss.item() / (cox_loss.item() + self.eps)
            sigma_cox_val = 1.0
            sigma_focal_val = math.sqrt(ratio) * sigma_cox_val

            def inv_softplus(x):
                return math.log(math.exp(x) - 1 + self.eps)

            with torch.no_grad():
                self.log_sigma_focal.data = torch.tensor(inv_softplus(sigma_focal_val))
                self.log_sigma_cox.data = torch.tensor(inv_softplus(sigma_cox_val))
                self.auto_initialized = True
                print(f"[AutoInit] log_sigma_focal: {self.log_sigma_focal.item():.4f}, log_sigma_cox: {self.log_sigma_cox.item():.4f}")

        # ----- Uncertainty-weighted Total Loss -----
        sigma_focal = F.softplus(self.log_sigma_focal) + self.eps
        sigma_cox = F.softplus(self.log_sigma_cox) + self.eps

        total_loss = (1 / (2 * sigma_focal ** 2)) * focal_loss + (1 / (2 * sigma_cox ** 2)) * cox_loss
        total_loss += self.log_sigma_focal.detach() + self.log_sigma_cox.detach()

        return total_loss


class BERTMLPClassifier(nn.Module):
    """
    A hybrid BERT-based classifier that integrates text features from DistilBERT and continuous features,
    using multi-head attention for feature refinement.

    This model consists of:
        - **BERT Encoder:** Processes textual input using a pre-trained DistilBERT model.
        - **MLP for Continuous Features:** Processes numerical structured data.
        - **Feature Fusion Layer:** Concatenates text and numerical features, followed by a Linear transformation.
        - **Multi-Head Attention Module:** Enhances the combined feature representations.
        - **Two Independent MLPs:** One for binary classification, one for TTE prediction.

    Args:
        bert_model (nn.Module): Pre-trained BERT model (e.g., DistilBERT).
        num_continuous_features (int): Number of structured numerical features.
        dropout_rate (float): Dropout rate for regularization. Default is 0.5.
        num_attention_heads (int): Number of heads in the multi-head attention mechanism. Default is 6.
        num_attention_layers (int): Number of stacked multi-head attention layers. Default is 6.

    """

    def __init__(self, bert_model, num_continuous_features, dropout_rate=0.5, num_attention_heads=6, num_attention_layers=6):
        super(BERTMLPClassifier, self).__init__()

        # ===== Fixed Parameters =====
        self.bert_hidden_size = 768  # BERT output size (fixed for DistilBERT)
        self.mlp_hidden_sizes = (128, 64)  # Hidden sizes for MLP processing continuous features
        self.classifier_hidden_sizes = (384, 192)  # Hidden sizes for classifier layers
        self.temperature = 0.5  # Fixed temperature scaling

        # ===== Model Components =====
        # 1. BERT Text Encoder
        self.bert = bert_model

        # 2. MLP for Continuous Features
        self.extra_features_fc = nn.Sequential(
            nn.Linear(num_continuous_features, self.mlp_hidden_sizes[0]),  # 128
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.mlp_hidden_sizes[0], self.mlp_hidden_sizes[1]),  # 64
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 3. Feature Fusion Layer (Replaces text_weight & continuous_weight)
        self.fusion_layer = nn.Linear(self.bert_hidden_size + self.mlp_hidden_sizes[-1], self.bert_hidden_size)

        # 4. Stacked Multi-Head Attention Layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.bert_hidden_size, num_heads=num_attention_heads, dropout=dropout_rate)
            for _ in range(num_attention_layers)
        ])

        # 5. Layer Normalization
        self.layer_norm = nn.LayerNorm(self.bert_hidden_size)

        # 6. Separate MLP for Classification
        self.classifier_mlp = nn.Sequential(
            nn.Linear(self.bert_hidden_size, self.mlp_hidden_sizes[0]),  # 768 → 384
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.mlp_hidden_sizes[0], self.mlp_hidden_sizes[1]),  # 384 → 192
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.mlp_hidden_sizes[1], 2)  # 192 → 2 (Binary classification)
        )

        # 7. Separate MLP for TTE Prediction
        self.tte_mlp=nn.Sequential(
            nn.Linear(self.bert_hidden_size, self.mlp_hidden_sizes[0]),  # 768 → 384
            nn.SELU(),
            nn.AlphaDropout(dropout_rate),
            nn.Linear(self.mlp_hidden_sizes[0], self.mlp_hidden_sizes[1]),  # 384 → 192
            nn.SELU(),
            nn.AlphaDropout(dropout_rate),
            nn.Linear(self.mlp_hidden_sizes[1], 1)  # 192 → 1
        )

    def forward(self, text_input_ids, text_attention_mask, continuous_features):
        """
        Forward pass for the BERTMLPClassifier.

        Args:
            text_input_ids (torch.Tensor): Input IDs for text, shape (batch_size, seq_length).
            text_attention_mask (torch.Tensor): Attention mask for text, shape (batch_size, seq_length).
            continuous_features (torch.Tensor): Continuous features, shape (batch_size, num_continuous_features).

        Returns:
            tuple: (logits, tte_scores)
                logits (torch.Tensor): Logits of shape (batch_size, 2) for classification.
                tte_scores (torch.Tensor): Risk scores for TTE of shape (batch_size, 1).
        """
        expected_shape = (continuous_features.shape[0], self.extra_features_fc[0].in_features)
        assert continuous_features.shape == expected_shape, f"Expected shape {expected_shape}, but got {continuous_features.shape}"
        # Step 1: Process text features through BERT
        bert_output = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        cls_output = bert_output.last_hidden_state[:, 0, :]  # Extract CLS token representation

        # Step 2: Process continuous features through MLP
        extra_features = self.extra_features_fc(continuous_features)  # Shape: (batch_size, 64)

        # Step 3: Feature Fusion (Replaces text_weight * CLS + continuous_weight * MLP)
        combined_features = torch.cat([cls_output, extra_features], dim=1)  # Shape: (batch_size, 768 + 64)
        combined_features = self.fusion_layer(combined_features)  # Linear transformation to (batch_size, 768)

        # Step 4: Multi-Head Attention with Residual Connections
        combined_features = combined_features.unsqueeze(1).permute(1, 0, 2)  # Reshape for attention layers
        residual = combined_features.clone()  # Save original features for residual connection

        for attention_layer in self.attention_layers:
            attended_features, _ = attention_layer(combined_features, combined_features, combined_features)
            combined_features = attended_features + residual  # Add residual connection
            combined_features = self.layer_norm(combined_features)  # Normalize after each attention layer

        # Step 5: Remove Sequence Dimension
        attended_features = combined_features.squeeze(0)  # Shape: (batch_size, 768)

        # Step 6: Compute Outputs using Separate MLPs
        logits = self.classifier_mlp(attended_features) / self.temperature  # Shape: (batch_size, 2)
        tte_scores = self.tte_mlp(attended_features)  # Shape: (batch_size, 1)

        return logits, tte_scores



def calculate_best_f1(labels, probs, threshold_range=(0.1, 0.9, 0.01), mode='f1'):
    """
    Efficiently calculates the best F1 score, macro F1 score, and weighted F1 score over a range of thresholds.

    Args:
        labels (list or np.array): Ground truth binary labels.
        probs (list or np.array): Predicted probabilities for the positive class.
        threshold_range (tuple): A range of thresholds in the format (start, stop, step). Default is (0.1, 0.9, 0.01).
        mode (str): Optimization mode. Choose 'f1', 'macro_f1', or 'weighted_f1'. Default is 'f1'.

    Returns:
        tuple:
            - Best F1 score according to the chosen mode.
            - Corresponding threshold for the chosen best F1.
    """
    if mode not in ['f1', 'macro_f1', 'weighted_f1']:
        raise ValueError("Invalid mode. Choose 'f1', 'macro_f1', or 'weighted_f1'.")

    probs=np.array(probs)
    thresholds=np.array(np.arange(*threshold_range))

    # Expand dimensions for broadcasting: (num_thresholds, num_samples)
    preds_matrix = (probs[None, :] >= thresholds[:, None]).astype(int)

    # Compute F1 scores for all thresholds at once
    f1_scores = np.array([f1_score(labels, preds) for preds in preds_matrix])
    macro_f1_scores = np.array([f1_score(labels, preds, average='macro') for preds in preds_matrix])
    weighted_f1_scores = np.array([f1_score(labels, preds, average='weighted') for preds in preds_matrix])

    # Get best F1 for each type
    best_idx_f1 = np.argmax(f1_scores)
    best_idx_macro_f1 = np.argmax(macro_f1_scores)
    best_idx_weighted_f1 = np.argmax(weighted_f1_scores)

    best_f1, best_threshold_f1 = f1_scores[best_idx_f1], thresholds[best_idx_f1]
    best_macro_f1, best_threshold_macro_f1 = macro_f1_scores[best_idx_macro_f1], thresholds[best_idx_macro_f1]
    best_weighted_f1, best_threshold_weighted_f1 = weighted_f1_scores[best_idx_weighted_f1], thresholds[best_idx_weighted_f1]

    # Print all F1 scores and corresponding thresholds
    print(f"Best F1 Score: {best_f1:.4f} at threshold: {best_threshold_f1:.2f}")
    print(f"Best Macro F1 Score: {best_macro_f1:.4f} at threshold: {best_threshold_macro_f1:.2f}")
    print(f"Best Weighted F1 Score: {best_weighted_f1:.4f} at threshold: {best_threshold_weighted_f1:.2f}")

    # Select the best F1 and threshold based on the chosen mode
    if mode == 'f1':
        return best_f1, best_threshold_f1
    elif mode == 'macro_f1':
        return best_macro_f1, best_threshold_macro_f1
    else:  # mode == 'weighted_f1'
        return best_weighted_f1, best_threshold_weighted_f1


def train_model(model, train_loader, val_loader, epochs=100, freeze=20, patience=5, class_weights=None, gamma=2,
                lr=3e-4, weight_decay=1e-4, threshold_range=(0.1, 0.9, 0.1), f1_mode='f1', save_path="model.pt"):

    model.to(device)
    criterion = FocalCoxLoss(class_weight=class_weights, gamma=gamma)

    model_params = list(model.parameters())
    sigma_params = [criterion.log_sigma_focal, criterion.log_sigma_cox]

    optimizer = optim.AdamW(
        model_params,
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Initialize tracking variables
    best_val_auc, best_val_c, best_val_f1, best_threshold, early_stop_counter = 0, 0, 0, 0.5, 0

    for epoch in range(epochs):
        if epoch == freeze:
            for param in sigma_params:
                param.requires_grad = True
            optimizer = optim.AdamW(
                list(model.parameters()) + sigma_params,
                lr=lr, weight_decay=weight_decay
            )
            print(f"🚀 Sigma unfreeze")

        # ========================= TRAIN =========================
        model.train()
        running_loss = 0.0
        all_train_labels, all_train_probs, all_train_tte, all_train_event = [], [], [], []

        for batch_idx, (text_input_ids, text_attention_mask, continuous_features, labels, tte) in enumerate(
                train_loader):
            text_input_ids, text_attention_mask, continuous_features, labels, tte = (
                text_input_ids.to(device),
                text_attention_mask.to(device),
                continuous_features.to(device).float(),
                labels.to(device),
                tte.to(device).float()
            )

            optimizer.zero_grad()
            logits, tte_scores = model(text_input_ids, text_attention_mask, continuous_features)
            loss = criterion(logits, labels, tte_scores, tte)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()
            running_loss += loss.item()

            # Collect predictions and labels for metrics
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_probs.extend(probs)
            all_train_tte.extend(tte.cpu().numpy())
            all_train_event.extend(tte_scores.detach().cpu().numpy())

            # Print training progress
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                batch_loss = running_loss / (batch_idx + 1)
                train_auc = roc_auc_score(all_train_labels, np.nan_to_num(all_train_probs, nan=1e-8))
                train_c_index = concordance_index(all_train_tte, -np.array(all_train_event))
                sigma_focal_value = torch.exp(criterion.log_sigma_focal).item()
                sigma_cox_value = torch.exp(criterion.log_sigma_cox).item()
                print(f"📌 Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}] - "
                      f"Loss: {batch_loss:.4f}, Train AUC: {train_auc:.4f}, Train C-Index: {train_c_index:.4f}, "
                      f"Sigma Focal: {sigma_focal_value:.4f}, Sigma Cox: {sigma_cox_value:.4f}")

        # Compute training metrics
        epoch_loss = running_loss / len(train_loader)
        train_auc = roc_auc_score(all_train_labels, np.nan_to_num(all_train_probs, nan=1e-8))
        train_c_index = concordance_index(all_train_tte, -np.array(all_train_event))
        train_f1, train_threshold = calculate_best_f1(all_train_labels, all_train_probs, threshold_range, mode=f1_mode)
        sigma_focal_value=torch.exp(criterion.log_sigma_focal).item()
        sigma_cox_value=torch.exp(criterion.log_sigma_cox).item()
        print(f"✅ Epoch {epoch + 1}/{epochs} - Training Loss: {epoch_loss:.4f}, "
              f"Train AUC: {train_auc:.4f}, Train C-Index: {train_c_index:.4f}, "
              f"Train F1: {train_f1:.4f} at threshold {train_threshold:.4f}, "
              f"Sigma Focal: {sigma_focal_value:.4f}, Sigma Cox: {sigma_cox_value:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_labels, all_val_probs, all_val_tte, all_val_event = [], [], [], []

        with torch.no_grad():
            for batch_idx, (text_input_ids, text_attention_mask, continuous_features, labels, tte) in enumerate(val_loader):
                text_input_ids, text_attention_mask, continuous_features, labels, tte = (
                    text_input_ids.to(device),
                    text_attention_mask.to(device),
                    continuous_features.to(device).float(),
                    labels.to(device),
                    tte.to(device).float()
                )

                logits, tte_scores = model(text_input_ids, text_attention_mask, continuous_features)
                loss = criterion(logits, labels, tte_scores, tte)
                val_loss += loss.item()

                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs)
                all_val_tte.extend(tte.cpu().numpy())
                all_val_event.extend(tte_scores.detach().cpu().numpy())

                # Print validation progress
                if (batch_idx + 1) % max(1, len(val_loader) // 10) == 0:
                    all_val_probs = list(np.nan_to_num(all_val_probs, nan=1e-8, posinf=1.0, neginf=0.0))
                    val_auc = roc_auc_score(all_val_labels, all_val_probs)
                    val_c_index = concordance_index(all_val_tte, -np.array(all_val_event))
                    print(f"Validation Progress - Batch [{batch_idx + 1}/{len(val_loader)}] - "
                          f"Loss: {val_loss / (batch_idx + 1):.4f}, Validate AUC: {val_auc:.4f}, Validate C-Index: {val_c_index:.4f}")

        # Compute validation metrics
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(all_val_labels, np.nan_to_num(all_val_probs, nan=1e-8))
        val_c_index = concordance_index(all_val_tte, -np.array(all_val_event))
        val_f1, val_threshold = calculate_best_f1(all_val_labels, all_val_probs, threshold_range, mode=f1_mode)

        print(f"🔹 Validation - Loss: {val_loss:.4f}, Validate AUC: {val_auc:.4f}, Validate C-Index: {val_c_index:.4f}, "
              f"Validate F1: {val_f1:.4f} at threshold {val_threshold:.4f}")

        # Early stopping and model saving
        if val_auc > best_val_auc:
            best_val_auc, best_val_c, best_val_f1, best_threshold = val_auc, val_c_index, val_f1, val_threshold
            torch.save(model.state_dict(), save_path)
            early_stop_counter=0
            print(f"🟢 Epoch {epoch + 1}: Model saved - Validate AUC: {val_auc:.4f}, C-Index: {val_c_index:.4f}, "
                  f"F1: {val_f1:.4f} at threshold {val_threshold:.4f}")
        else:
            early_stop_counter+=1
            print(f"⚠️ Epoch {epoch + 1}: No improvement in validation metrics for {early_stop_counter} epochs.")

        # Step scheduler
        scheduler.step(epoch + 1)

        if early_stop_counter >= patience:
            print(f"⛔ Early stopping at epoch {epoch + 1}, no improvement for {patience} consecutive epochs.")
            break


def evaluate(model, test_loader, threshold_range=(0.1, 0.9, 0.1), path=None):
    """
    Evaluate a model on the test dataset.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        threshold_range (tuple): Range of thresholds for F1 score calculation. Default is (0.1, 0.9, 0.1).
        path (str): Path to save evaluation data for visualization. If None, no data is saved.

    Returns:
        dict: A dictionary containing evaluation metrics, including C-Index for TTE.
    """
    model.to(device)
    model.eval()
    all_labels, all_probs, all_tte, all_event = [], [], [], []

    with torch.no_grad():
        for batch_idx, (text_input_ids, text_attention_mask, continuous_features, labels, tte) in enumerate(test_loader):
            # Move data to device
            text_input_ids, text_attention_mask, continuous_features, labels, tte = (
                text_input_ids.to(device),
                text_attention_mask.to(device),
                continuous_features.to(device).float(),
                labels.to(device),
                tte.to(device).float()
            )

            # Model prediction
            logits, tte_scores = model(text_input_ids, text_attention_mask, continuous_features)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_tte.extend(tte.cpu().numpy())
            all_event.extend(tte_scores.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_tte = np.array(all_tte, dtype=np.float32)
    all_event = np.array(all_event, dtype=np.float32)

    # Calculate final metrics
    auc_score = roc_auc_score(all_labels, all_probs)
    c_index = concordance_index(all_tte, -all_event)
    best_f1, best_threshold = calculate_best_f1(all_labels, all_probs, threshold_range)

    final_preds = (all_probs >= best_threshold).astype(int)
    accuracy = accuracy_score(all_labels, final_preds)
    precision = precision_score(all_labels, final_preds)
    recall = recall_score(all_labels, final_preds)
    specificity = recall_score(all_labels, final_preds, pos_label=0)

    precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
    auc_pr = auc(recall_vals, precision_vals)

    f1 = f1_score(all_labels, final_preds)
    brier_score = np.mean((all_probs - all_labels) ** 2)

    # Print final evaluation metrics
    print(f"ROC AUC: {auc_score:.4f}")
    print(f"Precision-Recall AUC: {auc_pr:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Best Threshold for F1: {best_threshold:.4f} with F1 Score: {best_f1:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"C-Index for TTE: {c_index:.4f}")

    # If path is provided, save data for later visualization
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        evaluation_results = {
            "all_labels": all_labels,
            "all_probs": all_probs,
            "all_tte": all_tte,
            "all_event": all_event,
            "roc_auc": auc_score,
            "pr_auc": auc_pr,
            "brier_score": brier_score,
            "c_index": c_index,
            "best_threshold": best_threshold,
            "best_f1": best_f1
        }

        with open(path, "wb") as f:
            pickle.dump(evaluation_results, f)
        print(f"✅ Evaluation results saved at: {path}")

    return {
        "roc_auc": round(auc_score, 4),
        "pr_auc": round(auc_pr, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "best_threshold": round(best_threshold, 4),
        "best_f1": round(best_f1, 4),
        "brier_score": round(brier_score, 4),
        "c_index": round(c_index, 4)
    }


class DualLogger:
    """
    A logger that outputs messages to both the console and a file.
    """
    def __init__(self, file_path):
        self.console = sys.stdout
        self.file = open(file_path, 'w')

    def write(self, message):
        self.console.write(message)  # Print to console
        self.file.write(message)    # Save to file

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def BERTMLP_run(gender, path, features, label='cvd_q', batch=128, epoch=10, freeze=20, patience=3,
                      search_trials=20, threshold_range=(0.1, 0.9, 0.001), f1_mode='f1'):
    """
    Run semi-supervised training and hyperparameter tuning for a BERT-based hybrid model,
    with dual logging to both console and a .txt file.

    Args:
        gender (str): Dataset group (e.g., male/female).
        path (str): Path to the dataset and save directories.
        features (list): List of continuous feature column names.
        label (str): Name of the label column. Default is 'cvd_q'.
        batch (int): Batch size for training and evaluation. Default is 128.
        epoch (int): Number of training epochs. Default is 10.
        freeze (int): Number of epochs to freeze the sigma. Default is 20.
        patience (int): Early stopping patience. Default is 3.
        search_trials (int): Number of Optuna trials for hyperparameter optimization. Default is 20.
        threshold_range (tuple): Range of thresholds for F1 score calculation. Default is (0.1, 0.9, 0.001).
        f1_mode (str): Optimization mode for F1 calculation ('f1' or 'macro_f1'). Default is 'f1'.

    Returns:
        tuple: (Best trained model, Evaluation metrics for train/val/test sets, Best hyperparameters)
    """
    # Define paths
    log_file_path = f"{path}/saved/{gender}/{label}/training_log.txt"
    best_result_path = f"{path}/saved/{gender}/{label}/best_result.txt"
    best_model_path = f"{path}/saved/{gender}/{label}/best_model.pt"

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = DualLogger(log_file_path)
    sys.stdout = logger

    try:
        # Load dataset
        df = pd.read_csv(f"{path}/data/{gender}.csv")
        train_loader, val_loader, test_loader, class_weights = preprocess(
            df, feature=features, label_column=label, test_size=0.15, validation_size=0.15, batch_size=batch
        )

        # Initialize best scores and hyperparameters
        best_params = None
        best_val_auc, best_val_f1 = 0, 0
        tolerance = 1

        if search_trials > 0:
            def objective(trial):
                """ Optuna optimization function for hyperparameter tuning. """
                # Define hyperparameter search space
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)

                print(f"\nTrial hyperparameters: dropout_rate={dropout_rate}")

                # Initialize model
                bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
                model = BERTMLPClassifier(
                    bert_model=bert_model,
                    num_continuous_features=len(features),
                    dropout_rate=dropout_rate
                ).to(device)

                # Train model
                train_model(
                    model, train_loader, val_loader, epochs=epoch, freeze=freeze, patience=patience,
                    class_weights=class_weights, threshold_range=threshold_range, f1_mode=f1_mode,
                    save_path=f"{path}/saved/{gender}/{label}/temp_model.pt"
                )

                # Load best model checkpoint and evaluate
                model.load_state_dict(torch.load(f"{path}/saved/{gender}/{label}/temp_model.pt", map_location=device))
                val_metrics = evaluate(model, val_loader, threshold_range=threshold_range)

                # Update best model if F1 improves, with AUC tolerance
                nonlocal best_val_f1, best_val_auc
                if val_metrics['best_f1'] > best_val_f1 or (val_metrics['best_f1'] == best_val_f1 and val_metrics['auc'] > best_val_auc - tolerance):
                    best_val_f1 = val_metrics['best_f1']
                    best_val_auc = val_metrics['roc_auc']
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Updated best model with AUC: {best_val_auc:.4f} and F1: {best_val_f1:.4f}")

                return val_metrics['best_f1']

            # Start Optuna hyperparameter tuning
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=search_trials)

            # Get best hyperparameters
            best_trial = study.best_trial
            best_params = best_trial.params
            print(f"\nBest trial hyperparameters: {best_params}")

        else:
            print("\nSkipping hyperparameter search. Loading existing model...")

        # Load best model for final evaluation
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        best_model = BERTMLPClassifier(
            bert_model=bert_model,
            num_continuous_features=len(features),
            dropout_rate=best_params['dropout_rate'] if best_params else 0.3
        ).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))

        train_result_path=f"{path}/saved/{gender}/{label}/train_result.pkl"
        validate_result_path=f"{path}/saved/{gender}/{label}/validate_result.pkl"
        test_result_path=f"{path}/saved/{gender}/{label}/test_result.pkl"

        # Final evaluation on train, val, and test sets
        print("\nFinal Evaluation on Train Set:")
        train_metrics = evaluate(best_model, train_loader, threshold_range=threshold_range, path=train_result_path)

        print("\nFinal Evaluation on Validation Set:")
        val_metrics = evaluate(best_model, val_loader, threshold_range=threshold_range, path=validate_result_path)

        print("\nFinal Evaluation on Test Set:")
        test_metrics = evaluate(best_model, test_loader, threshold_range=threshold_range, path=test_result_path)

        # Save results
        with open(best_result_path, "w") as f:
            f.write(f"Best Hyperparameters: {best_params}\n")
            f.write(f"Train Metrics: {train_metrics}\n")
            f.write(f"Validation Metrics: {val_metrics}\n")
            f.write(f"Test Metrics: {test_metrics}\n")

    finally:
        sys.stdout = logger.console
        logger.close()

    return best_model, {"train": train_metrics, "val": val_metrics, "test": test_metrics}, best_params

