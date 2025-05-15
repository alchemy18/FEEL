import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.utils import resample
import random
import numpy as np
import pandas as pd
import json
import warnings 
import logging
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 1)
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, _ = x.shape
        squeezed = self.squeeze(x).view(batch_size, channels)
        excitation = self.excitation(squeezed)
        return x * excitation.view(batch_size, channels, 1)

class FeatureExtractor(nn.Module):
    def __init__(self, output_dim, window_sizes=[5, 9, 13], init_filters=32):
        super().__init__()
        
        self.input_channels = 1
        self.output_dim = output_dim
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.input_channels, init_filters, kernel_size=ws, padding='same'),
                nn.BatchNorm1d(init_filters),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(init_filters, init_filters*2, kernel_size=ws, padding='same'),
                nn.BatchNorm1d(init_filters*2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool1d(kernel_size=2)
            ) for ws in window_sizes
        ])
        
        self.se_block = SqueezeExcitationBlock(len(window_sizes) * init_filters * 2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.projection = nn.Sequential(
            nn.Linear(len(window_sizes) * init_filters * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        conv_outputs = []
        for conv_block in self.conv_layers:
            conv_outputs.append(conv_block(x))
        
        multi_scale_features = torch.cat(conv_outputs, dim=1)
        attended_features = self.se_block(multi_scale_features)
        pooled_features = self.adaptive_pool(attended_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        return self.projection(pooled_features)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class EmotionTransformer(nn.Module):
    def __init__(self, num_emotions=2, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.feature_extraction = FeatureExtractor(output_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotions)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, x):
        features = self.feature_extraction(x)
        features = features.unsqueeze(0)
        features = self.pos_encoder(features)
        encoded = self.transformer_encoder(features)
        output = encoded.mean(dim=0)
        return F.softmax(self.classifier(output), dim=1)

    def fit(self, x_train, y_train, batch_size=16, nb_epochs=50):
        self.to(self.device)
        train_data = list(zip(x_train, y_train))
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: (
                torch.stack([item[0] for item in batch]).to(self.device),
                torch.stack([item[1] for item in batch]).to(self.device)
            )
        )

        for epoch in range(nb_epochs):
            self.train()
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            x_test = x_test.to(self.device)
            outputs = self(x_test)
            return torch.argmax(outputs, dim=1)

def create_windows_and_labels(data, labels, window_size, window_overlap):
    assert 0 <= window_overlap < 1, "window_overlap must be between 0 (no overlap) and 1 (exclusive)."
    step = int(window_size * (1 - window_overlap))
    step = max(1, step)
    
    new_data = []
    new_labels = []
    for i, row in enumerate(data):
        L = row.shape[0]
        for j in range(0, L, step):
            window = row[j:j+window_size]
            if window.shape[0] < window_size:
                pad_size = window_size - window.shape[0]
                window = F.pad(window, (0, pad_size), "constant", 0)
            new_data.append(window)
            new_labels.append(labels[i])
            if j + window_size >= L:
                break
    new_data = torch.stack(new_data)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    return new_data, new_labels

def to_tensor_from_series(series, pad_value=0):
    """
    Convert a pandas Series (or list) of sequences (numpy arrays)
    into a padded torch tensor.

    Args:
        series (pd.Series or list): A series or list where each element is a numpy array.
        pad_value (float): The value to use for padding.

    Returns:
        tensor (torch.Tensor): A tensor of shape (num_sequences, max_length).
    """
    sequences = series.to_list() if hasattr(series, "to_list") else series
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [
        np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=pad_value)
        for seq in sequences
    ]
    return torch.tensor(padded_sequences, dtype=torch.float32)

def train_transformer_arousal(X_train, y_train_arousal, X_test, y_test_arousal, window_size, window_overlap):
    X_train = to_tensor_from_series(X_train).unsqueeze(1)
    X_test = to_tensor_from_series(X_test).unsqueeze(1)
    y_train_arousal = torch.tensor(y_train_arousal.values, dtype=torch.long)
    y_test_arousal = torch.tensor(y_test_arousal.values, dtype=torch.long)

    X_train, y_train_arousal = create_windows_and_labels(X_train, y_train_arousal, window_size, window_overlap)
    X_test, y_test_arousal = create_windows_and_labels(X_test, y_test_arousal, window_size, window_overlap)
    
    model_arousal = EmotionTransformer(num_emotions=2)
    model_arousal.fit(X_train, y_train_arousal)

    y_pred_arousal = model_arousal.predict(X_test)
    y_test_arousal = y_test_arousal.cpu().numpy()
    y_pred_arousal = y_pred_arousal.cpu().numpy()
    
    accuracy_arousal = accuracy_score(y_test_arousal, y_pred_arousal)
    balanced_acc_arousal = balanced_accuracy_score(y_test_arousal, y_pred_arousal)
    f1_a = f1_score(y_test_arousal, y_pred_arousal)
    r2_a = r2_score(y_test_arousal, y_pred_arousal)
    mse_a = mean_squared_error(y_test_arousal, y_pred_arousal)

    return accuracy_arousal, balanced_acc_arousal, r2_a, mse_a, f1_a

def Arousal(PathFile, output_file=None, window_size=60, window_overlap=0.5):
    if not(PathFile[-11:]=="Raw_EDA.csv" or PathFile[-11:]=="Raw_PPG.csv"):
        raise AssertionError("Path file is not Raw_EDA.csv or Raw_PPG.csv")
    if PathFile[-11:] == "Raw_EDA.csv":
        signal_type = "EDA"
    elif PathFile[-11:] == "Raw_PPG.csv":
        signal_type = "PPG"
        
    X_train = pd.read_csv(PathFile)
    X_train = X_train.fillna(0)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)    
    X_train['Data'] = X_train['Data'].apply(lambda x: np.array(json.loads(x), dtype=np.float64))
    results = []
    
    accuracy_arousal_trans, balanced_acc_arousal_trans, r2_a, mse_a, f1_a = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data = X_train[X_train['PID']==pi]
        train_data = X_train[X_train['PID']!=pi]        
        train_x = train_data['Data']
        train_arousal = train_data["arousal_category"]
        test_x = test_data['Data']
        test_arousal = test_data["arousal_category"]

        train_data_combined = pd.concat([train_x, train_arousal], axis=1)
        
        minority_class = train_data_combined[train_arousal.name].value_counts().idxmin()
        majority_class = train_data_combined[train_arousal.name].value_counts().idxmax()
        
        minority_data = train_data_combined[train_data_combined[train_arousal.name] == minority_class]
        majority_data = train_data_combined[train_data_combined[train_arousal.name] == majority_class]
        
        minority_oversampled = resample(minority_data, 
                                      replace=True,
                                      n_samples=len(majority_data),
                                      random_state=42)
        
        train_data_balanced = pd.concat([majority_data, minority_oversampled])
        
        train_x = train_data_balanced.drop(columns=[train_arousal.name])
        train_arousal = train_data_balanced[train_arousal.name]
        train_x = train_x['Data']
        
        accuracy_arousal_trans[pi], balanced_acc_arousal_trans[pi], r2_a[pi], mse_a[pi], f1_a[pi] = train_transformer_arousal(
            train_x, train_arousal, test_x, test_arousal, window_size, window_overlap
        )
        
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Raw Signal',
            'Model Name': 'Transformer',
            'Classification': 'Arousal',
            'Test_Accuracy': accuracy_arousal_trans[pi],
            'Test_F1': f1_a[pi]
        })
        
    # print("------ Average accuracy of Transformer on arousal ----------")
    acc_aro = sum(accuracy_arousal_trans.values())/len(accuracy_arousal_trans.values())

    # print("------ Average f1 of Transformer on arousal ----------")
    f1_aro = sum(f1_a.values())/len(f1_a.values())

    if output_file:
    # Convert the results list into a DataFrame
        df_results = pd.DataFrame(results)
        
        if os.path.exists(output_file):
            # Read the existing CSV into a DataFrame
            existing_df = pd.read_csv(output_file)
            # Append the new results to the existing DataFrame
            df_results = pd.concat([existing_df, df_results], ignore_index=True)
        
        # Write the combined DataFrame to CSV (always write header since it's a full overwrite)
        df_results.to_csv(output_file, index=False)
        
    return acc_aro, f1_aro

def train_transformer_valence(X_train, y_train_valence, X_test, y_test_valence, window_size, window_overlap):
    X_train = to_tensor_from_series(X_train).unsqueeze(1)
    X_test = to_tensor_from_series(X_test).unsqueeze(1)
    y_train_valence = torch.tensor(y_train_valence.values, dtype=torch.long)
    y_test_valence = torch.tensor(y_test_valence.values, dtype=torch.long)

    X_train, y_train_valence = create_windows_and_labels(X_train, y_train_valence, window_size, window_overlap)
    X_test, y_test_valence = create_windows_and_labels(X_test, y_test_valence, window_size, window_overlap)
    
    model_valence = EmotionTransformer(num_emotions=2)
    model_valence.fit(X_train, y_train_valence)

    y_pred_valence = model_valence.predict(X_test)
    y_test_valence = y_test_valence.cpu().numpy()
    y_pred_valence = y_pred_valence.cpu().numpy()
    
    accuracy_valence = accuracy_score(y_test_valence, y_pred_valence)
    balanced_acc_valence = balanced_accuracy_score(y_test_valence, y_pred_valence)
    f1_v = f1_score(y_test_valence, y_pred_valence)
    r2_v = r2_score(y_test_valence, y_pred_valence)
    mse_v = mean_squared_error(y_test_valence, y_pred_valence)

    return accuracy_valence, balanced_acc_valence, r2_v, mse_v, f1_v

def Valence(PathFile, output_file=None, window_size=60, window_overlap=0.5):
    if not(PathFile[-11:]=="Raw_EDA.csv" or PathFile[-11:]=="Raw_PPG.csv"):
        raise AssertionError("Path file is not Raw_EDA.csv or Raw_PPG.csv")
    if PathFile[-11:] == "Raw_EDA.csv":
        signal_type = "EDA"
    elif PathFile[-11:] == "Raw_PPG.csv":
        signal_type = "PPG"
        
    X_train = pd.read_csv(PathFile)
    X_train = X_train.fillna(0)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)
    X_train['Data'] = X_train['Data'].apply(lambda x: np.array(json.loads(x), dtype=np.float64))
    
    results = []
    accuracy_valence_trans, balanced_acc_valence_trans, r2_v, mse_v, f1_v = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data = X_train[X_train['PID']==pi]
        train_data = X_train[X_train['PID']!=pi]

        train_x = train_data["Data"]
        train_valence = train_data["valence_category"]
        test_x = test_data["Data"]
        test_valence = test_data["valence_category"]

        train_data_combined = pd.concat([train_x, train_valence], axis=1)
        
        minority_class = train_data_combined[train_valence.name].value_counts().idxmin()
        majority_class = train_data_combined[train_valence.name].value_counts().idxmax()
        
        minority_data = train_data_combined[train_data_combined[train_valence.name] == minority_class]
        majority_data = train_data_combined[train_data_combined[train_valence.name] == majority_class]
        
        minority_oversampled = resample(minority_data, 
                                      replace=True,
                                      n_samples=len(majority_data),
                                      random_state=42)
        
        train_data_balanced = pd.concat([majority_data, minority_oversampled])
        
        train_x = train_data_balanced.drop(columns=[train_valence.name])
        train_valence = train_data_balanced[train_valence.name]
        train_x = train_x['Data']
        
        accuracy_valence_trans[pi], balanced_acc_valence_trans[pi], r2_v[pi], mse_v[pi], f1_v[pi] = train_transformer_valence(
            train_x, train_valence, test_x, test_valence, window_size, window_overlap
        )
        
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Raw Signal',
            'Model Name': 'Transformer',
            'Classification': 'Valence',
            'Test_Accuracy': accuracy_valence_trans[pi],
            'Test_F1': f1_v[pi]
        })
    
    # print("------ Average accuracy of Transformer on valence ----------")
    acc_va = sum(accuracy_valence_trans.values())/len(accuracy_valence_trans.values())

    # print("------ Average f1 of Transformer on valence ----------")
    f1_va = sum(f1_v.values())/len(f1_v.values())

    if output_file:
    # Convert the results list into a DataFrame
        df_results = pd.DataFrame(results)
        
        if os.path.exists(output_file):
            # Read the existing CSV into a DataFrame
            existing_df = pd.read_csv(output_file)
            # Append the new results to the existing DataFrame
            df_results = pd.concat([existing_df, df_results], ignore_index=True)
        
        # Write the combined DataFrame to CSV (always write header since it's a full overwrite)
        df_results.to_csv(output_file, index=False)
        
    return acc_va, f1_va

# p = "Raw_EDA.csv"
# t = "ankush.csv"
# Arousal(PathFile = p,output_file=t)