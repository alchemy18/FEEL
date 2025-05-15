import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
from sklearn.utils import resample
import random
import numpy as np
import pandas as pd
import json
import warnings 
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def create_windows_and_labels(data, labels, window_size, window_overlap):
    # Ensure window_overlap is within valid range
    assert 0 <= window_overlap < 1, "window_overlap must be between 0 (no overlap) and 1 (exclusive)."
    
    # Compute step size. If window_overlap is 0, step equals window_size (no overlap).
    step = int(window_size * (1 - window_overlap))
    # Ensure step is at least 1 to avoid infinite loops
    step = max(1, step)
    
    new_data = []
    new_labels = []
    for i, row in enumerate(data):
        L = row.shape[0]
        # Use a sliding window with the computed step size
        for j in range(0, L, step):
            window = row[j:j+window_size]
            if window.shape[0] < window_size:
                # Pad the window on the right with zeros if it is shorter than window_size
                pad_size = window_size - window.shape[0]
                window = F.pad(window, (0, pad_size), "constant", 0)
            new_data.append(window)
            new_labels.append(labels[i])
            # Break early if this window reached the end of the row to avoid duplicate padded windows
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

class ClassifierLstmMlp(nn.Module):
    def __init__(self, nb_classes):
        super(ClassifierLstmMlp, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout_rate = 0.3

        # LSTM Module
        self.lstm = nn.LSTM(input_size=1, 
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           dropout=self.dropout_rate)
        
        # NN Module
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classification layer
        self.fc = nn.Linear(128, nb_classes)
        self.softmax = nn.Softmax(dim=1)
        
        # Training setup
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)  # [batch, features, 1]
        
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        x = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        x = self.mlp(x)
        
        x = self.fc(x)
        return self.softmax(x)

    def collate_fn(self, batch):
        data, label = zip(*batch)
        data_tensor = torch.stack(data)
        label_tensor = torch.stack(label)
        return data_tensor, label_tensor
    
    def fit(self, x_train, y_train, batch_size=8, nb_epochs=100, shuffle=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        train_data = list(zip(x_train, y_train))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                                 shuffle=shuffle, collate_fn=self.collate_fn)

        for epoch in range(nb_epochs):
            self.train()
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

        return None
    
    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            x_test = x_test.to(self.device)
            outputs = self.forward(x_test)
            return torch.argmax(outputs, dim=1)


def train_lstm_mlp_arousal(X_train, y_train_arousal, X_test, y_test_arousal, window_size, window_overlap):
    X_train = to_tensor_from_series(X_train).unsqueeze(1)
    X_test = to_tensor_from_series(X_test).unsqueeze(1)
    y_train_arousal = torch.tensor(y_train_arousal.values, dtype=torch.long)
    y_test_arousal = torch.tensor(y_test_arousal.values, dtype=torch.long)

    X_train, y_train_arousal = create_windows_and_labels(X_train, y_train_arousal, window_size, window_overlap)
    X_test,  y_test_arousal  = create_windows_and_labels(X_test, y_test_arousal, window_size, window_overlap)
    
    model_arousal = ClassifierLstmMlp(nb_classes=2)
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

def Arousal(PathFile, output_file=None, window_size = 60, window_overlap = 0.5):
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
    
    accuracy_arousal_LSTM, balanced_acc_arousal_LSTM, r2_a, mse_a, f1_a = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data=X_train[X_train['PID']==pi]
        train_data=X_train[X_train['PID']!=pi]        
        train_x=train_data['Data']
        train_arousal=train_data["arousal_category"]
        test_x=test_data['Data']
        test_arousal=test_data["arousal_category"]

        train_data_combined = pd.concat([train_x, train_arousal], axis=1)
        
        minority_class = train_data_combined[train_arousal.name].value_counts().idxmin()
        majority_class = train_data_combined[train_arousal.name].value_counts().idxmax()
        
        minority_data = train_data_combined[train_data_combined[train_arousal.name] == minority_class]
        majority_data = train_data_combined[train_data_combined[train_arousal.name] == majority_class]
        
        minority_oversampled = resample(minority_data, 
                                        replace=True,     # With replacement
                                        n_samples=len(majority_data),  # Match the number of majority class samples
                                        random_state=42)  # Ensure reproducibility
        
        train_data_balanced = pd.concat([majority_data, minority_oversampled])
        
        train_x = train_data_balanced.drop(columns=[train_arousal.name])
        train_arousal = train_data_balanced[train_arousal.name]
        train_x=train_x['Data']
        
        accuracy_arousal_LSTM[pi], balanced_acc_arousal_LSTM[pi], r2_a[pi], mse_a[pi], f1_a[pi]=train_lstm_mlp_arousal(train_x,train_arousal,test_x,test_arousal,window_size,window_overlap)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Raw Signal',
            'Model Name': 'LSTM',
            'Classification': 'Arousal',
            'Test_Accuracy': accuracy_arousal_LSTM[pi],
            'Test_F1': f1_a[pi]
        })
    # print("------ Average accuracy of LSTM on arousal ----------")
    acc_aro = sum(accuracy_arousal_LSTM.values())/len(accuracy_arousal_LSTM.values())

    # print("------ Average f1 of LSTM on arousal ----------")
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


def train_lstm_valence(X_train, y_train_valence, X_test, y_test_valence, window_size, window_overlap):
    # Fixed tensor conversion using to_list() instead of .values
    X_train = to_tensor_from_series(X_train).unsqueeze(1)
    X_test = to_tensor_from_series(X_test).unsqueeze(1)
    y_train_valence = torch.tensor(y_train_valence.values, dtype=torch.long)
    y_test_valence = torch.tensor(y_test_valence.values, dtype=torch.long)

    X_train, y_train_valence = create_windows_and_labels(X_train, y_train_valence, window_size, window_overlap)
    X_test, y_test_valence = create_windows_and_labels(X_test, y_test_valence, window_size, window_overlap)
    
    model_valence = ClassifierLstmMlp(nb_classes=2)
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

def Valence(PathFile,output_file=None, window_size = 60, window_overlap = 0.5):
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
    accuracy_valence_LSTM, balanced_acc_valence_LSTM, r2_v, mse_v, f1_v = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data=X_train[X_train['PID']==pi]
        train_data=X_train[X_train['PID']!=pi]

        train_x=train_data["Data"]
        train_valence=train_data["valence_category"]

        test_x=test_data["Data"]
        test_valence=test_data["valence_category"]

        train_data_combined = pd.concat([train_x, train_valence], axis=1)
        
        minority_class = train_data_combined[train_valence.name].value_counts().idxmin()
        majority_class = train_data_combined[train_valence.name].value_counts().idxmax()
        
        minority_data = train_data_combined[train_data_combined[train_valence.name] == minority_class]
        majority_data = train_data_combined[train_data_combined[train_valence.name] == majority_class]
        
        minority_oversampled = resample(minority_data, 
                                        replace=True,     # With replacement
                                        n_samples=len(majority_data),  # Match the number of majority class samples
                                        random_state=42)  # Ensure reproducibility
        
        train_data_balanced = pd.concat([majority_data, minority_oversampled])
        
        train_x = train_data_balanced.drop(columns=[train_valence.name])
        train_valence = train_data_balanced[train_valence.name]
        train_x=train_x['Data']
        
        accuracy_valence_LSTM[pi], balanced_acc_valence_LSTM[pi], r2_v[pi], mse_v[pi], f1_v[pi]=train_lstm_valence(train_x,train_valence,test_x,test_valence,window_size,window_overlap)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Raw Signal',
            'Model Name': 'LSTM',
            'Classification': 'Valence',
            'Test_Accuracy': accuracy_valence_LSTM[pi],
            'Test_F1': f1_v[pi]
        })
        
    # print("------ Average accuracy of LSTM on valence ----------")
    acc_va = sum(accuracy_valence_LSTM.values())/len(accuracy_valence_LSTM.values())

    # print("------ Average f1 of LSTM on valence ----------")
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
