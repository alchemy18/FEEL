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

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, nb_classes):
        super(TransformerClassifier, self).__init__()
        self.hidden_dim = 128
        self.nhead = 4
        
        # First layer to project features
        self.input_layer = nn.Linear(input_dim, self.hidden_dim)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.nhead,
            batch_first=True
        )
        
        # Neural network layers
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classification layer
        self.fc = nn.Linear(128, nb_classes)
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Project input features
        x = self.input_layer(x)
        
        # Reshape for attention (batch_size, 1, hidden_dim)
        x = x.unsqueeze(1)
        
        # Apply self-attention
        x, _ = self.self_attention(x, x, x)
        
        # Squeeze back to (batch_size, hidden_dim)
        x = x.squeeze(1)
        
        # Apply MLP
        x = self.mlp(x)
        
        # Final classification
        x = self.fc(x)
        return self.softmax(x)

    def fit(self, x_train, y_train, batch_size=16, nb_epochs=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        for epoch in range(nb_epochs):
            self.train()
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
            outputs = self(x_test)
            return torch.argmax(outputs, dim=1)

def train_transformer_arousal(X_train, y_train_arousal, X_test, y_test_arousal):
    input_dim = X_train.shape[1]
    
    model_arousal = TransformerClassifier(input_dim=input_dim, nb_classes=2)
    model_arousal.fit(X_train.values, y_train_arousal.values)

    y_pred_arousal = model_arousal.predict(X_test.values)
    y_pred_arousal = y_pred_arousal.cpu().numpy()
    
    accuracy_arousal = accuracy_score(y_test_arousal, y_pred_arousal)
    balanced_acc_arousal = balanced_accuracy_score(y_test_arousal, y_pred_arousal)
    f1_a = f1_score(y_test_arousal, y_pred_arousal)
    r2_a = r2_score(y_test_arousal, y_pred_arousal)
    mse_a = mean_squared_error(y_test_arousal, y_pred_arousal)

    return accuracy_arousal, balanced_acc_arousal, r2_a, mse_a, f1_a

def Arousal(PathFile, output_file=None):
    if not(PathFile[-16:]=="Features_EDA.csv" or PathFile[-16:]=="Features_PPG.csv" or PathFile[-21:]=="Features_Combined.csv"):
        raise AssertionError(f"Path file is not valid, please check {PathFile[-16:]}")
    
    if PathFile[-16:] == "Features_EDA.csv":
        signal_type = "EDA"
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    elif PathFile[-16:] == "Features_PPG.csv":
        signal_type = "PPG"
        relevent_features = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    else:
        signal_type = "Combined"
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR','BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
               
    X_train = pd.read_csv(PathFile)
    X_train = X_train.fillna(0)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)
    results = []
    
    accuracy_arousal_trans, balanced_acc_arousal_trans, r2_a, mse_a, f1_a = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data = X_train[X_train['PID']==pi]
        train_data = X_train[X_train['PID']!=pi]

        train_x = train_data[relevent_features]
        train_arousal = train_data["arousal_category"]
        test_x = test_data[relevent_features]
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
        
        accuracy_arousal_trans[pi], balanced_acc_arousal_trans[pi], r2_a[pi], mse_a[pi], f1_a[pi] = train_transformer_arousal(
            train_x, train_arousal, test_x, test_arousal
        )
        
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
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

def train_transformer_valence(X_train, y_train_valence, X_test, y_test_valence):
    input_dim = X_train.shape[1]
    
    model_valence = TransformerClassifier(input_dim=input_dim, nb_classes=2)
    model_valence.fit(X_train.values, y_train_valence.values)

    y_pred_valence = model_valence.predict(X_test.values)
    y_pred_valence = y_pred_valence.cpu().numpy()
    
    accuracy_valence = accuracy_score(y_test_valence, y_pred_valence)
    balanced_acc_valence = balanced_accuracy_score(y_test_valence, y_pred_valence)
    f1_v = f1_score(y_test_valence, y_pred_valence)
    r2_v = r2_score(y_test_valence, y_pred_valence)
    mse_v = mean_squared_error(y_test_valence, y_pred_valence)

    return accuracy_valence, balanced_acc_valence, r2_v, mse_v, f1_v

def Valence(PathFile, output_file=None):
    if not(PathFile[-16:]=="Features_EDA.csv" or PathFile[-16:]=="Features_PPG.csv" or PathFile[-21:]=="Features_Combined.csv"):
        raise AssertionError(f"Path file is not valid, please check {PathFile[-16:]}")
    
    if PathFile[-16:] == "Features_EDA.csv":
        signal_type = "EDA"
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    elif PathFile[-16:] == "Features_PPG.csv":
        signal_type = "PPG"
        relevent_features = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    else:
        signal_type = "Combined"
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR','BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
                
    X_train = pd.read_csv(PathFile)
    X_train = X_train.fillna(0)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)
    results = []
    
    accuracy_valence_trans, balanced_acc_valence_trans, r2_v, mse_v, f1_v = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data = X_train[X_train['PID']==pi]
        train_data = X_train[X_train['PID']!=pi]

        train_x = train_data[relevent_features]
        train_valence = train_data["valence_category"]
        test_x = test_data[relevent_features]
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
        
        accuracy_valence_trans[pi], balanced_acc_valence_trans[pi], r2_v[pi], mse_v[pi], f1_v[pi] = train_transformer_valence(
            train_x, train_valence, test_x, test_valence
        )
        
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
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

# p = "Feature_EDA.csv"
# t = "ankush.csv"
# Valence(PathFile = p,output_file=t)