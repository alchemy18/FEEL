import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
from sklearn.utils import resample

import warnings
warnings.filterwarnings('ignore')
random_seed = 42

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_sizes[0], padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_sizes[1], padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_sizes[2], padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
        self.shortcut_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += shortcut
        return F.relu(x)


class ClassifierResNet(nn.Module):
    def __init__(self, nb_classes):
        super(ClassifierResNet, self).__init__()
        n_feature_maps = 64

        self.models = nn.ModuleList()
        layers = [ResNetBlock(in_channels=1, out_channels=n_feature_maps * 2, 
                                kernel_sizes=[8,5,3]), ResNetBlock(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2, 
                                kernel_sizes=[8,5,3])]
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.models.append(nn.Sequential(*layers))

        self.fc = nn.Linear((n_feature_maps * 2), nb_classes)
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # print(f"forward shape:{inputs.shape}" )
        outputs = [self.models[0](inputs).squeeze(-1)]
        x = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        x = self.fc(x)
        return self.softmax(x)

    def collate_fn(self,batch):
        data, label = zip(*batch)
        data_tensor = torch.stack(data)
        label_tensor=torch.stack(label)
        return data_tensor , label_tensor
    
    def fit(self, x_train, y_train, batch_size=16, nb_epochs=100,shuffle=True):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        train_data = list(zip(x_train, y_train))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,collate_fn=self.collate_fn)

        for epoch in range(nb_epochs):
            self.train()
            epoch_loss = 0
            for inputs , labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # print(inputs.shape)
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # print(f"Epoch {epoch+1}/{nb_epochs}, Loss: {epoch_loss:.4f}")

        return None
    
    
    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            x_test = x_test.to(self.device)
            outputs = self.forward(x_test)
            return torch.argmax(outputs, dim=1)
    


def train_resnet_arousal(X_train, y_train_arousal, X_test, y_test_arousal):
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    y_train_arousal = torch.tensor(y_train_arousal.values, dtype=torch.long)
    y_test_arousal = torch.tensor(y_test_arousal.values, dtype=torch.long)
    
    # print(X_train.shape,X_test.shape,y_train_arousal.shape,y_test_arousal.shape)
    # Create and fit the resnet model for Arousal

    model_arousal_resnet = ClassifierResNet(nb_classes = 2)
    model_arousal_resnet.fit(X_train, y_train_arousal)

    # Predict on the test set for Arousal and Valence
    y_pred_arousal_resnet = model_arousal_resnet.predict(X_test)
    # print(type(y_pred_arousal_resnet))
    y_test_arousal = y_test_arousal.cpu().numpy()
    y_pred_arousal_resnet = y_pred_arousal_resnet.cpu().numpy()
    # Calculate accuracy for Arousal and Valence classification
    accuracy_arousal_resnet = accuracy_score(y_test_arousal, y_pred_arousal_resnet)
    balanced_acc_arousal_resnet = balanced_accuracy_score(y_test_arousal, y_pred_arousal_resnet)

    f1_a = f1_score(y_test_arousal, y_pred_arousal_resnet)

    r2_a = r2_score(y_test_arousal, y_pred_arousal_resnet)

    mse_a = mean_squared_error(y_test_arousal, y_pred_arousal_resnet)

    return accuracy_arousal_resnet, balanced_acc_arousal_resnet, r2_a, mse_a, f1_a


def Arousal(PathFile,output_file=None):
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
    
    accuracy_arousal_resnet, balanced_acc_arousal_resnet, r2_a, mse_a, f1_a = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data=X_train[X_train['PID']==pi]
        train_data=X_train[X_train['PID']!=pi]

        train_x=train_data[relevent_features]
        train_arousal=train_data["arousal_category"]

        test_x=test_data[relevent_features]
        test_arousal=test_data["arousal_category"]
        
        train_data_combined = pd.concat([train_x, train_arousal], axis=1)
        
        # Separate the minority and majority classes
        minority_class = train_data_combined[train_arousal.name].value_counts().idxmin()
        majority_class = train_data_combined[train_arousal.name].value_counts().idxmax()
        
        minority_data = train_data_combined[train_data_combined[train_arousal.name] == minority_class]
        majority_data = train_data_combined[train_data_combined[train_arousal.name] == majority_class]
        
        # Oversample the minority class by duplicating samples
        minority_oversampled = resample(minority_data, 
                                        replace=True,     # With replacement
                                        n_samples=len(majority_data),  # Match the number of majority class samples
                                        random_state=42)  # Ensure reproducibility
        
        # Combine oversampled minority class with majority class
        train_data_balanced = pd.concat([majority_data, minority_oversampled])
        
        # Separate features and labels after balancing
        train_x = train_data_balanced.drop(columns=[train_arousal.name])
        train_arousal = train_data_balanced[train_arousal.name]
        
        
        accuracy_arousal_resnet[pi], balanced_acc_arousal_resnet[pi], r2_a[pi], mse_a[pi], f1_a[pi]=train_resnet_arousal(train_x,train_arousal,test_x,test_arousal)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
            'Model Name': 'Resnet',
            'Classification': 'Arousal',
            'Test_Accuracy': accuracy_arousal_resnet[pi],
            'Test_F1': f1_a[pi]
        })
    # print("------ Average accuracy of resnet on arousal ----------")
    acc_aro = sum(accuracy_arousal_resnet.values())/len(accuracy_arousal_resnet.values())

    # print("------ Average balaned accuracy of resnet on arousal ----------")
    # print(sum(balanced_acc_arousal_resnet.values())/len(balanced_acc_arousal_resnet.values()))

    # print("------ Average r2 of resnet on arousal ----------")
    # print(sum(r2_a.values())/len(r2_a.values()))

    # print("------ Average mse of resnet on arousal ----------")
    # print(sum(mse_a.values())/len(mse_a.values()))

    # print("------ Average f1 of resnet on arousal ----------")
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

def train_resnet_valence(X_train, y_train_valence, X_test, y_test_valence):
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    y_train_valence = torch.tensor(y_train_valence.values, dtype=torch.long)
    y_test_valence = torch.tensor(y_test_valence.values, dtype=torch.long)
    # Create and fit the resnet model for valence

    model_valence_resnet = ClassifierResNet(nb_classes = 2)
    model_valence_resnet.fit(X_train, y_train_valence)

    # Predict on the test set for valence and Valence
    y_pred_valence_resnet = model_valence_resnet.predict(X_test)
    y_test_valence = y_test_valence.cpu().numpy()
    y_pred_valence_resnet = y_pred_valence_resnet.cpu().numpy()
    # Calculate accuracy for valence and Valence classification
    accuracy_valence_resnet = accuracy_score(y_test_valence, y_pred_valence_resnet)
    balanced_acc_valence_resnet = balanced_accuracy_score(y_test_valence, y_pred_valence_resnet)

    f1_a = f1_score(y_test_valence, y_pred_valence_resnet)

    r2_a = r2_score(y_test_valence, y_pred_valence_resnet)

    mse_a = mean_squared_error(y_test_valence, y_pred_valence_resnet)

    return accuracy_valence_resnet, balanced_acc_valence_resnet, r2_a, mse_a, f1_a


def Valence(PathFile,output_file=None):
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
    accuracy_valence_resnet, balanced_acc_valence_resnet, r2_v, mse_v, f1_v = {}, {}, {}, {}, {}
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data=X_train[X_train['PID']==pi]
        train_data=X_train[X_train['PID']!=pi]

        train_x=train_data[relevent_features]
        train_valence=train_data["valence_category"]

        test_x=test_data[relevent_features]
        test_valence=test_data["valence_category"]

        train_data_combined = pd.concat([train_x, train_valence], axis=1)
        
        # Separate the minority and majority classes
        minority_class = train_data_combined[train_valence.name].value_counts().idxmin()
        majority_class = train_data_combined[train_valence.name].value_counts().idxmax()
        
        minority_data = train_data_combined[train_data_combined[train_valence.name] == minority_class]
        majority_data = train_data_combined[train_data_combined[train_valence.name] == majority_class]
        
        # Oversample the minority class by duplicating samples
        minority_oversampled = resample(minority_data, 
                                        replace=True,     # With replacement
                                        n_samples=len(majority_data),  # Match the number of majority class samples
                                        random_state=42)  # Ensure reproducibility
        
        # Combine oversampled minority class with majority class
        train_data_balanced = pd.concat([majority_data, minority_oversampled])
        
        # Separate features and labels after balancing
        train_x = train_data_balanced.drop(columns=[train_valence.name])
        train_valence = train_data_balanced[train_valence.name]
        
        accuracy_valence_resnet[pi], balanced_acc_valence_resnet[pi], r2_v[pi], mse_v[pi], f1_v[pi]=train_resnet_valence(train_x,train_valence,test_x,test_valence)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
            'Model Name': 'Resnet',
            'Classification': 'Valence',
            'Test_Accuracy': accuracy_valence_resnet[pi],
            'Test_F1': f1_v[pi]
        })
        
    # print("------ Average accuracy of resnet on valence ----------")
    acc_va = sum(accuracy_valence_resnet.values())/len(accuracy_valence_resnet.values())

    # print("------ Average balaned accuracy of resnet on valence ----------")
    # print(sum(balanced_acc_valence_resnet.values())/len(balanced_acc_valence_resnet.values()))

    # print("------ Average r2 of resnet on valence ----------")
    # print(sum(r2_v.values())/len(r2_v.values()))

    # print("------ Average mse of resnet on valence ----------")
    # print(sum(mse_v.values())/len(mse_v.values()))

    # print("------ Average f1 of resnet on valence ----------")
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

