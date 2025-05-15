from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
import warnings
warnings.filterwarnings('ignore')
random_seed = 42
def train_MLP_arousal(X_train, y_train_arousal, X_test, y_test_arousal, hidden_layer_sizes=(100,), random_seed=42):
    # Create and fit the MLP model for Arousal

    model_arousal_mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_seed)
    model_arousal_mlp.fit(X_train, y_train_arousal)

    # Predict on the test set for Arousal and Valence
    y_pred_arousal_mlp = model_arousal_mlp.predict(X_test)

    # Calculate accuracy for Arousal and Valence classification
    accuracy_arousal_mlp = accuracy_score(y_test_arousal, y_pred_arousal_mlp)
    balanced_acc_arousal_mlp = balanced_accuracy_score(y_test_arousal, y_pred_arousal_mlp)

    f1_a = f1_score(y_test_arousal, y_pred_arousal_mlp)

    r2_a = r2_score(y_test_arousal, y_pred_arousal_mlp)

    mse_a = mean_squared_error(y_test_arousal, y_pred_arousal_mlp)

    return accuracy_arousal_mlp, balanced_acc_arousal_mlp, r2_a, mse_a, f1_a


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
    accuracy_arousal_mlp, balanced_acc_arousal_mlp, r2_a, mse_a, f1_a = {}, {}, {}, {}, {}
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
        
        
        accuracy_arousal_mlp[pi], balanced_acc_arousal_mlp[pi], r2_a[pi], mse_a[pi], f1_a[pi]=train_MLP_arousal(train_x,train_arousal,test_x,test_arousal)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
            'Model Name': 'MLP',
            'Classification': 'Arousal',
            'Test_Accuracy': accuracy_arousal_mlp[pi],
            'Test_F1': f1_a[pi]
        })
    # print("------ Average accuracy of mlp on arousal ----------")
    acc_aro = sum(accuracy_arousal_mlp.values())/len(accuracy_arousal_mlp.values())

    # print("------ Average balaned accuracy of mlp on arousal ----------")
    # print(sum(balanced_acc_arousal_mlp.values())/len(balanced_acc_arousal_mlp.values()))

    # print("------ Average r2 of mlp on arousal ----------")
    # print(sum(r2_a.values())/len(r2_a.values()))

    # print("------ Average mse of mlp on arousal ----------")
    # print(sum(mse_a.values())/len(mse_a.values()))

    # print("------ Average f1 of mlp on arousal ----------")
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

def train_MLP_valence(X_train, y_train_valence, X_test, y_test_valence, hidden_layer_sizes=(100,), random_seed=42):

    # Create and fit the MLP model for Valence
    model_valence_mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_seed)
    model_valence_mlp.fit(X_train, y_train_valence)

    # Predict on the test set for Arousal and Valence
    y_pred_valence_mlp = model_valence_mlp.predict(X_test)

    # Calculate accuracy for Arousal and Valence classification
    accuracy_valence_mlp = accuracy_score(y_test_valence, y_pred_valence_mlp)
    balanced_acc_valence_mlp = balanced_accuracy_score(y_test_valence, y_pred_valence_mlp)

    f1_v = f1_score(y_test_valence, y_pred_valence_mlp)

    r2_v = r2_score(y_test_valence, y_pred_valence_mlp)

    mse_v = mean_squared_error(y_test_valence, y_pred_valence_mlp)

    return accuracy_valence_mlp, balanced_acc_valence_mlp, r2_v, mse_v, f1_v

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
    
    accuracy_valence_mlp, balanced_acc_valence_mlp, r2_v, mse_v, f1_v = {}, {}, {}, {}, {}
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
        
        accuracy_valence_mlp[pi], balanced_acc_valence_mlp[pi], r2_v[pi], mse_v[pi], f1_v[pi]=train_MLP_valence(train_x,train_valence,test_x,test_valence)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
            'Model Name': 'MLP',
            'Classification': 'Valence',
            'Test_Accuracy': accuracy_valence_mlp[pi],
            'Test_F1': f1_v[pi]
        })
    # print("------ Average accuracy of mlp on valence ----------")
    acc_va = sum(accuracy_valence_mlp.values())/len(accuracy_valence_mlp.values())

    # print("------ Average balaned accuracy of mlp on valence ----------")
    # print(sum(balanced_acc_valence_mlp.values())/len(balanced_acc_valence_mlp.values()))

    # print("------ Average r2 of mlp on valence ----------")
    # print(sum(r2_v.values())/len(r2_v.values()))

    # print("------ Average mse of mlp on valence ----------")
    # print(sum(mse_v.values())/len(mse_v.values()))

    # print("------ Average f1 of mlp on valence ----------")
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

