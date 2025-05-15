from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
from sklearn.utils import resample
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import warnings 
import random
import torch

warnings.filterwarnings('ignore')

seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def train_random_forest_arousal(X_train, y_train_arousal, X_test, y_test_arousal):

    model_arousal_RF = RandomForestClassifier(n_estimators=100, random_state=seed ,n_jobs=5)
    model_arousal_RF.fit(X_train, y_train_arousal)


    # Predict on the test set for Arousal
    y_pred_arousal_RF = model_arousal_RF.predict(X_test)

    # Calculate accuracy for Arousal classification
    accuracy_arousal_RF = accuracy_score(y_test_arousal, y_pred_arousal_RF)
    balanced_acc_arousal_RF = balanced_accuracy_score(y_test_arousal, y_pred_arousal_RF)

    f1_a = f1_score(y_test_arousal, y_pred_arousal_RF)


    r2_a = r2_score(y_test_arousal, y_pred_arousal_RF)

    mse_a = mean_squared_error(y_test_arousal, y_pred_arousal_RF)

    return accuracy_arousal_RF, balanced_acc_arousal_RF, r2_a, mse_a, f1_a

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
    
    accuracy_arousal_RF, balanced_acc_arousal_RF, r2_a, mse_a, f1_a= {}, {}, {}, {}, {}
    
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

        accuracy_arousal_RF[pi], balanced_acc_arousal_RF[pi], r2_a[pi], mse_a[pi], f1_a[pi]=train_random_forest_arousal(train_x,train_arousal,test_x,test_arousal)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
            'Model Name': 'Random Forest',
            'Classification': 'Arousal',
            'Test_Accuracy': accuracy_arousal_RF[pi],
            'Test_F1': f1_a[pi]
        })

    # print("------ Average accuracy of RF on arousal ----------")
    acc_aro = sum(accuracy_arousal_RF.values())/len(accuracy_arousal_RF.values())

    # print("------ Average balaned accuracy of RF on arousal ----------")
    # print(sum(balanced_acc_arousal_RF.values())/len(balanced_acc_arousal_RF.values()))

    # print("------ Average r2 of RF on arousal ----------")
    # print(sum(r2_a.values())/len(r2_a.values()))

    # print("------ Average mse of RF on arousal ----------")
    # print(sum(mse_a.values())/len(mse_a.values()))

    # print("------ Average f1 of RF on arousal ----------")
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

def train_random_forest_valence(X_train, y_train_valence, X_test , y_test_valence):

    # Create and fit the RF model for Valence with k=9
    model_valence_RF = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=5)
    model_valence_RF.fit(X_train, y_train_valence)

    # Predict on the test set for Valence
    y_pred_valence_RF = model_valence_RF.predict(X_test)

    accuracy_valence_RF = accuracy_score(y_test_valence, y_pred_valence_RF)
    balanced_acc_valence_RF = balanced_accuracy_score(y_test_valence, y_pred_valence_RF)

    f1_v = f1_score(y_test_valence, y_pred_valence_RF)


    r2_v = r2_score(y_test_valence, y_pred_valence_RF)

    mse_v = mean_squared_error(y_test_valence, y_pred_valence_RF)

    return accuracy_valence_RF, balanced_acc_valence_RF, r2_v, mse_v, f1_v

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
    accuracy_valence_RF, balanced_acc_valence_RF, r2_v, mse_v, f1_v= {}, {}, {}, {}, {}
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
        

        accuracy_valence_RF[pi], balanced_acc_valence_RF[pi], r2_v[pi], mse_v[pi], f1_v[pi]=train_random_forest_valence(train_x,train_valence,test_x,test_valence)
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
            'Model Name': 'Random Forest',
            'Classification': 'Valence',
            'Test_Accuracy': accuracy_valence_RF[pi],
            'Test_F1': f1_v[pi]
        })

    # print("------ Average accuracy of RF on valence ----------")
    acc_va = sum(accuracy_valence_RF.values())/len(accuracy_valence_RF.values())

    # print("------ Average balaned accuracy of RF on valence ----------")
    # print(sum(balanced_acc_valence_RF.values())/len(balanced_acc_valence_RF.values()))

    # print("------ Average r2 of RF on valence ----------")
    # print(sum(r2_v.values())/len(r2_v.values()))

    # print("------ Average mse of RF on valence ----------")
    # print(sum(mse_v.values())/len(mse_v.values()))

    # print("------ Average f1 of RF on valence ----------")
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


