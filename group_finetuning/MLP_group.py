from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
import warnings
import torch
import pickle

warnings.filterwarnings('ignore')
random_seed = 42

def train_MLP_arousal(X_train, y_train_arousal, X_test, y_test_arousal,X_test1,y_test_arousal1, hidden_layer_sizes=(100,), random_seed=42,gr = None):
    # Create and fit the MLP model for Arousal

    model_arousal_mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_seed)
    model_arousal_mlp.fit(X_train, y_train_arousal)

    # Predict on the test set for Arousal and Valence
    y_pred_arousal_mlp = model_arousal_mlp.predict(X_test)
    accuracy_arousal_mlp = accuracy_score(y_test_arousal, y_pred_arousal_mlp)
    f1_a = f1_score(y_test_arousal, y_pred_arousal_mlp)

    y_pred_arousal_mlp1 = model_arousal_mlp.predict(X_test1)
    accuracy_arousal_mlp1 = accuracy_score(y_test_arousal1, y_pred_arousal_mlp1)
    f1_a1 = f1_score(y_test_arousal1, y_pred_arousal_mlp1)

    with open("<path to save the model>",'wb+') as file:
        pickle.dump(model_arousal_mlp, file)

    return accuracy_arousal_mlp*100, f1_a, accuracy_arousal_mlp1*100, f1_a1


def Arousal(PathFile, test_path_list):
    if not(PathFile[-16:]=="Features_EDA.csv" or PathFile[-16:]=="Features_PPG.csv" or PathFile[-21:]=="Features_combined.csv"):
        raise AssertionError(f"Path file is not valid, please check {PathFile[-16:]}")
    
    if PathFile[-16:] == "Features_EDA.csv":
        signal_type = "EDA"
        gr = PathFile.split("_")[0][-3:] + "_EDA" 
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    elif PathFile[-16:] == "Features_PPG.csv":
        signal_type = "PPG"
        gr = PathFile.split("_")[0][-3:] + "_PPG" 
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
        gr = PathFile.split("_")[0][-3:] + "_Combined" 
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
    
    X_test = pd.read_csv(test_path_list[0])
    X_test = X_test.fillna(0)
    X_test.replace([np.inf, -np.inf], 0, inplace=True)

    X_test1 = pd.read_csv(test_path_list[1])
    X_test1 = X_test1.fillna(0)
    X_test1.replace([np.inf, -np.inf], 0, inplace=True)
    
    test_data=X_test
    train_data=X_train
    test_data1=X_test1

    train_x=train_data[relevent_features]
    train_arousal=train_data["arousal_category"]

    test_x=test_data[relevent_features]
    test_arousal=test_data["arousal_category"]
    test_x1=test_data1[relevent_features]
    test_arousal1=test_data1["arousal_category"]

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
        
        
    accuracy_arousal_RF,f1_a, accuracy_arousal_RF1, f1_a1 =train_MLP_arousal(train_x,train_arousal,test_x,test_arousal,test_x1,test_arousal1,gr = gr)
    
    return round(accuracy_arousal_RF,3), round(f1_a,3), round(accuracy_arousal_RF1,3), round(f1_a1,3)

def train_MLP_valence(X_train, y_train_valence, X_test, y_test_valence, X_test1,y_test_valence1, hidden_layer_sizes=(100,), random_seed=42, gr = None):

    # Create and fit the MLP model for Valence
    model_valence_mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_seed)
    model_valence_mlp.fit(X_train, y_train_valence)

    # Predict on the test set for Arousal and Valence
    y_pred_valence_mlp = model_valence_mlp.predict(X_test)
    accuracy_valence_mlp = accuracy_score(y_test_valence, y_pred_valence_mlp)
    f1_v = f1_score(y_test_valence, y_pred_valence_mlp)

    y_pred_valence_mlp1 = model_valence_mlp.predict(X_test1)
    accuracy_valence_mlp1 = accuracy_score(y_test_valence1, y_pred_valence_mlp1)
    f1_v1 = f1_score(y_test_valence1, y_pred_valence_mlp1)

    with open("<path to save the model>",'wb+') as file:
        pickle.dump(model_valence_mlp, file)

    return accuracy_valence_mlp*100, f1_v, accuracy_valence_mlp1*100, f1_v1

def Valence(PathFile, test_path_list):
    if not(PathFile[-16:]=="Features_EDA.csv" or PathFile[-16:]=="Features_PPG.csv" or PathFile[-21:]=="Features_combined.csv"):
        raise AssertionError(f"Path file is not valid, please check {PathFile[-16:]}")
    
    if PathFile[-16:] == "Features_EDA.csv":
        signal_type = "EDA"
        gr = PathFile.split("_")[0][-3] + "_EDA" 
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    elif PathFile[-16:] == "Features_PPG.csv":
        signal_type = "PPG"
        gr = PathFile.split("_")[0][-3] + "_PPG"
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
        gr = PathFile.split("_")[0][-3] + "_Combined"
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
    
    X_test = pd.read_csv(test_path_list[0])
    X_test = X_test.fillna(0)
    X_test.replace([np.inf, -np.inf], 0, inplace=True)

    X_test1 = pd.read_csv(test_path_list[1])
    X_test1 = X_test1.fillna(0)
    X_test1.replace([np.inf, -np.inf], 0, inplace=True)                

    test_data=X_test
    train_data=X_train 
    test_data1=X_test1

    
    train_x=train_data[relevent_features]
    train_valence=train_data["valence_category"]

    test_x=test_data[relevent_features]
    test_valence=test_data["valence_category"]

    test_x1=test_data1[relevent_features]
    test_valence1=test_data1["valence_category"]

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
        
    accuracy_valence_LDA,f1_v, accuracy_valence_LDA1, f1_v1 =train_MLP_valence(train_x,train_valence,test_x,test_valence,test_x1,test_valence1,gr = gr)
    
    
    return round(accuracy_valence_LDA,3),  round(f1_v,3), round(accuracy_valence_LDA1,3), round(f1_v1,3)

