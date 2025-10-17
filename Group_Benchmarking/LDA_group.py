from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
import warnings 
import random
import torch
import pickle

warnings.filterwarnings('ignore')

seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def train_LDA_arousal(X_train, y_train_arousal, X_test, y_test_arousal, X_test1,y_test_arousal1 ,gr = None):

    model_arousal_LDA = LinearDiscriminantAnalysis(n_components=1)
    # print(y_train_arousal.min(),y_train_arousal.max())
    model_arousal_LDA.fit(X_train, y_train_arousal)

    # Predict on the test set for Arousal
    y_pred_arousal_LDA = model_arousal_LDA.predict(X_test)

    # Calculate accuracy for Arousal classification
    accuracy_arousal_LDA = accuracy_score(y_test_arousal, y_pred_arousal_LDA)
    f1_a = f1_score(y_test_arousal, y_pred_arousal_LDA)

    y_pred_arousal_LDA1 = model_arousal_LDA.predict(X_test1)

    # Calculate accuracy for Arousal classification
    accuracy_arousal_LDA1 = accuracy_score(y_test_arousal1, y_pred_arousal_LDA1)
    f1_a1 = f1_score(y_test_arousal1, y_pred_arousal_LDA1)
    with open("<path to save the model>",'wb+') as file:
        pickle.dump(model_arousal_LDA, file)

    return accuracy_arousal_LDA*100, f1_a, accuracy_arousal_LDA1*100, f1_a1

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
    test_data1=X_test1
    train_data=X_train
        
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
        # print(pi)
    train_arousal = train_data_balanced[train_arousal.name]
        # print(train_arousal.max(),train_arousal.min())
    accuracy_arousal_LDA, f1_a, accuracy_arousal_LDA1, f1_a1 =train_LDA_arousal(train_x,train_arousal,test_x,test_arousal,test_x1,test_arousal1,gr)

    # print("------ Average accuracy of LDA on arousal ----------")
    
        
    return round(accuracy_arousal_LDA,3), round(f1_a,3), round(accuracy_arousal_LDA1,3), round(f1_a1,3)


def train_LDA_valence(X_train, y_train_valence, X_test , y_test_valence,X_test1,y_test_valence1, gr = None):

    # Create and fit the LDA model for Valence with k=9
    model_valence_LDA = LinearDiscriminantAnalysis(n_components=1)
    model_valence_LDA.fit(X_train, y_train_valence)

    # Predict on the test set for Valence
    y_pred_valence_LDA = model_valence_LDA.predict(X_test)
    accuracy_valence_LDA = accuracy_score(y_test_valence, y_pred_valence_LDA)
    f1_v = f1_score(y_test_valence, y_pred_valence_LDA)

    y_pred_valence_LDA1 = model_valence_LDA.predict(X_test1)
    accuracy_valence_LDA1 = accuracy_score(y_test_valence1, y_pred_valence_LDA1)
    f1_v1 = f1_score(y_test_valence1, y_pred_valence_LDA1)

    with open("<path to save the model>",'wb+') as file:
        pickle.dump(model_valence_LDA, file)


    return accuracy_valence_LDA*100, f1_v, accuracy_valence_LDA1*100, f1_v1

def Valence(PathFile, test_path_list):

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
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_CMSEn', 'HRV_RCMSEn',
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
    test_data1=X_test1
    train_data=X_train    

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
        

    accuracy_valence_LDA,f1_v, accuracy_valence_LDA1, f1_v1 =train_LDA_valence(train_x,train_valence,test_x,test_valence,test_x1,test_valence1,gr)


        
    return round(accuracy_valence_LDA,3),  round(f1_v,3), round(accuracy_valence_LDA1,3), round(f1_v1,3)

