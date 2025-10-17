import pandas as pd
import pickle
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import neurokit2 as nk
import scipy.stats as stats
# import cvxEDA.src.cvxEDA as cvxEDA
from sklearn.preprocessing import MinMaxScaler
import warnings
import sys
import finetuning_CLSP_group_CNN_representative
import finetuning_CLSP_group_representative
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




def call_mlp(perc,gr,label_n,modality, tests):
    feature_EDA = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    feature_PPG = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    feature_Combined = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR','BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    # Train CoCoOp with pretrained CLSP
    modalities = {
        "EDA": {
            "feature_cols": feature_EDA,
            "csv": "<path to the csv for feature eda for group (gr) as according to the function call>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda and arousal category>",
                "valence_category": "<path for the clsp model for eda and valence category>"
            },
            'test_list': tests        },
        "PPG": {
            "feature_cols": feature_PPG,
            "csv": "<path to the csv for feature ppg for group (gr) as according to the function call>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for ppg and arousal category>",
                "valence_category": "<path for the clsp model for ppg and valence category>"
            },
            'test_list': tests
        },
        "Combined": {
            "feature_cols": feature_Combined,
            "csv": "<path to the csv for feature eda+ppg for group (gr) as according to the function call>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda+ppg and arousal category>",
                "valence_category": "<path for the clsp model for eda+ppg and valence category>"
            },
            'test_list': tests
        }
    }

    percentages = perc
    label = "arousal_category" if label_n == "arousal" else "valence_category"
    

    modality_data =  modalities[modality]
    test_acc_1, test_f1_1, test_acc_2, test_f1_2 = finetuning_CLSP_group_representative.train_cocoop(
        dataset_name=gr,
        csv_path=modality_data["csv"],
        test_list = modality_data["test_list"],
        ckpt_path={modality_data['ckpt'][label]},
        feature_cols=modality_data["feature_cols"],
        label_col=label,
        percentage=percentages,
        batch_size=4,           # Custom batch size
        epochs=15,             # Custom number of epochs
        lr=54-5,                # Custom learning rate
        n_ctx=16,                # Custom number of context tokens
        meta_hidden_dim= 32,     # Custom meta-net hidden dimension
        device='cuda'
    )
                                    
    return test_acc_1, test_f1_1, test_acc_2, test_f1_2


def call_cnn(perc,gr,label_n,modality, tests):
    feature_EDA = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    feature_PPG = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    feature_Combined = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR','BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    # Train CoCoOp with pretrained CLSP
    modalities = {
        "EDA": {
            "feature_cols": feature_EDA,
            "csv": "<path to the csv for feature eda for group (gr) as according to the function call, will be done automatically>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda and arousal category>",
                "valence_category": "<path for the clsp model for eda and valence category>"
            },
            'test_list': tests        },
        "PPG": {
            "feature_cols": feature_PPG,
            "csv": "<path to the csv for feature ppg for group (gr) as according to the function call, will be done automatically>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for ppg and arousal category>",
                "valence_category": "<path for the clsp model for ppg and valence category>"
            },
            'test_list': tests
        },
        "Combined": {
            "feature_cols": feature_Combined,
            "csv": "<path to the csv for feature eda+ppg for group (gr) as according to the function call, will be done automatically>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda+ppg and arousal category>",
                "valence_category": "<path for the clsp model for eda+ppg and valence category>"
            },
            'test_list': tests
        }
    }

    percentages = perc
    label = "arousal_category" if label_n == "arousal" else "valence_category"
    

    modality_data =  modalities[modality]
    test_acc_1, test_f1_1, test_acc_2, test_f1_2 = finetuning_CLSP_group_CNN_representative.train_cocoop(
        dataset_name= gr ,
        csv_path=modality_data["csv"],
        test_list = modality_data["test_list"],
        
        ckpt_path={modality_data['ckpt'][label]},
        feature_cols=modality_data["feature_cols"],
        label_col=label,
        percentage=percentages,
        batch_size=4,           # Custom batch size
        epochs=15,             # Custom number of epochs
        lr=5e-5,                # Custom learning rate
        n_ctx=24,                # Custom number of context tokens
        meta_hidden_channel= 24,     # Custom meta-net hidden channels
        device='cuda'
    )
                                    
    return test_acc_1, test_f1_1, test_acc_2, test_f1_2

Groups_eda = [] # list of paths to groups for eda features
Groups_ppg = [] # list of paths to groups for ppg features
Groups_combined = [] # list of paths to groups for eda+ppg features
perc = [5,25,50]
labels_n = ["arousal", "valence"]
modalitys = ['EDA', 'PPG', 'Combined']
for modality in modalitys:
    if modality=='EDA':
        Groups = Groups_ppg
    elif modality=='PPG':
        Groups = Groups_eda
    elif modality=='Combined':
        Groups = Groups_combined
    for i in Groups:
        for per in perc:
            for label_n in labels_n:
                tests =  Groups.copy()
                tests.remove(i)
                print(f"Running CLSP-MLP {per}% with Training as {i} on {modality} for {label_n}")
                ar, f1, ar_1, f1_1 = call_mlp(perc = per,gr = i,label_n = label_n ,modality = modality, tests = tests)
                print(f"The result on {tests[0]} Category :: Accuracy - {round(ar*100,3)}% F1 - {round(f1,3)}")
                print(f"The result on {tests[1]} Category :: Accuracy - {round(ar_1*100,3)}% F1 - {round(f1_1,3)}")
    print()
    print("-"*30)
    print()
for modality in modalitys:
    if modality=='EDA':
        Groups = Groups_ppg
    elif modality=='PPG':
        Groups = Groups_eda
    elif modality=='Combined':
        Groups = Groups_combined
    for i in Groups:
        for per in perc:
            for label_n in labels_n:
                tests =  Groups.copy()
                tests.remove(i)
                print(f"Running CLSP-CNN {per}% with Training as {i} on {modality} for {label_n}")
                ar, f1, ar_1, f1_1 = call_cnn(perc = per,gr = i,label_n = label_n ,modality = modality, tests = tests)
                print(f"The result on {tests[0]} Category :: Accuracy - {round(ar*100,3)}% F1 - {round(f1,3)}")
                print(f"The result on {tests[1]} Category :: Accuracy - {round(ar_1*100,3)}% F1 - {round(f1_1,3)}")
    print()
    print("-"*30)
    print()


    