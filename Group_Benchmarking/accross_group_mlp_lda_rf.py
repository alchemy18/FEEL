import pandas as pd
import pickle
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import neurokit2 as nk
import scipy.stats as stats
import warnings
import sys
import LDA_group
import RandomForest_group
import MLP_group
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



csv_folder = "/mnt/drive/home/pragyas/Pragya/EEVR_Extension/CoCoOp/dataset"

def make_csvs():
    eda_files_group1 = [] # list of tuple of (paths to eda feature files for datasets in group1 ,dataset name)
    eda_files_group2 = [] # list of tuple of (paths to eda feature files for datasets in group2 ,dataset name)
    eda_files_group3 = [] # list of tuple of (paths to eda feature files for datasets in group3 ,dataset name)

    ppg_files_group1 = [] # list of tuple of (paths to ppg feature files for datasets in group1 ,dataset name)
    ppg_files_group2 = [] # list of tuple of (paths to ppg feature files for datasets in group2 ,dataset name)
    ppg_files_group3 = [] # list of tuple of (paths to ppg feature files for datasets in group3 ,dataset name)

    combined_files_group1 = [] # list of tuple of (paths to eda+ppg feature files for datasets in group1 ,dataset name)
    combined_files_group2 = [] # list of tuple of (paths to eda+ppg feature files for datasets in group2 ,dataset name)
    combined_files_group3 = [] # list of tuple of (paths to eda+ppg feature files for datasets in group3 ,dataset name)

    np.set_printoptions(threshold=np.inf)

    eda_group1 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in eda_files_group1], ignore_index=True)
    eda_group2 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in eda_files_group2], ignore_index=True)
    eda_group3 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in eda_files_group3], ignore_index=True)

    ppg_group1 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in ppg_files_group1], ignore_index=True)
    ppg_group2 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in ppg_files_group2], ignore_index=True)
    ppg_group3 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in ppg_files_group3], ignore_index=True)

    combined_group1 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in combined_files_group1], ignore_index=True)
    combined_group2 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in combined_files_group2], ignore_index=True)
    combined_group3 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in combined_files_group3], ignore_index=True)


    eda_group1.to_csv("<path to save the concatinated file of eda from group1>", index=False)
    eda_group2.to_csv("<path to save the concatinated file of eda from group2>", index=False)
    eda_group3.to_csv("<path to save the concatinated file of eda from group3>", index=False)
    ppg_group1.to_csv("<path to save the concatinated file of ppg from group1>", index=False)
    ppg_group2.to_csv("<path to save the concatinated file of ppg from group2>", index=False)
    ppg_group3.to_csv("<path to save the concatinated file of ppg from group3>", index=False)
    combined_group1.to_csv("<path to save the concatinated file of eda+ppg from group1>", index=False)
    combined_group2.to_csv("<path to save the concatinated file of eda+ppg from group2>", index=False)
    combined_group3.to_csv("<path to save the concatinated file of eda+ppg from group3>", index=False)
    print("CSV files created successfully.")

    return 

eda_group1_path = "<path to the concatinated file of eda from group1>"
eda_group2_path = "<path to the concatinated file of eda from group2>"
eda_group3_path = "<path to the concatinated file of eda from group3>"
ppg_group1_path = "<path to the concatinated file of ppg from group1>"
ppg_group2_path = "<path to the concatinated file of ppg from group2>"
ppg_group3_path = "<path to the concatinated file of ppg from group3>"
combined_group1_path = "<path to save the concatinated file of eda+ppg from group1>"
combined_group2_path = "<path to save the concatinated file of eda+ppg from group2>"
combined_group3_path = "<path to save the concatinated file of eda+ppg from group3>"


def run_LDA():
    print("Running LDA_group\n")

    print("Running on lab Category, EDA")
    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = eda_group1_path , test_path_list = [eda_group2_path, eda_group3_path])
    print(f"The result of LDA_group (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = eda_group1_path , test_path_list = [eda_group2_path, eda_group3_path])
    print(f"The result of LDA_group on EDA (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on EDA (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group2 Category, EDA")

    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = eda_group2_path , test_path_list = [eda_group1_path, eda_group3_path])
    print(f"The result of LDA_group on EDA (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on EDA  (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = eda_group2_path , test_path_list = [eda_group1_path, eda_group3_path])
    print(f"The result of LDA_group on EDA (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on EDA (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, EDA")

    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = eda_group3_path , test_path_list = [eda_group2_path, eda_group1_path])
    print(f"The result of LDA_group on EDA (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on EDA (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = eda_group3_path , test_path_list = [eda_group2_path, eda_group1_path])
    print(f"The result of LDA_group on EDA (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on EDA (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on lab Category, PPG")
    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = ppg_group1_path , test_path_list = [ppg_group2_path, ppg_group3_path])
    print(f"The result of LDA_group on PPG (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on PPG (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = ppg_group1_path , test_path_list = [ppg_group2_path, ppg_group3_path])
    print(f"The result of LDA_group on PPG (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on PPG (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group2 Category, PPG")

    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = ppg_group2_path , test_path_list = [ppg_group1_path, ppg_group3_path])
    print(f"The result of LDA_group on PPG (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on PPG (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = ppg_group2_path , test_path_list = [ppg_group1_path, ppg_group3_path])
    print(f"The result of LDA_group on PPG (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on PPG (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, PPG")

    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = ppg_group3_path , test_path_list = [ppg_group2_path, ppg_group1_path])
    print(f"The result of LDA_group on PPG (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on PPG (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = ppg_group3_path , test_path_list = [ppg_group2_path, ppg_group1_path])
    print(f"The result of LDA_group on PPG (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on PPG (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on lab Category, Combined")

    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = combined_group1_path , test_path_list = [combined_group2_path, combined_group3_path])
    print(f"The result of LDA_group on Combined (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on Combined (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = combined_group1_path , test_path_list = [combined_group2_path, combined_group3_path])
    print(f"The result of LDA_group on Combined (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on Combined (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()
 
    print("Running on group2 Category, Combined")

    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = combined_group2_path , test_path_list = [combined_group1_path, combined_group3_path])
    print(f"The result of LDA_group on Combined (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on Combined (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = combined_group2_path , test_path_list = [combined_group1_path, combined_group3_path])
    print(f"The result of LDA_group on Combined of group2 Category (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on Combined of group2 Category (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, Combined")

    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile = combined_group3_path , test_path_list = [combined_group2_path, combined_group1_path])
    print(f"The result of LDA_group on Combined (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on Combined (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()
    
    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile = combined_group3_path , test_path_list = [combined_group2_path, combined_group1_path])
    print(f"The result of LDA_group on Combined (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of LDA_group on Combined (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print("-"*50)
    print()


def run_RF():
    print("Running RandomForest_group\n")

    print("Running on lab Category, EDA")
    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = eda_group1_path , test_path_list = [eda_group2_path, eda_group3_path])
    print(f"The result of RandomForest_group (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = eda_group1_path , test_path_list = [eda_group2_path, eda_group3_path])
    print(f"The result of RandomForest_group on EDA (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on EDA (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group2 Category, EDA")

    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = eda_group2_path , test_path_list = [eda_group1_path, eda_group3_path])
    print(f"The result of RandomForest_group on EDA (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on EDA  (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = eda_group2_path , test_path_list = [eda_group1_path, eda_group3_path])
    print(f"The result of RandomForest_group on EDA (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on EDA (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, EDA")

    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = eda_group3_path , test_path_list = [eda_group2_path, eda_group1_path])
    print(f"The result of RandomForest_group on EDA (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on EDA (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = eda_group3_path , test_path_list = [eda_group2_path, eda_group1_path])
    print(f"The result of RandomForest_group on EDA (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on EDA (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on lab Category, PPG")
    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = ppg_group1_path , test_path_list = [ppg_group2_path, ppg_group3_path])
    print(f"The result of RandomForest_group on PPG (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on PPG (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = ppg_group1_path , test_path_list = [ppg_group2_path, ppg_group3_path])
    print(f"The result of RandomForest_group on PPG (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on PPG (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group2 Category, PPG")

    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = ppg_group2_path , test_path_list = [ppg_group1_path, ppg_group3_path])
    print(f"The result of RandomForest_group on PPG (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on PPG (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = ppg_group2_path , test_path_list = [ppg_group1_path, ppg_group3_path])
    print(f"The result of RandomForest_group on PPG (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on PPG (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, PPG")

    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = ppg_group3_path , test_path_list = [ppg_group2_path, ppg_group1_path])
    print(f"The result of RandomForest_group on PPG (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on PPG (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = ppg_group3_path , test_path_list = [ppg_group2_path, ppg_group1_path])
    print(f"The result of RandomForest_group on PPG (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on PPG (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on lab Category, Combined")

    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = combined_group1_path , test_path_list = [combined_group2_path, combined_group3_path])
    print(f"The result of RandomForest_group on Combined (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on Combined (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = combined_group1_path , test_path_list = [combined_group2_path, combined_group3_path])
    print(f"The result of RandomForest_group on Combined (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on Combined (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()
 
    print("Running on group2 Category, Combined")

    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = combined_group2_path , test_path_list = [combined_group1_path, combined_group3_path])
    print(f"The result of RandomForest_group on Combined (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on Combined (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = combined_group2_path , test_path_list = [combined_group1_path, combined_group3_path])
    print(f"The result of RandomForest_group on Combined of group2 Category (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on Combined of group2 Category (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, Combined")

    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile = combined_group3_path , test_path_list = [combined_group2_path, combined_group1_path])
    print(f"The result of RandomForest_group on Combined (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on Combined (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()
    
    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile = combined_group3_path , test_path_list = [combined_group2_path, combined_group1_path])
    print(f"The result of RandomForest_group on Combined (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of RandomForest_group on Combined (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print("-"*50)
    print()

def run_MLP():
    print("Running MLP_group\n")

    print("Running on lab Category, EDA")
    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = eda_group1_path , test_path_list = [eda_group2_path, eda_group3_path])
    print(f"The result of MLP_group (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = eda_group1_path , test_path_list = [eda_group2_path, eda_group3_path])
    print(f"The result of MLP_group on EDA (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on EDA (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group2 Category, EDA")

    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = eda_group2_path , test_path_list = [eda_group1_path, eda_group3_path])
    print(f"The result of MLP_group on EDA (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on EDA  (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = eda_group2_path , test_path_list = [eda_group1_path, eda_group3_path])
    print(f"The result of MLP_group on EDA (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on EDA (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, EDA")

    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = eda_group3_path , test_path_list = [eda_group2_path, eda_group1_path])
    print(f"The result of MLP_group on EDA (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on EDA (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = eda_group3_path , test_path_list = [eda_group2_path, eda_group1_path])
    print(f"The result of MLP_group on EDA (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on EDA (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on lab Category, PPG")
    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = ppg_group1_path , test_path_list = [ppg_group2_path, ppg_group3_path])
    print(f"The result of MLP_group on PPG (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on PPG (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = ppg_group1_path , test_path_list = [ppg_group2_path, ppg_group3_path])
    print(f"The result of MLP_group on PPG (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on PPG (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group2 Category, PPG")

    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = ppg_group2_path , test_path_list = [ppg_group1_path, ppg_group3_path])
    print(f"The result of MLP_group on PPG (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on PPG (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = ppg_group2_path , test_path_list = [ppg_group1_path, ppg_group3_path])
    print(f"The result of MLP_group on PPG (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on PPG (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, PPG")

    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = ppg_group3_path , test_path_list = [ppg_group2_path, ppg_group1_path])
    print(f"The result of MLP_group on PPG (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on PPG (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = ppg_group3_path , test_path_list = [ppg_group2_path, ppg_group1_path])
    print(f"The result of MLP_group on PPG (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on PPG (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on lab Category, Combined")

    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = combined_group1_path , test_path_list = [combined_group2_path, combined_group3_path])
    print(f"The result of MLP_group on Combined (Arousal) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on Combined (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = combined_group1_path , test_path_list = [combined_group2_path, combined_group3_path])
    print(f"The result of MLP_group on Combined (Valence) on group2 Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on Combined (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()
 
    print("Running on group2 Category, Combined")

    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = combined_group2_path , test_path_list = [combined_group1_path, combined_group3_path])
    print(f"The result of MLP_group on Combined (Arousal) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on Combined (Arousal) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = combined_group2_path , test_path_list = [combined_group1_path, combined_group3_path])
    print(f"The result of MLP_group on Combined of group2 Category (Valence) on lab Category :: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on Combined of group2 Category (Valence) on group3 Category :: Accuracy - {ar_1}% F1 - {f1_1}")
    print()

    print("Running on group3 Category, Combined")

    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile = combined_group3_path , test_path_list = [combined_group2_path, combined_group1_path])
    print(f"The result of MLP_group on Combined (Arousal) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on Combined (Arousal) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print()
    
    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile = combined_group3_path , test_path_list = [combined_group2_path, combined_group1_path])
    print(f"The result of MLP_group on Combined (Valence) on group2 Category:: Accuracy - {ar}% F1 - {f1}")
    print(f"The result of MLP_group on Combined (Valence) on lab Category:: Accuracy - {ar_1}% F1 - {f1_1}")
    print("-"*50)
    print()


make_csvs()
run_LDA()
run_RF()
run_MLP()
