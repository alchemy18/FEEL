# Group_running.py
import pandas as pd
import pickle
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import neurokit2 as nk
import scipy.stats as stats
import warnings
import sys
import random
import torch

import LDA_group
import RandomForest_group
import MLP_group

warnings.filterwarnings('ignore')

# -----------------------------
# Set random seeds for reproducibility
# -----------------------------
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# -----------------------------
# Base CSV Folder Path
# -----------------------------
csv_folder = "/mnt/drive/home/pragyas/Pragya/EEVR_Extension/CoCoOp/dataset"

# -----------------------------
# Function to create concatenated CSVs for groups
# -----------------------------
def make_csvs():
    # Define file lists (replace <path> and <dataset_name> accordingly)
    eda_files_group1 = []  # list of tuples (path_to_csv, dataset_name)
    eda_files_group2 = []
    eda_files_group3 = []

    ppg_files_group1 = []
    ppg_files_group2 = []
    ppg_files_group3 = []

    combined_files_group1 = []
    combined_files_group2 = []
    combined_files_group3 = []

    np.set_printoptions(threshold=np.inf)

    # Concatenate all group files
    eda_group1 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in eda_files_group1], ignore_index=True)
    eda_group2 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in eda_files_group2], ignore_index=True)
    eda_group3 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in eda_files_group3], ignore_index=True)

    ppg_group1 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in ppg_files_group1], ignore_index=True)
    ppg_group2 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in ppg_files_group2], ignore_index=True)
    ppg_group3 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in ppg_files_group3], ignore_index=True)

    combined_group1 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in combined_files_group1], ignore_index=True)
    combined_group2 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in combined_files_group2], ignore_index=True)
    combined_group3 = pd.concat([pd.read_csv(f).assign(dataset=x) for f, x in combined_files_group3], ignore_index=True)

    # Save concatenated files
    eda_group1.to_csv("<path_to_save_EDA_group1>", index=False)
    eda_group2.to_csv("<path_to_save_EDA_group2>", index=False)
    eda_group3.to_csv("<path_to_save_EDA_group3>", index=False)

    ppg_group1.to_csv("<path_to_save_PPG_group1>", index=False)
    ppg_group2.to_csv("<path_to_save_PPG_group2>", index=False)
    ppg_group3.to_csv("<path_to_save_PPG_group3>", index=False)

    combined_group1.to_csv("<path_to_save_COMBINED_group1>", index=False)
    combined_group2.to_csv("<path_to_save_COMBINED_group2>", index=False)
    combined_group3.to_csv("<path_to_save_COMBINED_group3>", index=False)

    print("✅ CSV files created successfully.")


# -----------------------------
# Define paths to concatenated group CSVs
# -----------------------------
eda_group1_path = "<path_to_EDA_group1>"
eda_group2_path = "<path_to_EDA_group2>"
eda_group3_path = "<path_to_EDA_group3>"

ppg_group1_path = "<path_to_PPG_group1>"
ppg_group2_path = "<path_to_PPG_group2>"
ppg_group3_path = "<path_to_PPG_group3>"

combined_group1_path = "<path_to_COMBINED_group1>"
combined_group2_path = "<path_to_COMBINED_group2>"
combined_group3_path = "<path_to_COMBINED_group3>"


# -----------------------------
# Run LDA on grouped datasets
# -----------------------------
def run_LDA():
    print("\n========== Running LDA_group ==========\n")

    print(">> Group 1 (Lab) → Group 2 & 3 (Cross-domain)")
    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile=eda_group1_path, test_path_list=[eda_group2_path, eda_group3_path])
    print(f"LDA_group (EDA Arousal) G2: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile=eda_group1_path, test_path_list=[eda_group2_path, eda_group3_path])
    print(f"LDA_group (EDA Valence) G2: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    print(">> Group 2 → Group 1 & 3")
    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile=eda_group2_path, test_path_list=[eda_group1_path, eda_group3_path])
    print(f"LDA_group (EDA Arousal) G1: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile=eda_group2_path, test_path_list=[eda_group1_path, eda_group3_path])
    print(f"LDA_group (EDA Valence) G1: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    print(">> Group 3 → Group 1 & 2")
    ar, f1, ar_1, f1_1 = LDA_group.Arousal(PathFile=eda_group3_path, test_path_list=[eda_group1_path, eda_group2_path])
    print(f"LDA_group (EDA Arousal) G1: {ar}% {f1} | G2: {ar_1}% {f1_1}\n")

    ar, f1, ar_1, f1_1 = LDA_group.Valence(PathFile=eda_group3_path, test_path_list=[eda_group1_path, eda_group2_path])
    print(f"LDA_group (EDA Valence) G1: {ar}% {f1} | G2: {ar_1}% {f1_1}\n")

    print("✅ LDA_group completed successfully.\n")


# -----------------------------
# Run Random Forest on grouped datasets
# -----------------------------
def run_RF():
    print("\n========== Running RandomForest_group ==========\n")
    ar, f1, ar_1, f1_1 = RandomForest_group.Arousal(PathFile=combined_group1_path, test_path_list=[combined_group2_path, combined_group3_path])
    print(f"RandomForest_group (Combined Arousal) G2: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    ar, f1, ar_1, f1_1 = RandomForest_group.Valence(PathFile=combined_group1_path, test_path_list=[combined_group2_path, combined_group3_path])
    print(f"RandomForest_group (Combined Valence) G2: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    print("✅ RandomForest_group completed successfully.\n")


# -----------------------------
# Run MLP on grouped datasets
# -----------------------------
def run_MLP():
    print("\n========== Running MLP_group ==========\n")
    ar, f1, ar_1, f1_1 = MLP_group.Arousal(PathFile=ppg_group1_path, test_path_list=[ppg_group2_path, ppg_group3_path])
    print(f"MLP_group (PPG Arousal) G2: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    ar, f1, ar_1, f1_1 = MLP_group.Valence(PathFile=ppg_group1_path, test_path_list=[ppg_group2_path, ppg_group3_path])
    print(f"MLP_group (PPG Valence) G2: {ar}% {f1} | G3: {ar_1}% {f1_1}\n")

    print("✅ MLP_group completed successfully.\n")


import argparse
import pandas as pd

def main(eda_fet_path, ppg_fet_path, com_fet_path, out_path_benchmark):
    # Load feature files
    eda_features = pd.read_csv(eda_fet_path)
    ppg_features = pd.read_csv(ppg_fet_path)
    com_features = pd.read_csv(com_fet_path)

    # --- Add your benchmarking / analysis code here ---
    # For example, combining features or running models
    combined = pd.concat([eda_features, ppg_features, com_features], axis=1)

    # Save benchmark results
    combined.to_csv(out_path_benchmark, index=False)
    print(f"Benchmarking results saved to {out_path_benchmark}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking on physiological features.")
    parser.add_argument("--eda_fet_path", type=str, required=True, help="Path to EDA features CSV")
    parser.add_argument("--ppg_fet_path", type=str, required=True, help="Path to PPG features CSV")
    parser.add_argument("--com_fet_path", type=str, required=True, help="Path to combined features CSV")
    parser.add_argument("--out_path_benchmark", type=str, required=True, help="Path to save benchmark results CSV")

    args = parser.parse_args()
    main(args.eda_fet_path, args.ppg_fet_path, args.com_fet_path, args.out_path_benchmark)
