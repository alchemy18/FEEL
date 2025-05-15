import pandas as pd
import pickle
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import neurokit2 as nk
import scipy.stats as stats
import cvxEDA.src.cvxEDA as cvxEDA
from sklearn.preprocessing import MinMaxScaler
import warnings
import sys
current_dir = os.getcwd()
scripts_dir = os.path.join(current_dir, "..", "Scripts")
sys.path.append(scripts_dir)
import LDA
# RF - Random Forest Model 
# (n_estimators=100, random_state=random_seed,n_jobs=5, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
import RandomForest
# HC + NN (MLP)
# MLPClassifier of Scikit learn 
# (hidden_layer_sizes=100, random_state=42, activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
import MLP
# HC+ ResNet
# ResNetBlock(in_channels=1, out_channels=n_feature_maps * 2, kernel_sizes=[8,5,3])
import resnet_features
# HC + LSTM + NN
# LSTM Module::
# - Hidden Size: 128
# - Number of Layers: 2
# - Dropout: 0.3 (applied between LSTM layers)
# MLP Module::
# - Linear Transformation: Maps from 128 (LSTM's hidden size) to 256 neurons.
# - Activation: ReLU
# - Dropout: 0.4
import lstm_features
# HC+ Attention + NN
# Self-Attention::
# Incorporates a multi-head self-attention layer (4 heads) to capture inter-feature dependencies.
# MLP Layers::
# Consists of two linear layers with ReLU activations and dropout for regularization:
# - 128 → 256 (with ReLU and 0.4 dropout)
# - 256 → 128 (with ReLU and 0.3 dropout)
import transformer_features
# Signal+ ResNet
# ResNetBlock(in_channels=1, out_channels=n_feature_maps * 2, kernel_sizes=[8,5,3])
import resnet_signals
# Signal + LSTM + NN
# LSTM Module::
# - Hidden Size: 128
# - Number of Layers: 2
# - Dropout: 0.3 (applied between LSTM layers)
# MLP Module::
# - Linear Transformation: Maps from 128 (LSTM's hidden size) to 256 neurons.
# - Activation: ReLU
# - Dropout: 0.4
import lstm_signals
import transformer_signals
import clsp_com
import clsp_ppg
import clsp_eda
import argparse

warnings.filterwarnings('ignore')

# Set up argument parser
parser = argparse.ArgumentParser(description="Benchmarking the Scientisst dataset models")
parser.add_argument('--eda_fet_path', type=str, required=True, help='Path to the EDA feature file')
parser.add_argument('--ppg_fet_path', type=str, required=True, help='Path to the PPG feature file')
parser.add_argument('--com_fet_path', type=str, required=True, help='Path to the Combined feature file')
parser.add_argument('--eda_raw_path', type=str, required=True, help='Path to the EDA raw file')
parser.add_argument('--ppg_raw_path', type=str, required=True, help='Path to the PPG raw file')
parser.add_argument('--out_path_benchmark', type=str, required=True, help='Path to save the benchmarking results')

# Parse arguments
args = parser.parse_args()

# Assign input paths from arguments
eda_fet_path = args.eda_fet_path
ppg_fet_path = args.ppg_fet_path
com_fet_path = args.com_fet_path
eda_raw_path = args.eda_raw_path
ppg_raw_path = args.ppg_raw_path
out_path_benchmark = args.out_path_benchmark


print("Running LDA")
ar, f1 = LDA.Arousal(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of LDA on Combined (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = LDA.Valence(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of LDA on Combined (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = LDA.Arousal(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of LDA on PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = LDA.Valence(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of LDA on PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = LDA.Arousal(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of LDA on EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = LDA.Valence(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of LDA on EDA (Valence):: Accuracy - {ar}% F1 - {f1}")



print("Running RandomForest")
ar, f1 = RandomForest.Arousal(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of RandomForest on Combined (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = RandomForest.Valence(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of RandomForest on Combined (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = RandomForest.Arousal(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of RandomForest on PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = RandomForest.Valence(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of RandomForest on PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = RandomForest.Arousal(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of RandomForest on EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = RandomForest.Valence(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of RandomForest on EDA (Valence):: Accuracy - {ar}% F1 - {f1}")

# # # ML models has been trained and infered as per LOSO, individual participant wise results are stored in Model_Benchmark.csv in result folder. Don't forget to update Excel Sheet

# # # BenchMarking DL Models on the Features

print("Running MLP")
ar, f1 = MLP.Arousal(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of MLP on Combined (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = MLP.Valence(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of MLP on Combined (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = MLP.Arousal(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of MLP on PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = MLP.Valence(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of MLP on PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = MLP.Arousal(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of MLP on EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = MLP.Valence(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of MLP on EDA (Valence):: Accuracy - {ar}% F1 - {f1}")


print("Running Resnet")
ar, f1 = resnet_features.Arousal(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of Resnet on Combined (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_features.Valence(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of Resnet on Combined (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_features.Arousal(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of Resnet on PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_features.Valence(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of Resnet on PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_features.Arousal(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of Resnet on EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_features.Valence(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of Resnet on EDA (Valence):: Accuracy - {ar}% F1 - {f1}")

print("Running LSTM + NN")
ar, f1 = lstm_features.Arousal(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of LSTM + NN on Combined (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_features.Valence(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of LSTM + NN on Combined (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_features.Arousal(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of LSTM + NN on PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_features.Valence(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of LSTM + NN on PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_features.Arousal(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of LSTM + NN on EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_features.Valence(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of LSTM + NN on EDA (Valence):: Accuracy - {ar}% F1 - {f1}")


print("Running Attention + NN")
ar, f1 = transformer_features.Arousal(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of Attention + NN on Combined (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_features.Valence(PathFile = com_fet_path,output_file=out_path_benchmark)
print(f"The result of Attention + NN on Combined (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_features.Arousal(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of Attention + NN on PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_features.Valence(PathFile = ppg_fet_path,output_file=out_path_benchmark)
print(f"The result of Attention + NN on PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_features.Arousal(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of Attention + NN on EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_features.Valence(PathFile = eda_fet_path,output_file=out_path_benchmark)
print(f"The result of Attention + NN on EDA (Valence):: Accuracy - {ar}% F1 - {f1}")

# # DL Models on the Feature Data has been trained. Participant wise result has been updated in the Model_Benchmark.csv. Complete excel sheets with the average result of LOSO. 

# # BenchMarking DL Models on the Raw Data


print("Running Resnet")

ar, f1 = resnet_signals.Arousal(PathFile = ppg_raw_path,output_file=out_path_benchmark)
print(f"The result of Resnet on Raw PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_signals.Valence(PathFile = ppg_raw_path,output_file=out_path_benchmark)
print(f"The result of Resnet on Raw PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_signals.Arousal(PathFile = eda_raw_path,output_file=out_path_benchmark)
print(f"The result of Resnet on Raw EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = resnet_signals.Valence(PathFile = eda_raw_path,output_file=out_path_benchmark)
print(f"The result of Resnet on Raw EDA (Valence):: Accuracy - {ar}% F1 - {f1}")


print("Running LSTM + NN")

ar, f1 = lstm_signals.Arousal(PathFile = ppg_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of LSTM + NN on Raw PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_signals.Valence(PathFile = ppg_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of LSTM + NN on Raw PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_signals.Arousal(PathFile = eda_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of LSTM + NN on Raw EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = lstm_signals.Valence(PathFile = eda_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of LSTM + NN on Raw EDA (Valence):: Accuracy - {ar}% F1 - {f1}")


print("Running CNN + Transformer")

ar, f1 = transformer_signals.Arousal(PathFile = ppg_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of CNN + Transformer on Raw PPG (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_signals.Valence(PathFile = ppg_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of CNN + Transformer on Raw PPG (Valence):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_signals.Arousal(PathFile = eda_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of CNN + Transformer on Raw EDA (Arousal):: Accuracy - {ar}% F1 - {f1}")

ar, f1 = transformer_signals.Valence(PathFile = eda_raw_path,output_file=out_path_benchmark, window_size = 60, window_overlap = 0.5)
print(f"The result of CNN + Transformer on Raw EDA (Valence):: Accuracy - {ar}% F1 - {f1}")




