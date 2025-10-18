from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time
import torch
import pandas as pd
import numpy as np
from thop import profile
from lstm_features import ClassifierLstmMlp
from resnet_features import ClassifierResNet
from transformer_features import TransformerClassifier

# ============================================================
# 1. Load Dataset
# ============================================================
def load_dataset(data_path, signal_type='EDA'):
    """
    Load dataset and extract features based on signal type.

    Args:
        data_path (str): Path to the CSV file containing features.
        signal_type (str): Type of signal ('EDA', 'PPG', or 'Combined').

    Returns:
        sample_input (torch.Tensor): Sample input tensor from dataset.
        num_features (int): Number of features in the input.
    """
    # Define relevant features based on signal type
    if signal_type == 'EDA':
        relevant_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy',
                            'insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR',
                            'meanRespSCR','sumAmpSCR','sumRespSCR']
    elif signal_type == 'PPG':
        relevant_features = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
                            'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 
                            'HRV_VHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 
                            'HRV_CVI', 'HRV_PSS', 'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 
                            'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
                            'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                            'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
                            'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
                            'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    else:  # Combined
        relevant_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy',
                            'insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR',
                            'meanRespSCR','sumAmpSCR','sumRespSCR','BPM', 'PPG_Rate_Mean', 
                            'HRV_MedianNN', 'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 
                            'HRV_LF', 'HRV_VHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 
                            'HRV_CVI', 'HRV_PSS', 'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 
                            'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
                            'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                            'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
                            'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
                            'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    
    # Load dataset
    df = pd.read_csv(data_path)
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Extract features
    X = df[relevant_features].values
    num_features = len(relevant_features)
    Y = df["valence_category"].values
    
    # Convert to tensor with shape (batch_size=1, channels=1, num_features)
    sample_input = torch.tensor(X[0:1], dtype=torch.float32).unsqueeze(1)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Signal type: {signal_type}")
    print(f"Number of features: {num_features}")
    print(f"Input shape: {sample_input.shape}\n")
    
    return sample_input, num_features
# ============================================================
# sklearn MODEL UTILITIES
# ============================================================

def measure_latency_sklearn(model, X_sample, n_runs=500):
    """Measure average inference latency (ms) for sklearn model."""
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(X_sample)
    end = time.perf_counter()
    avg_latency = (end - start) * 1000 / n_runs
    return avg_latency


def count_parameters_sklearn(model):
    """Estimate number of parameters for sklearn model."""
    if hasattr(model, 'coef_'):  # Linear models, LDA, LogisticRegression, etc.
        params = np.prod(model.coef_.shape)
        if hasattr(model, 'intercept_'):
            params += model.intercept_.size
        return params
    elif hasattr(model, 'coefs_'):  # MLP
        return sum(np.prod(w.shape) for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    elif hasattr(model, 'estimators_'):  # Random Forest
        # Count all tree nodes as approximate parameters
        return sum(est.tree_.node_count for est in model.estimators_)
    else:
        return 0


def estimate_flops_sklearn(model, X_sample):
    """Rough FLOP estimation for sklearn models."""
    n_features = X_sample.shape[1]
    
    if isinstance(model, LinearDiscriminantAnalysis):
        # dot product per sample per class: 2 * n_features * n_classes
        n_classes = len(model.classes_)
        flops = 2 * n_features * n_classes -1 
    elif isinstance(model, MLPClassifier):
        # sum of 2 * (input * hidden + hidden * output)
        layer_shapes = [(a.shape[0], a.shape[1]) for a in model.coefs_]
        flops = sum(2 * m * n for m, n in layer_shapes)
    elif isinstance(model, RandomForestClassifier):
        # Each sample passes through ~depth comparisons per tree
        depths = [est.get_depth() for est in model.estimators_]
        flops = np.mean(depths) * len(model.estimators_)
    else:
        flops = np.nan
    return flops / 1e6  # MFLOPs


def benchmark_sklearn_model(model_name, model, X_sample, y_sample):
    """Benchmark sklearn model similar to PyTorch models."""
    print(f"\n{'='*60}")
    print(f"Benchmarking sklearn model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Fit model
        model.fit(X_sample, y_sample)
        
        # Measure latency
        avg_latency = measure_latency_sklearn(model, X_sample[:1])
        
        # Parameters & FLOPs
        params = count_parameters_sklearn(model)
        flops_m = estimate_flops_sklearn(model, X_sample[:1])
        
        print(f"[Latency] {avg_latency:.4f} ms")
        print(f"[Model Size] FLOPs: {flops_m:.2f} MFLOPs | Params: {params/1e6:.2f}M")
        
        results = {
            'model_name': model_name,
            'latency_ms': avg_latency,
            'flops_mflops': flops_m,
            'params': params / 1e6,
            'status': 'success'
        }
    except Exception as e:
        print(f"[ERROR] Failed to benchmark {model_name}: {e}")
        results = {
            'model_name': model_name,
            'latency_ms': None,
            'flops_mflops': None,
            'params': None,
            'status': f'failed: {e}'
        }
    
    return results


# ============================================================
# 8. Run sklearn model benchmarks
# ============================================================
if __name__ == "__main__":
    # Load dataset (same as before)
    DATA_PATH = "<path to eda/ppg/combined csv>"
    SIGNAL_TYPE = 'Combined'  # Options: 'EDA', 'PPG', 'Combined'
    NUM_CLASSES = 2
    df = pd.read_csv(DATA_PATH).fillna(0).replace([np.inf, -np.inf], 0)

    if SIGNAL_TYPE == 'EDA':
        relevant_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy',
                            'insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR',
                            'meanRespSCR','sumAmpSCR','sumRespSCR']
    elif SIGNAL_TYPE == 'PPG':
        relevant_features = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
                            'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 
                            'HRV_VHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 
                            'HRV_CVI', 'HRV_PSS', 'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 
                            'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
                            'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                            'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
                            'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
                            'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    else:  # Combined
        relevant_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy',
                            'insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR',
                            'meanRespSCR','sumAmpSCR','sumRespSCR','BPM', 'PPG_Rate_Mean', 
                            'HRV_MedianNN', 'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 
                            'HRV_LF', 'HRV_VHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 
                            'HRV_CVI', 'HRV_PSS', 'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 
                            'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
                            'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                            'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
                            'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
                            'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    
    
    # Prepare training samples for sklearn
    X = df[relevant_features]
    y = df["valence_category"]
    
    sklearn_models = {
        'LDA': LinearDiscriminantAnalysis(n_components=1),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42 ,n_jobs=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
    }

    sklearn_results = []
    for name, model in sklearn_models.items():
        result = benchmark_sklearn_model(name, model, X, y)
        sklearn_results.append(result)
    
    sklearn_df = pd.DataFrame(sklearn_results)
    
    print("\n" + "="*80)
    print("SKLEARN BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(sklearn_df.to_string(index=False))
    print("="*80)
