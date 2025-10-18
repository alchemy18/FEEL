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
    
    # Convert to tensor with shape (batch_size=1, channels=1, num_features)
    sample_input = torch.tensor(X[0:1], dtype=torch.float32).unsqueeze(1)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Signal type: {signal_type}")
    print(f"Number of features: {num_features}")
    print(f"Input shape: {sample_input.shape}\n")
    
    return sample_input, num_features


# ============================================================
# 2. Measure Inference Latency
# ============================================================
def measure_total_latency(model, input_tensor, warmup_runs=10, timed_runs=500):
    """
    Measure total inference latency of a PyTorch model using CUDA events.
    Works for both GPU and CPU, more accurate on GPU.

    Args:
        model (torch.nn.Module): The model to profile.
        input_tensor (torch.Tensor): Input tensor for the model.
        warmup_runs (int): Number of warm-up iterations (not measured).
        timed_runs (int): Number of timed iterations.

    Returns:
        float: Average latency in milliseconds.
    """
    device = next(model.parameters()).device
    model.eval()

    # Warm-up to stabilize performance
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # CUDA timing (preferred for GPU)
    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        import time
        starter, ender = time.perf_counter, time.perf_counter

    # Timed runs
    total_time = 0.0
    with torch.no_grad():
        for _ in range(timed_runs):
            if device.type == "cuda":
                starter.record()
                _ = model(input_tensor)
                ender.record()
                torch.cuda.synchronize()
                total_time += starter.elapsed_time(ender)
            else:
                start = starter()
                _ = model(input_tensor)
                end = ender()
                total_time += (end - start) * 1000

    avg_latency = total_time / timed_runs
    return avg_latency


# ============================================================
# 3. Measure FLOPs and Parameters
# ============================================================
def measure_flops_and_params(model, dummy_input):
    """
    Calculate FLOPs and parameter count using THOP.

    Args:
        model (torch.nn.Module): The model to profile.
        dummy_input (torch.Tensor): Sample input for FLOP calculation.

    Returns:
        tuple: (flops in MFLOPs, params in millions)
    """
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_m = flops / 1e6
    return flops_m, params


# ============================================================
# 4. Benchmark Single Model
# ============================================================
def benchmark_model(model_name, model, sample_input, device):
    """
    Benchmark a single model for latency, FLOPs, and parameters.

    Args:
        model_name (str): Name of the model.
        model (torch.nn.Module): The model to benchmark.
        sample_input (torch.Tensor): Input tensor for the model.
        device (torch.device): Device to run the model on.

    Returns:
        dict: Dictionary containing benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Move model and input to device
    if model_name == "TransformerClassifier":
        sample_input = sample_input.squeeze(0)
    model = model.to(device)
    sample_input = sample_input.to(device)
    
    try:
        # Measure latency
        avg_latency = measure_total_latency(model, sample_input)
        print(f"[Latency] {avg_latency:.4f} ms")
        
        # Measure FLOPs and parameters
        flops_m, params = measure_flops_and_params(model, sample_input)
        print(f"[Model Size] FLOPs: {flops_m:.2f} MFLOPs | Params: {params:.2f}M")
        
        results = {
            'model_name': model_name,
            'latency_ms': avg_latency,
            'flops_mflops': flops_m,
            'params': params,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to benchmark {model_name}: {str(e)}")
        results = {
            'model_name': model_name,
            'latency_ms': None,
            'flops_mflops': None,
            'params_millions': None,
            'status': f'failed: {str(e)}'
        }
    
    return results


# ============================================================
# 5. Benchmark Multiple Models
# ============================================================
def benchmark_multiple_models(models_dict, sample_input, device):
    """
    Benchmark multiple models and collect results.

    Args:
        models_dict (dict): Dictionary of {model_name: model_instance}.
        sample_input (torch.Tensor): Input tensor for the models.
        device (torch.device): Device to run the models on.

    Returns:
        pd.DataFrame: DataFrame containing all benchmark results.
    """
    results_list = []
    
    for model_name, model in models_dict.items():
        result = benchmark_model(model_name, model, sample_input, device)
        results_list.append(result)
        
        # Clear CUDA cache between models if using GPU
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    
    return results_df


# ============================================================
# 6. Main Execution
# ============================================================
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "<path to eda/ppg/combined csv>"
    SIGNAL_TYPE = 'EDA'  # Options: 'EDA', 'PPG', 'Combined'
    NUM_CLASSES = 2
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load dataset
    sample_input, num_features = load_dataset(DATA_PATH, signal_type=SIGNAL_TYPE)
    
    # ============================================================
    # Define models to benchmark
    # ============================================================
    models_to_benchmark = {
        'ClassifierLstmMlp': ClassifierLstmMlp(nb_classes=NUM_CLASSES),
        'ClassifierResNet' : ClassifierResNet(nb_classes=NUM_CLASSES),
        'TransformerClassifier' : TransformerClassifier(nb_classes=NUM_CLASSES,input_dim=15)
    }
    
    print(f"Total models to benchmark: {len(models_to_benchmark)}\n")
    
    # ============================================================
    # Run benchmarks
    # ============================================================
    results_df = benchmark_multiple_models(models_to_benchmark, sample_input, device)
    
    # ============================================================
    # Display results
    # ============================================================
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    

    
    # Display sorted by latency
    if results_df['latency_ms'].notna().any():
        print("\n" + "="*80)
        print("MODELS RANKED BY LATENCY (fastest to slowest)")
        print("="*80)
        sorted_df = results_df[results_df['status'] == 'success'].sort_values('latency_ms')
        print(sorted_df[['model_name', 'latency_ms', 'flops_mflops', 'params']].to_string(float_format='%.2f',index=False))
        print("="*80)