import torch
import pandas as pd
import numpy as np
import json
from thop import profile
import torch.nn.functional as F
from resnet_signals import ClassifierResNet
from lstm_signals import ClassifierLstmMlp
from transformer_signals import EmotionTransformer


# ============================================================
# Helper Functions for Raw Data Processing
# ============================================================
def to_tensor_from_series(series, pad_value=0):
    """
    Convert a pandas Series of sequences (numpy arrays) into a padded torch tensor.
    """
    sequences = series.to_list() if hasattr(series, "to_list") else series
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [
        np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=pad_value)
        for seq in sequences
    ]
    return torch.tensor(padded_sequences, dtype=torch.float32)


def create_single_window(data, window_size):
    """
    Create a single window from the data for profiling purposes.
    
    Args:
        data: Tensor of shape (sequence_length,)
        window_size: int, the length of the window
    
    Returns:
        window: Tensor of shape (window_size,)
    """
    L = data.shape[0]
    if L >= window_size:
        # Take the first window
        window = data[:window_size]
    else:
        # Pad if shorter than window_size
        pad_size = window_size - L
        window = F.pad(data, (0, pad_size), "constant", 0)
    return window


# ============================================================
# 1. Load Raw Dataset
# ============================================================
def load_raw_dataset(data_path, window_size=60):
    """
    Load raw signal dataset.

    Args:
        data_path (str): Path to the CSV file containing raw signals.
        window_size (int): Size of the sliding window.

    Returns:
        sample_input (torch.Tensor): Sample input tensor from dataset.
        signal_type (str): Type of signal (EDA or PPG).
    """
    # Determine signal type from path
    if "Raw_EDA" in data_path:
        signal_type = "EDA"
    elif "Raw_PPG" in data_path:
        signal_type = "PPG"
    else:
        raise ValueError("Path must contain 'Raw_EDA' or 'Raw_PPG'")
    
    # Load dataset
    df = pd.read_csv(data_path)
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Parse JSON-encoded data column
    df['Data'] = df['Data'].apply(lambda x: np.array(json.loads(x), dtype=np.float64))
    
    # Extract a sample sequence
    sample_sequence = df['Data'].iloc[0]
    
    # Convert to tensor and create a window
    sample_tensor = torch.tensor(sample_sequence, dtype=torch.float32)
    sample_window = create_single_window(sample_tensor, window_size)
    
    # Add batch and channel dimensions: (1, 1, window_size)
    sample_input = sample_window.unsqueeze(0).unsqueeze(0)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Signal type: {signal_type}")
    print(f"Window size: {window_size}")
    print(f"Sample sequence length: {len(sample_sequence)}")
    print(f"Input shape: {sample_input.shape}\n")
    
    return sample_input, signal_type


# ============================================================
# 2. Measure Inference Latency
# ============================================================
def measure_total_latency(model, input_tensor, warmup_runs=10, timed_runs=50):
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
    params_m = params / 1e6
    return flops_m, params_m


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
    model = model.to(device)
    sample_input = sample_input.to(device)
    
    try:
        # Measure latency
        avg_latency = measure_total_latency(model, sample_input)
        print(f"[Latency] {avg_latency:.4f} ms")
        
        # Measure FLOPs and parameters
        flops_m, params_m = measure_flops_and_params(model, sample_input)
        print(f"[Model Size] FLOPs: {flops_m:.2f} MFLOPs | Params: {params_m:.2f}M")
        
        results = {
            'model_name': model_name,
            'latency_ms': avg_latency,
            'flops_mflops': flops_m,
            'params_millions': params_m,
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
    DATA_PATH = "<path to eda/ppg csv>"
    
    NUM_CLASSES = 2
    WINDOW_SIZE = 60  # Must match the window size used in training
    
    print("="*60)
    print("Raw Signal Model Benchmarking - Multiple Models")
    print("="*60)
    print()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load dataset
    sample_input, signal_type = load_raw_dataset(DATA_PATH, window_size=WINDOW_SIZE)
    
    # ============================================================
    # Define models to benchmark
    # ============================================================
    models_to_benchmark = {
        'ResNet': ClassifierResNet(nb_classes=NUM_CLASSES),
        'ClassifierLstmMlp' : ClassifierLstmMlp(nb_classes=NUM_CLASSES),
        'EmotionTranformer' : EmotionTransformer(num_emotions=NUM_CLASSES)
    }
    
    print(f"Total models to benchmark: {len(models_to_benchmark)}\n")
    
    # ============================================================
    # Run benchmarks
    # ============================================================
    print("="*60)
    print("Starting Performance Measurements...")
    print("="*60)
    
    results_df = benchmark_multiple_models(models_to_benchmark, sample_input, device)
    
    # ============================================================
    # Display results
    # ============================================================
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"Signal Type: {signal_type}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Input Shape: {sample_input.shape}")
    print("-"*80)
    print(results_df.to_string(index=False))
    print("="*80)
    

    
    # Display sorted by latency
    if results_df['latency_ms'].notna().any():
        print("\n" + "="*80)
        print("MODELS RANKED BY LATENCY (fastest to slowest)")
        print("="*80)
        sorted_df = results_df[results_df['status'] == 'success'].sort_values('latency_ms')
        print(sorted_df[['model_name', 'latency_ms', 'flops_mflops', 'params_millions']].to_string(index=False))
        print("="*80)
    
    