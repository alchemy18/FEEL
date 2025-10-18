import torch
import pandas as pd
import numpy as np
from thop import profile
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

text_class = ["The participant felt a strong physical reaction, like a racing heart or tense body, and experienced high-energy emotions such as excitement, enthusiasm, surprise, anger, and nervousness.","The participant felt low energy and relaxed, with calm emotions like peacefulness, relaxed, neutral, boredom, and lack of interest."]
label1 = text_class[0]
label2 = text_class[1]
# ============================================================
# CLSP finetune Model Classes
# ============================================================
class Text_Encoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable
            
        # Initialize tokenizer once
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def text_tokens(self, batch):
        text_embeddings = []
        for texts in batch:
            # Tokenize and get embeddings
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Extract embeddings from the last hidden state
            embeddings = model_output.last_hidden_state

            # Mean pooling the token embeddings to get sentence-level embeddings
            sentence_embeddings = torch.mean(embeddings, dim=1)
            text_embeddings.append(sentence_embeddings)

        return text_embeddings

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=100, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, batch):
        projected = self.projection(batch)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()

        # Define hidden layer dimensions
        hidden_dims = [50, 100]

        # Create sequential layers using nn.Linear and nn.ReLU activations
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def get_hidden_embedding(self, x):
        x = self.layer1(x)
        return self.layer2(x)

# --- CoCoOp components ---
class CoCoOpPromptLearner(nn.Module):
    def __init__(self, feature_dim=100, n_ctx=16, output_dim=768):
        super().__init__()
        self.n_ctx = n_ctx
        # Static context vectors (from CoOp)
        self.ctx = nn.Parameter(torch.empty(n_ctx, output_dim))
        nn.init.normal_(self.ctx, std=0.02)
        # Meta-Net for generating dynamic, instance-specific tokens
        self.meta_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, ts_features):
        # Generate instance-conditional token
        bias = self.meta_net(ts_features).unsqueeze(1)  # [batch, 1, output_dim]
        return self.ctx.unsqueeze(0) + bias  # [batch, n_ctx, output_dim]

class CoCoOpCLSP(nn.Module):
    def __init__(self, feature_dim, n_classes=2, n_ctx=4, device='cuda'):
        super().__init__()
        # Feature encoder - renamed from encoder to eda_encoder for consistency
        self.eda_encoder = CustomMLP(feature_dim, 100)
        # CoCoOp components
        self.prompt_learner = CoCoOpPromptLearner(100, n_ctx)
        self.text_encoder_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = Text_Encoder()  # Use the Text_Encoder class
        self.text_projection = ProjectionHead()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.device = device
        
    def forward(self, x, encoded_label1, encoded_label2):
        """Modified forward to accept pre-encoded labels for benchmarking"""
        features = self.eda_encoder(x['data'] if isinstance(x, dict) else x)
        
        # Simple similarity computation with pre-encoded labels
        similarity1 = torch.sum(features * encoded_label1, dim=1)
        similarity2 = torch.sum(features * encoded_label2, dim=1)
        
        # Stack similarities for binary classification
        logits = torch.stack([similarity1, similarity2], dim=1)
        return logits
    
    def forward_with_text(self, x, class_text):
        """Original forward method with text encoding"""
        features = self.eda_encoder(x)
        ctx = self.prompt_learner(features)

        text_tokens = self.text_encoder_tokenizer(
            class_text, padding=True, truncation=True, return_tensors='pt'
        ).to(features.device)

        # Use word_embeddings from the text encoder
        text_embedding = self.text_encoder.model.embeddings.word_embeddings(text_tokens["input_ids"])

        # Concatenate prompt and embeddings
        final_embedding = torch.cat((ctx, text_embedding), dim=1)
        attention_mask = torch.ones(final_embedding.shape[0], final_embedding.shape[1]).to(final_embedding.device)

        # Forward pass through the encoder
        outputs = self.text_encoder.model(inputs_embeds=final_embedding, attention_mask=attention_mask).last_hidden_state

        # Forward pass through the projection head
        ctx = self.text_projection(outputs)
        ctx_pooled = torch.mean(ctx, dim=1)
        
        logits = torch.sum(ctx_pooled * features, dim=1)
        return logits

# ============================================================
# Dataset Loading
# ============================================================
def load_dataset(data_path, signal_type='EDA'):
    if signal_type == 'EDA':
        relevant_features = ['ku_eda', 'sk_eda', 'dynrange', 'slope', 'variance', 'entropy',
                            'insc', 'fd_mean', 'max_scr', 'min_scr', 'nSCR', 'meanAmpSCR',
                            'meanRespSCR', 'sumAmpSCR', 'sumRespSCR']
    elif signal_type == 'PPG':
        relevant_features = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN', 'HRV_Prc20NN', 'HRV_MinNN',
                            'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF', 'HRV_LFn', 'HRV_HFn',
                            'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS', 'HRV_PAS', 'HRV_PI',
                            'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width',
                            'HRV_MFDFA_alpha1_Peak', 'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                            'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
                            'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
                            'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    else:  # Combined
        relevant_features = ['ku_eda', 'sk_eda', 'dynrange', 'slope', 'variance', 'entropy',
                            'insc', 'fd_mean', 'max_scr', 'min_scr', 'nSCR', 'meanAmpSCR',
                            'meanRespSCR', 'sumAmpSCR', 'sumRespSCR', 'BPM', 'PPG_Rate_Mean',
                            'HRV_MedianNN', 'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN',
                            'HRV_LF', 'HRV_VHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2',
                            'HRV_CVI', 'HRV_PSS', 'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
                            'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
                            'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
                            'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
                            'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
                            'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    
    df = pd.read_csv(data_path)
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    X = df[relevant_features].values
    num_features = len(relevant_features)
    
    sample_input = torch.tensor(X[0:1], dtype=torch.float32).unsqueeze(1)
    sample_batch = {
        'data': torch.tensor(X[0:1], dtype=torch.float32),
        'target': torch.tensor([0], dtype=torch.long),
        'text': ["Data of physiological signal"]
    }
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Signal type: {signal_type}, Features: {num_features}\n")
    
    return sample_input, num_features, sample_batch

# ============================================================
# Latency Measurement
# ============================================================
def preprocess_labels(model, device, warmup_runs=5):
    """Preprocess labels and measure preprocessing latency."""

    
    model.eval()
    model = model.to(device)
    
    # Warm-up
    for _ in range(warmup_runs):
        with torch.no_grad():
            text_embeddings = model.text_encoder.text_tokens([label1])
            text_embeddings = torch.stack(text_embeddings).to(device)
            text_embeddings = model.text_projection(text_embeddings)
            _ = text_embeddings.squeeze(1)
    
    # Time preprocessing
    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
    else:
        import time
        start_time = time.perf_counter()
    
    with torch.no_grad():
        # Label 1
        text_embeddings = model.text_encoder.text_tokens([label1])
        text_embeddings = torch.stack(text_embeddings).to(device)
        text_embeddings = model.text_projection(text_embeddings)
        encoded_label1 = text_embeddings.squeeze(1)
        
        # Label 2
        text_embeddings = model.text_encoder.text_tokens([label2])
        text_embeddings = torch.stack(text_embeddings).to(device)
        text_embeddings = model.text_projection(text_embeddings)
        encoded_label2 = text_embeddings.squeeze(1)
    
    if device.type == "cuda":
        ender.record()
        torch.cuda.synchronize()
        preprocessing_time = starter.elapsed_time(ender)
    else:
        preprocessing_time = (time.perf_counter() - start_time) * 1000
    
    return encoded_label1, encoded_label2, preprocessing_time


def measure_inference_latency(model, input_tensor, encoded_label1, encoded_label2, 
                              warmup_runs=10, timed_runs=500):
    """Measure per-inference latency (excluding preprocessing)."""
    device = next(model.parameters()).device
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in tqdm(range(warmup_runs), desc="Warm-up"):
            _ = model(input_tensor, encoded_label1, encoded_label2)
    
    # Time inference
    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    else:
        import time
    
    total_time = 0.0
    with torch.no_grad():
        for _ in tqdm(range(timed_runs), desc="Timing"):
            if device.type == "cuda":
                starter.record()
                _ = model(input_tensor, encoded_label1, encoded_label2)
                ender.record()
                torch.cuda.synchronize()
                total_time += starter.elapsed_time(ender)
            else:
                start = time.perf_counter()
                _ = model(input_tensor, encoded_label1, encoded_label2)
                end = time.perf_counter()
                total_time += (end - start) * 1000
    
    avg_latency = total_time / timed_runs
    return avg_latency

# ============================================================
# FLOPs and Parameters Measurement (UPDATED)
# ============================================================
def measure_clsp_metrics(model, sample_batch, device):
    """Calculate FLOPs and parameters for CLSP model."""
    
    print("\n" + "="*60)
    print("CALCULATING FLOPs AND PARAMETERS")
    print("="*60)
    
    preprocessing_flops = 0
    inference_flops = 0
    
    # ============================================================
    # PARAMETER BREAKDOWN - ALL COMPONENTS
    # ============================================================
    print("\n[COMPLETE MODEL PARAMETERS]")
    
    # Text Encoder
    text_encoder_params = sum(p.numel() for p in model.text_encoder.parameters())
    print(f"  Text Encoder (DistilBERT): {text_encoder_params/1e6:.4f} M params")
    
    # Projection Head
    projection_params = sum(p.numel() for p in model.text_projection.parameters())
    print(f"  Projection Head:           {projection_params/1e6:.4f} M params")
    
    # EDA Encoder
    eda_params = sum(p.numel() for p in model.eda_encoder.parameters())
    print(f"  EDA Encoder:               {eda_params/1e6:.4f} M params")
    
    # CoCoOp Prompt Learner
    cocoop_params = sum(p.numel() for p in model.prompt_learner.parameters())
    ctx_params = model.prompt_learner.ctx.numel()
    meta_net_params = sum(p.numel() for p in model.prompt_learner.meta_net.parameters())
    print(f"  CoCoOp Prompt Learner:     {cocoop_params/1e6:.4f} M params")
    print(f"    - Static Context (ctx):  {ctx_params} params")
    print(f"    - Meta-Net:              {meta_net_params} params")
    
    # Logit Scale
    logit_scale_params = model.logit_scale.numel()
    print(f"  Logit Scale:               {logit_scale_params} params")
    
    # Total
    total_params = (text_encoder_params + projection_params + eda_params + 
                    cocoop_params + logit_scale_params)
    print(f"\n  → TOTAL MODEL PARAMETERS:  {total_params/1e6:.4f} M")
    
    # ============================================================
    # PREPROCESSING: Text Encoding & Projection (One-time)
    # ============================================================
    print("\n[PREPROCESSING - One-time operations]")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_input = tokenizer(label1, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Text Encoder
    try:
        text_encoder_flops, _ = profile(
            model.text_encoder.model,
            inputs=(encoded_input['input_ids'], encoded_input['attention_mask']),
            verbose=False
        )
    except:
        seq_length = encoded_input['input_ids'].shape[1]
        text_encoder_flops = 2 * text_encoder_params * seq_length
    
    print(f"  Text Encoder: {text_encoder_flops/1e6:.2f} MFLOPs, {text_encoder_params/1e6:.4f}M params")
    
    # Projection Head
    proj_input_dim, proj_hidden_dim = 768, 100
    batch_size = 1
    projection_flops = (
        2 * batch_size * proj_input_dim * proj_hidden_dim +  # Linear1: 768->100
        2 * batch_size * proj_hidden_dim * proj_hidden_dim +  # Linear2: 100->100
        5 * proj_hidden_dim +  # LayerNorm
        8 * proj_hidden_dim    # GELU
    )
    
    print(f"  Projection Head: {projection_flops/1e6:.4f} MFLOPs, {projection_params/1e6:.4f}M params")
    
    # Total preprocessing (2 labels)
    preprocessing_flops = 2 * (text_encoder_flops + projection_flops)
    preprocessing_params = text_encoder_params + projection_params
    
    print(f"  → TOTAL PREPROCESSING: {preprocessing_flops/1e6:.2f} MFLOPs")
    print(f"  → PREPROCESSING PARAMS: {preprocessing_params/1e6:.4f} M")
    
    # ============================================================
    # PER-INFERENCE: Simplified forward() - Current Implementation
    # ============================================================
    print("\n[PER-INFERENCE - Current simplified forward()]")
    
    dummy_eda = sample_batch['data'].to(device)
    input_dim = dummy_eda.shape[1]
    batch_size = dummy_eda.shape[0]
    
    # EDA Encoder (MLP: input_dim -> 50 -> 100)
    eda_encoder_flops = (
        2 * batch_size * input_dim * 50 +   # Linear1
        2 * batch_size * 50 * 100            # Linear2
    )
    
    print(f"  EDA Encoder: {eda_encoder_flops/1e6:.4f} MFLOPs, {eda_params/1e6:.4f}M params")
    
    # Matrix Multiplications (2x)
    eda_hidden_dim = 100
    matmul_flops = 2 * (2 * batch_size * eda_hidden_dim * 1)
    
    print(f"  Matrix Multiplications (2x): {matmul_flops/1e6:.6f} MFLOPs")
    
    inference_flops = eda_encoder_flops + matmul_flops
    inference_params_simple = eda_params
    
    print(f"  → TOTAL PER-INFERENCE: {inference_flops/1e6:.4f} MFLOPs")
    print(f"  → INFERENCE PARAMS:    {inference_params_simple/1e6:.4f} M")
    
    # ============================================================
    # PER-INFERENCE: Full forward_with_text() - With CoCoOp
    # ============================================================
    print("\n[PER-INFERENCE - Full forward_with_text() with CoCoOp]")
    print("  (If using the complete pipeline)")
    
    # CoCoOp Prompt Learner
    # Meta-Net: Linear(100, 32) + Linear(32, 768)
    meta_net_flops = (
        2 * batch_size * 100 * 32 +    # Linear1
        2 * batch_size * 32 * 768      # Linear2
    )
    print(f"  Meta-Net (CoCoOp): {meta_net_flops/1e6:.4f} MFLOPs, {meta_net_params/1e6:.4f}M params")
    
    # Full inference with CoCoOp
    full_inference_flops = eda_encoder_flops + meta_net_flops + text_encoder_flops + projection_flops
    full_inference_params = eda_params + cocoop_params
    
    print(f"  EDA Encoder:       {eda_encoder_flops/1e6:.4f} MFLOPs")
    print(f"  Text Processing:   {(text_encoder_flops + projection_flops)/1e6:.4f} MFLOPs")
    print(f"  → TOTAL FULL INFERENCE: {full_inference_flops/1e6:.4f} MFLOPs")
    print(f"  → FULL INFERENCE PARAMS: {full_inference_params/1e6:.4f} M")
    
    # ============================================================
    # Summary
    # ============================================================
    whole_pipeline_flops = preprocessing_flops + inference_flops
    one_time_params = preprocessing_params + inference_params_simple
    
    print("\n" + "="*60)
    print("SUMMARY - CURRENT SIMPLIFIED IMPLEMENTATION")
    print("="*60)
    print(f"\nTotal Model Parameters (All components): {total_params/1e6:.4f} M")
    print(f"  - Text Encoder:         {text_encoder_params/1e6:.4f} M")
    print(f"  - Projection Head:      {projection_params/1e6:.4f} M")
    print(f"  - EDA Encoder:          {eda_params/1e6:.4f} M")
    print(f"  - CoCoOp (ctx+meta):    {cocoop_params/1e6:.4f} M")
    print(f"  - Logit Scale:          {logit_scale_params} params")
    
    print(f"\n--- ONE-TIME INFERENCE (1st prediction) ---")
    print(f"Parameters Used: {one_time_params/1e6:.4f} M")
    print(f"  - Preprocessing:  {preprocessing_params/1e6:.4f} M (Text Encoder + Projection)")
    print(f"  - Inference:      {inference_params_simple/1e6:.4f} M (EDA Encoder only)")
    print(f"FLOPs: {whole_pipeline_flops/1e6:.2f} MFLOPs")
    print(f"  - Preprocessing:  {preprocessing_flops/1e6:.2f} MFLOPs")
    print(f"  - One Inference:  {inference_flops/1e6:.4f} MFLOPs")
    
    print(f"\n--- SUBSEQUENT INFERENCES (2nd+) ---")
    print(f"Parameters Used: {inference_params_simple/1e6:.4f} M (EDA Encoder only)")
    print(f"FLOPs: {inference_flops/1e6:.4f} MFLOPs per prediction")
    
    print(f"\n--- IF USING FULL COCOOP PIPELINE ---")
    print(f"Parameters per inference: {full_inference_params/1e6:.4f} M (EDA + CoCoOp)")
    print(f"FLOPs per inference: {full_inference_flops/1e6:.4f} MFLOPs")
    
    print(f"\nNote: CoCoOp parameters ({cocoop_params/1e6:.4f}M) are currently")
    print(f"      NOT used in the simplified forward() method.")
    print("="*60 + "\n")
    
    return {
        'whole_pipeline_mflops': whole_pipeline_flops / 1e6,
        'per_inference_mflops': inference_flops / 1e6,
        'preprocessing_mflops': preprocessing_flops / 1e6,
        
        # Parameter counts
        'total_params_m': total_params / 1e6,
        'one_time_params_m': one_time_params / 1e6,
        'inference_params_m': inference_params_simple / 1e6,
        'full_inference_params_m': full_inference_params / 1e6,
        
        # Component breakdown
        'text_encoder_params_m': text_encoder_params / 1e6,
        'projection_params_m': projection_params / 1e6,
        'eda_encoder_params_m': eda_params / 1e6,
        'cocoop_params_m': cocoop_params / 1e6,
        'preprocessing_params_m': preprocessing_params / 1e6,
        
        # Detailed CoCoOp breakdown
        'ctx_params': ctx_params,
        'meta_net_params': meta_net_params,
        'logit_scale_params': logit_scale_params
    }

# ============================================================
# Benchmark Function
# ============================================================
def benchmark_clsp_model(model, sample_batch, device):
    """Benchmark CLSP model for latency, FLOPs, and parameters."""
    print(f"\n{'='*60}")
    print(f"Benchmarking CLSP Model")
    print(f"{'='*60}")
    
    model = model.to(device)
    input_data = {
        'data': sample_batch['data'].to(device),
        'target': sample_batch['target'].to(device),
        'text': sample_batch['text']
    }
    
    try:
        # Measure preprocessing latency and get encoded labels
        print("\n[1/3] Measuring preprocessing latency...")
        encoded_label1, encoded_label2, preprocessing_latency = preprocess_labels(model, device)
        print(f"Preprocessing latency: {preprocessing_latency:.4f} ms")
        
        # Measure per-inference latency
        print("\n[2/3] Measuring per-inference latency...")
        inference_latency = measure_inference_latency(
            model, input_data, encoded_label1, encoded_label2
        )
        print(f"Per-inference latency: {inference_latency:.4f} ms")
        
        # Calculate total pipeline latency
        total_pipeline_latency = preprocessing_latency + inference_latency
        print(f"Total pipeline latency (1st prediction): {total_pipeline_latency:.4f} ms")
        
        # Measure FLOPs and parameters
        print("\n[3/3] Calculating FLOPs and parameters...")
        metrics = measure_clsp_metrics(model, sample_batch, device)
        
        # FIXED: Include ALL metrics from measure_clsp_metrics
        results = {
            'model_name': 'CLIPModel',
            'preprocessing_latency_ms': preprocessing_latency,
            'inference_latency_ms': inference_latency,
            'total_pipeline_latency_ms': total_pipeline_latency,
            'whole_pipeline_mflops': metrics['whole_pipeline_mflops'],
            'per_inference_mflops': metrics['per_inference_mflops'],
            'preprocessing_mflops': metrics['preprocessing_mflops'],
            
            # Parameter counts
            'total_params_m': metrics['total_params_m'],
            'one_time_params_m': metrics['one_time_params_m'],
            'inference_params_m': metrics['inference_params_m'],
            'full_inference_params_m': metrics['full_inference_params_m'],
            
            # Component breakdown - ADD THESE
            'text_encoder_params_m': metrics['text_encoder_params_m'],
            'projection_params_m': metrics['projection_params_m'],
            'eda_encoder_params_m': metrics['eda_encoder_params_m'],
            'cocoop_params_m': metrics['cocoop_params_m'],
            'preprocessing_params_m': metrics['preprocessing_params_m'],
            
            # Detailed CoCoOp breakdown - ADD THESE
            'ctx_params': metrics['ctx_params'],
            'meta_net_params': metrics['meta_net_params'],
            'logit_scale_params': metrics['logit_scale_params'],
            
            'status': 'success'
        }
        
    except Exception as e:
        print(f"[ERROR] Benchmarking failed: {str(e)}")
        results = {'model_name': 'CLIPModel', 'status': f'failed: {str(e)}'}
    
    return results
# ============================================================
# Main Execution (UPDATED RESULTS DISPLAY)
# ============================================================
if __name__ == "__main__":
    DATA_PATH = "<path to eda/ppg/combined csv>"
    SIGNAL_TYPE = 'Combined'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load dataset
    sample_input, num_features, sample_batch = load_dataset(DATA_PATH, signal_type=SIGNAL_TYPE)
    
    # Initialize model with correct parameters
    model = CoCoOpCLSP(feature_dim=num_features, n_classes=2, device=device)
    
    # Run benchmark
    results = benchmark_clsp_model(model, sample_batch, device)
    
    # Display final results
    print("\n" + "="*80)
    print("FINAL BENCHMARK RESULTS")
    print("="*80)
    
    if results['status'] == 'success':
        print(f"\n{'METRIC':<50} {'VALUE':>15}")
        print("-" * 65)
        
        # Latency
        print(f"{'Preprocessing Latency (one-time):':<50} {results['preprocessing_latency_ms']:>12.4f} ms")
        print(f"{'Per-Inference Latency:':<50} {results['inference_latency_ms']:>12.4f} ms")
        print(f"{'Total Pipeline Latency (1st prediction):':<50} {results['total_pipeline_latency_ms']:>12.4f} ms")
        
        # FLOPs
        print()
        print(f"{'Whole Pipeline FLOPs (1st prediction):':<50} {results['whole_pipeline_mflops']:>12.2f} MFLOPs")
        print(f"{'Per-Inference FLOPs (2nd+ predictions):':<50} {results['per_inference_mflops']:>12.4f} MFLOPs")
        
        # Parameters
        print()
        print(f"{'--- PARAMETER BREAKDOWN ---':<50}")
        print(f"{'Total Model Parameters (all components):':<50} {results['total_params_m']:>12.4f} M")
        print(f"{'  - Text Encoder:':<50} {results['text_encoder_params_m']:>12.4f} M")
        print(f"{'  - Projection Head:':<50} {results['projection_params_m']:>12.4f} M")
        print(f"{'  - EDA Encoder:':<50} {results['eda_encoder_params_m']:>12.4f} M")
        print(f"{'  - CoCoOp (ctx + meta-net):':<50} {results['cocoop_params_m']:>12.4f} M")
        print(f"{'  - Logit Scale:':<50} {results['logit_scale_params']:>15} params")
        
        print()
        print(f"{'One-Time Inference Parameters (1st prediction):':<50} {results['one_time_params_m']:>12.4f} M")
        print(f"{'  - Preprocessing (Text Encoder + Projection):':<50} {results['preprocessing_params_m']:>12.4f} M")
        print(f"{'  - Inference (EDA Encoder only):':<50} {results['inference_params_m']:>12.4f} M")
        
        print()
        print(f"{'Subsequent Inference Parameters (2nd+):':<50} {results['inference_params_m']:>12.4f} M")
        
        print()
        print(f"{'If using full CoCoOp pipeline per inference:':<50} {results['full_inference_params_m']:>12.4f} M")
        
        print("\n" + "="*80)
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv('clsp_benchmark_results.csv', index=False)
        print("\nResults saved to: clsp_benchmark_results.csv")
    else:
        print(f"\nBenchmarking failed: {results['status']}")
    
    print("="*80)