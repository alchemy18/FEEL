import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import math 
# --- CLSP components ---

class Text_Encoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name).to(torch.device("cuda"))
        for p in self.model.parameters():
            p.requires_grad = trainable

    def text_tokens(self,batch):
        text_embeddings = []
        for i in range(len(batch)):
            texts = batch[i]
            # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased').to(torch.device("cuda"))
            model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(torch.device("cuda"))
            
            # Tokenize and get embeddings
            # encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(torch.device("cuda"))
            with torch.no_grad():
                model_output = model(texts)

            # Extract embeddings from the last hidden state
            embeddings = model_output.last_hidden_state

            # Get the embeddings for the [CLS] token (the first token)
            cls_embeddings = embeddings[:, 0, :]

            # Alternatively, mean pooling the token embeddings to get sentence-level embeddings
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
        # text_embeddings = []
        
        projected = self.projection(batch)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        # text_embeddings.append(x)
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
        # self.output_layer = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.output_layer(x)
        return x

    def get_hidden_embedding(self, x):
        x = self.layer1(x)
        return self.layer2(x)

class CLSPModel(nn.Module):
    def __init__(self, mlp_input_dim, mlp_output_dim, device):
        super().__init__()
        self.text_encoder = Text_Encoder().to(device)
        self.enocder = CustomMLP(mlp_input_dim, mlp_output_dim).to(device)
        self.text_projection = ProjectionHead(embedding_dim=768, projection_dim=100).to(device)
        self.device = device

# --- CoCoOp components ---
class CoCoOpPromptLearner(nn.Module):
    def __init__(self, feature_dim=100, n_ctx=4,meta_hidden_dim=None,output_dim = 768):
        super().__init__()
        self.n_ctx = n_ctx
        # Static context vectors (from CoOp)
        self.ctx = nn.Parameter(torch.empty(n_ctx, output_dim))
        nn.init.normal_(self.ctx, std=0.02)
        # Meta-Net for generating dynamic, instance-specific tokens
        self.meta_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=meta_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=meta_hidden_dim, out_channels=1, kernel_size=3, padding=1)
        )
        self.proj = nn.Linear(feature_dim, output_dim)

    def forward(self, ts_features):
        # print(ts_features.shape)
        # Generate instance-conditional token
        x = self.meta_net(ts_features.unsqueeze(1))  # [batch, 1, feature_dim]
        # print(x.shape)
        bias = self.proj(x.squeeze(1)).unsqueeze(1)
        return self.ctx.unsqueeze(0) + bias  # [batch, n_ctx, feature_dim]

class CoCoOpCLSP(nn.Module):
    def __init__(self, feature_dim, n_classes=2, n_ctx=4, meta_hidden_dim=None):
        super().__init__()
        # Feature encoder
        self.encoder = CustomMLP(feature_dim, 100)
        # CoCoOp components
        self.prompt_learner = CoCoOpPromptLearner(100, n_ctx, meta_hidden_dim=meta_hidden_dim)
        self.text_encoder_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_projection = ProjectionHead()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        
    def forward(self, x, class_text):
        features = self.encoder(x)
        ctx = self.prompt_learner(features)

        text_tokens = self.text_encoder_tokenizer(
            class_text, padding=True, truncation=True, return_tensors='pt'
        ).to(features.device)

        # Use word_embeddings from the text encoder - it's frozen, but we keep it in the graph
        text_embedding = self.text_encoder.embeddings.word_embeddings(text_tokens["input_ids"])

        # Concatenate prompt and embeddings
        final_embedding = torch.cat((ctx, text_embedding), dim=1)
        attention_mask = torch.ones(final_embedding.shape[0], final_embedding.shape[1]).to(final_embedding.device)

        # Forward pass through the frozen encoder (gradients will not flow into it)
        outputs = self.text_encoder(inputs_embeds=final_embedding, attention_mask=attention_mask).last_hidden_state

        # Forward pass through the frozen projection head
        ctx = self.text_projection(outputs)

        ctx_pooled = torch.mean(ctx, dim=1)
        
        logits = torch.sum(ctx_pooled * features, dim=1)

        return logits

# --- Data pipeline ---

class DatasetClass(Dataset):
    def __init__(self, data, feature_cols, label_col, from_dataframe=False):
        if from_dataframe:
            df = data  # Assume data is already a DataFrame
        else:
            df = pd.read_csv(data)  # Otherwise, treat as path
        
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx]),
            'label': torch.tensor(self.labels[idx])
        }

def load_pretrained_clsp(ckpt_path, feature_dim, n_ctx=4, meta_hidden_dim=None, device='cuda'):
    """Load CLSP's feature encoder only"""
    model = CoCoOpCLSP(feature_dim, n_ctx=n_ctx, meta_hidden_dim=meta_hidden_dim).to(device)

    try:
        # Try to load pretrained weights
        pretrained_dict = torch.load(ckpt_path, map_location=device)
        
        # Extract encoder weights
        encoder_weights = {}
        projection_head_weights = {}
        for k, v in pretrained_dict.items():
            if 'eda_encoder' in k or 'ppg_encoder' in k or 'combined_encoder' in k:
                if 'eda_encoder' in k:
                    new_k = k.replace('eda_encoder.', '')
                elif 'ppg_encoder' in k:
                    new_k = k.replace('ppg_encoder.', '')
                elif 'combined_encoder' in k:
                    new_k = k.replace('combined_encoder.', '')
                else:
                    print("ISSUE")
                if new_k.startswith('layer1') or new_k.startswith('layer2'):
                    encoder_weights[new_k] = v
            if 'text_projection' in k:
                new_k = k.replace('text_projection.', '')
                projection_head_weights[new_k] = v
        
        # Load weights into encoder
        model.encoder.load_state_dict(encoder_weights, strict=False)
        model.text_projection.load_state_dict(projection_head_weights)
        
        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.text_projection.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        
        # print("Loaded pretrained encoder weights")
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        print("Training from scratch")
    
    return model

from sklearn.metrics import classification_report

# --- Training and evaluation ---
def evaluate_model(model, dataloader, device,text_class):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].float().to(device)  # (B, F)
            labels = batch['label'].to(device)               # (B,)
            B = features.shape[0]

            # Repeat features and texts
            features = features.unsqueeze(1).repeat(1, 2, 1)  # (B, 2, F)
            features = features.view(-1, features.shape[-1])  # (B*2, F)
            texts = text_class * B     # (B*2,)

            # Get logits
            logits = model(features, texts)                   # (B*2, 1)
            logits = logits.view(-1, 2)                       # (B, 2)

            preds = torch.argmax(logits, dim=1)               # Choose class with higher score

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # print("\n" + "-"*30 + " Classification Report " + "-"*30)
    # report = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'])
    # print(report)
    # print("-"*80)

    return accuracy, f1

def apply_oversampling(df, feature_cols, label_col, oversampling_method='smote', random_state=42):
    """
    Apply oversampling to balance the dataset.
    
    Args:
        df: DataFrame containing the data
        feature_cols: List of feature column names
        label_col: Name of the label column
        oversampling_method: Type of oversampling ('smote', 'random')
        random_state: Random state for reproducibility
    
    Returns:
        DataFrame with oversampled data
    """
    X = df[feature_cols].values
    y = df[label_col].values
    
    # print(f"Before oversampling: {Counter(y)}")
    
    if oversampling_method == 'smote':
        # Use SMOTE for synthetic sample generation
        oversampler = SMOTE(random_state=random_state)
    elif oversampling_method == 'random':
        # Use random oversampling (duplicate existing samples)
        oversampler = RandomOverSampler(random_state=random_state)
    else:
        raise ValueError("oversampling_method must be one of: 'smote', 'random' ")
    
    try:
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        # print(f"After oversampling: {Counter(y_resampled)}")
        
        # Create new DataFrame with oversampled data
        df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
        df_resampled[label_col] = y_resampled
        
        # If original DataFrame had PID column, we need to handle it
        if 'PID' in df.columns:
            # For new synthetic samples, we'll assign them to existing PIDs randomly
            original_pids = df['PID'].values
            n_original = len(original_pids)
            n_synthetic = len(y_resampled) - n_original
            
            # For original samples, keep original PIDs
            # For synthetic samples, randomly assign existing PIDs
            if n_synthetic > 0:
                synthetic_pids = np.random.choice(original_pids, size=n_synthetic, replace=True)
                all_pids = np.concatenate([original_pids, synthetic_pids])
                df_resampled['PID'] = all_pids
            else:
                df_resampled['PID'] = original_pids
                
        return df_resampled
        
    except Exception as e:
        print(f"Error during oversampling: {e}")
        print("Returning original dataframe without oversampling")
        return df


def run_training_once(df_train_full, df_test_full, feature_cols, label_col, ckpt_path, percentage,
                      batch_size, epochs, lr, n_ctx, meta_hidden_dim, device, run_name="Run", 
                      save_path=None, early_stopping_patience=25, use_oversampling=True, 
                      oversampling_method='smote'):

    # Print original class distribution
    # print(f"\n{run_name} - Original training data class distribution:")
    # original_counts = df_train_full[label_col].value_counts().sort_index()
    # print(f"Class 0: {original_counts[0]} samples")
    # print(f"Class 1: {original_counts[1]} samples")
    # print(f"Total: {len(df_train_full)} samples")
    df_train_full = df_train_full.fillna(0)

    # Step 1: Apply oversampling to the full training data if requested
    if use_oversampling:
        # print(f"\n{run_name} - Applying {oversampling_method} oversampling to full training data...")
        df_train_balanced = apply_oversampling(df_train_full, feature_cols, label_col, 
                                             oversampling_method=oversampling_method, 
                                             random_state=42)
    else:
        df_train_balanced = df_train_full.copy()
    
    # Step 2: Take x% from each class separately to ensure exact balance
    df_0_balanced = df_train_balanced[df_train_balanced[label_col] == 0]
    df_1_balanced = df_train_balanced[df_train_balanced[label_col] == 1]

    # Calculate samples per class to maintain overall x% of original data
    total_original_samples = len(df_train_full)
    samples_per_class = math.ceil((total_original_samples * percentage / 100) / 2)
    
    print(f"\n{run_name} - Sampling {samples_per_class} samples from each class:")
    print(f"Available Class 0 samples: {len(df_0_balanced)}")
    print(f"Available Class 1 samples: {len(df_1_balanced)}")
    df_0_small = df_0_balanced.sample(n=min(samples_per_class, len(df_0_balanced)), random_state=42)
    df_1_small = df_1_balanced.sample(n=min(samples_per_class, len(df_1_balanced)), random_state=42)
    df_train_small = pd.concat([df_0_small, df_1_small]).sample(frac=1.0, random_state=42)


    train_dataset = DatasetClass(df_train_small, feature_cols, label_col, from_dataframe=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DatasetClass(df_test_full, feature_cols, label_col, from_dataframe=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    feature_dim = len(feature_cols)
    n_classes = 2

    if ckpt_path:
        model = load_pretrained_clsp(ckpt_path, feature_dim, n_ctx, meta_hidden_dim, device)
    else:
        model = CoCoOpCLSP(feature_dim, n_classes, n_ctx, meta_hidden_dim).to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    if label_col == "arousal_category":
        text_class = ["The participant felt low energy and relaxed, with calm emotions like peacefulness, relaxed, neutral, boredom, and lack of interest.","The participant felt a strong physical reaction, like a racing heart or tense body, and experienced high-energy emotions such as excitement, enthusiasm, surprise, anger, and nervousness."]
    else:
        text_class = ["The participant felt bad and was in a negative mood, with emotions like sadness, fear, anger, worry, hopelessness, and frustration.", "The participant experienced a positive mood characterized by emotions such as happy, joy, gratitude, serenity, interest, hope, pride, amusement, inspiration, awe, and love."]
      
    # print(f"\n{run_name} Training Progress:")
    # print("-"*60)
    # print(f"{'Epoch':^10}|{'Train Loss':^15}|{'Test Acc':^15}|{'Test F1':^15}")
    # print("-"*60)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].float().to(device)
            labels = batch['label'].long().to(device)
            B = features.shape[0]

            # Repeat features for each prompt
            features = features.unsqueeze(1).repeat(1, 2, 1)  # (B, 2, F)
            features = features.view(-1, features.shape[-1])  # (B*2, F)
            texts = text_class * B  # (B*2,)

            logits = model(features, texts)  # (B*2, 1)
            logits = logits.view(-1, 2)      # (B, 2)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on test set
        test_acc, test_f1 = evaluate_model(model, test_loader, device, text_class)

        # Early stopping check
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                # print(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
                break

    # Load the best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n" + "-"*30 + f" {run_name} Final Results " + "-"*30)
    test_acc, test_f1 = evaluate_model(model, test_loader, device, text_class)
    
    print(f"Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"✓ Model successfully saved to {save_path}")
    
    return test_acc, test_f1

    # test_acc, test_f1 = evaluate_model(model, test_loader, device, text_class)
    # trainpercentage_acc, trainpercentage_f1 = evaluate_model(model, remaining_train_loader, device, text_class)

    # print(f"{run_name} - Final Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
    # print(f"{run_name} - Remaining Train ({100-percentage}%) Accuracy: {trainpercentage_acc:.4f}, F1-score: {trainpercentage_f1:.4f}")

    # if save_path:
    #     torch.save(model.state_dict(), save_path)
    #     print(f"Model saved to {save_path}")

    # return test_acc, test_f1, trainpercentage_acc, trainpercentage_f1

def train_cocoop(dataset_name, csv_path, ckpt_path=None, feature_cols=None, label_col=None, percentage=None,
                 batch_size=4, epochs=300, lr=1e-5, n_ctx=4, meta_hidden_dim=None, device='cuda',
                 use_oversampling=True, oversampling_method='smote'):

    feature_name = "EDA" if "eda" in csv_path.lower() else "PPG" if "ppg" in csv_path.lower() else "COM"
    # print(f"Running for {percentage}% and {label_col} ")
    # print(f"Parameters: batch_size={batch_size}, epochs={epochs}, lr={lr}, n_ctx={n_ctx}, meta_hidden_dim={meta_hidden_dim}")

    print("\n" + "="*80)
    print(f"EXPERIMENT: {dataset_name} - {feature_name} - {label_col} - {percentage}% Training Data")
    # if use_oversampling:
    #     print(f"OVERSAMPLING: {oversampling_method.upper()}")
    print("-"*80)
    print(f"PARAMETERS:")
    print(f"  • Batch Size: {batch_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Learning Rate: {lr}")
    print(f"  • Context Tokens (n_ctx): {n_ctx}")
    print(f"  • Meta-Net Hidden Dim: {meta_hidden_dim}")
    print(f"  • Oversampling: {use_oversampling} ({oversampling_method if use_oversampling else 'None'})")
    print("-"*80)

    df = pd.read_csv(csv_path)
    unique_pids = df['PID'].unique()
    pid_train, pid_test = train_test_split(unique_pids, test_size=0.5, random_state=42)

    df_train_full = df[df['PID'].isin(pid_train)].copy()
    df_test_full = df[df['PID'].isin(pid_test)].copy()

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['PID', 'arousal_category', 'valence_category']]

    run_name_1 = f"Run 1"
    run_name_2 = f"Run 2"

    model_name_1 = f"./{dataset_name}_CNN/{dataset_name}_{feature_name}_{label_col}_{percentage}_{oversampling_method}_run1.pt"
    model_name_2 = f"./{dataset_name}_CNN/{dataset_name}_{feature_name}_{label_col}_{percentage}_{oversampling_method}_run2.pt"

    test_acc_1, test_f1_1 = run_training_once(
        df_train_full, df_test_full, feature_cols, label_col,
        ckpt_path, percentage, batch_size, epochs, lr, n_ctx, meta_hidden_dim, device, run_name_1,
        save_path=model_name_1, use_oversampling=use_oversampling, oversampling_method=oversampling_method
    )

    test_acc_2, test_f1_2 = run_training_once(
        df_test_full, df_train_full, feature_cols, label_col,
        ckpt_path, percentage, batch_size, epochs, lr, n_ctx, meta_hidden_dim, device, run_name_2,
        save_path=model_name_2, use_oversampling=use_oversampling, oversampling_method=oversampling_method
    )

    avg_test_acc = (test_acc_1 + test_acc_2) / 2
    avg_test_f1 = (test_f1_1 + test_f1_2) / 2

    print("\n" + "-"*40 + " FINAL RESULTS " + "-"*40)
    print(f"│ Average Test Accuracy:              {avg_test_acc:.4f}  │")
    print(f"│ Average Test F1-Score:              {avg_test_f1:.4f}  │")
    print("="*80 + "\n")
    
    return avg_test_acc, avg_test_f1


# --- Usage ---

if __name__ == "__main__":
    
    # Feature columns from your CSV
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
    dataset_name = "dataset name"

    modalities = {
        "EDA": {
            "feature_cols": feature_EDA,
            "csv": "<path to the eda features csv>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda and arousal category>",
                "valence_category": "<path for the clsp model for eda and arousal category>"
            }
        },
        "PPG": {
            "feature_cols": feature_PPG,
            "csv": "<path to the ppg features csv>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda and arousal category>",
                "valence_category": "<path for the clsp model for eda and arousal category>"
            }
        },
        "Combined": {
            "feature_cols": feature_Combined,
            "csv": "<path to the combined features csv>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda and arousal category>",
                "valence_category": "<path for the clsp model for eda and arousal category>"
            }
        }
    }

    percentages = [5,25,50]
    labels = ["valence_category"]
    
    results = []

    oversampling_method = ['smote','random']

    for modality_name, modality_data in modalities.items():
        for label in labels:
            for oversampling in oversampling_method:
                for pct in percentages:
                    for bc in [4]:
                        for ep in [15]:
                            for lr in [5e-5]: 
                                for n_ctx in [24]:
                                    for me in [24]:
                                        print(f"Label : {label}, Modality : {modality_name}, Percentage : {pct}, Meta-Net Channel Dim : {me}")
                            
                                        model = train_cocoop(
                                            dataset_name=dataset_name,
                                            csv_path=modality_data["csv"],
                                            ckpt_path=f"/path to the base checkpoints directory/{modality_data['ckpt'][label]}",
                                            feature_cols=modality_data["feature_cols"],
                                            label_col=label,
                                            percentage=pct,
                                            batch_size=bc,           # Custom batch size
                                            epochs=ep,             # Custom number of epochs
                                            lr=lr,                # Custom learning rate
                                            n_ctx=n_ctx,                # Custom number of context tokens
                                            meta_hidden_dim= me,     # Custom meta-net hidden dimension
                                            device='cuda',
                                            use_oversampling=True,   # Enable oversampling
                                            oversampling_method=oversampling  # Options: 'smote', 'random'
                                        )
                                        
                                        print("=" * 25)
            
        
        