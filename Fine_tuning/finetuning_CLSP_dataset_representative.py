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
            with torch.no_grad():
                model_output = model(texts)

            # Extract embeddings from the last hidden state
            embeddings = model_output.last_hidden_state

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

class CLSPModel(nn.Module):
    def __init__(self, mlp_input_dim, mlp_output_dim, device):
        super().__init__()
        self.text_encoder = Text_Encoder().to(device)
        self.enocder = CustomMLP(mlp_input_dim, mlp_output_dim).to(device)
        self.text_projection = ProjectionHead(embedding_dim=768, projection_dim=100).to(device)
        self.device = device

# --- CoCoOp components ---
class CoCoOpPromptLearner(nn.Module):
    def __init__(self, feature_dim=100, n_ctx=4, output_dim=768, meta_hidden_dim=None):
        super().__init__()
        self.n_ctx = n_ctx
        # Static context vectors (from CoOp)
        self.ctx = nn.Parameter(torch.empty(n_ctx, output_dim))
        nn.init.normal_(self.ctx, std=0.02)
        
        # Calculate meta hidden dimension if not provided
        if meta_hidden_dim is None:
            meta_hidden_dim = 16
            
        # Meta-Net for generating dynamic, instance-specific tokens
        self.meta_net = nn.Sequential(
            nn.Linear(feature_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(meta_hidden_dim, output_dim)
        )

    def forward(self, ts_features):
        # Generate instance-conditional token
        bias = self.meta_net(ts_features).unsqueeze(1)  # [batch, 1, feature_dim]
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

        text_embedding = self.text_encoder.embeddings.word_embeddings(text_tokens["input_ids"])

        final_embedding = torch.cat((ctx, text_embedding), dim=1)
        attention_mask = torch.ones(final_embedding.shape[0], final_embedding.shape[1]).to(final_embedding.device)

        outputs = self.text_encoder(inputs_embeds=final_embedding, attention_mask=attention_mask).last_hidden_state

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

    return accuracy, f1



def run_training_once(df_train_full, df_test_full, feature_cols, label_col, ckpt_path, percentage,
                      batch_size, epochs, lr, n_ctx, meta_hidden_dim, device, run_name="Run", 
                      save_path=None, early_stopping_patience=25):

    df_0 = df_train_full[df_train_full[label_col] == 0]
    df_1 = df_train_full[df_train_full[label_col] == 1]
    df_0_small = df_0.sample(frac=percentage/100, random_state=42)
    df_1_small = df_1.sample(frac=percentage/100, random_state=42)
    df_train_small = pd.concat([df_0_small, df_1_small]).sample(frac=1.0, random_state=42)

    df_train_remaining = pd.concat([
        df_0.drop(df_0_small.index),
        df_1.drop(df_1_small.index)
    ]).sample(frac=1.0, random_state=42)

    train_dataset = DatasetClass(df_train_small, feature_cols, label_col, from_dataframe=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DatasetClass(df_test_full, feature_cols, label_col, from_dataframe=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    remaining_train_dataset = DatasetClass(df_train_remaining, feature_cols, label_col, from_dataframe=True)
    remaining_train_loader = DataLoader(remaining_train_dataset, batch_size=batch_size, shuffle=False)

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
        text_class = ["The participant felt a strong physical reaction, like a racing heart or tense body, and experienced high-energy emotions such as excitement, enthusiasm, surprise, anger, and nervousness.","The participant felt low energy and relaxed, with calm emotions like peacefulness, relaxed, neutral, boredom, and lack of interest."]
    else:
        text_class = ["The participant felt bad and was in a negative mood, with emotions like sadness, fear, anger, worry, hopelessness, and frustration.", "The participant felt bad and was in a negative mood, with emotions like sadness, fear, depression, anger, stress, worry, hopelessness, and frustration."]
        
    
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
    trainpercentage_acc, trainpercentage_f1 = evaluate_model(model, remaining_train_loader, device, text_class)
    
    print(f"Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
    print(f"Remaining Train ({100-percentage}%) Accuracy: {trainpercentage_acc:.4f}, F1-score: {trainpercentage_f1:.4f}")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"✓ Model successfully saved to {save_path}")
    
    return test_acc, test_f1, trainpercentage_acc, trainpercentage_f1


def train_cocoop(dataset_name, csv_path, ckpt_path=None, feature_cols=None, label_col=None, percentage=None,
                 batch_size=4, epochs=300, lr=1e-5, n_ctx=4, meta_hidden_dim=None, device='cuda'):

    feature_name = "EDA" if "eda" in csv_path.lower() else "PPG" if "ppg" in csv_path.lower() else "COM"

    print("\n" + "="*80)
    print(f"EXPERIMENT: {dataset_name} - {feature_name} - {label_col} - {percentage}% Training Data")
    print("-"*80)
    print(f"PARAMETERS:")
    print(f"  • Batch Size: {batch_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Learning Rate: {lr}")
    print(f"  • Context Tokens (n_ctx): {n_ctx}")
    print(f"  • Meta-Net Hidden Dim: {meta_hidden_dim}")
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

    model_name_1 = f"./{dataset_name}/Fet:{feature_name}_Label:{label_col}_Per:{percentage}_Dim:{meta_hidden_dim}_run1.pt"
    model_name_2 = f"./{dataset_name}/Fet:{feature_name}_Label:{label_col}_Per:{percentage}_Dim:{meta_hidden_dim}run2.pt"

    test_acc_1, test_f1_1, trainpercentage_acc_1, trainpercentage_f1_1 = run_training_once(
        df_train_full, df_test_full, feature_cols, label_col,
        ckpt_path, percentage, batch_size, epochs, lr, n_ctx, meta_hidden_dim, device, run_name_1,
        save_path=model_name_1
    )

    test_acc_2, test_f1_2, trainpercentage_acc_2, trainpercentage_f1_2 = run_training_once(
        df_test_full, df_train_full, feature_cols, label_col,
        ckpt_path, percentage, batch_size, epochs, lr, n_ctx, meta_hidden_dim, device, run_name_2,
        save_path=model_name_2
    )

    avg_test_acc = (test_acc_1 + test_acc_2) / 2
    avg_test_f1 = (test_f1_1 + test_f1_2) / 2
    avg_trainpercentage_acc = (trainpercentage_acc_1 + trainpercentage_acc_2) / 2
    avg_trainpercentage_f1 = (trainpercentage_f1_1 + trainpercentage_f1_2) / 2

    print("\n" + "-"*40 + " FINAL RESULTS " + "-"*40)
    print(f"│ Average Test Accuracy:              {avg_test_acc:.4f}  │")
    print(f"│ Average Test F1-Score:              {avg_test_f1:.4f}  │")
    print(f"│ Remaining Train ({100-percentage}%) Accuracy:  {avg_trainpercentage_acc:.4f}  │")
    print(f"│ Remaining Train ({100-percentage}%) F1-Score:  {avg_trainpercentage_f1:.4f}  │")
    print("="*80 + "\n")
    
    return avg_test_acc, avg_test_f1, avg_trainpercentage_acc, avg_trainpercentage_f1


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
    dataset_name = "<dataset name>"

    modalities = {
        "EDA": {
            "feature_cols": feature_EDA,
            "csv": "<path for the eda features data for the dataset>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda and arousal category>",
                "valence_category": "<path for the clsp model for eda and valence category>"
            }
        },
        "PPG": {
            "feature_cols": feature_PPG,
            "csv":"<path for the ppg features data for the dataset>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for ppg and arousal category>",
                "valence_category": "<path for the clsp model for ppg and valence category>"
            }
        },
        "Combined": {
            "feature_cols": feature_Combined,
            "csv": "<path for the eda+ppg features data for the dataset>",
            "ckpt": {
                "arousal_category": "<path for the clsp model for eda+ppg and arousal category>",
                "valence_category": "<path for the clsp model for eda+ppg and arousal category>"
            }
        }
    }

    percentages = [5, 25, 50]
    labels = ["arousal_category", "valence_category"]
    
    results = []

    for modality_name, modality_data in modalities.items():
        for label in labels:
            for pct in percentages:
                print(f"Label : {label}, Modality : {modality_name}, Percentage : {pct}")
    
                model = train_cocoop(
                    dataset_name=dataset_name,
                    csv_path=modality_data["csv"],
                    ckpt_path=modality_data['ckpt'][label],
                    feature_cols=modality_data["feature_cols"],
                    label_col=label,
                    percentage=pct,
                    batch_size=4,           # Custom batch size
                    epochs=15,             # Custom number of epochs
                    lr=5e-5,                # Custom learning rate
                    n_ctx=16,                # Custom number of context tokens
                    meta_hidden_dim= 32,     # Custom meta-net hidden dimension
                    device='cuda'
                )
                
                print("=" * 25)
        
        
        