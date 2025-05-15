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
    def __init__(self, feature_dim=100, n_ctx=4,meta_hidden_channel=None,output_dim = 768):
        super().__init__()
        self.n_ctx = n_ctx
        # Static context vectors (from CoOp)
        self.ctx = nn.Parameter(torch.empty(n_ctx, output_dim))
        nn.init.normal_(self.ctx, std=0.02)
        # Meta-Net for generating dynamic, instance-specific tokens
        self.meta_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=meta_hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=meta_hidden_channel, out_channels=1, kernel_size=3, padding=1)
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
    def __init__(self, feature_dim, n_classes=2, n_ctx=4, meta_hidden_channel=None):
        super().__init__()
        # Feature encoder
        self.encoder = CustomMLP(feature_dim, 100)
        # CoCoOp components
        self.prompt_learner = CoCoOpPromptLearner(100, n_ctx, meta_hidden_channel=meta_hidden_channel)
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

def load_pretrained_clsp(ckpt_path, feature_dim, n_ctx=4, meta_hidden_channel=None, device='cuda'):
    """Load CLSP's feature encoder only"""
    model = CoCoOpCLSP(feature_dim, n_ctx=n_ctx, meta_hidden_channel=meta_hidden_channel).to(device)

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

def sample_settingwise(df, percentage):
    return (
        df.groupby('setting', group_keys=False)
            .apply(lambda g: g.sample(frac=percentage / 100, random_state=42))
    )


def run_training_once(df_train_full, df_test_full1,df_test_full2, feature_cols, label_col, ckpt_path, percentage,
                      batch_size, epochs, lr, n_ctx, meta_hidden_channel, device, run_name="Run", 
                      save_path=None, early_stopping_patience=25):

    df_0 = df_train_full[df_train_full[label_col] == 0]
    df_1 = df_train_full[df_train_full[label_col] == 1]
    df_0_small = sample_settingwise(df_0, percentage)
    df_1_small = sample_settingwise(df_1, percentage)
    df_train_small = pd.concat([df_0_small, df_1_small]).sample(frac=1.0, random_state=42)

    df_train_remaining = pd.concat([
        df_0.drop(df_0_small.index),
        df_1.drop(df_1_small.index)
    ]).sample(frac=1.0, random_state=42)

    train_dataset = DatasetClass(df_train_small, feature_cols, label_col, from_dataframe=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset1 = DatasetClass(df_test_full1, feature_cols, label_col, from_dataframe=True)
    test_loader1 = DataLoader(test_dataset1, batch_size=batch_size, shuffle=False)

    test_dataset2 = DatasetClass(df_test_full2, feature_cols, label_col, from_dataframe=True)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)

    remaining_train_dataset = DatasetClass(df_train_remaining, feature_cols, label_col, from_dataframe=True)
    remaining_train_loader = DataLoader(remaining_train_dataset, batch_size=batch_size, shuffle=False)

    feature_dim = len(feature_cols)
    n_classes = 2

    if ckpt_path:
        model = load_pretrained_clsp(ckpt_path, feature_dim, n_ctx, meta_hidden_channel, device)
    else:
        model = CoCoOpCLSP(feature_dim, n_classes, n_ctx, meta_hidden_channel).to(device)

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
        test_acc1, test_f1_1 = evaluate_model(model, test_loader1, device, text_class)
        test_acc2, test_f1_2 = evaluate_model(model, test_loader2, device, text_class)

        # Early stopping check
        if test_f1_1 > best_f1:
            best_f1 = test_f1_1
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

    # print("\n" + "-"*30 + f" {run_name} Final Results " + "-"*30)
    test_acc1, test_f1_1 = evaluate_model(model, test_loader1, device, text_class)
    test_acc2, test_f1_2 = evaluate_model(model, test_loader2, device, text_class)

    # print(f"Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
    # print(f"Remaining Train ({100-percentage}%) Accuracy: {trainpercentage_acc:.4f}, F1-score: {trainpercentage_f1:.4f}")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        # print(f"✓ Model successfully saved to {save_path}")
    
    return test_acc1, test_f1_1, test_acc2, test_f1_2

    # test_acc, test_f1 = evaluate_model(model, test_loader, device, text_class)
    # trainpercentage_acc, trainpercentage_f1 = evaluate_model(model, remaining_train_loader, device, text_class)

    # print(f"{run_name} - Final Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
    # print(f"{run_name} - Remaining Train ({100-percentage}%) Accuracy: {trainpercentage_acc:.4f}, F1-score: {trainpercentage_f1:.4f}")

    # if save_path:
    #     torch.save(model.state_dict(), save_path)
    #     print(f"Model saved to {save_path}")

    # return test_acc, test_f1, trainpercentage_acc, trainpercentage_f1

def train_cocoop(dataset_name, csv_path,test_list, ckpt_path=None, feature_cols=None, label_col=None, percentage=None,
                 batch_size=4, epochs=300, lr=1e-5, n_ctx=4, meta_hidden_channel=None, device='cuda'):

    feature_name = "EDA" if "eda" in csv_path.lower() else "PPG" if "ppg" in csv_path.lower() else "COM"
    # print(f"Running for {percentage}% and {label_col} ")
    # print(f"Parameters: batch_size={batch_size}, epochs={epochs}, lr={lr}, n_ctx={n_ctx}, meta_hidden_channel={meta_hidden_channel}")

    # print("\n" + "="*80)
    # print(f"EXPERIMENT: {dataset_name} - {feature_name} - {label_col} - {percentage}% Training Data")
    # print("-"*80)
    # print(f"PARAMETERS:")
    # print(f"  • Batch Size: {batch_size}")
    # print(f"  • Epochs: {epochs}")
    # print(f"  • Learning Rate: {lr}")
    # print(f"  • Context Tokens (n_ctx): {n_ctx}")
    # print(f"  • Meta-Net Hidden Dim: {meta_hidden_channel}")
    # print("-"*80)

    df_train_full  = pd.read_csv(csv_path)
    df_test_full1 =  pd.read_csv(test_list[0]) 
    df_test_full2 =  pd.read_csv(test_list[1])

    if feature_cols is None:
        feature_cols = [col for col in df_train_full.columns if col not in ['PID', 'arousal_category', 'valence_category','setting']]

    run_name_1 = f"Run 1"

    model_name_1 = "<path to save the model>"

    test_acc_1, test_f1_1, test_acc_2, test_f1_2 = run_training_once(
        df_train_full, df_test_full1,df_test_full2, feature_cols, label_col,
        ckpt_path, percentage, batch_size, epochs, lr, n_ctx, meta_hidden_channel, device, run_name_1,
        save_path=model_name_1
    )

    # print("\n" + "-"*40 + " FINAL RESULTS " + "-"*40)
    # print(f"│ Average Test Accuracy:              {test_acc_1:.4f}  │")
    # print(f"│ Average Test F1-Score:              {test_f1_1:.4f}  │")
    # print("="*80 + "\n")
    
    return test_acc_1, test_f1_1, test_acc_2, test_f1_2


   
        