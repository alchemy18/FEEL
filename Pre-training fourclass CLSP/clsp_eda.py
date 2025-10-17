import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import itertools
import random
import os
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import warnings
warnings.filterwarnings("ignore")

# Set random seed
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def csv_read(path):
    return pd.read_csv(path)

def lowercase(data):
    return data.lower()

def stop_words(text):
    stop_words_set = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words_set]
    return " ".join(filtered_sentence)

def punctuations(data):
    no_punct = [words for words in data if words not in string.punctuation]
    words_wo_punct = ''.join(no_punct)
    return words_wo_punct

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return " ".join(lemmatized_text)

def preprcosess(text_data):
    text_data['Text'] = text_data['Text Description'].apply(lambda x: lemmatize(x))
    text_data['Text'] = text_data['Text'].apply(lambda x: stop_words(x))
    text_data['Text'] = text_data['Text'].apply(lambda x: lowercase(x))
    text_data['Text'] = text_data['Text'].apply(lambda x: punctuations(x))
    return text_data

class Text_Encoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name).to(torch.device("cuda"))
        for p in self.model.parameters():
            p.requires_grad = trainable
    def text_tokens(self, batch):
        text_embeddings = []
        for i in range(len(batch)):
            texts = batch[i]
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(torch.device("cuda"))
            encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(torch.device("cuda"))
            with torch.no_grad():
                model_output = model(**encoded_input)
            embeddings = model_output.last_hidden_state
            sentence_embeddings = torch.mean(embeddings, dim=1)
            text_embeddings.append(sentence_embeddings)
        return text_embeddings

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=100, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    def forward(self, batch):
        text_embeddings = []
        for i in range(len(batch)):
            x = batch[i]
            projected = self.projection(x)
            x = self.gelu(projected)
            x = self.fc(x)
            x = self.dropout(x)
            x = x + projected
            x = self.layer_norm(x)
            text_embeddings.append(x)
        return text_embeddings

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dims = [50, 100]
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(hidden_dims[1], output_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x
    def get_hidden_embedding(self, x):
        x = self.layer1(x)
        return self.layer2(x)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, text):
        self.data = data
        self.targets = targets
        self.text = text
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        item = {}
        item['data'] = torch.from_numpy(self.data[index]).float()
        item['target'] = self.targets[index]
        item['text'] = self.text[index]
        return item

def windowed_preprocess(df):
    processed_data = []
    for index, row in df.iterrows():
        row_as_array = np.array([np.array(value) for value in row.to_numpy()])
        processed_data.append(row_as_array)
    return processed_data

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    def reset(self):
        self.avg, self.sum, self.count = [0] * 3
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class CLIPModel(nn.Module):
    def __init__(self, mlp_input_dim, mlp_output_dim, device):
        super().__init__()
        self.text_encoder = Text_Encoder().to(device)
        self.ppg_encoder = CustomMLP(mlp_input_dim, mlp_output_dim).to(device)
        self.text_projection = ProjectionHead(embedding_dim=768, projection_dim=100).to(device)
        self.device = device
    def ppg_train(self, learning_rate, beta1, beta2, epsilon, train_dataloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.ppg_encoder.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        for batch in tqdm(train_dataloader):
            data = batch['data']
            target = batch['target']
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.ppg_encoder(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
    def forward(self, batch):
        data_values = batch['data']
        ppg_embeddings = self.ppg_encoder.get_hidden_embedding(data_values).to(self.device)
        text_embeddings = self.text_encoder.text_tokens(batch["text"])
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(self.device)
        text_embeddings = text_embeddings.squeeze(1)
        logits = torch.matmul(text_embeddings, ppg_embeddings.T)
        ppg_similarity = torch.matmul(ppg_embeddings, ppg_embeddings.T)
        texts_similarity = torch.matmul(text_embeddings, text_embeddings.T)
        targets = F.softmax((ppg_similarity + texts_similarity) / 2, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device=torch.device("cuda")):
    model.to(device)
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        for key in batch.keys():
            if key != "text":
                batch[key] = batch[key].to(device)
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        count = batch["data"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def four_class_infer(eda_path):
    """
    Inference for four-class EDA zero-shot using CLIP-style model.
    Only uses the EDA features file—no text file or extra processing needed.
    """
    relevant_features_eda = [
        'ku_eda','sk_eda','dynrange','slope','variance','entropy','insc',
        'fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR',
        'sumAmpSCR','sumRespSCR'
    ]
    eda_df = csv_read(eda_path)
    eda_df = eda_df.fillna(0)
    eda_df.replace([np.inf, -np.inf], 0, inplace=True)

    input_dim = len(relevant_features_eda)
    output_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_eda_four_class_model.pt"   # Set your model path

    # Prepare test matrices
    X_test = eda_df[relevant_features_eda]
    test_y = eda_df['four_class'].to_list()  # Column must match your training target

    # Define the 4 class label descriptions (ensure same as training)
    labels = [
        "High Arousal and Low Valence: Strong physical reaction with intense negative emotions like anger, fear, frustration, anxiety, or panic.",
        "High Arousal and High Valence: Strong physical activation with energizing positive emotions like joy, enthusiasm, exhilaration, or amusement.",
        "Low Arousal and Low Valence: Low energy with subdued negative emotions like sadness, boredom, tiredness, disappointment, or hopelessness.",
        "Low Arousal and High Valence: Calm and relaxed with subtle positive emotions like contentment, peace, satisfaction, and mild happiness."
    ]
    # Preprocess labels (same as in training)
    processed_labels = []
    for label in labels:
        lab = lemmatize(label)
        lab = stop_words(lab)
        lab = lowercase(lab)
        lab = punctuations(lab)
        processed_labels.append(lab)

    # Build/Load model
    best_model = CLIPModel(
        mlp_input_dim=input_dim,
        mlp_output_dim=output_dim,
        device=device
    ).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.eval()

    # Encode all class text labels
    encoded_labels = []
    for label in processed_labels:
        text_embeddings = best_model.text_encoder.text_tokens([label])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label = text_embeddings.squeeze(1)
        encoded_labels.append(encoded_label.to(device))

    # Inference prediction loop
    X_test = np.array(X_test)
    pred = []
    for feat in X_test:
        features = best_model.ppg_encoder.get_hidden_embedding(
            torch.from_numpy(feat).float().to(device)
        )
        similarities = [torch.matmul(features, cl.T).item() for cl in encoded_labels]
        predicted_class = int(np.argmax(similarities))
        pred.append(predicted_class)

    # Output metrics
    print("EDA Four-Class Prediction")
    print(f"Prediction: {pred}")
    print(f"True Values: {test_y}")
    print(classification_report(test_y, pred))
    accuracy = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1}")

# Usage example (edit as needed)
# EDA_FourClass(eda_path="your_EDA_features_file.csv")



def main():
    # ==== PATHS: CHANGE as needed! ====
    eda_path = 
    text_file = 
    # ==== FEATURE LIST: CHANGE as needed! ====
    relevant_features_eda = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']

    eda_df = csv_read(eda_path)
    # eda_df = eda_df[eda_df['CMA'] != 'Baseline']  # Uncomment if baseline ignore relevant

    text_data = pd.read_csv(text_file)
    text_data = preprcosess(text_data)

    # Map text to EDA rows
    p = {}
    pi = list(set(text_data['Participant ID'].tolist()))
    vi = list(set(text_data['Video ID'].tolist()))
    for i in pi:
        temp = {}
        df_pi = text_data[text_data['Participant ID'] == i]
        for j in vi:
            df_vi = df_pi[df_pi['Video ID'] == j]['Text']
            if len(df_vi) > 0:
                temp[j] = df_vi.tolist()[0]
            else:
                temp[j] = ""
        p[i] = temp
    eda_df['Text'] = pd.NA
    for index, row in tqdm(eda_df.iterrows(), total=len(eda_df)):
        i = row['Participant ID']
        j = row['Video ID']
        try:
            eda_df.at[index, 'Text'] = p[i][int(str(j)[-1])]
        except Exception:
            eda_df.at[index, 'Text'] = ""
    eda_df = eda_df.reset_index(drop=True)

    target_column = 'four_class'  # CHANGE IF NEEDED!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    input_dim = len(relevant_features_eda)
    output_dim = 4  # 4-class classification

    fet = relevant_features_eda

    train_data, test_data = train_test_split(
        eda_df, test_size=0.2, random_state=seed, stratify=eda_df[target_column]
    )
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    X_train = train_data[fet]
    X_text = train_data['Text']
    X_test = test_data[fet]
    X_train_processed = windowed_preprocess(X_train)
    X_test_processed = windowed_preprocess(X_test)
    train_y = train_data[target_column].to_list()
    test_y = test_data[target_column].to_list()

    custom_dataset = CustomDataset(X_train_processed, train_y, X_text)
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    model = CLIPModel(mlp_input_dim=input_dim, mlp_output_dim=output_dim, device=device)

    params = [
        {"params": model.text_encoder.parameters(), "lr": 1e-3},
        {"params": model.ppg_encoder.parameters(), "lr": 1e-3},
        {"params": itertools.chain(model.text_projection.parameters()), "lr": 1e-3, "weight_decay": 1e-3}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.8)
    step = "epoch"
    best_loss = float('inf')
    model_path = f"best_eda_four_class_model.pt"

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        model.ppg_train(learning_rate, beta1, beta2, epsilon, train_dataloader)
        train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, step, device)
        model.eval()
        valid_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, step, device)
        avg = valid_loss.avg if valid_loss else float('inf')
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), model_path)
            print("Saved EDA Best Model!")
        lr_scheduler.step(avg)

    labels = [
        "High Arousal and Low Valence: Strong physical reaction with intense negative emotions like anger, fear, frustration, anxiety, or panic.",
        "High Arousal and High Valence: Strong physical activation with energizing positive emotions like joy, enthusiasm, exhilaration, or amusement.",
        "Low Arousal and Low Valence: Low energy with subdued negative emotions like sadness, boredom, tiredness, disappointment, or hopelessness.",
        "Low Arousal and High Valence: Calm and relaxed with subtle positive emotions like contentment, peace, satisfaction, and mild happiness."
    ]
    processed_labels = []
    for label in labels:
        processed_label = lemmatize(label)
        processed_label = stop_words(processed_label)
        processed_label = lowercase(processed_label)
        processed_label = punctuations(processed_label)
        processed_labels.append(processed_label)

    best_model = CLIPModel(mlp_input_dim=input_dim, mlp_output_dim=output_dim, device=device).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.eval()

    encoded_labels = []
    for label in processed_labels:
        text_embeddings = best_model.text_encoder.text_tokens([label])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label = text_embeddings.squeeze(1)
        encoded_labels.append(encoded_label.to(device))

    pred = []
    for i in X_test_processed:
        features = best_model.ppg_encoder.get_hidden_embedding(torch.from_numpy(i).float().to(device))
        similarities = []
        for encoded_label in encoded_labels:
            similarity = torch.matmul(features, encoded_label.T)
            similarities.append(similarity)
        predicted_class = max(enumerate(similarities), key=lambda x: x[1])[0]
        pred.append(predicted_class)

    print(f"Prediction: {pred}")
    print(f"True Values: {test_y}")
    print(classification_report(test_y, pred))
    accuracy = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1}")

# if __name__ == "__main__":
#     main()
