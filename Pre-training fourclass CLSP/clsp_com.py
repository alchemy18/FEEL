import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from tqdm import tqdm
import itertools
import random
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report



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
    no_punct = [ch for ch in data if ch not in string.punctuation]
    return ''.join(no_punct)

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
    for _, row in df.iterrows():
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
        self.tabular_encoder = CustomMLP(mlp_input_dim, mlp_output_dim).to(device)
        self.text_projection = ProjectionHead(embedding_dim=768, projection_dim=100).to(device)
        self.device = device
    def train_tabular(self, learning_rate, beta1, beta2, epsilon, train_dataloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.tabular_encoder.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        for batch in tqdm(train_dataloader):
            data = batch['data']
            target = batch['target']
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.tabular_encoder(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
    def forward(self, batch):
        data_values = batch['data']
        tabular_embeddings = self.tabular_encoder.get_hidden_embedding(data_values).to(self.device)
        text_embeddings = self.text_encoder.text_tokens(batch["text"])
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(self.device)
        text_embeddings = text_embeddings.squeeze(1)
        logits = torch.matmul(text_embeddings, tabular_embeddings.T)
        tabular_similarity = torch.matmul(tabular_embeddings, tabular_embeddings.T)
        texts_similarity = torch.matmul(text_embeddings, text_embeddings.T)
        targets = F.softmax((tabular_similarity + texts_similarity) / 2, dim=-1)
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
    
def four_class_infer(com_path):
        # Load data and features
    combined_df = pd.read_csv(com_path)
    combined_df = combined_df.fillna(0)
    combined_df.replace([np.inf, -np.inf], 0, inplace=True)
    combined_df['IBI'] = 0

    relevant_features_combined = ['IBI', 'BPM', 'PPG_Rate_Mean', 'HRV_MedianNN', 'HRV_Prc20NN',
    'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF', 'HRV_LFn',
    'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS', 'HRV_PAS',
    'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
    'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
    'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
    'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
    'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
    'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC', 'ku_eda', 'sk_eda', 'dynrange', 'slope',
    'variance', 'entropy', 'insc', 'fd_mean', 'max_scr', 'min_scr', 'nSCR',
    'meanAmpSCR', 'meanRespSCR', 'sumAmpSCR', 'sumRespSCR']

    target_column = 'four_class_eda'

    model_path = "best_com_four_class_model.pt"

    X_test = combined_df[relevant_features_combined]
    test_y = combined_df[target_column].to_list()

    # Four-class zero-shot label descriptions (tailor as needed)
    labels = [
        "High Arousal and Low Valence: Strong physical reaction with intense negative emotions like anger, fear, frustration, anxiety, or panic.",
        "High Arousal and High Valence: Strong physical activation with energizing positive emotions like joy, enthusiasm, exhilaration, or amusement.",
        "Low Arousal and Low Valence: Low energy with subdued negative emotions like sadness, boredom, tiredness, disappointment, or hopelessness.",
        "Low Arousal and High Valence: Calm and relaxed with subtle positive emotions like contentment, peace, satisfaction, and mild happiness."
    ]

    # Preprocessing (using your attached functions)
    processed_labels = []
    for label in labels:
        l = lemmatize(label)
        l = stop_words(l)
        l = lowercase(l)
        l = punctuations(l)
        processed_labels.append(l)

    # Build model and load weights
    input_dim = len(relevant_features_combined)
    output_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = CLIPModel(mlp_input_dim=input_dim, mlp_output_dim=output_dim, device=device).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.eval()

    # Encode all class text labels
    encoded_labels = []
    for pl in processed_labels:
        text_embeddings = best_model.text_encoder.text_tokens([pl])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label = text_embeddings.squeeze(1)
        encoded_labels.append(encoded_label.to(device))

    # Prepare feature data
    X_test_arr = np.array(X_test)
    pred = []
    for j in X_test_arr:
        features = best_model.tabular_encoder.get_hidden_embedding(torch.from_numpy(j).float().to(device))
        similarities = [torch.matmul(features, enc_label.T) for enc_label in encoded_labels]
        predicted_class = int(np.argmax([sim.item() for sim in similarities]))
        pred.append(predicted_class)

    print("Four-Class Prediction")
    print(f"Prediction: {pred}")
    print(f"True Values: {test_y}")
    print(classification_report(test_y, pred))
    accuracy = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1}")

# USAGE:
# four_class_infer(com_path="your_combined_feature_csv.csv", model_path="best_combined_clf.pt", target_column="four_class_eda", relevant_features_combined=[...])
