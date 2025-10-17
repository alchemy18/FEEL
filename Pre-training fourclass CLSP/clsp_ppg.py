import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm.autonotebook import tqdm
import itertools
import random 
import os
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def csv_read(path):
    df = pd.read_csv(path)
    return df

def lowercase(data):
    temp = data.lower()
    return temp

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
                
            # Tokenize and get embeddings
            encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(torch.device("cuda"))
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Extract embeddings from the last hidden state
            embeddings = model_output.last_hidden_state
            sentence_embeddings = torch.mean(embeddings, dim=1)
            text_embeddings.append(sentence_embeddings)

        return text_embeddings

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=100,
        dropout=0.1
    ):
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
        super(CustomMLP, self).__init__()

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
        text = f"{self.name}: {self.avg:.4f}"
        return text

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
        text_values = batch['text']

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


def four_class_infer(ppg_path):
    # Feature columns, should match training
    fet= ['BPM', 'IBI', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    ppg_df = csv_read(ppg_path)
    ppg_df['IBI'] = 0
    ppg_df = ppg_df.fillna(0)
    ppg_df.replace([np.inf, -np.inf], 0, inplace=True)
    input_dim = len(fet)
    output_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_ppg_four_class_model.pt"  # use your actual model path

    X_test = ppg_df[fet]
    test_y = ppg_df['four_class'].to_list()  # this column must match your training target

    # Four-class label texts (must be the same as used in training)
    labels = [
        "High Arousal and Low Valence: Strong physical reaction with intense negative emotions like anger, fear, frustration, anxiety, or panic.",
        "High Arousal and High Valence: Strong physical activation with energizing positive emotions like joy, enthusiasm, exhilaration, or amusement.",
        "Low Arousal and Low Valence: Low energy with subdued negative emotions like sadness, boredom, tiredness, disappointment, or hopelessness.",
        "Low Arousal and High Valence: Calm and relaxed with subtle positive emotions like contentment, peace, satisfaction, and mild happiness."
    ]
    # Preprocess label texts
    processed_labels = []
    for label in labels:
        lab = lemmatize(label)
        lab = stop_words(lab)
        lab = lowercase(lab)
        lab = punctuations(lab)
        processed_labels.append(lab)

    # Load model
    best_model = CLIPModel(mlp_input_dim=input_dim, mlp_output_dim=output_dim, device=device).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.eval()

    # Encode label texts
    encoded_labels = []
    for label in processed_labels:
        text_embeddings = best_model.text_encoder.text_tokens([label])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label = text_embeddings.squeeze(1)
        encoded_labels.append(encoded_label.to(device))

    # Prediction loop
    X_test = np.array(X_test)
    pred = []
    for j in X_test:
        features = best_model.ppg_encoder.get_hidden_embedding(torch.from_numpy(j).float().to(device))
        similarities = [torch.matmul(features, enc.T).item() for enc in encoded_labels]
        predicted_class = int(np.argmax(similarities))
        pred.append(predicted_class)

    print("PPG Four-Class Prediction")
    print(f"Prediction: {pred}")
    print(f"True Values: {test_y}")
    print(classification_report(test_y, pred))
    accuracy = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1}")

# Example usage:
# PPG_FourClass(ppg_path="your_ppg_features_file.csv")
