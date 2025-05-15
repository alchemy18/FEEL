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
from sklearn.model_selection import LeaveOneOut
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
  temp  = data.lower()
  return temp

def stop_words(text):
  stop_words_set = set(stopwords.words('english'))
  word_tokens = word_tokenize(text)
  filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words_set]
  return " ".join(filtered_sentence)

def punctuations(data):
  no_punct=[words for words in data if words not in string.punctuation]
  words_wo_punct=''.join(no_punct)
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

def balanced_df(eda_df):
    filtered_rows = []
    for index, row in tqdm(eda_df.iterrows()):
        if row['arousal_category'] == 1:
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    balance_a_eda_df = pd.concat([eda_df, filtered_df], ignore_index=True)
    balance_a_eda_df  = balance_a_eda_df.reset_index(drop=True)
    return balance_a_eda_df

class Text_Encoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name).to(torch.device("cuda"))
            
        for p in self.model.parameters():
            p.requires_grad = trainable
        
        for p in self.model.parameters():
            p.requires_grad = trainable
        


    def text_tokens(self,batch):
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

            # Get the embeddings for the [CLS] token (the first token)
            cls_embeddings = embeddings[:, 0, :]

            # Alternatively, mean pooling the token embeddings to get sentence-level embeddings
            sentence_embeddings = torch.mean(embeddings, dim=1)

            text_embeddings.append(sentence_embeddings)

        return text_embeddings
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=100,  # Change projection_dim to 100
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
        self.text  = text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {}
        item['data'] = torch.from_numpy(self.data[index]).float()
        item['target'] = self.targets[index]
        item['text'] = self.text[index]
        return item
    
def windowed_preprocess(train_df, test_df):
    train = []
    test = []
    for index, row in train_df.iterrows():
        # Convert each row (Series) into a list of NumPy arrays
        row_as_list1 = np.array([np.array(value) for value in row.to_numpy()])
        train.append(row_as_list1)

    for index, row in test_df.iterrows():
        # Convert each row (Series) into a list of NumPy arrays
        row_as_list2 = np.array([np.array(value) for value in row.to_numpy()])
        test.append(row_as_list2)

    return train, test

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
        self.eda_encoder = CustomMLP(mlp_input_dim, mlp_output_dim).to(device)
        self.text_projection = ProjectionHead(embedding_dim=768, projection_dim=100).to(device)
        self.device = device
        
    def eda_train(self,learning_rate, beta1, beta2 , epsilon , train_dataloader):
        criterion= nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam( self.eda_encoder.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        optimizer.zero_grad()
        for batch in tqdm(train_dataloader):
            data = batch['data']
            target = batch['target']

            data, target = data.to(self.device), target.to(self.device)  # Move data and target to GPU
            optimizer.zero_grad()

            output = self.eda_encoder(data)

            target = target.unsqueeze(1).float()
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

    def forward(self, batch):
        data_values = batch['data']
        text_values = batch['text']

        eda_embeddings = self.eda_encoder.get_hidden_embedding(data_values).to(self.device)
        text_embeddings  = self.text_encoder.text_tokens(batch["text"])#.to(self.device)
        text_embeddings = self.text_projection(text_embeddings)#.to(self.device)
        text_embeddings = torch.stack(text_embeddings).to(self.device)
        eda_tensor = eda_embeddings

        # Calculating the Loss
        text_embeddings = text_embeddings.squeeze(1)
        logits = torch.matmul(text_embeddings, eda_tensor.T)
        # print("Logits ",logits.shape)
        eda_similarity = torch.matmul(eda_tensor, eda_tensor.T)
        # print("eda_similarity ",eda_similarity.shape)
        input_size = text_embeddings.size(0) * text_embeddings.size(1)
        # print("Text Embedding ",text_embeddings.shape)
        texts_similarity = torch.matmul(text_embeddings, text_embeddings.T)
        targets = F.softmax((eda_similarity + texts_similarity) / 2, dim=-1) 
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device=torch.device("cuda")):
    # device = torch.device("cuda")
    model.to(device)
    loss_meter = AvgMeter()
    # print(f"train_loader: {train_loader}")
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        for key in batch.keys():
            if key != "text":
                # print(key)
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


def Arousal(eda_path):
    # loading eda_df
    relevant_features_eda = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    eda_df = csv_read(eda_path)
    identifiers = ['Participant ID']
    eda_df = eda_df.fillna(0)
    eda_df.replace([np.inf, -np.inf], 0, inplace=True)

    input_dim =len(relevant_features_eda)
    output_dim = 1
    device = torch.device("cuda")
    
    model_path = "/mnt/drive/home/pragyas/Pragya/Benchmarking_IMWUT/CLSP_Models/Zer-shot/best_eda_arousal_category_42.pt"
    X_test = eda_df[relevant_features_eda]
    test_y = eda_df['arousal_category'].to_list()
    label1 = "Data of High and more arousal_category"
    label2 = "Data of Low and less arousal_category"
    label1 = lemmatize(label1)
    label1 = stop_words(label1)
    label1 = lowercase(label1)
    label1 = punctuations(label1)

    label2 = lemmatize(label2)
    label2 = stop_words(label2)
    label2 = lowercase(label2)
    label2 = punctuations(label2)

    

    try :
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        best_model = CLIPModel(mlp_input_dim = input_dim, mlp_output_dim = output_dim, device = device).to(device)
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        best_model.eval()
        pred = []
        
        text_embeddings  = best_model.text_encoder.text_tokens([label1])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label1 = text_embeddings.squeeze(1)
        encoded_label1 = encoded_label1.to(device)
        

        text_embeddings  = best_model.text_encoder.text_tokens([label2])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label2 = text_embeddings.squeeze(1)
        encoded_label2 = encoded_label2.to(device)
        # print(encoded_label2, encoded_label1)
        
        X_test = np.array(X_test)
        

        for j in X_test:
            if isinstance(j, np.ndarray):
                features = best_model.eda_encoder.get_hidden_embedding(torch.from_numpy(j).float().to(device))
                features.to(device)
                a = torch.matmul(features, encoded_label1.T) 
                b = torch.matmul(features, encoded_label2.T)
                # print("a:",a)
                # print("b:",b)
                value = 1 if a > b else 0
                
                pred.append(value)
            else:
                print(f"Skipping invalid entry: {j}")
            
        # print(f"Prediction: {pred}")
        # print(f"True Values: {test_y}")
        # print(classification_report(test_y, pred))

        accuracy = accuracy_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        print(f"EDA Arousal")
        print(f"accuracy {accuracy}")
        print(f"f1 {f1}")


    
    except Exception as e:
        print(e)

def Valence(eda_path):
    # loading eda_df
    relevant_features_eda = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    eda_df = csv_read(eda_path)
    identifiers = ['Participant ID']
    eda_df = eda_df.fillna(0)
    eda_df.replace([np.inf, -np.inf], 0, inplace=True)

    input_dim =len(relevant_features_eda)
    output_dim = 1
    device = torch.device("cuda")
    
    model_path = "/mnt/drive/home/pragyas/Pragya/Benchmarking_IMWUT/CLSP_Models/Zer-shot/best_eda_valence_category_42.pt"
    X_test = eda_df[relevant_features_eda]
    test_y = eda_df['valence_category'].to_list()
    label1 = "Data of High and more positive emotion"
    label2 = "Data of Low and less negative emotion"
    label1 = lemmatize(label1)
    label1 = stop_words(label1)
    label1 = lowercase(label1)
    label1 = punctuations(label1)

    label2 = lemmatize(label2)
    label2 = stop_words(label2)
    label2 = lowercase(label2)
    label2 = punctuations(label2)
    try :
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        best_model = CLIPModel(mlp_input_dim = input_dim, mlp_output_dim = output_dim, device = device).to(device)
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        best_model.eval()
        pred = []
        text_embeddings  = best_model.text_encoder.text_tokens([label1])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label1 = text_embeddings.squeeze(1)
        encoded_label1 = encoded_label1.to(device)

        text_embeddings  = best_model.text_encoder.text_tokens([label2])
        text_embeddings = best_model.text_projection(text_embeddings)
        text_embeddings = torch.stack(text_embeddings).to(device)
        encoded_label2 = text_embeddings.squeeze(1)
        encoded_label2 = encoded_label2.to(device)
        X_test = np.array(X_test)

        for j in X_test:
            if isinstance(j, np.ndarray):
                features = best_model.eda_encoder.get_hidden_embedding(torch.from_numpy(j).float().to(device))
                features.to(device)
                a = torch.matmul(features, encoded_label1.T) 
                b = torch.matmul(features, encoded_label2.T)
                # print(a)
                # print(b)
                value = 1 if a > b else 0
                pred.append(value)
            else:
                print(f"Skipping invalid entry: {j}")
            
        # print(f"Prediction: {pred}")
        # print(f"True Values: {test_y}")
        # print(classification_report(test_y, pred))

        accuracy = accuracy_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        print(f"EDA Valence")
        print(f"accuracy {accuracy}")
        print(f"f1 {f1}")

    except Exception as e:
        print(e)




# eda_path = "/mnt/drive/home/pragyas/Pragya/EEVR_Extension/zero-shot/emowear/emowear_EDA_scaled.csv"

# test_arousal(eda_path = eda_path)

# test_valence(eda_path = eda_path)

# eda_fet_path = "/mnt/drive/home/pragyas/Pragya/Benchmarking_IMWUT/Datasets/UBFC/Features_EDA.csv"
# Arousal(eda_path= eda_fet_path)
# Valence(eda_path=eda_fet_path)
