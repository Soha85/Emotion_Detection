import pandas as pd
import streamlit
import torch
from transformers import BertTokenizer, BertModel
import re
import emoji
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from TweetCNN import TweetCNN
import torch.optim as optim
import torch.nn as nn

# Load pre-trained BERT tokenizer and model

class Classify:
    def __init__(self):
        return

    def loadData(self):
        Tweets = pd.read_csv('2018-E-c-En-train.csv')
        labels = list(Tweets.columns[2:])
        return Tweets,labels

    def PreprocessData(self,Tweet):
        Tweet =  re.sub(r"http:\S+", '', Tweet.lower())
        Tweet = emoji.demojize(Tweet)
        Tweet = Tweet.translate(str.maketrans('', '',string.punctuation))
        return Tweet

    def SplitHashTags(self,Tweet):
        return re.findall(r"#(\w+)",Tweet)

    def Bert_Emdedding(self,Tweets):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        inputs = tokenizer(Tweets, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**inputs)
        hidden_states = outputs.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        return embeddings

    def TrainPreparing(self,embeddings, labels,test_size=0.2):
        streamlit.write(embeddings.shape)
        streamlit.write(labels.shape)
        #mlb = MultiLabelBinarizer()
        #labels = mlb.fit_transform(labels)
        streamlit.write(labels.shape)
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state=42)

        # Convert data to PyTorch tensors
        X_train, X_test = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_test.values, dtype=torch.float32)
        y_train, y_test = torch.tensor(y_train.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)

        # Prepare data loaders
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=2)
        return train_loader,test_loader,len(labels)

    def BuildModel(self,embed_dim,num_classes):
        model = TweetCNN(embed_dim=embed_dim, num_classes=num_classes)
        criterion = nn.BCELoss()  #Binary Cross Entropy
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        return model,criterion,optimizer

    def TrainModel(self,model,criterion,optimizer,num_epochs,train_loader):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            streamlit.write(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        return model

    def TestModel(self,model,test_loader):
        # Testing loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                predicted = outputs.round()  # Round predictions to get binary output for each label
                correct += (predicted == y_batch).sum().item()
                total += y_batch.numel()

        accuracy = correct / total
        return "Test Accuracy:" + str(accuracy)

