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
import matplotlib.pyplot as plt
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

        return hidden_states.mean(dim=1)

    def TrainPreparing(self,tweets_embeddings, labels, batch_size,test_size=0.2):

        # Split into train and test sets

        #X_train, X_test, y_train, y_test = train_test_split(tweets_embeddings, labels, test_size=test_size, random_state=42)
        # First split: Split data into training and temporary sets
        X_train, X_temp, y_train, y_temp = train_test_split(tweets_embeddings, labels,
                                                            test_size=(test_size * 2), random_state=42)


        # Second split: Split the temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=42)

        # Convert data to PyTorch tensors
        X_train, X_test,X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
        y_train, y_test, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32),  torch.tensor(y_val, dtype=torch.float32)

        # Prepare data loaders
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        val_data = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        return train_loader,test_loader,val_loader,len(labels)

    def BuildModel(self,embed_dim,num_classes):
        model = TweetCNN(embed_dim=embed_dim, num_classes=num_classes)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        return model,criterion,optimizer

    def TrainModel(self,model,criterion,optimizer,num_epochs,train_loader,val_loader):
        train_losses = []
        val_losses = []
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

            # Record average loss over training batches
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase (optional)
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Print or log progress
            streamlit.write(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        return model,train_losses, val_losses

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

    def plot_loss_curves(self,train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        streamlit.pyplot(plt)