from TransformerNN import TransformerOnBertEmbeddings,LSTMOnBertEmbeddings
import numpy as np
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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, hamming_loss



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
        # Tokenize the Tweets (no truncation, padding for the whole batch)
        inputs = tokenizer(Tweets, padding=True, truncation=True, max_length=128, return_tensors="pt")

        # Create a DataLoader to handle batching
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=500)

        # List to store hidden states
        all_hidden_states = []

        # Process the data in batches
        for batch in dataloader:
            input_ids, attention_mask = batch

            # Ensure that we're in evaluation mode
            bert_model.eval()

            # No need to compute gradients during inference
            with torch.no_grad():
                # Pass the batch through BERT
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract the hidden states (the last hidden state of the model)
            hidden_states = outputs.last_hidden_state

            # Average the hidden states along the sequence length (dim=1)
            avg_hidden_states = hidden_states.mean(dim=1)

            # Append the averaged hidden states for this batch
            all_hidden_states.append(avg_hidden_states)

        # Concatenate all the batches of hidden states
        all_hidden_states = torch.cat(all_hidden_states, dim=0)

        return all_hidden_states

    def TrainPreparing(self,tweets_embeddings, labels, batch_size,test_size=0.2):

        # Split into train and test sets

        #X_train, X_test, y_train, y_test = train_test_split(tweets_embeddings, labels, test_size=test_size, random_state=42)
        # First split: Split data into training and temporary sets
        X_train, X_temp, y_train, y_temp = train_test_split(tweets_embeddings, labels,
                                                            test_size=(test_size * 2), random_state=42)
        # Second split: Split the temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=42)

        # Convert data to PyTorch tensors
        X_train, X_test, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
        y_train, y_test, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

        # Prepare data loaders
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        val_data = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        return train_loader,test_loader,val_loader,len(labels)

    def BertCNNBuildModel(self,embed_dim,num_classes):
        model = TweetCNN(embed_dim=embed_dim, num_classes=num_classes)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        return model,criterion,optimizer

    def TransformerBuildModel(self,embed_dim,num_classes):
        model = TransformerOnBertEmbeddings(embed_dim=embed_dim, num_classes=num_classes)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        return model,criterion,optimizer

    def LSTMBuildModel(self,embed_dim,lstm_hidden_dim,num_classes):
        model = LSTMOnBertEmbeddings(embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim, num_classes=num_classes)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        return model,criterion,optimizer

    def TrainModel(self, model, criterion, optimizer, num_epochs, train_loader, val_loader, threshold=0.5):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_hamming_losses , train_hamming_scores = [],[]
        val_hamming_losses, val_hamming_scores = [], []

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            all_preds, all_labels = [],[]
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)  # Outputs: [batch_size, num_classes]
                # Compute loss
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # Compute accuracy (multi-label classification)
                preds = (outputs >= threshold).float()  # Predictions based on threshold
                correct_train += (preds == y_batch).all(dim=1).sum().item()  # Correctly classified samples
                total_train += y_batch.size(0)  # Total samples in batch
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train  # Accuracy for the epoch
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            train_hamming_losses.append(hamming_loss(np.vstack(all_labels), np.vstack(all_preds)))
            train_hamming_scores.append(self.hamming_score(np.vstack(all_labels), np.vstack(all_preds)))

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            val_pred, val_labels = [],[]

            with torch.no_grad():
                for X_v, y_v in val_loader:
                    outputs = model(X_v)
                    # Compute loss
                    loss = criterion(outputs, y_v)

                    val_running_loss += loss.item()
                    # Compute accuracy
                    preds = (outputs >= threshold).float()  # Predictions based on threshold
                    correct_val += (preds == y_v).all(dim=1).sum().item()
                    total_val += y_v.size(0)
                    val_pred.append(preds.cpu().numpy())
                    val_labels.append(y_v.cpu().numpy())

            avg_val_loss = val_running_loss / len(val_loader)
            val_accuracy = correct_val / total_val  # Accuracy for the epoch
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            val_hamming_losses.append(hamming_loss(np.vstack(val_labels), np.vstack(val_pred)))
            val_hamming_scores.append(self.hamming_score(np.vstack(val_labels), np.vstack(val_pred)))

            # Print or log progress
            streamlit.write(f"Epoch {epoch + 1}, "
                     f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, "
                     f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
                     f"Training Hamming loss:{train_hamming_losses[-1]:.4f}, Training Hamming Score:{train_hamming_scores[-1]:.4f}, "
                     f"Validation Hamming loss:{val_hamming_losses[-1]:.4f}, Validation Hamming Score:{val_hamming_scores[-1]:.4f}")

        # Return model and metrics
        return model, train_hamming_losses, val_hamming_losses, train_hamming_scores, val_hamming_scores

    def TestModel(self,model,test_loader,labels,threshold=0.5):
        # Testing loop
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [],[]
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                preds = (outputs >= threshold).float()  # Predictions based on threshold
                correct += (preds == y_batch).all(dim=1).sum().item()
                total += y_batch.size(0)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        class_report = classification_report(np.vstack(all_labels), np.vstack(all_preds),
                                             target_names=labels,
                                             zero_division=0)

        streamlit.write(f"Hamming loss:\n{hamming_loss(np.vstack(all_labels), np.vstack(all_preds))}")
        streamlit.write(f"Hamming Acc:\n{self.hamming_score(np.vstack(all_labels), np.vstack(all_preds))}")
        streamlit.write("Classification Report:")
        streamlit.code(class_report)
        return "Test Accuracy:" + str(correct / total)

    def hamming_score(self,y_true, y_pred):
        """
        Computes the Hamming Score, a.k.a. Subset Accuracy, for multi-label classification.
        """
        scores = []
        for true, pred in zip(y_true, y_pred):
            # Intersection divided by the union of true and predicted labels
            if np.sum(np.logical_or(true, pred)) == 0:
                scores.append(1)  # Avoid division by zero when no labels exist
            else:
                scores.append(np.sum(np.logical_and(true, pred)) / np.sum(np.logical_or(true, pred)))
        return np.mean(scores)

    def plot_curves(self,train_losses, val_losses, train_acc, val_acc):
        epochs = range(1, len(train_losses) + 1)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot losses in the first subplot
        ax1.plot(epochs, train_losses, label="Training Loss", color="blue")
        ax1.plot(epochs, val_losses, label="Validation Loss", color="orange")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid()

        # Plot accuracies in the second subplot
        ax2.plot(epochs, train_acc, label="Training Accuracy", color="green")
        ax2.plot(epochs, val_acc, label="Validation Accuracy", color="red")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid()

        # Adjust layout and display the figure in Streamlit
        plt.tight_layout()

        streamlit.pyplot(fig)