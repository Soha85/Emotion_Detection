import torch.nn as nn
import torch


class TweetCNN(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(TweetCNN, self).__init__()

        # Define CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.dropout = nn.Dropout(0.5)

        # Calculate the flattened dimension after convolutions and pooling
        flattened_dim = 128 * (embed_dim // 4)
        self.fc = nn.Linear(flattened_dim, num_classes)

        # Output layer for multi-label classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input shape is [batch_size, channels, embed_dim]
        x = x.unsqueeze(1)  # Add channel dimension (batch_size, 1, 768)

        # Convolutional and pooling layers
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        # x = torch.relu(self.conv2(x))
        # x = self.pool2(x)

        x = self.dropout(x)  # Apply dropout after pooling

        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer and output
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid for multi-label output

        return x
