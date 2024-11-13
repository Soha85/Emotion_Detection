import torch.nn as nn
import torch


class TweetCNN(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(TweetCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Adjust the fully connected layer input size based on the output size after convolutions and pooling
        self.fc = nn.Linear(64 * (embed_dim // 4), num_classes)  # Adjust based on pooling layers
        self.dropout = nn.Dropout(0.5)

        # Output layer for multi-label classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, embed_dim]

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.sigmoid(x)  # Apply sigmoid for multi-label output

        return x
