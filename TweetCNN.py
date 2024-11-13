import torch.nn as nn
import torch


class TweetCNN(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(TweetCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * (embed_dim // 4), num_classes)  # Adjust based on pooling layers
        self.dropout = nn.Dropout(0.5)

        # Output layer for multi-label classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)  # Ensure input shape [batch_size, embed_dim, seq_len]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.sigmoid(x)  # Apply sigmoid for multi-label output

        return x
