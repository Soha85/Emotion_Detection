import torch.nn as nn
import torch


class TweetCNN(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(TweetCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Adjusted fully connected layer input size to match flattened size
        self.fc = nn.Linear(64 * 192, num_classes)  # Output size matches number of classes (11)
        self.dropout = nn.Dropout(0.1)

    # Ensure the forward method is correctly defined here
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, embed_dim]

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))
