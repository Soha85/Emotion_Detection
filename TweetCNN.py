import torch.nn as nn
import torch
# Define a CNN model
class TweetCNN(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(TweetCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * (embed_dim // 2), num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc(x)
        x = self.softmax(x)
        return x