import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch

class TransformerOnBertEmbeddings(nn.Module):
    def __init__(self, embed_dim, num_classes, nhead=8, num_layers=2):
        super(TransformerOnBertEmbeddings, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        cls_output = x[0, :, :]  # Shape is [batch_size, embed_dim]
        x = self.dropout(cls_output)
        return torch.sigmoid(self.fc(x))
