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
        if x.dim() == 2:
            # If input is [batch_size, embed_dim], assume CLS embeddings only
            assert x.shape[1] == 768, f"Expected embedding dimension of 768, but got {x.shape[1]}"
            cls_output = x  # Shape is already [batch_size, embed_dim]
        elif x.dim() == 3:
            # If input is [batch_size, seq_len, embed_dim], transpose for TransformerEncoder
            assert x.shape[2] == 768, f"Expected embedding dimension of 768, but got {x.shape[2]}"
            x = x.transpose(0, 1)  # Transform to [seq_len, batch_size, embed_dim]
            x = self.transformer_encoder(x)
            cls_output = x[0, :, :]  # Extract CLS token, shape [batch_size, embed_dim]
        else:
            raise ValueError(
                "Unexpected input shape: should be [batch_size, embed_dim] or [batch_size, seq_len, embed_dim]")

            # Apply dropout and the fully connected layer
        x = self.dropout(cls_output)
        return torch.sigmoid(self.fc(x))  # For multi-label classification


class LSTMOnBertEmbeddings(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(LSTMOnBertEmbeddings, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, num_classes)  # 128*2 for bidirectional
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # LSTM returns: (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate the final hidden states from both directions
        final_hidden_state = torch.cat((h_n[-2], h_n[-1]), dim=1)  # Shape: [batch_size, 256]

        # Apply dropout and pass through the fully connected layer
        x = self.dropout(final_hidden_state)
        return torch.sigmoid(self.fc(x))  # Output shape: [batch_size, num_classes]
