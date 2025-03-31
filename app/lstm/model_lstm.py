import torch
import torch.nn as nn

class LinkPredictorLSTM(nn.Module):
    def __init__(self, embedding_dim=768, hidden_size=256):
        super().__init__()
        self.input_dim = embedding_dim * 3
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: (1, batch, hidden)
        out = self.dropout(hn.squeeze(0))
        return torch.sigmoid(self.fc(out))
