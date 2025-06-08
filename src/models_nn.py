import torch
from torch import nn

def _last_hidden(h):
    """Handle LSTM (h, c) tuple vs GRU tensor, return last layer hidden."""
    if isinstance(h, tuple):
        h = h[0]
    return h[-1]


class RNNClassifier(nn.Module):
    def __init__(self, cell="gru", hidden_dim=64, num_layers=1):
        super().__init__()
        cell_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[cell]
        self.rnn = cell_cls(
            input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, 200] -> [batch, 200, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, h = self.rnn(x)
        out = self.fc(_last_hidden(h))
        return out.squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, 200] -> [batch, 200, 1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.proj(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.fc(pooled).squeeze(-1)
