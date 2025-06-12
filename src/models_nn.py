'''
This module was supposed to be used for the neural network models, but we found
it more efficient to build the models directly in the notebook.
See `notebooks/modeling_nn.ipynb` for the actual implementations.
'''

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
        # x: [batch, 200, 2]
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
        # x: [batch, 200, 2]
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        x = self.proj(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.fc(pooled).squeeze(-1)

class TransformerClassifierWithCustomAttention(nn.Module):
    """
    TransformerClassifier that returns actual attention weights from the self-attention mechanism.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.proj = nn.Linear(1, d_model)
        
        # Create custom transformer layers that can return attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'linear1': nn.Linear(d_model, d_model * 4),
                'dropout': nn.Dropout(dropout),
                'linear2': nn.Linear(d_model * 4, d_model),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout1': nn.Dropout(dropout),
                'dropout2': nn.Dropout(dropout),
            })
            self.layers.append(layer)
        
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x, return_attention=False):
        # x: [batch, seq_len, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
            
        # Handle 2D input by taking first feature
        if x.shape[2] == 2:
            x = x[:, :, :1]
            
        x = self.proj(x)  # [batch, seq_len, d_model]
        
        attention_weights = []
        
        # Pass through each transformer layer
        for layer in self.layers:
            # Self-attention
            attn_output, attn_weights = layer['self_attn'](x, x, x)
            if return_attention:
                attention_weights.append(attn_weights.detach())
            
            # Add & Norm
            x = layer['norm1'](x + layer['dropout1'](attn_output))
            
            # Feed Forward
            ff_output = layer['linear2'](layer['dropout'](torch.relu(layer['linear1'](x))))
            
            # Add & Norm
            x = layer['norm2'](x + layer['dropout2'](ff_output))
        
        # Global average pooling
        pooled = x.mean(dim=1)  # [batch, d_model]
        logits = self.fc(pooled)  # [batch, 1]
        
        if return_attention:
            return logits.squeeze(-1), attention_weights
        else:
            return logits.squeeze(-1)
    
    def get_attention_weights(self, x):
        """Helper method to get attention weights"""
        return self.forward(x, return_attention=True)