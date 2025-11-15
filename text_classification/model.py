import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import PositionalEncoding


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, num_classes=9, dropout=0.1, pad_idx=0, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x == 0)  # padding mask
        emb = self.embed(x)  # (batch, seq_len, d_model)
        emb = self.pos(emb)
        # transformer expects (seq_len, batch, d_model)
        out = self.transformer_encoder(emb.transpose(0,1), src_key_padding_mask=mask)
        out = out.transpose(0,1)  # (batch, seq_len, d_model)
        # pool over seq_len
        out = out.transpose(1,2)  # (batch, d_model, seq_len)
        pooled = self.pool(out).squeeze(-1)  # (batch, d_model)
        logits = self.classifier(pooled)
        return logits
