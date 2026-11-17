"""
PyTorch Deep Learning Architectures for M250 Maraging Steel R2 Property Prediction.
Models: CNN1D, BiLSTM, Transformer, CNN+LSTM Hybrid
"""

import torch
import torch.nn as nn
import math


class CNN1DRegressor(nn.Module):
    """
    1D-CNN Regressor: treats input features as a 1D sequence.
    Conv1D(64, k=3) -> BN -> ReLU -> Conv1D(128, k=3) -> BN -> ReLU ->
    GlobalMaxPool -> Dense(64) -> Dropout(0.3) -> Dense(2)
    """
    def __init__(self, n_features, n_outputs=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_outputs)

    def forward(self, x):
        # x: (batch, n_features) -> reshape to (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class BiLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM Regressor: each feature = one time step.
    BiLSTM(128) -> Dropout(0.2) -> BiLSTM(64) -> Dense(32) -> Dense(2)
    """
    def __init__(self, n_features, n_outputs=2):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # last time step
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    """
    Transformer Regressor: self-attention over feature tokens.
    Feature embedding -> Positional Encoding -> MultiHeadAttention(4 heads) ->
    LayerNorm -> FFN(128) -> Dense(2)
    """
    def __init__(self, n_features, n_outputs=2, d_model=64, n_heads=4):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=n_features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(d_model * n_features, 128)
        self.fc2 = nn.Linear(128, n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.n_features = n_features

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, 1) -> embed -> (batch, n_features, d_model)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class CNNLSTMHybrid(nn.Module):
    """
    CNN + LSTM Hybrid: local pattern detection + sequential dependency.
    Conv1D(64) -> MaxPool -> BiLSTM(64) -> Dense(2)
    """
    def __init__(self, n_features, n_outputs=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(max(n_features // 2, 1))
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        pool_out = max(n_features // 2, 1)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, n_outputs)

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        # (batch, 64, pool_out) -> (batch, pool_out, 64) for LSTM
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class PINNRegressor(nn.Module):
    """
    Physics-Informed Neural Network Regressor.
    Standard Feed Forward NN structure but trained with physics constraints.
    Dense(128) -> BN -> ReLU -> Dense(64) -> BN -> ReLU -> Dense(32) -> Dense(n_outputs)
    """
    def __init__(self, n_features, n_outputs=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_outputs)
        )

    def forward(self, x):
        return self.net(x)

def get_model(name, n_features, n_outputs=2):
    """Factory function to get model by name."""
    models = {
        'cnn': CNN1DRegressor,
        'lstm': BiLSTMRegressor,
        'transformer': TransformerRegressor,
        'hybrid': CNNLSTMHybrid,
        'pinn': PINNRegressor,
    }
    if name.lower() not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name.lower()](n_features, n_outputs)
