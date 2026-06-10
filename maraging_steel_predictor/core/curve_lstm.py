"""
LSTM Curve Generator for stress-strain curve parameter prediction.
Trained on Option B dataset: R0 curve params -> R2 curve params.
"""

import torch
import torch.nn as nn


class CurveLSTM(nn.Module):
    """
    LSTM(64) -> Dense(32) -> Dense(6)
    Predicts R2 curve parameters from R0 curve parameters.
    Outputs: yield_strain, uts_strain, fracture_strain, yield_stress, UTS, area_under_curve
    """
    def __init__(self, n_inputs=8, n_outputs=6):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True, num_layers=2, dropout=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, n_inputs) -> (batch, n_inputs, 1)
        x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last time step
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


INPUT_COLS = [
    'R0_elastic_slope_MPa', 'R0_yield_strain', 'R0_uts_strain',
    'R0_fracture_strain', 'R0_yield_stress_MPa', 'R0_UTS_MPa',
    'R0_hardening_ratio', 'R0_area_under_curve'
]

TARGET_COLS = [
    'Target_R2_yield_strain', 'Target_R2_uts_strain', 'Target_R2_fracture_strain',
    'Target_R2_yield_stress_MPa', 'Target_R2_UTS_MPa', 'Target_R2_area_under_curve'
]
