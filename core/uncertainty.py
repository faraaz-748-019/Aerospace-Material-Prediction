"""
Monte Carlo Dropout Uncertainty Estimation.
Runs 50 forward passes with dropout enabled to estimate prediction uncertainty.
"""

import numpy as np
import torch


def mc_dropout_predict(model, x_tensor, n_passes=50, device='cpu'):
    """
    Run Monte Carlo Dropout inference.
    
    Args:
        model: PyTorch model with dropout layers
        x_tensor: input tensor (1, n_features) or (batch, n_features)
        n_passes: number of forward passes (default 50)
        device: 'cpu' or 'cuda'
    
    Returns:
        mean_pred: mean prediction (n_outputs,)
        std_pred: standard deviation (n_outputs,)
        all_preds: all predictions (n_passes, n_outputs)
        min_guaranteed: mean - 2*std (minimum guaranteed values)
    """
    model.to(device)
    model.train()  # Keep dropout ON

    x_tensor = x_tensor.to(device)
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(0)

    all_preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            # Enable dropout by keeping model in train mode
            # but disable gradient computation
            pred = model(x_tensor)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.array(all_preds).squeeze()  # (n_passes, n_outputs)
    if all_preds.ndim == 1:
        all_preds = all_preds.reshape(-1, 1)

    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    min_guaranteed = mean_pred - 2 * std_pred

    return mean_pred, std_pred, all_preds, min_guaranteed


def mc_dropout_curve(model, x_tensor, n_passes=50, device='cpu'):
    """
    MC Dropout for curve generator model.
    Returns mean and std for curve parameter predictions.
    """
    return mc_dropout_predict(model, x_tensor, n_passes, device)
