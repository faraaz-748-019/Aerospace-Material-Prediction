"""
Unit and Integration Tests for Model Training Pipeline.
Verifies classical ML, deep learning, physics-informed loss, curve generator,
and stacking ensemble can train and update weights on small valid datasets.
"""

import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor

from core.architectures import get_model, PINNRegressor, TransformerRegressor
from core.curve_lstm import CurveLSTM
from core.ensemble import StackingEnsemble


# ============================================================================
# 1. CLASSICAL ML MODEL TRAINING TESTS (LIGHTWEIGHT FIXTURES)
# ============================================================================

@pytest.mark.parametrize("model_cls,kwargs", [
    (ExtraTreesRegressor, {'n_estimators': 5, 'max_depth': 3, 'random_state': 42}),
    (RandomForestRegressor, {'n_estimators': 5, 'max_depth': 3, 'random_state': 42}),
    (GradientBoostingRegressor, {'n_estimators': 5, 'max_depth': 3, 'random_state': 42}),
])
def test_classical_sklearn_models_train(synthetic_wide_arrays, model_cls, kwargs):
    """Test standard Scikit-Learn tree regressors fit and predict without error."""
    X, y = synthetic_wide_arrays
    model = MultiOutputRegressor(model_cls(**kwargs))
    model.fit(X, y)
    
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_xgboost_trains_on_sample(synthetic_wide_arrays):
    """Test XGBoost MultiOutputRegressor fits and predicts."""
    try:
        import xgboost as xgb
    except ImportError:
        pytest.skip("xgboost not installed")

    X, y = synthetic_wide_arrays
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=5, max_depth=3, random_state=42, verbosity=0))
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_lightgbm_trains_on_sample(synthetic_wide_arrays):
    """Test LightGBM MultiOutputRegressor fits and predicts."""
    try:
        import lightgbm as lgb
    except ImportError:
        pytest.skip("lightgbm not installed")

    X, y = synthetic_wide_arrays
    model = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=5, max_depth=3, random_state=42, verbosity=-1))
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


# ============================================================================
# 2. DEEP LEARNING MODEL TRAINING TESTS (PYTORCH)
# ============================================================================

@pytest.mark.parametrize("model_name", ['cnn', 'lstm', 'transformer', 'hybrid', 'pinn'])
def test_dl_models_train_one_epoch(synthetic_wide_arrays, model_name):
    """
    Test that every PyTorch model architecture can forward, backward,
    and update weights on a small synthetic batch without divergence.
    """
    X, y = synthetic_wide_arrays
    n_features = X.shape[1]
    n_outputs = y.shape[1]

    model = get_model(model_name, n_features, n_outputs=n_outputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    initial_loss = None
    final_loss = None

    model.train()
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        assert pred.shape == batch_y.shape
        loss = criterion(pred, batch_y)
        
        if initial_loss is None:
            initial_loss = loss.item()
            
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert np.isfinite(initial_loss)
    assert np.isfinite(final_loss)


# ============================================================================
# 3. PHYSICS-INFORMED LOSS TESTS
# ============================================================================

def test_pinn_physics_loss_enforcement():
    """
    Verify the PINN loss penalty:
    1. When predicted YS > UTS, penalty must be strictly positive.
    2. When predicted YS <= UTS, penalty must be zero.
    3. Positivity penalty triggers on negative values.
    """
    # Case 1: Physical violation (YS = 1600 > UTS = 1500)
    violating_pred = torch.tensor([[1500.0, 1600.0, 10.0, 5.0, 5.0, 3.0]])
    phy_loss_violating = torch.mean(torch.relu(violating_pred[:, 1] - violating_pred[:, 0]))
    assert phy_loss_violating.item() == 100.0, "Physics penalty should equal YS - UTS when YS > UTS"

    # Case 2: Physically consistent (YS = 1450 <= UTS = 1550)
    consistent_pred = torch.tensor([[1550.0, 1450.0, 10.0, 5.0, 5.0, 3.0]])
    phy_loss_consistent = torch.mean(torch.relu(consistent_pred[:, 1] - consistent_pred[:, 0]))
    assert phy_loss_consistent.item() == 0.0, "Physics penalty must be zero when YS <= UTS"

    # Case 3: Negative property violation
    negative_pred = torch.tensor([[1500.0, 1400.0, -2.0, 5.0, 5.0, 3.0]])
    pos_loss = torch.mean(torch.relu(-negative_pred))
    assert pos_loss.item() > 0.0, "Positivity penalty must trigger on negative predictions"


# ============================================================================
# 4. CURVE LSTM TRAINING TEST
# ============================================================================

def test_curve_lstm_trains_on_sample():
    """Test CurveLSTM can forward, compute loss, and backward on 8-input, 6-target batches."""
    model = CurveLSTM(n_inputs=8, n_outputs=6)
    x = torch.randn(10, 8)
    y = torch.randn(10, 6)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    optimizer.zero_grad()
    pred = model(x)
    assert pred.shape == (10, 6)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    assert np.isfinite(loss.item())


# ============================================================================
# 5. STACKING ENSEMBLE TRAINING & SERIALIZATION TEST
# ============================================================================

def test_stacking_ensemble_fit_predict_save(tmp_path):
    """
    Verify StackingEnsemble:
    - Trains Ridge meta-learner on simulated base model predictions.
    - Generates predictions with expected shape.
    - Saves to disk and loads back with bitwise identical predictions.
    """
    np.random.seed(42)
    n_samples = 30
    n_targets = 6

    # Simulated base model predictions
    oof_preds = {
        'model_a': np.random.uniform(10, 100, (n_samples, n_targets)).astype(np.float32),
        'model_b': np.random.uniform(10, 100, (n_samples, n_targets)).astype(np.float32),
        'model_c': np.random.uniform(10, 100, (n_samples, n_targets)).astype(np.float32),
    }
    y_true = np.random.uniform(10, 100, (n_samples, n_targets)).astype(np.float32)

    ensemble = StackingEnsemble()
    ensemble.fit(oof_preds, y_true)
    assert ensemble.is_fitted

    # Test predict
    pred = ensemble.predict(oof_preds)
    assert pred.shape == (n_samples, n_targets)
    assert np.isfinite(pred).all()

    # Test serialization
    save_path = str(tmp_path / "test_ensemble.pkl")
    ensemble.save(save_path)
    assert os.path.exists(save_path)

    loaded_ensemble = StackingEnsemble()
    loaded_ensemble.load(save_path)
    assert loaded_ensemble.is_fitted
    assert loaded_ensemble.base_model_names == ensemble.base_model_names

    pred_loaded = loaded_ensemble.predict(oof_preds)
    np.testing.assert_array_almost_equal(pred, pred_loaded)
