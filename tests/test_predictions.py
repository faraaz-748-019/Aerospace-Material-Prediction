"""
Unit and Integration Tests for Model Predictions and Inference.
Tests serialized model loading, prediction output shapes, finite values,
feature dimension mismatch error handling, Monte Carlo Dropout, and curve reconstruction.
"""

import os
import pytest
import numpy as np
import joblib
import torch

from core.architectures import get_model
from core.uncertainty import mc_dropout_predict, mc_dropout_curve
from core.curve_generator import reconstruct_r2_curve, generate_smooth_curve
from core.ensemble import StackingEnsemble


# ============================================================================
# 1. SERIALIZED MODEL LOADING AND INFERENCE TESTS
# ============================================================================

@pytest.mark.parametrize("model_file", [
    'extratrees.pkl', 'randomforest.pkl', 'gradientboosting.pkl', 'xgboost.pkl', 'lightgbm.pkl'
])
def test_saved_classical_models_load_and_predict(real_models_dir, model_file):
    """Verify serialized classical ML model files can be loaded and produce finite predictions."""
    path = os.path.join(real_models_dir, 'ml', model_file)
    if not os.path.exists(path):
        pytest.skip(f"Model file {model_file} not found")

    model = joblib.load(path)
    
    # 14 input features matching feature_cols
    sample_input = np.array([[
        1617.0, 1573.0, 75.0, 10.0, 5.0, 2.0, 2.5,   # R0
        1560.0, 1518.0, 71.0, 9.0, 5.3, 2.1, 3.2    # R1
    ]], dtype=np.float32)

    pred = model.predict(sample_input)
    assert pred.shape == (1, 6), f"Expected shape (1, 6), got {pred.shape}"
    assert np.isfinite(pred).all(), f"Model {model_file} produced non-finite predictions"


@pytest.mark.parametrize("dl_file,arch_name", [
    ('cnn.pt', 'cnn'),
    ('lstm.pt', 'lstm'),
    ('hybrid.pt', 'hybrid'),
    ('pinn.pt', 'pinn'),
])
def test_saved_dl_models_load_and_predict(real_models_dir, dl_file, arch_name):
    """Verify saved PyTorch DL models can be loaded and produce valid output shapes."""
    path = os.path.join(real_models_dir, 'dl', dl_file)
    if not os.path.exists(path):
        pytest.skip(f"DL model checkpoint {dl_file} not found")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert 'model_state' in checkpoint
    assert 'n_features' in checkpoint

    n_features = checkpoint['n_features']
    model = get_model(arch_name, n_features, n_outputs=6)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    sample_input = torch.randn(1, n_features)
    with torch.no_grad():
        out = model(sample_input)

    assert out.shape == (1, 6)
    assert torch.isfinite(out).all()


def test_saved_stacking_ensemble_predicts(real_models_dir):
    """Verify pre-trained stacking ensemble loads and predicts."""
    path = os.path.join(real_models_dir, 'ml', 'stacking_ensemble.pkl')
    if not os.path.exists(path):
        pytest.skip("stacking_ensemble.pkl not found")

    ensemble = StackingEnsemble()
    ensemble.load(path)
    assert ensemble.is_fitted
    assert len(ensemble.base_model_names) > 0

    # Build mock prediction dictionary for base models
    mock_preds = {
        name: np.ones((2, 6), dtype=np.float32) * 1400.0
        for name in ensemble.base_model_names
    }

    pred = ensemble.predict(mock_preds)
    assert pred.shape == (2, 6)
    assert np.isfinite(pred).all()


# ============================================================================
# 2. FEATURE COUNT MISMATCH & ERROR HANDLING
# ============================================================================

def test_classical_model_rejects_wrong_feature_dimension(real_models_dir):
    """Verify that providing mismatched feature count (e.g. 5 instead of 14) raises an error."""
    path = os.path.join(real_models_dir, 'ml', 'xgboost.pkl')
    if not os.path.exists(path):
        pytest.skip("xgboost.pkl not found")

    model = joblib.load(path)
    wrong_input = np.random.randn(1, 5).astype(np.float32)
    
    with pytest.raises(Exception):
        model.predict(wrong_input)


def test_dl_model_rejects_wrong_input_tensor_dimension():
    """Verify PyTorch models with fixed feature dimensions (PINN, Transformer) raise RuntimeError on mismatched inputs."""
    pinn = get_model('pinn', n_features=14, n_outputs=6)
    wrong_tensor = torch.randn(1, 8)  # 8 instead of 14
    
    with pytest.raises(RuntimeError):
        pinn(wrong_tensor)

    transformer = get_model('transformer', n_features=14, n_outputs=6)
    with pytest.raises(RuntimeError):
        transformer(wrong_tensor)


def test_stacking_ensemble_raises_on_unfitted():
    """Verify StackingEnsemble raises RuntimeError when predict is called before fit."""
    ensemble = StackingEnsemble()
    with pytest.raises(RuntimeError, match="Ensemble not fitted"):
        ensemble.predict({'model_a': np.zeros((1, 6))})


def test_stacking_ensemble_raises_on_missing_base_model():
    """Verify StackingEnsemble raises KeyError with informative message if a base model is missing."""
    ensemble = StackingEnsemble()
    oof = {
        'model_a': np.zeros((10, 2)),
        'model_b': np.zeros((10, 2))
    }
    ensemble.fit(oof, np.zeros((10, 2)))
    
    # Missing 'model_b'
    incomplete_preds = {'model_a': np.zeros((2, 2))}
    with pytest.raises(KeyError, match="Missing base model prediction for 'model_b'"):
        ensemble.predict(incomplete_preds)


# ============================================================================
# 3. MONTE CARLO DROPOUT UNCERTAINTY QUANTIFICATION
# ============================================================================

def test_mc_dropout_predict_statistical_properties():
    """
    Test Monte Carlo Dropout:
    - Runs 50 stochastic passes.
    - Computes mean, std, and min_guaranteed (mean - 2*std).
    - Checks that std >= 0 and min_guaranteed <= mean.
    - Confirms that active dropout generates non-zero variance.
    """
    model = get_model('cnn', n_features=14, n_outputs=6)
    x = torch.randn(1, 14)

    mean_p, std_p, all_preds, min_guar = mc_dropout_predict(model, x, n_passes=50)

    assert mean_p.shape == (6,)
    assert std_p.shape == (6,)
    assert all_preds.shape == (50, 6)
    assert min_guar.shape == (6,)

    # Statistical invariants
    assert (std_p >= 0.0).all(), "Standard deviation must be non-negative"
    assert (min_guar <= mean_p).all(), "Minimum guaranteed must be <= expected mean"
    np.testing.assert_allclose(min_guar, mean_p - 2 * std_p, atol=1e-5)

    # Dropout must produce variance across stochastic passes
    assert np.any(std_p > 0.0), "Dropout should introduce variance across forward passes"


# ============================================================================
# 4. CONTINUOUS STRESS-STRAIN CURVE GENERATION TESTS
# ============================================================================

def test_reconstruct_r2_curve_outputs_and_physical_linearity(synthetic_curve_params):
    """
    Test continuous curve reconstruction:
    - Exactly 200 points generated.
    - Strain is strictly monotonic (increasing).
    - Initial linear elastic slope strictly adheres to Young's Modulus E = 190 GPa.
    """
    strain, stress = reconstruct_r2_curve(synthetic_curve_params, n_points=200)

    assert len(strain) == 200
    assert len(stress) == 200
    assert np.isfinite(strain).all()
    assert np.isfinite(stress).all()

    # Monotonicity check
    strain_diffs = np.diff(strain)
    assert (strain_diffs > 0).all(), "Strain values along curve must be strictly monotonically increasing"

    # Elastic region linearity check: E = stress / strain ~ 190,000 MPa (190 GPa)
    # Check at the first non-zero point
    first_non_zero_idx = 1
    slope = stress[first_non_zero_idx] / strain[first_non_zero_idx]
    assert abs(slope - 190000.0) < 1900.0, f"Elastic slope ({slope}) deviates by > 1% from 190 GPa Young's Modulus"


def test_generate_smooth_curve_handles_unsorted_and_duplicate_strains():
    """Verify generate_smooth_curve handles unsorted or duplicate strain values without crashing."""
    unsorted_strains = [0.0, 0.010, 0.005, 0.010, 0.015]
    stresses = [0.0, 1500.0, 950.0, 1500.0, 1400.0]

    s_strain, s_stress = generate_smooth_curve(unsorted_strains, stresses, n_points=50)
    assert len(s_strain) == 50
    assert len(s_stress) == 50
    # Must be sorted
    assert (np.diff(s_strain) >= 0).all()


def test_generate_smooth_curve_under_two_points():
    """Verify edge case with fewer than 2 points returns original coordinates safely."""
    strains = [0.005]
    stresses = [1000.0]
    out_strain, out_stress = generate_smooth_curve(strains, stresses)
    assert out_strain == strains
    assert out_stress == stresses


# ============================================================================
# 5. PDF REPORT GENERATOR & SHAP EXPLAINABILITY TESTS
# ============================================================================

def test_generate_pdf_report_output_file(tmp_path):
    """Verify generate_report creates a valid, non-empty PDF file with header %PDF."""
    from core.report_generator import generate_report
    output_pdf = str(tmp_path / "test_report.pdf")
    
    mock_data = {
        'r0_inputs': {'R0_UTS': 1617.0, 'R0_YS': 1573.0, 'R0_K1c': 75.0, 'R0_Elongation': 10.0},
        'r1_inputs': {'R1_UTS': 1560.0, 'R1_YS': 1518.0},
        'predicted_uts': 1448.0,
        'predicted_ys': 1366.0,
        'confidence_uts': 12.0,
        'confidence_ys': 15.0,
        'min_guaranteed_uts': 1436.0,
        'min_guaranteed_ys': 1351.0,
        'degradation_uts': 10.4,
        'degradation_ys': 13.1,
        'model_name': 'XGBoost (Tested)'
    }

    generate_report(output_pdf, mock_data)

    assert os.path.exists(output_pdf)
    assert os.path.getsize(output_pdf) > 2000
    with open(output_pdf, 'rb') as f:
        header = f.read(5)
        assert header == b'%PDF-', "Generated file must be a valid PDF"


def test_shap_explainer_computes_importance(real_models_dir):
    """Verify compute_shap_values calculates valid feature importance dictionary."""
    from core.shap_explainer import compute_shap_values, format_feature_name
    
    # Test feature name formatting
    assert format_feature_name("R0_UTS") == "R0 Uts"
    assert format_feature_name("Repair_Location_Center") == "Repair Location Center"

    # Test SHAP on trained model
    xgb_path = os.path.join(real_models_dir, 'ml', 'xgboost.pkl')
    if not os.path.exists(xgb_path):
        pytest.skip("xgboost.pkl not found for SHAP test")

    model = joblib.load(xgb_path)
    X_sample = np.random.uniform(10.0, 1600.0, size=(10, 14)).astype(np.float32)
    feature_names = [f"F_{i}" for i in range(14)]

    # Compute SHAP for target 0 (UTS)
    shap_vals, importance = compute_shap_values(model, X_sample, feature_names=feature_names, target_idx=0)

    assert importance is not None
    assert len(importance) == 14
    for feat, imp in importance.items():
        assert imp >= 0.0, f"Importance for {feat} must be non-negative"
        assert np.isfinite(imp)

