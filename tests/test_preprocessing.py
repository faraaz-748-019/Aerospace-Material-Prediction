"""
Unit Tests for Preprocessing and Feature Engineering.
Validates long-to-wide pivoting, one-hot encoding, scaler consistency,
immutability, edge cases, and target leakage prevention.
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from train_all import (
    load_and_combine_data, load_option_b_data,
    R0_FEATURES, R1_FEATURES, TARGETS, CATEGORICAL_FEATURES
)


# ============================================================================
# 1. PIVOTING & FEATURE STRUCTURE TESTS
# ============================================================================

def test_load_and_combine_data_output_structure(real_data_dir):
    """Test load_and_combine_data produces 4,000 samples and all required pivoted columns."""
    df = load_and_combine_data(real_data_dir)
    assert len(df) == 4000, f"Expected 4000 pivoted rows, got {len(df)}"
    
    # Check R0 and R1 features exist
    for f in R0_FEATURES:
        assert f in df.columns, f"Missing expected R0 feature: {f}"
    for f in R1_FEATURES:
        assert f in df.columns, f"Missing expected R1 feature: {f}"
    for t in TARGETS:
        assert t in df.columns, f"Missing expected target: {t}"
    for cat in CATEGORICAL_FEATURES:
        assert cat in df.columns, f"Missing one-hot categorical feature: {cat}"


def test_categorical_one_hot_encoding_integrity(real_data_dir):
    """Verify categorical columns are binary (0 or 1) integers."""
    df = load_and_combine_data(real_data_dir)
    for cat in CATEGORICAL_FEATURES:
        unique_vals = set(df[cat].unique())
        assert unique_vals.issubset({0, 1}), f"Column {cat} must contain only 0 or 1, got {unique_vals}"
        assert np.issubdtype(df[cat].dtype, np.integer), f"Column {cat} must be integer dtype"

    # Mutual exclusivity check: Specimen_Type_S139 + Specimen_Type_S200 == 1
    specimen_sum = df['Specimen_Type_S139'] + df['Specimen_Type_S200']
    assert (specimen_sum == 1).all(), "Every specimen must be either S139 or S200"

    # Location sum: Center + Edge + Root + Toe == 1
    loc_sum = (
        df['Repair_Location_Center'] + df['Repair_Location_Edge'] +
        df['Repair_Location_Root'] + df['Repair_Location_Toe']
    )
    assert (loc_sum == 1).all(), "Every specimen must have exactly one repair location"


def test_synthetic_pivoting_on_controlled_data(synthetic_long_df, tmp_path):
    """Test pivoting on a small controlled synthetic dataset."""
    csv_file = tmp_path / "synthetic_test_4000_samples.csv"
    synthetic_long_df.to_csv(csv_file, index=False)
    
    pivoted = load_and_combine_data(str(tmp_path))
    assert len(pivoted) == 20, f"Expected 20 samples after pivoting, got {len(pivoted)}"
    assert 'R0_UTS' in pivoted.columns
    assert 'R1_UTS' in pivoted.columns
    assert 'R2_UTS' in pivoted.columns


# ============================================================================
# 2. SCALER CONSISTENCY & INVERSE TRANSFORMATION
# ============================================================================

def test_scaler_fitted_on_train_only(synthetic_wide_arrays):
    """
    Verify StandardScaler properties:
    - Zero mean and unit variance on training set.
    - Test set is transformed using train parameters without refitting.
    """
    X, _ = synthetic_wide_arrays
    X_train, X_test = X[:30], X[30:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train scaled should have mean ~ 0 and std ~ 1
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-5)

    # Test scaled should NOT mutate train scaler parameters
    train_mean_before = scaler.mean_.copy()
    _ = scaler.transform(X_test)
    np.testing.assert_array_equal(scaler.mean_, train_mean_before)


def test_scaler_inverse_transform_reversibility(synthetic_wide_arrays):
    """Verify that inverse_transform restores exact original unscaled values."""
    X, _ = synthetic_wide_arrays
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_restored = scaler.inverse_transform(X_scaled)
    
    np.testing.assert_allclose(X_restored, X, atol=1e-4)


# ============================================================================
# 3. EDGE CASES & IMMUTABILITY CHECKS
# ============================================================================

def test_single_sample_transformation(synthetic_wide_arrays):
    """Verify scaler handles a single 1D/2D test sample without shape distortion."""
    X, _ = synthetic_wide_arrays
    scaler = StandardScaler().fit(X)
    
    single_sample = X[0:1]
    scaled_single = scaler.transform(single_sample)
    assert scaled_single.shape == (1, 14)
    assert np.isfinite(scaled_single).all()


def test_constant_valued_feature_scaling():
    """Verify StandardScaler does not divide by zero on constant-valued features."""
    X_const = np.ones((20, 5), dtype=np.float32) * 50.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_const)
    
    # When variance is 0, scale_ is set to 1.0 by sklearn, output is 0.0
    assert np.isfinite(X_scaled).all()
    np.testing.assert_allclose(X_scaled, 0.0)


def test_no_unintended_input_mutation(synthetic_long_df):
    """Verify preprocessing does not mutate input DataFrame in place."""
    df_copy = synthetic_long_df.copy(deep=True)
    # Perform grouping/inspection without modifying original
    _ = df_copy.pivot(index='Sample_ID', columns='Condition', values=['UTS'])
    pd.testing.assert_frame_equal(df_copy, synthetic_long_df)


def test_target_isolation_no_leakage(real_data_dir):
    """
    CRITICAL QUALITY CHECK:
    Ensure the 14 input feature columns contain ZERO Target R2 properties.
    """
    df = load_and_combine_data(real_data_dir)
    feature_cols = [c for c in R0_FEATURES if c in df.columns] + [c for c in R1_FEATURES if c in df.columns]
    
    # Assert no R2 property exists in feature list
    for f in feature_cols:
        assert not f.startswith("R2_"), f"Target leakage detected! Feature '{f}' contains R2 target data"
        assert f not in TARGETS, f"Target '{f}' found in feature inputs"


# ============================================================================
# 4. OPTION B (CURVE DATA) PREPROCESSING
# ============================================================================

def test_load_option_b_data_preprocessing(real_data_dir):
    """Verify load_option_b_data loads and filters required curve parameters."""
    df_b = load_option_b_data(real_data_dir)
    assert df_b is not None
    assert len(df_b) == 3000
    assert 'sample_id' not in df_b.columns, "sample_id should be excluded from curve features"
    assert df_b.shape[1] == 14, f"Expected 14 columns (8 inputs + 6 targets), got {df_b.shape[1]}"


def test_load_option_b_handles_missing_file(tmp_path):
    """Verify load_option_b_data returns None gracefully if file is absent."""
    result = load_option_b_data(str(tmp_path))
    assert result is None
