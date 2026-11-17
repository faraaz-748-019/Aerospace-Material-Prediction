"""
Pytest configuration and shared fixtures for Aerospace Materials Prediction testing suite.
Provides fast, lightweight synthetic datasets, real dataset paths, and helper fixtures.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_all import TARGETS, R0_FEATURES, R1_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def real_data_dir(project_root):
    return os.path.join(project_root, "data")


@pytest.fixture(scope="session")
def real_primary_csv_path(real_data_dir):
    path = os.path.join(real_data_dir, "M250_S139_S200_separate_dataset_4000_samples.csv")
    assert os.path.isfile(path), f"Primary dataset not found at {path}"
    return path


@pytest.fixture(scope="session")
def real_option_b_csv_path(real_data_dir):
    path = os.path.join(real_data_dir, "synthetic_option_B_curve_based_R0_to_R2_3000.csv")
    assert os.path.isfile(path), f"Option B dataset not found at {path}"
    return path


@pytest.fixture(scope="session")
def real_models_dir(project_root):
    return os.path.join(project_root, "models")


@pytest.fixture
def synthetic_long_df():
    """
    Generate a small, controlled valid dataset in raw long format (20 samples, 60 rows).
    Each sample has R0, R1, R2 conditions.
    """
    np.random.seed(42)
    rows = []
    specimen_types = ["S139", "S200"]
    locations = ["Root", "Toe", "Edge", "Center"]
    
    for sid in range(20):
        st = specimen_types[sid % 2]
        loc = locations[sid % 4]
        
        # Base R0 properties
        r0_uts = np.random.uniform(1600, 1660)
        r0_ys = r0_uts - np.random.uniform(30, 60)
        r0_k1c = np.random.uniform(70, 85)
        r0_elong = np.random.uniform(9.0, 12.0)
        r0_haz = np.random.uniform(4.0, 6.0)
        r0_depth = np.random.uniform(4.9, 5.2)
        r0_aust = np.random.uniform(2.0, 4.0)

        # R0 row
        rows.append({
            "Sample_ID": sid, "Specimen_Type": st, "Condition": "R0",
            "UTS": r0_uts, "YS": r0_ys, "K1c": r0_k1c, "Elongation": r0_elong,
            "HAZ_Width": r0_haz, "Gauge_Depth": r0_depth, "Repair_Location": loc,
            "Austenite_Content": r0_aust
        })

        # R1 row (degraded)
        r1_uts = r0_uts * np.random.uniform(0.95, 0.98)
        r1_ys = r0_ys * np.random.uniform(0.94, 0.97)
        rows.append({
            "Sample_ID": sid, "Specimen_Type": st, "Condition": "R1",
            "UTS": r1_uts, "YS": r1_ys, "K1c": r0_k1c * 0.95, "Elongation": r0_elong * 0.95,
            "HAZ_Width": r0_haz * 1.05, "Gauge_Depth": r0_depth * 1.01, "Repair_Location": loc,
            "Austenite_Content": r0_aust * 1.2
        })

        # R2 row (further degraded)
        r2_uts = r1_uts * np.random.uniform(0.94, 0.97)
        r2_ys = r1_ys * np.random.uniform(0.93, 0.96)
        rows.append({
            "Sample_ID": sid, "Specimen_Type": st, "Condition": "R2",
            "UTS": r2_uts, "YS": r2_ys, "K1c": r0_k1c * 0.88, "Elongation": r0_elong * 0.88,
            "HAZ_Width": r0_haz * 1.15, "Gauge_Depth": r0_depth * 1.02, "Repair_Location": loc,
            "Austenite_Content": r0_aust * 1.5
        })

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_wide_arrays():
    """
    Generate synthetic (X, y) arrays matching the 14 input features and 6 targets.
    Useful for quick training and metric unit tests.
    """
    np.random.seed(42)
    n_samples = 40
    # 14 features: 7 R0 + 7 R1
    X = np.random.uniform(10.0, 1600.0, size=(n_samples, 14)).astype(np.float32)
    # 6 targets: R2_UTS, R2_YS, R2_Elongation, R2_HAZ_Width, R2_Gauge_Depth, R2_Austenite_Content
    y = np.zeros((n_samples, 6), dtype=np.float32)
    y[:, 0] = X[:, 0] * 0.92  # R2_UTS
    y[:, 1] = X[:, 1] * 0.90  # R2_YS
    y[:, 2] = 8.5            # Elongation
    y[:, 3] = 6.2            # HAZ Width
    y[:, 4] = 5.1            # Gauge Depth
    y[:, 5] = 5.5            # Austenite Content
    return X, y


@pytest.fixture
def synthetic_curve_params():
    """Valid parameters for stress-strain curve generation."""
    return {
        'yield_strain': 0.007189,
        'uts_strain': 0.015,
        'fracture_strain': 0.018,
        'yield_stress_MPa': 1366.0,
        'UTS_MPa': 1448.0,
        'area_under_curve': 12.5
    }
