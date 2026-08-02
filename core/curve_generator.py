"""
Stress-Strain Curve Generator using Pchip Interpolation.
Reconstructs smooth curves from predicted key points.
Includes hardcoded reference curves for Parent Material, R0, R1, R2.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


# ═══════════════════════════════════════════════════════════
# Hardcoded reference curves from experimental data
# ═══════════════════════════════════════════════════════════

PARENT_MATERIAL_CURVE = {
    'strain': np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.009, 0.010, 0.012, 0.014, 0.016, 0.018]),
    'stress': np.array([0, 380, 760, 1140, 1520, 1710, 1750, 1820, 1815, 1800, 1780])
}

R0_CURVE = {
    'strain': np.array([0.0, 0.002, 0.004, 0.006, 0.00826, 0.010, 0.012, 0.014234, 0.016]),
    'stress': np.array([0, 380, 760, 1140, 1570, 1590, 1601, 1595, 1580])
}

R1_CURVE = {
    'strain': np.array([0.0, 0.002, 0.004, 0.006, 0.00793, 0.010, 0.012, 0.014, 0.015434, 0.017]),
    'stress': np.array([0, 380, 760, 1140, 1507, 1530, 1545, 1546, 1538, 1520])
}

R2_REFERENCE_CURVE = {
    'strain': np.array([0.0, 0.002, 0.004, 0.006, 0.0075, 0.010, 0.012, 0.014, 0.016634, 0.018]),
    'stress': np.array([0, 380, 760, 1140, 1425, 1455, 1470, 1477, 1465, 1440])
}


def generate_smooth_curve(key_strains, key_stresses, n_points=200):
    """
    Generate a smooth stress-strain curve using Pchip interpolation.
    
    Args:
        key_strains: array of key strain values
        key_stresses: array of key stress values
        n_points: number of points for smooth curve
    
    Returns:
        smooth_strain: array of strain values
        smooth_stress: array of stress values
    """
    if len(key_strains) < 2:
        return key_strains, key_stresses

    # Ensure monotonically increasing strain
    sort_idx = np.argsort(key_strains)
    key_strains = np.array(key_strains)[sort_idx]
    key_stresses = np.array(key_stresses)[sort_idx]

    # Remove duplicates
    unique_mask = np.diff(key_strains, prepend=-1) > 0
    key_strains = key_strains[unique_mask]
    key_stresses = key_stresses[unique_mask]

    if len(key_strains) < 2:
        return key_strains, key_stresses

    interpolator = PchipInterpolator(key_strains, key_stresses)
    smooth_strain = np.linspace(key_strains[0], key_strains[-1], n_points)
    smooth_stress = interpolator(smooth_strain)

    return smooth_strain, smooth_stress


def reconstruct_r2_curve(predicted_params, n_points=200):
    """
    Reconstruct R2 stress-strain curve from predicted parameters.
    
    Args:
        predicted_params: dict or array with keys/indices:
            yield_strain, uts_strain, fracture_strain,
            yield_stress_MPa, UTS_MPa, area_under_curve
    
    Returns:
        smooth_strain, smooth_stress arrays
    """
    if isinstance(predicted_params, dict):
        yield_strain = predicted_params['yield_strain']
        uts_strain = predicted_params['uts_strain']
        fracture_strain = predicted_params['fracture_strain']
        yield_stress = predicted_params['yield_stress_MPa']
        uts = predicted_params['UTS_MPa']
    else:
        yield_strain = predicted_params[0]
        uts_strain = predicted_params[1]
        fracture_strain = predicted_params[2]
        yield_stress = predicted_params[3]
        uts = predicted_params[4]

    # Build key points for curve reconstruction
    key_strains = [0.0, yield_strain * 0.3, yield_strain * 0.6, yield_strain,
                   (yield_strain + uts_strain) / 2, uts_strain,
                   (uts_strain + fracture_strain) / 2, fracture_strain]
    
    # Elastic region is linear, then work hardening, then plateau/fracture
    # Enforce physically consistent 190 GPa Young's Modulus for M250
    elastic_slope = 190000.0 
    # CRITICAL: Always calculate yield_strain from yield_stress and slope to avoid slope jumps
    yield_strain = yield_stress / elastic_slope
    key_stresses = [
        0.0,
        elastic_slope * yield_strain * 0.3,
        elastic_slope * yield_strain * 0.6,
        yield_stress,
        yield_stress + 0.7 * (uts - yield_stress),
        uts,
        uts * 0.98,
        uts * 0.95
    ]

    return generate_smooth_curve(key_strains, key_stresses, n_points)


def get_reference_curves():
    """
    Return all reference curves interpolated smoothly.
    Returns dict of {name: (strain, stress)} tuples.
    """
    curves = {}
    for name, data in [('Parent Material', PARENT_MATERIAL_CURVE),
                        ('R0', R0_CURVE), ('R1', R1_CURVE),
                        ('R2 Reference', R2_REFERENCE_CURVE)]:
        strain, stress = generate_smooth_curve(data['strain'], data['stress'])
        curves[name] = (strain, stress)
    return curves


def get_curve_markers(strain, stress, yield_strain, uts_strain, fracture_strain,
                      yield_stress, uts):
    """
    Return marker positions for yield point, UTS, and fracture point.
    """
    markers = {
        'Yield Point': (yield_strain, yield_stress),
        'UTS': (uts_strain, uts),
        'Fracture': (fracture_strain, stress[np.argmin(np.abs(
            np.linspace(strain[0], strain[-1], len(stress)) - fracture_strain
        ))] if len(stress) > 0 else 0)
    }
    return markers
