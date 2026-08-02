"""
SHAP Explainability Module — Multi-Target Support.
TreeExplainer for classical ML models, KernelExplainer fallback for DL models.
Supports per-target feature importance for all 6 mechanical properties.

Handles MultiOutputRegressor wrappers by extracting individual estimators.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Target display names (match train_all.TARGETS order)
TARGET_DISPLAY = {
    0: 'UTS',
    1: 'YS',
    2: 'Elongation',
    3: 'HAZ Width',
    4: 'Gauge Depth',
    5: 'Austenite Content',
}


def format_feature_name(col_name):
    """Format raw column names into readable labels."""
    return col_name.replace('_', ' ').title()


def _unwrap_model_for_target(model, target_idx):
    """
    Extract the single-output estimator for a given target index.

    sklearn MultiOutputRegressor wraps N independent estimators.
    TreeExplainer needs the raw tree model (e.g., ExtraTreesRegressor),
    not the MultiOutputRegressor wrapper.

    Returns (estimator, model_type) tuple.
    """
    from sklearn.multioutput import MultiOutputRegressor

    if isinstance(model, MultiOutputRegressor):
        estimators = model.estimators_
        if target_idx < len(estimators):
            est = estimators[target_idx]
        else:
            est = estimators[0]
        # Check if the estimator is tree-based
        tree_types = (
            'ExtraTreesRegressor', 'RandomForestRegressor',
            'GradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor',
        )
        est_type = type(est).__name__
        if est_type in tree_types:
            return est, 'tree'
        else:
            return est, 'kernel'
    else:
        return model, None  # caller decides type


def compute_shap_values(model, X_train, X_test=None, feature_names=None,
                        model_type='tree', target_idx=0):
    """
    Compute SHAP values for a model, for a specific target output.

    Args:
        model: trained model (sklearn or wrapper)
        X_train: training data for background (numpy array)
        X_test: optional test data to explain (defaults to X_train sample)
        feature_names: list of feature names
        model_type: 'tree' for TreeExplainer, 'kernel' for KernelExplainer
        target_idx: index of the target to explain (0-5)

    Returns:
        shap_values: SHAP values array for the selected target
        feature_importance: dict of {feature_name: mean_abs_shap}
    """
    if not SHAP_AVAILABLE:
        return None, None

    if X_test is None:
        X_test = X_train[:min(100, len(X_train))]

    # Try to unwrap MultiOutputRegressor to get per-target estimator
    unwrapped, unwrapped_type = _unwrap_model_for_target(model, target_idx)
    if unwrapped_type is not None:
        model_type = unwrapped_type
        use_model = unwrapped
    else:
        use_model = model

    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(use_model)
            shap_values_raw = explainer.shap_values(X_test)
        else:
            background = shap.kmeans(X_train, min(50, len(X_train)))

            # For DL or non-tree models, wrap predict to return scalar
            def predict_fn(x):
                pred = use_model.predict(x)
                if hasattr(pred, 'ndim') and pred.ndim > 1:
                    if target_idx < pred.shape[1]:
                        return pred[:, target_idx]
                    return pred[:, 0]
                return pred

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values_raw = explainer.shap_values(X_test, nsamples=100)
    except Exception:
        try:
            # Fallback: KernelExplainer with wrapped predict
            background = shap.kmeans(X_train, min(50, len(X_train)))

            def predict_fn_fallback(x):
                pred = use_model.predict(x)
                if hasattr(pred, 'ndim') and pred.ndim > 1:
                    if target_idx < pred.shape[1]:
                        return pred[:, target_idx]
                    return pred[:, 0]
                return pred

            explainer = shap.KernelExplainer(predict_fn_fallback, background)
            shap_values_raw = explainer.shap_values(X_test, nsamples=100)
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return None, None

    # Extract the SHAP values for the requested target
    sv = _extract_target_shap(shap_values_raw, target_idx, unwrapped_type)
    if sv is None:
        return None, None

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_train.shape[1])]

    # Compute mean absolute SHAP values per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=1)

    feature_importance = dict(zip(feature_names, mean_abs_shap))
    feature_importance = dict(sorted(feature_importance.items(),
                                      key=lambda x: x[1], reverse=True))

    return sv, feature_importance


def _extract_target_shap(shap_values_raw, target_idx, unwrapped_type=None):
    """
    Extract SHAP values for a single target from multi-output results.

    If the model was unwrapped (MultiOutputRegressor → single estimator),
    the SHAP values are already for that target — return directly.
    """
    # If we unwrapped to a single-output estimator, SHAP values are already
    # for the correct target. Just handle array shapes.
    if unwrapped_type is not None:
        if isinstance(shap_values_raw, list):
            # Some TreeExplainer versions return a list even for single-output
            return shap_values_raw[0] if len(shap_values_raw) == 1 else shap_values_raw[0]
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3:
                return shap_values_raw[:, :, 0]
            return shap_values_raw
        return shap_values_raw

    # Fallback: multi-output model that wasn't unwrapped
    if isinstance(shap_values_raw, list):
        if target_idx < len(shap_values_raw):
            return shap_values_raw[target_idx]
        else:
            return shap_values_raw[0]
    elif isinstance(shap_values_raw, np.ndarray):
        if shap_values_raw.ndim == 3:
            if target_idx < shap_values_raw.shape[2]:
                return shap_values_raw[:, :, target_idx]
            else:
                return shap_values_raw[:, :, 0]
        else:
            return shap_values_raw
    return shap_values_raw


def compute_all_targets_importance(model, X_train, X_test=None,
                                   feature_names=None, model_type='tree',
                                   n_targets=6):
    """
    Compute SHAP feature importance for ALL targets at once.

    Returns:
        dict: {target_idx: feature_importance_dict}
    """
    all_importance = {}
    for t in range(n_targets):
        _, importance = compute_shap_values(
            model, X_train, X_test, feature_names, model_type, target_idx=t)
        if importance is not None:
            all_importance[t] = importance
    return all_importance


def plot_shap_bar(feature_importance, title="SHAP Feature Importance",
                  save_path=None, top_n=15, dark_mode=True):
    """
    Create a horizontal bar chart of SHAP feature importance.

    Args:
        feature_importance: dict {feature_name: shap_value}
        title: chart title
        save_path: optional path to save figure
        top_n: number of top features to show
        dark_mode: whether to use dark theme

    Returns:
        fig: matplotlib Figure
    """
    if feature_importance is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'SHAP values not available',
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig

    names = list(feature_importance.keys())[:top_n]
    values = list(feature_importance.values())[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))

    if dark_mode:
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#1e1e2e')

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='none',
                   height=0.7)

    ax.set_yticks(range(len(names)))
    label_color = 'white' if dark_mode else 'black'
    ax.set_yticklabels(names, fontsize=10, color=label_color)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12, color=label_color)
    ax.set_title(title, fontsize=14, fontweight='bold', color=label_color)
    ax.invert_yaxis()
    ax.tick_params(colors=label_color)
    ax.grid(axis='x', alpha=0.2, color='gray')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('gray' if dark_mode else '#ccc')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())

    return fig


def plot_shap_comparison(all_importance, feature_names=None, top_n=8,
                         dark_mode=True):
    """
    Create a grouped bar chart comparing feature importance across all targets.

    Args:
        all_importance: dict {target_idx: {feature: value}}
        feature_names: optional list (unused — names come from dicts)
        top_n: top features to show (based on average across targets)
        dark_mode: use dark theme

    Returns:
        fig: matplotlib Figure
    """
    if not all_importance:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No SHAP data available', ha='center', va='center',
                fontsize=14, color='white' if dark_mode else 'black')
        ax.set_axis_off()
        return fig

    # Collect all feature names across targets and compute average importance
    all_features = set()
    for imp in all_importance.values():
        all_features.update(imp.keys())

    avg_importance = {}
    for feat in all_features:
        vals = [all_importance[t].get(feat, 0.0) for t in all_importance]
        avg_importance[feat] = np.mean(vals)

    top_features = sorted(avg_importance, key=avg_importance.get, reverse=True)[:top_n]

    n_targets = len(all_importance)
    target_labels = [TARGET_DISPLAY.get(t, f'Target {t}') for t in sorted(all_importance.keys())]
    x = np.arange(len(top_features))
    bar_width = 0.8 / n_targets

    fig, ax = plt.subplots(figsize=(12, 6))
    if dark_mode:
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#1e1e2e')

    cmap = plt.cm.Set2
    for i, t in enumerate(sorted(all_importance.keys())):
        vals = [all_importance[t].get(f, 0.0) for f in top_features]
        ax.bar(x + i * bar_width, vals, bar_width, label=target_labels[i],
               color=cmap(i / max(n_targets - 1, 1)), edgecolor='none')

    label_color = 'white' if dark_mode else 'black'
    ax.set_xticks(x + bar_width * (n_targets - 1) / 2)
    ax.set_xticklabels([f.replace(' ', '\n') if len(f) > 12 else f
                       for f in top_features],
                       fontsize=9, color=label_color, rotation=0)
    ax.set_ylabel('Mean |SHAP Value|', fontsize=11, color=label_color)
    ax.set_title('Feature Importance Across All Targets', fontsize=14,
                 fontweight='bold', color=label_color)
    ax.tick_params(colors=label_color)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax.grid(axis='y', alpha=0.2, color='gray')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('gray')
    fig.tight_layout()
    return fig
