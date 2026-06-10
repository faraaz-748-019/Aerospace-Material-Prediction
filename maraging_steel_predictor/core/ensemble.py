"""
Stacking Ensemble with Ridge Regression meta-learner.
Collects out-of-fold predictions from all base models and uses Ridge to combine them.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import joblib
import os


class StackingEnsemble:
    """
    Meta-learner stacking ensemble.
    - Collects out-of-fold predictions
    - Trains a Ridge meta-learner on OOF predictions
    """
    def __init__(self):
        self.meta_learner = Ridge(alpha=1.0)
        self.base_model_names = []
        self.is_fitted = False

    def fit(self, base_predictions_oof, y_true):
        """
        Train the meta-learner on out-of-fold predictions.
        base_predictions_oof: dict of {model_name: (n_samples, n_targets)} OOF predictions
        """
        self.base_model_names = list(base_predictions_oof.keys())
        X_meta = np.hstack([base_predictions_oof[name] for name in self.base_model_names])
        self.meta_learner.fit(X_meta, y_true)
        self.is_fitted = True

    def predict(self, base_predictions):
        """
        Predict using the meta-learner.
        base_predictions: dict of {model_name: (n_samples, n_targets)} predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        X_meta = np.hstack([base_predictions[name] for name in self.base_model_names])
        return self.meta_learner.predict(X_meta)

    def save(self, path):
        """Save the ensemble to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'meta_learner': self.meta_learner,
            'base_names': self.base_model_names,
            'is_fitted': self.is_fitted,
        }, path)

    def load(self, path):
        """Load the ensemble from disk."""
        data = joblib.load(path)
        self.meta_learner = data['meta_learner']
        self.base_model_names = data['base_names']
        self.is_fitted = data['is_fitted']


def generate_oof_predictions(models, X, y, n_splits=5):
    """
    Generate out-of-fold predictions for stacking.
    models: dict of {name: fitted_model_or_predict_fn}
    Returns: dict of {name: oof_predictions array}
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = {name: np.zeros((len(X), 2)) for name in models}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for name, model in models.items():
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            preds = model.predict(X_val)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            if preds.shape[1] == 1:
                preds = np.column_stack([preds, preds])
            oof[name][val_idx] = preds[:, :2]

    return oof
