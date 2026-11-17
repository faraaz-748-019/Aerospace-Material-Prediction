"""
Full ML + DL Training Pipeline for M250 Maraging Steel R2 Property Prediction.
Trains all classical ML and deep learning models, builds stacking ensemble,
trains LSTM curve generator, and reports evaluation metrics.
"""

import os
import sys
import glob
import json
import warnings
import traceback
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.architectures import get_model
from core.curve_lstm import CurveLSTM, INPUT_COLS, TARGET_COLS
from core.ensemble import StackingEnsemble

# ─── Configuration ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ML_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ml')
DL_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'dl')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Target variables (6 properties to predict for condition R2)
TARGETS = ['R2_UTS', 'R2_YS', 'R2_Elongation', 'R2_HAZ_Width', 'R2_Gauge_Depth', 'R2_Austenite_Content']

# Input features (R0 condition, R1 condition, and Constants)
R0_FEATURES = ['R0_UTS', 'R0_YS', 'R0_K1c', 'R0_Elongation', 'R0_HAZ_Width', 'R0_Gauge_Depth', 'R0_Austenite_Content']
R1_FEATURES = ['R1_UTS', 'R1_YS', 'R1_K1c', 'R1_Elongation', 'R1_HAZ_Width', 'R1_Gauge_Depth', 'R1_Austenite_Content']
CATEGORICAL_FEATURES = [
    'Specimen_Type_S139', 'Specimen_Type_S200', 
    'Repair_Location_Center', 'Repair_Location_Edge', 
    'Repair_Location_Root', 'Repair_Location_Toe'
]

def log(msg, callback=None):
    """Print log message and optionally call callback."""
    print(msg)
    if callback:
        callback(msg)


def load_and_combine_data(data_dir=None, callback=None):
    """Load the new 4000-sample dataset, pivot to wide format, and one-hot encode."""
    if data_dir is None:
        data_dir = DATA_DIR

    log(f"[DATA] Searching for new 4000-sample datset in {data_dir}...", callback)
    
    # Locate the file
    if data_dir and os.path.isfile(data_dir) and data_dir.lower().endswith('.csv'):
        files = [data_dir]
    else:
        files = glob.glob(os.path.join(data_dir, '*4000_samples*.csv'))
        # Fallback to parent directory if not in data dir
        if not files:
            parent_dir = os.path.dirname(BASE_DIR)
            files = glob.glob(os.path.join(parent_dir, '*4000_samples*.csv'))
        
    if not files:
        raise ValueError("Could not find the 4000-samples dataset file!")
    
    csv_path = files[0]
    log(f"[DATA] Loading {csv_path}...", callback)
    df = pd.read_csv(csv_path)
    
    # Pivot logic
    meas_cols = ['UTS', 'YS', 'K1c', 'Elongation', 'HAZ_Width', 'Gauge_Depth', 'Austenite_Content']
    pvt = df.pivot(index='Sample_ID', columns='Condition', values=meas_cols)
    pvt.columns = [f"{col[1]}_{col[0]}" for col in pvt.columns]
    
    # Extract constants
    constants = df[['Sample_ID', 'Specimen_Type', 'Repair_Location']].drop_duplicates().set_index('Sample_ID')
    final_df = pvt.join(constants)
    
    # One-hot encode categoricals
    final_df = pd.get_dummies(final_df, columns=['Specimen_Type', 'Repair_Location'])
    
    # Ensure all categorical feature columns exist (fill missing with False/0)
    for col in CATEGORICAL_FEATURES:
        if col not in final_df.columns:
            final_df[col] = False
    
    # Convert bool to int for safely passing to ML models
    for col in CATEGORICAL_FEATURES:
        final_df[col] = final_df[col].astype(int)
        
    log(f"[DATA] Successfully pivoted dataset: {final_df.shape} shape", callback)
    return final_df


def load_option_b_data(data_dir=None, callback=None):
    """Load Option B (curve-based) data."""
    if data_dir is None:
        data_dir = DATA_DIR

    files = glob.glob(os.path.join(data_dir, '*option_B*.csv'))
    if not files:
        log("[DATA] No Option B CSV found — skipping curve model training", callback)
        return None

    df = pd.read_csv(files[0])
    if 'sample_id' in df.columns:
        df = df.drop(columns=['sample_id'])

    available_inputs = [c for c in INPUT_COLS if c in df.columns]
    available_targets = [c for c in TARGET_COLS if c in df.columns]

    if len(available_inputs) < 4 or len(available_targets) < 4:
        log("[DATA] Option B missing required columns", callback)
        return None

    log(f"[DATA] Loaded Option B: {len(df)} rows, {len(available_inputs)} inputs, "
        f"{len(available_targets)} targets", callback)
    return df[available_inputs + available_targets].dropna()


def stratified_split(X, y, test_size=0.2, random_state=42):
    """Split data with stratification by binned UTS ranges."""
    uts_bins = pd.qcut(y[:, 0], q=10, labels=False, duplicates='drop')
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=uts_bins)


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R², and approximate accuracy % for all targets."""
    # Ensure shapes match
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
        
    mae_arr = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    rmse_arr = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    r2_arr = r2_score(y_true, y_pred, multioutput='raw_values')
    r2_arr = np.nan_to_num(r2_arr, nan=0.0)
    if np.ndim(r2_arr) == 0:
        r2_arr = np.full_like(mae_arr, float(r2_arr))
    
    # Accuracy as 1 - MAPE (capped at 0)
    mape_arr = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)), axis=0) * 100
    accuracy_arr = np.maximum(0, 100 - mape_arr)
    
    # Overall averages
    mae_avg = float(np.mean(mae_arr))
    rmse_avg = float(np.mean(rmse_arr))
    r2_avg = float(np.mean(r2_arr))
    acc_avg = float(np.mean(accuracy_arr))
    
    metrics = {
        'MAE': round(mae_avg, 4), 'RMSE': round(rmse_avg, 4),
        'R2': round(r2_avg, 6), 'Accuracy%': round(acc_avg, 2),
        'UTS_RMSE': round(float(rmse_arr[0]), 4) if len(rmse_arr) > 0 else 0,
        'YS_RMSE': round(float(rmse_arr[1]), 4) if len(rmse_arr) > 1 else 0,
        'UTS_Acc%': round(float(accuracy_arr[0]), 2) if len(accuracy_arr) > 0 else 0,
        'YS_Acc%': round(float(accuracy_arr[1]), 2) if len(accuracy_arr) > 1 else 0,
        'Target_Metrics': {}
    }
    
    for i, target in enumerate(TARGETS):
        if i < len(mae_arr):
            # Ensure linter recognizes this as a dictionary
            tm = metrics.get('Target_Metrics')
            if isinstance(tm, dict):
                tm[target] = {
                    'MAE': round(float(mae_arr[i]), 4),
                    'RMSE': round(float(rmse_arr[i]), 4),
                    'R2': round(float(r2_arr[i]), 6),
                    'Accuracy%': round(float(accuracy_arr[i]), 2)
                }
            
    return metrics


# ─── Classical ML Training ───────────────────────────────────

def train_classical_models(X_train, y_train, X_test, y_test, callback=None):
    """Train all classical ML models with hyperparameter tuning."""
    results = {}
    models = {}

    # Define models and param grids
    model_configs = {
        'ExtraTrees': {
            'model': MultiOutputRegressor(ExtraTreesRegressor(random_state=42)),
            'params': {
                'estimator__n_estimators': [100, 200, 300, 500],
                'estimator__max_depth': [None, 10, 20, 30],
                'estimator__min_samples_split': [2, 5, 10],
                'estimator__min_samples_leaf': [1, 2, 4],
            },
            'tune': True,
        },
        'RandomForest': {
            'model': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
            'params': {
                'estimator__n_estimators': [100, 200, 300, 500],
                'estimator__max_depth': [None, 10, 20, 30],
                'estimator__min_samples_split': [2, 5, 10],
            },
            'tune': True,
        },
        'GradientBoosting': {
            'model': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
            'params': {
                'estimator__n_estimators': [100, 200, 300],
                'estimator__learning_rate': [0.01, 0.05, 0.1],
                'estimator__max_depth': [3, 5, 7],
            },
            'tune': True,
        },
    }

    if HAS_XGB:
        model_configs['XGBoost'] = {
            'model': MultiOutputRegressor(xgb.XGBRegressor(
                random_state=42, verbosity=0, n_jobs=1)),
            'params': {
                'estimator__n_estimators': [100, 200, 300, 500],
                'estimator__learning_rate': [0.01, 0.05, 0.1],
                'estimator__max_depth': [3, 5, 7, 10],
                'estimator__subsample': [0.7, 0.8, 0.9, 1.0],
            },
            'tune': True,
        }

    if HAS_LGB:
        model_configs['LightGBM'] = {
            'model': MultiOutputRegressor(lgb.LGBMRegressor(
                random_state=42, verbosity=-1, n_jobs=1)),
            'params': {
                'estimator__n_estimators': [100, 200, 300, 500],
                'estimator__learning_rate': [0.01, 0.05, 0.1],
                'estimator__max_depth': [3, 5, 7, -1],
                'estimator__num_leaves': [31, 50, 70],
            },
            'tune': True,
        }

    for name, config in model_configs.items():
        log(f"[ML] Training {name}...", callback)
        try:
            if config['tune']:
                search = RandomizedSearchCV(
                    config['model'], config['params'],
                    n_iter=min(50, max(10, len(config['params']) * 5)),
                    cv=5, scoring='r2', random_state=42, n_jobs=1,
                    error_score='raise'
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                log(f"  Best params: {search.best_params_}", callback)
            else:
                model = config['model']
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            results[name] = metrics
            models[name] = model
            log(f"  {name}: R²={metrics['R2']}, MAE={metrics['MAE']}, "
                f"RMSE={metrics['RMSE']}, Acc={metrics['Accuracy%']}%", callback)

            # Save model
            os.makedirs(ML_MODEL_DIR, exist_ok=True)
            joblib.dump(model, os.path.join(ML_MODEL_DIR, f'{name.lower()}.pkl'))
        except Exception as e:
            log(f"  Error training {name}: {e}", callback)
            traceback.print_exc()

    return models, results


# ─── Deep Learning Training ──────────────────────────────────

def train_dl_model(model, train_loader, val_X, val_y, name, n_epochs=300,
                   patience=20, callback=None):
    """Train a PyTorch model with early stopping."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                            factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    val_X_t = torch.FloatTensor(val_X).to(DEVICE)
    val_y_t = torch.FloatTensor(val_y).to(DEVICE)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            
            if name == 'pinn' and pred.shape[1] >= 2:
                # Physics Loss: YS (index 1) should not exceed UTS (index 0)
                phy_loss = torch.mean(torch.relu(pred[:, 1] - pred[:, 0]))
                # Positivity constraint
                pos_loss = torch.mean(torch.relu(-pred))
                
                alpha = 0.5
                loss = loss + alpha * (phy_loss + pos_loss)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = criterion(val_pred, val_y_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            log(f"    Epoch {epoch+1}/{n_epochs} — train_loss={train_loss/len(train_loader):.6f}, "
                f"val_loss={val_loss:.6f}", callback)

        if patience_counter >= patience:
            log(f"    Early stopping at epoch {epoch+1}", callback)
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


def train_dl_models(X_train, y_train, X_test, y_test, scaler, callback=None):
    """Train all DL models."""
    results = {}
    models = {}

    n_features = X_train.shape[1]
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    dl_model_names = ['cnn', 'lstm', 'transformer', 'hybrid', 'pinn']
    dl_display_names = {
        'cnn': 'CNN', 'lstm': 'LSTM',
        'transformer': 'Transformer', 'hybrid': 'CNN+LSTM Hybrid',
        'pinn': 'Physics-Informed NN'
    }

    n_outputs = len(TARGETS)

    for name in dl_model_names:
        display = dl_display_names[name]
        log(f"[DL] Training {display}...", callback)
        try:
            model = get_model(name, n_features, n_outputs=n_outputs)
            model = train_dl_model(model, train_loader, X_test, y_test, name,
                                   callback=callback)

            # Evaluate
            model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(X_test).to(DEVICE)
                y_pred = model(test_tensor).cpu().numpy()

            metrics = compute_metrics(y_test, y_pred)
            results[display] = metrics
            models[display] = model
            log(f"  {display}: R²={metrics['R2']}, MAE={metrics['MAE']}, "
                f"Acc={metrics['Accuracy%']}%", callback)

            # Save model
            os.makedirs(DL_MODEL_DIR, exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'model_name': name,
                'n_features': n_features,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
            }, os.path.join(DL_MODEL_DIR, f'{name}.pt'))
        except Exception as e:
            log(f"  Error training {display}: {e}", callback)
            traceback.print_exc()

    return models, results


# ─── LSTM Curve Generator ────────────────────────────────────

def train_curve_lstm(data_dir=None, callback=None):
    """Train the LSTM curve generator on Option B data."""
    df_b = load_option_b_data(data_dir, callback)
    if df_b is None:
        return None, None

    available_inputs = [c for c in INPUT_COLS if c in df_b.columns]
    available_targets = [c for c in TARGET_COLS if c in df_b.columns]

    X = df_b[available_inputs].values.astype(np.float32)
    y = df_b[available_targets].values.astype(np.float32)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = CurveLSTM(n_inputs=len(available_inputs), n_outputs=len(available_targets))
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    log("[CURVE] Training LSTM Curve Generator...", callback)
    model = train_dl_model(model, loader, X_test, y_test, 'curve_lstm',
                           n_epochs=300, patience=20, callback=callback)

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)
    metrics = compute_metrics(y_test_original, y_pred_original)
    log(f"  Curve LSTM: R²={metrics['R2']}, MAE={metrics['MAE']}", callback)

    # Save model and scalers
    os.makedirs(DL_MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'n_inputs': len(available_inputs),
        'n_outputs': len(available_targets),
        'scaler_x_mean': scaler_x.mean_.tolist(),
        'scaler_x_scale': scaler_x.scale_.tolist(),
        'scaler_y_mean': scaler_y.mean_.tolist(),
        'scaler_y_scale': scaler_y.scale_.tolist(),
        'input_cols': available_inputs,
        'target_cols': available_targets,
    }, os.path.join(DL_MODEL_DIR, 'curve_lstm.pt'))

    return model, metrics


# ─── Stacking Ensemble ───────────────────────────────────────

def build_stacking_ensemble(ml_models, dl_models, X_train, y_train, X_test, y_test,
                             scaler, callback=None):
    """Build stacking ensemble from all trained models."""
    log("[ENSEMBLE] Building stacking ensemble...", callback)

    # Collect predictions from all models
    all_preds_train = {}
    all_preds_test = {}
    
    n_outputs = len(TARGETS)

    for name, model in ml_models.items():
        try:
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            if pred_train.ndim == 1:
                pred_train = pred_train.reshape(-1, 1)
                pred_test = pred_test.reshape(-1, 1)
            all_preds_train[name] = pred_train[:, :n_outputs]
            all_preds_test[name] = pred_test[:, :n_outputs]
        except Exception as e:
            log(f"  Skip {name} in ensemble: {e}", callback)

    for name, model in dl_models.items():
        try:
            model.eval()
            with torch.no_grad():
                # DL models expect SCALED data!
                X_tr_scaled = scaler.transform(X_train) if scaler else X_train
                X_te_scaled = scaler.transform(X_test) if scaler else X_test
                train_t = torch.FloatTensor(X_tr_scaled).to(DEVICE)
                test_t = torch.FloatTensor(X_te_scaled).to(DEVICE)
                pred_train = model(train_t).cpu().numpy()
                pred_test = model(test_t).cpu().numpy()
            all_preds_train[name] = pred_train[:, :n_outputs]
            all_preds_test[name] = pred_test[:, :n_outputs]
        except Exception as e:
            log(f"  Skip {name} in ensemble: {e}", callback)

    if len(all_preds_train) < 2:
        log("  Not enough models for ensemble", callback)
        return None, None

    ensemble = StackingEnsemble()
    ensemble.fit(all_preds_train, y_train)

    y_pred = ensemble.predict(all_preds_test)
    metrics = compute_metrics(y_test, y_pred)
    log(f"  Stacking Ensemble: R²={metrics['R2']}, MAE={metrics['MAE']}, "
        f"Acc={metrics['Accuracy%']}%", callback)

    ensemble.save(os.path.join(ML_MODEL_DIR, 'stacking_ensemble.pkl'))

    return ensemble, metrics


# ─── Main Training Pipeline ──────────────────────────────────

def run_full_training(data_dir=None, callback=None, progress_callback=None):
    """
    Run the complete training pipeline.
    Returns: (all_results, best_model_name, scaler)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    all_results = {}

    # Step 1: Load data
    if progress_callback:
        progress_callback(5)
    df = load_and_combine_data(data_dir, callback)

    # Determine available features
    feature_cols = [c for c in R0_FEATURES if c in df.columns]
    r1_cols = [c for c in R1_FEATURES if c in df.columns]
    target_cols = [c for c in TARGETS if c in df.columns]

    if len(target_cols) < 2:
        raise ValueError(f"Missing target columns. Available: {list(df.columns)}")

    # Use R0+R1 features if available
    if len(r1_cols) == len(R1_FEATURES):
        use_features = feature_cols + r1_cols
        log(f"[CONFIG] Using R0+R1 features: {len(use_features)} features", callback)
    else:
        use_features = feature_cols
        log(f"[CONFIG] Using R0-only features: {len(use_features)} features", callback)

    X = df[use_features].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    # Step 2: Stratified split
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    log(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}", callback)

    if progress_callback:
        progress_callback(10)

    # Step 3: Scale data for DL models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    os.makedirs(ML_MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(ML_MODEL_DIR, 'scaler.pkl'))
    joblib.dump(use_features, os.path.join(ML_MODEL_DIR, 'feature_cols.pkl'))

    # Step 4: Train classical ML models (on unscaled data)
    if progress_callback:
        progress_callback(15)
    ml_models, ml_results = train_classical_models(X_train, y_train, X_test, y_test, callback)
    all_results.update(ml_results)
    if progress_callback:
        progress_callback(50)

    # Step 5: Train DL models (on scaled data)
    dl_models, dl_results = train_dl_models(X_train_scaled, y_train, X_test_scaled, y_test,
                                             scaler, callback)
    all_results.update(dl_results)
    if progress_callback:
        progress_callback(75)

    # Step 6: Build stacking ensemble
    # ML models use unscaled, DL use scaled — for ensemble, use the same data format
    # We'll build a wrapper approach using test predictions directly
    ensemble, ens_metrics = build_stacking_ensemble(
        ml_models, dl_models, X_train, y_train, X_test, y_test, scaler, callback)
    if ens_metrics:
        all_results['Stacking Ensemble'] = ens_metrics
    if progress_callback:
        progress_callback(85)

    # Step 7: Train LSTM curve generator
    curve_model, curve_metrics = train_curve_lstm(data_dir, callback)
    if curve_metrics:
        all_results['Curve LSTM'] = curve_metrics
    if progress_callback:
        progress_callback(95)

    # Step 8: Load baseline model for comparison
    baseline_path = os.path.join(data_dir, 'option_A_revised_best_model.pkl')
    if os.path.exists(baseline_path):
        try:
            baseline = joblib.load(baseline_path)
            log("[BASELINE] Loaded option_A_revised_best_model.pkl", callback)
            try:
                bl_pred = baseline.predict(X_test)
                bl_metrics = compute_metrics(y_test, bl_pred)
                all_results['Baseline (pkl)'] = bl_metrics
                log(f"  Baseline: R²={bl_metrics['R2']}, MAE={bl_metrics['MAE']}", callback)
            except Exception as e:
                log(f"  Baseline evaluation failed: {e}", callback)
        except Exception as e:
            log(f"  Could not load baseline: {e}", callback)

    # Step 9: Select best model
    best_name = max(all_results, key=lambda k: all_results[k].get('R2', -999))
    log(f"\n{'='*60}", callback)
    log(f"[BEST] {best_name} — R²={all_results[best_name]['R2']}, "
        f"Accuracy={all_results[best_name]['Accuracy%']}%", callback)
    log(f"{'='*60}\n", callback)

    # Save results summary
    results_path = os.path.join(ML_MODEL_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({'results': all_results, 'best_model': best_name,
                   'features': use_features}, f, indent=2)

    if progress_callback:
        progress_callback(100)

    return all_results, best_name, scaler


def retrain_stacking_ensemble_predictions(ml_models, dl_models, X, y, scaler):
    """Helper to get stacking predictions for a new input."""
    preds = {}
    n_out = len(TARGETS)
    for name, model in ml_models.items():
        try:
            p = model.predict(X)
            if p.ndim == 1: p = p.reshape(-1, 1)
            preds[name] = p[:, :n_out]
        except:
            pass
    X_scaled = scaler.transform(X) if scaler else X
    for name, model in dl_models.items():
        try:
            model.eval()
            with torch.no_grad():
                t = torch.FloatTensor(X_scaled).to(DEVICE)
                p = model(t).cpu().numpy()
            preds[name] = p[:, :n_out]
        except:
            pass
    return preds


if __name__ == '__main__':
    print("=" * 60)
    print(" M250 Maraging Steel — Full Training Pipeline")
    print("=" * 60)
    results, best, scaler = run_full_training()
    print("\n\nAll Results:")
    for name, metrics in sorted(results.items(), key=lambda x: x[1].get('R2', 0), reverse=True):
        print(f"  {name:25s} — R²={metrics['R2']:.6f}  MAE={metrics['MAE']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  Acc={metrics['Accuracy%']:.2f}%")
