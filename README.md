# M250 Maraging Steel — Weld Repair Property Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)

> **Institution:** Vikram Sarabhai Space Centre (VSSC/ISRO)

A professional desktop application that predicts R2 (second repair) mechanical properties of M250 Maraging Steel welds from R0 (original) and optionally R1 (first repair) measurements.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the application
python main.py
```

On first launch, the application automatically trains all models on the provided datasets. This may take 5–15 minutes depending on hardware.

## Input Properties

| Property | Unit | Typical Range |
|----------|------|---------------|
| R0 UTS (S139, S200) | MPa | 1585–1650 |
| R0 Yield Strength (S139, S200) | MPa | 1550–1596 |
| R0 Fracture Toughness (KIC) | MPa√m | 60–100 |
| R1 UTS (S139, S200) — optional | MPa | 1510–1610 |
| R1 Yield Strength (S139, S200) — optional | MPa | 1475–1565 |

## Model Architecture

### Classical ML (sklearn / xgboost / lightgbm)
- Extra Trees Regressor
- Random Forest Regressor
- XGBoost with early stopping
- LightGBM
- Gradient Boosting Regressor
- Hyperparameter tuning via RandomizedSearchCV (50 iterations, 5-fold CV)

### Deep Learning (PyTorch)
- **1D-CNN**: Conv1D→BN→ReLU→GlobalMaxPool→Dense
- **BiLSTM**: Bidirectional LSTM→Dense
- **Transformer**: Feature embeddings + positional encoding → multi-head self-attention (4 heads)
- **CNN+LSTM Hybrid**: Conv1D→MaxPool→BiLSTM→Dense

### Meta-Learner
- **Stacking Ensemble**: Out-of-fold predictions from all models → Ridge regression

### Curve Prediction
- **LSTM Curve Generator**: Predicts R2 stress-strain curve parameters from R0 curve data
- Smooth curve reconstruction via scipy Pchip interpolation

### Uncertainty & Explainability
- **Monte Carlo Dropout**: 50 forward passes → mean ± 2σ confidence intervals
- **SHAP**: TreeExplainer for ML models, KernelExplainer for DL models

## GUI Tabs

1. **Predict R2** — Input R0/R1 values, select model, view predicted UTS/YS/confidence/degradation
2. **Stress-Strain** — Interactive matplotlib curves with uncertainty bands
3. **Training** — Train all models with progress bar and performance comparison
4. **XAI** — SHAP feature importance charts
5. **Data Explorer** — Browse CSV data with statistics and scatter plots
6. **About/Help** — Project documentation

## Running Automated Tests & Validation

A formal 70-test QA suite validates dataset integrity, preprocessing, model architectures, and metrics:

```bash
# Run the complete test suite
pytest -v

# Run with code coverage report
pytest --cov=core --cov=train_all --cov-report=term-missing

# Run specific testing modules
pytest -v tests/test_data_validation.py
pytest -v tests/test_preprocessing.py
pytest -v tests/test_training.py
pytest -v tests/test_predictions.py
pytest -v tests/test_metrics.py
```

For complete QA audit results and methodology, see [`docs/testing-report.md`](docs/testing-report.md) and [`docs/testing-strategy.md`](docs/testing-strategy.md).

## Model Accuracy & Benchmark Verification

On the verified 4,000-sample multi-property dataset across 800 unseen test specimens:
- **Best Single Model (XGBoost / LightGBM)**:
  - **UTS $R^2$**: `0.8993` | **UTS MAE**: `8.41 MPa` | **UTS Accuracy**: `99.43%`
  - **Overall Average MAE**: `3.797 MPa` (Verified `< 5 MPa`)
  - **Overall Multi-Target $R^2$**: `0.8398` across all 6 properties
- **Stacking Meta-Ensemble**: Combines 10 classical and deep learning models with Ridge regression.
- *(Note: Legacy $R^2 > 0.99$ was observed on the earlier 2-target linear synthetic prototype).*

## Project Structure

```
├── main.py                     # Entry point (GUI Launcher)
├── train_all.py                # Full ML+DL pipeline script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── core/                       # Core ML/DL architecture and logic
│   ├── architectures.py        # CNN, BiLSTM, Transformer, Hybrid
│   ├── curve_lstm.py           # LSTM curve generator
│   ├── ensemble.py             # Stacking ensemble
│   ├── uncertainty.py          # MC Dropout
│   ├── curve_generator.py      # Pchip curve reconstruction
│   ├── shap_explainer.py       # SHAP computation
│   └── report_generator.py     # PDF export
│
├── gui/                        # Application frontend components
│   ├── app.py                  # Main window + tabs
│   ├── tab_predict.py
│   ├── tab_curves.py
│   ├── tab_training.py
│   ├── tab_xai.py
│   └── tab_explorer.py
│
├── data/                       # Datasets (CSV, Excel) + baseline models
├── models/                     # Trained models (.pkl, .pt)
├── notebooks/                  # Jupyter notebooks for EDA and Mechanical Analysis
├── docs/                       # Technical reports, PowerPoint presentations, and architecture diagrams
├── scripts/                    # Utility scripts (e.g., test transforms, document generation)
└── logs/                       # Application logs and text outputs
```
