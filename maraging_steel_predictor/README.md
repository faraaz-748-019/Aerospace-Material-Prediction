# M250 Maraging Steel — Weld Repair Property Predictor

> **Institution:** Vikram Sarabhai Space Centre (VSSC/ISRO)

A professional desktop application that predicts R2 (second repair) mechanical properties of M250 Maraging Steel welds from R0 (original) and optionally R1 (first repair) measurements.

## Quick Start

```bash
cd maraging_steel_predictor
pip install -r requirements.txt
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

## Expected Accuracy

With the provided synthetic datasets, typical results:
- **Stacking Ensemble**: R² > 0.99, MAE < 5 MPa
- **Best Single Model**: R² > 0.98
- **DL Models**: R² > 0.95

## Project Structure

```
maraging_steel_predictor/
├── main.py                     # Entry point
├── train_all.py                # Full ML+DL pipeline
├── requirements.txt
├── README.md
├── core/
│   ├── architectures.py        # CNN, BiLSTM, Transformer, Hybrid
│   ├── curve_lstm.py           # LSTM curve generator
│   ├── ensemble.py             # Stacking ensemble
│   ├── uncertainty.py          # MC Dropout
│   ├── curve_generator.py      # Pchip curve reconstruction
│   ├── shap_explainer.py       # SHAP computation
│   └── report_generator.py     # PDF export
├── gui/
│   ├── app.py                  # Main window + tabs
│   ├── tab_predict.py
│   ├── tab_curves.py
│   ├── tab_training.py
│   ├── tab_xai.py
│   └── tab_explorer.py
├── data/                       # CSV files + baseline .pkl
└── models/
    ├── ml/                     # Trained .pkl models
    └── dl/                     # Trained .pt models
```

## License

Internal use — Vikram Sarabhai Space Centre (VSSC/ISRO)
