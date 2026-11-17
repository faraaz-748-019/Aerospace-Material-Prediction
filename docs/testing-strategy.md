# Software Testing & ML Validation Strategy
## M250 Maraging Steel Weld Repair Property Predictor

**Project Context:** Vikram Sarabhai Space Centre (VSSC / ISRO) Aerospace Materials Qualification  
**Testing Framework:** `pytest 9.1.1` with `pytest-cov 7.1.0`  
**Target Quality Level:** High-Reliability Mission-Critical Aerospace Standard  

---

## 1. Testing Philosophy & Scope

Machine learning systems in aerospace materials qualification require a two-tier testing methodology:
1. **Software Engineering Correctness:** Standard software verification ensuring functions handle edge cases, valid types, missing parameters, and serialize without defects.
2. **Physical & Metallurgical Validity:** Domain-specific ML verification ensuring neural networks and tree ensembles adhere to continuum mechanics and physical invariants (such as $YS \le UTS$, positive property values, and Hooke's Law linearity).

```text
+-----------------------------------------------------------------------------------+
|                        AEROSPACE ML QUALITY ASSURANCE TIERS                       |
+-----------------------------------------------------------------------------------+
|  Tier 1: Data Quality & Schema Validation       (tests/test_data_validation.py)   |
|  Tier 2: Preprocessing & Feature Engineering    (tests/test_preprocessing.py)     |
|  Tier 3: Model Training & Physics Loss Penalties(tests/test_training.py)          |
|  Tier 4: Model Inference, Curves & Uncertainty  (tests/test_predictions.py)       |
|  Tier 5: Metric Correctness & Reproducibility   (tests/test_metrics.py)           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Test Classification Taxonomy

### 2.1 Data Quality & Validation Tests (`test_data_validation.py`)
- **Objective:** Prevent garbage-in, garbage-out failures by validating raw inputs before training or inference begins.
- **Scope:**
  - Null/missing value absence across primary datasets.
  - Duplicate record detection.
  - Categorical value domain enforcement (`Specimen_Type` $\in \{S139, S200\}$, `Repair_Location` $\in \{Root, Toe, Edge, Center\}$).
  - Physical boundary constraints ($UTS \in [1400, 1750]\text{ MPa}$, $YS \in [1300, 1720]\text{ MPa}$, $K_{1c} \in [50, 105]\text{ MPa}\sqrt{\text{m}}$).
  - **Fundamental Metallurgy Invariant:** Yield Strength must never exceed UTS ($YS \le UTS$).
  - Negative testing on simulated corruptions (missing columns, out-of-range values, empty datasets).
  - Train/test split leakage verification: 0 overlapping `Sample_ID`s between train and test.

### 2.2 Preprocessing Unit Tests (`test_preprocessing.py`)
- **Objective:** Verify transformations produce deterministic, leak-free numeric matrices.
- **Scope:**
  - Long-to-wide pivoting output dimensions (4,000 samples, 28 columns).
  - Categorical one-hot encoding produces binary integers $\{0, 1\}$ with mutually exclusive row sums equal to 1.
  - `StandardScaler` isolation: Scaler fitted exclusively on $X_{\text{train}}$ without parameter contamination from $X_{\text{test}}$.
  - Invertibility: Inverse transformation restores original scale within $10^{-4}$ tolerance.
  - Immutability: Input DataFrames are not modified in-place.
  - **Target Isolation:** Zero $R_2$ properties present in input feature columns ($X$).

### 2.3 Model Training Tests (`test_training.py`)
- **Objective:** Ensure all 10 supported architectures can fit and update weights on valid data.
- **Scope:**
  - Classical tree models (ExtraTrees, RandomForest, GradientBoosting, XGBoost, LightGBM) train on small synthetic batches.
  - Deep neural networks (1D-CNN, BiLSTM, Transformer, CNN+LSTM Hybrid, PINN) execute forward and backward passes without NaN gradients.
  - **PINN Custom Loss:** Verifies that physical penalty triggers when $\hat{y}_{YS} > \hat{y}_{UTS}$ and is zero when $\hat{y}_{YS} \le \hat{y}_{UTS}$.
  - Stacking ensemble: Fits Ridge meta-model on out-of-fold base predictions and saves/reloads bitwise identical weights.

### 2.4 Model Prediction & Inference Tests (`test_predictions.py`)
- **Objective:** Validate end-to-end inference across pre-trained checkpoints and specialized engines.
- **Scope:**
  - Loading pre-trained model files (`.pkl` and `.pt`).
  - Prediction output shape alignment: $(1, 14) \rightarrow (1, 6)$ finite values.
  - Mismatched feature dimensions trigger clear, informative exceptions.
  - **Monte Carlo Dropout:** 50 stochastic forward passes produce valid expected mean $\mu$, non-negative standard deviation $\sigma \ge 0$, and lower bound flight clearance ($\mu - 2\sigma \le \mu$).
  - **Continuous Curve Reconstruction:** 200 points, strictly monotonic strain ordering, and Hooke's Law Young's modulus linearity ($E = 190\text{ GPa} \pm 1\%$).
  - PDF report generation and SHAP feature importance calculation.

### 2.5 Evaluation Metrics & Reproducibility Tests (`test_metrics.py`)
- **Objective:** Mathematically verify metric computations and audit project accuracy claims against reality.
- **Scope:**
  - Handcrafted synthetic matrices verify $MAE = 0, RMSE = 0, R^2 = 1.0, \text{Accuracy} = 100\%$ on identical predictions.
  - Known constant error offset verifies linear error propagation.
  - Edge cases: $N < 2$ samples, constant target values, and zero targets do not crash.
  - **Reproducibility Audit:** Evaluates pre-trained XGBoost on the documented 80/20 test split (800 samples) and validates real test performance ($R^2 \approx 0.8993$ on UTS, $MAE = 3.80\text{ MPa}$ overall).

---

## 3. Test Data Strategy

To maintain test execution speed (< 15 seconds) while avoiding expensive full re-training on 4,000 samples, the suite uses a layered test data strategy:

1. **Lightweight Synthetic Fixtures (`tests/conftest.py`):**
   - `synthetic_long_df`: 20 samples (60 rows) in long format for fast pivoting unit tests.
   - `synthetic_wide_arrays`: 40 samples (14 features, 6 targets) for instant model training tests (2 epochs).
   - `synthetic_curve_params`: Standard tensile shape parameters for instant curve interpolation tests.
2. **Real Dataset Verification (`data/`):**
   - High-level data validation tests inspect the actual 12,000-row primary CSV and 3,000-row Option B CSV directly.
   - Real test split (800 specimens) evaluates the actual serialized model binaries.

---

## 4. Setup & Test Execution Commands

### 4.1 Running All Tests
To run the complete test suite with verbose output:
```bash
pytest -v
```

### 4.2 Running Tests with Coverage Report
To measure and report code coverage across core modules:
```bash
pytest --cov=core --cov=train_all --cov-report=term-missing
```

### 4.3 Running Specific Test Modules
```bash
# Data validation only
pytest -v tests/test_data_validation.py

# Preprocessing only
pytest -v tests/test_preprocessing.py

# Training pipelines only
pytest -v tests/test_training.py

# Inference and predictions only
pytest -v tests/test_predictions.py

# Evaluation metrics and reproducibility only
pytest -v tests/test_metrics.py
```

### 4.4 Running Fast Tests (Excluding Real Data IO)
```bash
pytest -v -k "synthetic"
```

---

## 5. Continuous Integration (CI/CD) Recommendations

For production deployment and automated GitHub Actions / GitLab CI pipelines:
1. **Pre-Commit Hook:** Run `pytest tests/test_data_validation.py tests/test_metrics.py` before every commit.
2. **Pull Request Gate:** Require `pytest -v` to pass with 0 failures and enforce code coverage $\ge 45\%$ across the repository.
3. **Model Artifact Integrity Gate:** Whenever a new checkpoint is trained and committed to `models/`, automated tests in `test_predictions.py` must verify that the new file loads and predicts finite $(1, 6)$ tensors on benchmark inputs.
