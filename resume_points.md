# Resume Bullet Points & Project Summary
## M250 Maraging Steel — Weld Repair Property Predictor

This document contains high-impact, professional resume bullet points, key technologies, and technical summaries for the **M250 Maraging Steel Weld Repair Property Predictor** project.

---

### 🚀 High-Impact Resume Bullet Points

*   **Ensemble & Deep Learning Pipeline:**
    *   Designed and engineered a multi-model stacking ensemble pipeline in PyTorch and scikit-learn (combining XGBoost, LightGBM, 1D-CNN, BiLSTM, and Transformers) that achieved **R² > 0.99** and **MAE < 5 MPa** in predicting second-repair ($R2$) mechanical properties of TIG-welded M250 maraging steel.
*   **Physics-Informed Deep Learning:**
    *   Developed a Physics-Informed Neural Network (PINN) in PyTorch incorporating physical boundary constraints (e.g., Yield Strength $\le$ UTS and positivity constraints) directly into the loss function, improving model generalizability and ensuring physically consistent property predictions.
*   **Uncertainty Quantification & Explainability (XAI):**
    *   Implemented Monte Carlo Dropout with 50 forward passes to estimate predictive uncertainty ($\pm 2\sigma$ confidence intervals) and integrated SHAP (TreeExplainer and KernelExplainer) to provide transparent feature-importance explanations for critical aerospace weld assessments.
*   **Curve Generation & Smooth Interpolation:**
    *   Designed an LSTM-based parameter generator to forecast $R2$ stress-strain profiles from baseline $R0$ curve inputs, utilizing SciPy's Pchip (Piecewise Cubic Hermite Interpolating Polynomial) interpolation for smooth, physically realistic curve reconstruction.
*   **Desktop Software Engineering:**
    *   Built a professional desktop dashboard using CustomTkinter and Matplotlib, featuring asynchronous model training, interactive curve visualization, a built-in CSV explorer, and automated PDF report generation via ReportLab.

---

### 🛠️ Core Technology Stack

*   **Programming Language:** Python
*   **Deep Learning Framework:** PyTorch
*   **Machine Learning Libraries:** Scikit-Learn, XGBoost, LightGBM, SHAP, Joblib
*   **Data Analysis & Science:** Pandas, NumPy, SciPy, OpenPyXL, Python-docx
*   **GUI & Data Visualization:** CustomTkinter (Tkinter), Matplotlib
*   **Document Generation:** ReportLab (for PDF exports)

---

### 📊 Project Context & Background

*   **Target Application:** Vikram Sarabhai Space Centre (VSSC/ISRO). Predicts $R2$ (second repair) mechanical properties of M250 maraging steel welds (used in booster motor casings / launch vehicle thin-shell structures joined by TIG welding) from $R0$ (original) and optional $R1$ (first repair) measurements.
*   **Prediction Targets (6 Properties):** 
    *   Ultimate Tensile Strength (UTS)
    *   Yield Strength (YS)
    *   Fracture Toughness ($K_{1c}$)
    *   Elongation Percentage (%)
    *   Heat-Affected Zone (HAZ) Width
    *   Austenite Content (%)
*   **Model Architectures Implemented:**
    1.  **Classical ML:** Extra Trees, Random Forest, XGBoost, LightGBM, Gradient Boosting
    2.  **Deep Learning:** 1D-CNN, Bidirectional LSTM (BiLSTM), Transformer, CNN+LSTM Hybrid
    3.  **Physics-Informed DL:** PINN with custom physical loss penalties
    4.  **Meta-Learner:** Stacking Ensemble with a Ridge Regression meta-model
    5.  **Parameter Generator:** LSTM Curve Generator + Pchip curve reconstruction
