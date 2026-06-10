"""Generate a technical PPT presentation for the Maraging Steel Predictor project."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
BG = RGBColor(0x0D, 0x11, 0x17)
ACCENT = RGBColor(0x00, 0x9E, 0xFF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0xB0, 0xBE, 0xC5)
DARK2 = RGBColor(0x14, 0x1A, 0x23)
GREEN = RGBColor(0x4C, 0xAF, 0x50)
ORANGE = RGBColor(0xFF, 0x98, 0x00)
PURPLE = RGBColor(0x9C, 0x27, 0xB0)
RED = RGBColor(0xE9, 0x1E, 0x63)

def add_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG

def add_text(slide, left, top, width, height, text, size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = align
    return tf

def add_bullet_slide(slide, left, top, width, height, items, size=16, color=WHITE):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(6)
    return tf

def add_rect(slide, left, top, width, height, fill_color):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape

def add_card(slide, left, top, width, height, title, body, title_color=ACCENT):
    add_rect(slide, left, top, width, height, DARK2)
    add_text(slide, left+0.2, top+0.15, width-0.4, 0.4, title, size=16, color=title_color, bold=True)
    add_bullet_slide(slide, left+0.2, top+0.55, width-0.4, height-0.7, body, size=13, color=GRAY)

# ═══════════ SLIDE 1: Title ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_rect(sl, 0, 2.0, 13.333, 0.06, ACCENT)
add_text(sl, 0.5, 2.3, 12, 1.2, "M250 Maraging Steel", size=44, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
add_text(sl, 0.5, 3.3, 12, 0.8, "Weld Repair Property Predictor — Technical Overview", size=24, color=ACCENT, align=PP_ALIGN.CENTER)
add_text(sl, 0.5, 4.5, 12, 0.5, "Vikram Sarabhai Space Centre (VSSC / ISRO)", size=18, color=GRAY, align=PP_ALIGN.CENTER)
add_text(sl, 0.5, 5.2, 12, 0.5, "ML + Deep Learning + Physics-Informed Neural Networks + Explainable AI", size=14, color=GRAY, align=PP_ALIGN.CENTER)

# ═══════════ SLIDE 2: Problem Statement ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "Problem Statement & Motivation", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

items = [
    "• M250 maraging steel is critical for aerospace launch vehicle thin-shell structures (TIG welding — LSLF)",
    "• Weld repairs degrade mechanical properties across cycles: R0 → R1 → R2",
    "• R0 (Original): UTS ~1585–1650 MPa, YS ~1550–1596 MPa, KIC ~75 MPa√m",
    "• R1 (First Repair): UTS ~1510–1610 MPa, YS ~1475–1565 MPa",
    "• R2 (Second Repair): UTS ~1448 MPa, YS ~1366 MPa — the prediction target",
    "",
    "• Goal: Predict 6 mechanical properties after 2nd repair (R2) from R0/R1 measurements",
    "• Target Properties: UTS, Yield Strength, Elongation, HAZ Width, Gauge Depth, Austenite Content",
    "• Challenge: Limited physical test data → need robust ML/DL generalization + uncertainty quantification"
]
add_bullet_slide(sl, 0.7, 1.3, 11.5, 5.5, items, size=17, color=WHITE)

# ═══════════ SLIDE 3: System Architecture ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "System Architecture Overview", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

add_card(sl, 0.5, 1.3, 3.8, 2.5, "Data Pipeline", [
    "• 4000-sample synthetic dataset",
    "• Pivot: long → wide format",
    "• One-hot: Specimen & Location",
    "• R0 + R1 feature concatenation",
    "• Stratified train/test split (80/20)",
])
add_card(sl, 4.6, 1.3, 3.8, 2.5, "Model Training", [
    "• 5 Classical ML models",
    "• 5 Deep Learning models",
    "• Stacking Ensemble (Ridge meta)",
    "• LSTM Curve Generator",
    "• RandomizedSearchCV (50 iter, 5-CV)",
])
add_card(sl, 8.7, 1.3, 4.1, 2.5, "Inference & GUI", [
    "• CustomTkinter 6-tab interface",
    "• MC Dropout uncertainty (50 passes)",
    "• SHAP explainability (Tree/Kernel)",
    "• PDF report generation",
    "• Stress-strain curve reconstruction",
])
add_card(sl, 0.5, 4.1, 6.0, 2.8, "Input Features (20 total)", [
    "R0: UTS, YS, K1c, Elongation, HAZ Width, Gauge Depth, Austenite Content (7)",
    "R1: UTS, YS, K1c, Elongation, HAZ Width, Gauge Depth, Austenite Content (7)",
    "Categorical: Specimen_Type (S139/S200), Repair_Location (Center/Edge/Root/Toe) → one-hot (6)",
])
add_card(sl, 6.8, 4.1, 6.0, 2.8, "Output Targets (6 properties)", [
    "R2_UTS — Ultimate Tensile Strength (MPa)",
    "R2_YS — Yield Strength (MPa)",
    "R2_Elongation — Elongation at break (%)",
    "R2_HAZ_Width — Heat-Affected Zone width (mm)",
    "R2_Gauge_Depth — Gauge depth (mm)",
    "R2_Austenite_Content — Retained austenite (%)",
])

# ═══════════ SLIDE 4: Classical ML Models ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "Classical ML Models", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

add_card(sl, 0.5, 1.3, 4.0, 2.6, "Extra Trees Regressor", [
    "• Extremely Randomized Trees",
    "• n_estimators: [100–500]",
    "• max_depth: [None, 10, 20, 30]",
    "• Wrapped in MultiOutputRegressor",
    "• Best overall R² in most runs",
], title_color=GREEN)

add_card(sl, 4.8, 1.3, 4.0, 2.6, "Random Forest", [
    "• Bootstrap aggregation ensemble",
    "• n_estimators: [100–500]",
    "• max_depth: [None, 10, 20, 30]",
    "• min_samples_split: [2, 5, 10]",
    "• Robust baseline model",
], title_color=GREEN)

add_card(sl, 9.1, 1.3, 3.7, 2.6, "Gradient Boosting", [
    "• Sequential error correction",
    "• learning_rate: [0.01, 0.05, 0.1]",
    "• n_estimators: [100–300]",
    "• max_depth: [3, 5, 7]",
    "• Strong on complex patterns",
], title_color=GREEN)

add_card(sl, 0.5, 4.2, 4.0, 2.6, "XGBoost", [
    "• Extreme Gradient Boosting",
    "• subsample: [0.7–1.0]",
    "• learning_rate: [0.01–0.1]",
    "• Regularized (L1/L2)",
    "• GPU-accelerated option",
], title_color=ORANGE)

add_card(sl, 4.8, 4.2, 4.0, 2.6, "LightGBM", [
    "• Gradient-based One-Side Sampling",
    "• num_leaves: [31, 50, 70]",
    "• Histogram-based splitting",
    "• Very fast training",
    "• Leaf-wise tree growth",
], title_color=ORANGE)

add_card(sl, 9.1, 4.2, 3.7, 2.6, "Training Strategy", [
    "• RandomizedSearchCV",
    "• 50 iterations, 5-fold CV",
    "• Scoring metric: R²",
    "• Stratified split by UTS bins",
    "• Metrics: MAE, RMSE, R², Acc%",
], title_color=PURPLE)

# ═══════════ SLIDE 5: Deep Learning Architectures ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "Deep Learning Architectures", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

add_card(sl, 0.5, 1.3, 4.0, 2.8, "1D-CNN Regressor", [
    "Conv1d(1→64, k=3) → BN → ReLU",
    "Conv1d(64→128, k=3) → BN → ReLU",
    "AdaptiveMaxPool1d(1)",
    "Dense(128→64) → Dropout(0.3)",
    "Dense(64→6 outputs)",
    "Treats features as 1D spatial signal",
], title_color=RED)

add_card(sl, 4.8, 1.3, 4.0, 2.8, "Bidirectional LSTM", [
    "BiLSTM(1→128) → Dropout(0.2)",
    "BiLSTM(256→64)",
    "Last time-step extraction",
    "Dense(128→32) → Dense(32→6)",
    "Each feature = one time step",
    "Captures sequential dependencies",
], title_color=RED)

add_card(sl, 9.1, 1.3, 3.7, 2.8, "Transformer Regressor", [
    "Linear(1→64) feature embedding",
    "Sinusoidal positional encoding",
    "2× TransformerEncoderLayer",
    "4-head self-attention, FFN(128)",
    "Flatten → Dense(128) → Dense(6)",
    "Global feature-to-feature attention",
], title_color=RED)

add_card(sl, 0.5, 4.4, 4.0, 2.7, "CNN + LSTM Hybrid", [
    "Conv1d(1→64, k=3) → BN → ReLU",
    "AdaptiveMaxPool1d(n_feat//2)",
    "BiLSTM(64→64) on pooled output",
    "Dense(128→32) → Dense(32→6)",
    "CNN: local patterns",
    "LSTM: sequential dependencies",
], title_color=PURPLE)

add_card(sl, 4.8, 4.4, 4.0, 2.7, "Physics-Informed NN (PINN)", [
    "Dense(n→128) → BN → ReLU → Drop(0.2)",
    "Dense(128→64) → BN → ReLU → Drop(0.2)",
    "Dense(64→32) → ReLU → Dense(32→6)",
    "Physics loss: UTS ≥ YS constraint",
    "Positivity: all outputs ≥ 0",
    "Loss = MSE + α·(physics + positivity)",
], title_color=PURPLE)

add_card(sl, 9.1, 4.4, 3.7, 2.7, "DL Training Config", [
    "Optimizer: Adam (lr=1e-3, wd=1e-5)",
    "Scheduler: ReduceLROnPlateau",
    "Max epochs: 300, patience: 20",
    "Batch size: 32",
    "Gradient clipping: max_norm=1.0",
    "StandardScaler normalization",
], title_color=ACCENT)

# ═══════════ SLIDE 6: Stacking Ensemble ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "Stacking Ensemble & Curve Generation", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

add_card(sl, 0.5, 1.3, 6.0, 3.0, "Stacking Ensemble — Ridge Meta-Learner", [
    "• Collects predictions from ALL 10 base models (5 ML + 5 DL)",
    "• Horizontally stacks predictions → (N, 10×6) meta-feature matrix",
    "• Ridge Regression (α=1.0) as meta-learner on stacked predictions",
    "• ML models predict on raw features; DL models on scaled features",
    "• Automatically balances model strengths across all 6 targets",
    "• Consistently achieves best R² by combining complementary models",
], title_color=ACCENT)

add_card(sl, 6.8, 1.3, 6.0, 3.0, "LSTM Curve Generator", [
    "• Predicts stress-strain curve parameters from R0 → R2",
    "• Architecture: LSTM(1→64, 2-layer) → Dense(64→32) → Dense(32→6)",
    "• Input: R0 curve params (elastic slope, yield/UTS/fracture strain, etc.)",
    "• Output: R2 yield_strain, uts_strain, fracture_strain, yield_stress, UTS, AUC",
    "• Pchip interpolation for smooth curve reconstruction",
    "• Enforces 190 GPa Young's Modulus (M250 physical constant)",
], title_color=ORANGE)

add_card(sl, 0.5, 4.6, 6.0, 2.5, "Stress-Strain Curve Reconstruction", [
    "• 8 key points: origin → elastic → yield → hardening → UTS → fracture",
    "• PchipInterpolator (monotone cubic) for 200-point smooth curves",
    "• Reference curves: Parent Material, R0, R1, R2 (from experimental data)",
    "• Markers: Yield Point, UTS, Fracture Point annotated on curves",
], title_color=GREEN)

add_card(sl, 6.8, 4.6, 6.0, 2.5, "Evaluation Metrics", [
    "• MAE — Mean Absolute Error (per-target & averaged)",
    "• RMSE — Root Mean Squared Error",
    "• R² — Coefficient of Determination",
    "• Accuracy% — 100 − MAPE (per-target, capped at 0%)",
    "• Per-target breakdown for all 6 mechanical properties",
], title_color=GREEN)

# ═══════════ SLIDE 7: XAI & Uncertainty ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "Explainability (XAI) & Uncertainty Quantification", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

add_card(sl, 0.5, 1.3, 6.0, 3.0, "SHAP Explainability Module", [
    "• TreeExplainer: exact SHAP values for tree-based ML models (fast)",
    "• KernelExplainer: model-agnostic fallback for DL models (~30s)",
    "• MultiOutputRegressor unwrapping for per-target tree analysis",
    "• Per-target feature importance: select any of 6 output properties",
    "• Cross-target comparison: grouped bar chart across all targets",
    "• Caching system: (model_name, target_idx) → importance dict",
    "• Dynamic 4-paragraph insight text explaining top-3 features",
], title_color=PURPLE)

add_card(sl, 6.8, 1.3, 6.0, 3.0, "Monte Carlo Dropout Uncertainty", [
    "• Keeps dropout layers ON during inference (model.train() mode)",
    "• Runs 50 stochastic forward passes per prediction",
    "• Computes: mean prediction, std deviation, all 50 predictions",
    "• 95% CI: mean ± 2σ → minimum guaranteed values",
    "• Applied to all DL models (CNN, LSTM, Transformer, Hybrid, PINN)",
    "• Captures epistemic uncertainty from model parameter space",
    "• Displayed alongside predictions in GUI with ±CI bars",
], title_color=RED)

add_card(sl, 0.5, 4.6, 12.3, 2.5, "XAI Visualization & Insights", [
    "• Horizontal bar chart: top-15 features ranked by mean |SHAP value| (dark theme, magma colormap)",
    "• Grouped bar chart: 6-target comparison with color-coded targets (top-10 features)",
    "• Feature ranking panel with rank badges (🥇🥈🥉), hover tooltips for metallurgical context",
    "• Auto-generated text: 4 paragraphs explaining SHAP methodology, graph interpretation, top features, and metallurgical alignment",
    "• Supports all models: ML (TreeExplainer), DL via wrapper class (KernelExplainer), Ensemble (best ML proxy)",
])

# ═══════════ SLIDE 8: GUI & Application ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "GUI Application — 7-Tab Interface", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

tabs = [
    ("🔮 Predict R2", "R0/R1 input fields, model selector, 6-property prediction cards, R0→R1→R2 comparison table, PDF export", GREEN),
    ("📈 Stress-Strain", "Interactive curve plots (Parent, R0, R1, R2 predicted), Pchip interpolation, key-point markers", ORANGE),
    ("⚙️ Training", "Full pipeline trigger, real-time progress, per-model results table, dataset file selector", RED),
    ("🧠 XAI", "SHAP computation, per-target/cross-target charts, feature ranking, dynamic insight paragraphs", PURPLE),
    ("📊 Data Explorer", "Dataset visualization, feature distributions, correlation analysis", ACCENT),
    ("📊 Detailed Metrics", "Per-model, per-target MAE/RMSE/R²/Acc%, comparison dashboard", RGBColor(0x00, 0xBC, 0xD4)),
    ("ℹ️ About", "Tool description, weld repair cycle info, model architecture summary, input ranges", GRAY),
]

for i, (name, desc, col) in enumerate(tabs):
    row = i // 2
    c = i % 2
    left = 0.5 + c * 6.4
    top = 1.3 + row * 1.5
    add_card(sl, left, top, 6.0, 1.3, name, [desc], title_color=col)

# ═══════════ SLIDE 9: Tech Stack ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_text(sl, 0.5, 0.3, 12, 0.6, "Technology Stack & Dependencies", size=32, color=ACCENT, bold=True)
add_rect(sl, 0.5, 1.0, 12.3, 0.04, ACCENT)

add_card(sl, 0.5, 1.3, 4.0, 2.8, "Deep Learning", [
    "• PyTorch — model architectures",
    "• CUDA support (auto-detect)",
    "• Custom loss functions (PINN)",
    "• Adam + ReduceLROnPlateau",
    "• Gradient clipping",
], title_color=RED)

add_card(sl, 4.8, 1.3, 4.0, 2.8, "Machine Learning", [
    "• scikit-learn — RF, ET, GBR, Ridge",
    "• XGBoost — gradient boosting",
    "• LightGBM — fast tree ensemble",
    "• RandomizedSearchCV tuning",
    "• MultiOutputRegressor wrapper",
], title_color=GREEN)

add_card(sl, 9.1, 1.3, 3.7, 2.8, "Data & Visualization", [
    "• Pandas — data wrangling",
    "• NumPy — numerical ops",
    "• Matplotlib — plotting",
    "• SciPy — Pchip interpolation",
    "• SHAP — explainability",
], title_color=ORANGE)

add_card(sl, 0.5, 4.4, 6.0, 2.7, "GUI & Reporting", [
    "• CustomTkinter — modern dark-mode desktop GUI",
    "• tkinter fallback for compatibility",
    "• ReportLab — PDF report generation",
    "• Threaded inference — non-blocking UI",
    "• FigureCanvasTkAgg — embedded matplotlib charts",
], title_color=PURPLE)

add_card(sl, 6.8, 4.4, 6.0, 2.7, "Project Structure", [
    "• main.py — entry point, auto-train on first run",
    "• train_all.py — full ML+DL training pipeline (690 LOC)",
    "• core/ — architectures, ensemble, SHAP, uncertainty, curves",
    "• gui/ — 7 tabs (predict, curves, training, XAI, explorer, metrics, about)",
    "• models/ — saved ML (.pkl) and DL (.pt) checkpoints",
], title_color=ACCENT)

# ═══════════ SLIDE 10: Summary ═══════════
sl = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(sl)
add_rect(sl, 0, 2.8, 13.333, 0.06, ACCENT)
add_text(sl, 0.5, 1.0, 12, 0.8, "Summary", size=36, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)

items = [
    "✅  10 models (5 ML + 5 DL) with hyperparameter-tuned training pipeline",
    "✅  Stacking ensemble with Ridge meta-learner for optimal predictions",
    "✅  Physics-Informed Neural Network enforcing UTS ≥ YS and non-negativity",
    "✅  Monte Carlo Dropout for 95% confidence intervals on predictions",
    "✅  SHAP explainability with per-target and cross-target analysis",
    "✅  LSTM-based stress-strain curve generation with Pchip reconstruction",
    "✅  Professional 7-tab GUI with dark theme, PDF export, and real-time training",
    "✅  Predicts 6 mechanical properties of M250 maraging steel after 2nd weld repair",
]
add_bullet_slide(sl, 1.5, 3.2, 10, 4.0, items, size=18, color=WHITE)

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Maraging_Steel_Predictor_Technical.pptx")
prs.save(out_path)
print(f"PPT saved to: {out_path}")
