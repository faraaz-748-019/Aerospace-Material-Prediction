import docx
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import datetime

def add_heading(doc, text, level=1):
    heading = doc.add_heading(text, level=level)
    return heading

def add_paragraph(doc, text):
    return doc.add_paragraph(text)

def main():
    doc = docx.Document()
    
    # Title
    title = doc.add_heading('M250 Maraging Steel: Weld Repair Property Predictor', 0)
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_paragraph('Methodology and Technical Approach Document').alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_paragraph(f'Date: {datetime.datetime.now().strftime("%Y-%m-%d")}').alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    doc.add_page_break()

    # Introduction
    add_heading(doc, '1. Introduction', level=1)
    p = add_paragraph(doc, "The M250 Maraging Steel Weld Repair Property Predictor is a sophisticated desktop application aimed at predicting the second repair (R2) mechanical properties of Maraging Steel welds based on original (R0) and first repair (R1) characteristics. The tool encapsulates classical machine learning approaches alongside advanced deep learning architectures within an intuitive graphical user interface (GUI) developed in Python.")

    # Data Handling and Processing
    add_heading(doc, '2. Data Handling and Processing', level=1)
    
    add_heading(doc, '2.1 Dataset Structure', level=2)
    p = add_paragraph(doc, "The primary dataset leverages synthetic or historical data comprising varying sample properties across Original (R0), Repair 1 (R1), and Repair 2 (R2) conditions. The properties tracked include Ultimate Tensile Strength (UTS), Yield Strength (YS), Fracture Toughness (K1c), Elongation, HAZ Width, Gauge Depth, and Austenite Content.")
    
    add_heading(doc, '2.2 Preprocessing Pipeline', level=2)
    p = add_paragraph(doc, "Data is preprocessed dynamically within the training pipeline. The steps involve:")
    doc.add_paragraph("• Pivoting: Raw data collected per sample per condition is pivoted into a broad format so features from R0 and R1 act as inputs, and R2 data acts as targets.", style='List Bullet')
    doc.add_paragraph("• Categorical Variables: Categorical columns such as 'Specimen_Type' (S139, S200) and 'Repair_Location' (Center, Edge, Root, Toe) are one-hot encoded for ingestion by all predictive models.", style='List Bullet')
    doc.add_paragraph("• Scaling: Inputs targetting deep-learning models are inherently standard-scaled (zero mean, unit variance) using Scikit-Learn's StandardScaler.", style='List Bullet')

    # Modeling Architecture
    add_heading(doc, '3. Modeling Architecture', level=1)
    p = add_paragraph(doc, "To ensure optimal predictive capabilities, the application trains a multitude of models across a multi-target regression schema (predicting up to 6 targets simultaneously).")

    add_heading(doc, '3.1 Classical Machine Learning', level=2)
    p = add_paragraph(doc, "We employ Scikit-Learn's multi-output wrapper over established supervised regressors:")
    doc.add_paragraph("• Extra Trees and Random Forest Regressors", style='List Bullet')
    doc.add_paragraph("• Gradient Boosting Regressor", style='List Bullet')
    doc.add_paragraph("• XGBoost Regressor (with early stopping characteristics)", style='List Bullet')
    doc.add_paragraph("• LightGBM Regressor", style='List Bullet')
    p2 = add_paragraph(doc, "To achieve best predictive potential, hyperparameter tuning uses RandomizedSearchCV spanning 50 iterations over a 5-fold cross-validation methodology, relying on R-squared metric acting as the scoring base.")

    add_heading(doc, '3.2 Deep Learning Modules', level=2)
    p = add_paragraph(doc, "A comprehensive PyTorch-based module builds several architectures aiming at discovering subtle, non-linear relationships across conditions:")
    doc.add_paragraph("• 1D-CNN: Convolutional feature extractions via Conv1D blocks mixed with global max-pooling.", style='List Bullet')
    doc.add_paragraph("• BiLSTM: Capturing sequential deterioration mappings using Bidirectional Long-Short Term Memory networks.", style='List Bullet')
    doc.add_paragraph("• Transformer: Employs multi-head self-attention mechanisms leveraging continuous encodings.", style='List Bullet')
    doc.add_paragraph("• Hybrid CNN-LSTM: Combined extraction using CNNs for local property changes and LSTMs for overarching conditional dependencies.", style='List Bullet')
    doc.add_paragraph("• Physics-Informed Neural Networks (PINNs): Restricts anomalies by embedding a custom physics loss during training (e.g. Guaranteeing Yield Strength never exceeds Ultimate Tensile Strength radially).", style='List Bullet')

    add_heading(doc, '3.3 Meta Learner (Stacking Ensemble)', level=2)
    p = add_paragraph(doc, "Out-of-fold predicted features generated through the best classical and deep learning approaches are aggregated into a Ridge-regression based Stacking Ensemble. This leverages overlapping predictive capabilities from disparate network families to augment baseline accuracy.")

    add_heading(doc, '3.4 LSTM Curve Generator', level=2)
    p = add_paragraph(doc, "A secondary predictive dimension utilizes an independent LSTM curve generator modeling complete stress-strain graphical progression based on corresponding continuous data coordinates alongside SciPy's PCHIP monotonic curve interpolations.")

    # Uncertainty estimation
    add_heading(doc, '4. Uncertainty & Explainability', level=1)
    p = add_paragraph(doc, "Confidence in physical engineering pipelines stands paramount.")
    
    add_heading(doc, '4.1 Monte Carlo Dropout Uncertainty', level=2)
    p = add_paragraph(doc, "Uncertainty approximations rely on enabling dropout layers sequentially during inference steps (Monte Carlo Dropout). Executing 50 forward passes allows the extrapolation of mean values bracketed by ± 2 sigma deviation bounds, explicitly mapping output confidence margins per predictive sample.")

    add_heading(doc, '4.2 eXplainable AI (XAI) using SHAP', level=2)
    p = add_paragraph(doc, "The interpretability is enhanced via SHAP (SHapley Additive exPlanations):")
    doc.add_paragraph("• TreeExplainer is leveraged for Random Forest and Boosting models.", style='List Bullet')
    doc.add_paragraph("• KernelExplainer maps deep learning representations.", style='List Bullet')
    p = add_paragraph(doc, "Graphical representations reflect the respective impact size each feature imparts strictly towards R2 target computations giving metallurgy researchers insights over causal features.")

    # GUI and Implementation
    add_heading(doc, '5. GUI & Modular Implementation', level=1)
    p = add_paragraph(doc, "The tool integrates CustomTkinter to deliver a modern visual experience compartmentalized cleanly:")
    doc.add_paragraph("• Predictive Tab: Rapid multi-target projection.", style='List Bullet')
    doc.add_paragraph("• Curve Interpolation Tab: Generates comprehensive real-world graphical representations mirroring lab stress-strain curves.", style='List Bullet')
    doc.add_paragraph("• Training Flow: Houses dynamic progress bars attached to isolated pipeline tasks, supporting real-time cross-accuracy tracking.", style='List Bullet')
    doc.add_paragraph("• XAI Tab: Graphically reflects SHAP variables relative to a given localized selection.", style='List Bullet')
    p = add_paragraph(doc, "Strict abstraction is kept using a 'core' engine dealing structurally with mathematical and architecture duties while mapping directly to a 'gui' package driving display callbacks.")

    doc.save("c:\\Users\\Acer\\Desktop\\MATERIAL SCI-anti\\maraging_steel_predictor\\Methodology_and_Technical_Approach.docx")

if __name__ == '__main__':
    main()
