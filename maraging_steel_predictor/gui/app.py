"""
Main GUI Application Window with 6 Tabbed Views.
Uses CustomTkinter for a modern, professional look.
"""

import os
import sys
import threading

# Try CustomTkinter, fall back to tkinter
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    CTK = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk
    CTK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MainApp:
    """Main application window with tabbed interface."""

    def __init__(self):
        if CTK:
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()

        self.root.title("M250 Maraging Steel — Weld Repair Property Predictor")
        self.root.geometry("1280x800")
        self.root.minsize(1000, 650)

        # Application state
        self.models = {}
        self.dl_models = {}
        self.scaler = None
        self.feature_cols = None
        self.training_results = {}
        self.best_model_name = None
        self.ensemble = None
        self.tabview = None
        self.predict_tab = None
        self.curves_tab = None
        self.training_tab = None
        self.xai_tab = None
        self.explorer_tab = None
        self.metrics_tab = None

        self._load_existing_models()
        self._build_ui()

    def _load_existing_models(self):
        """Load previously trained models if they exist."""
        import json
        import joblib
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ml_dir = os.path.join(base, 'models', 'ml')
        dl_dir = os.path.join(base, 'models', 'dl')

        # Load scaler
        scaler_path = os.path.join(ml_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # Load feature columns
        feat_path = os.path.join(ml_dir, 'feature_cols.pkl')
        if os.path.exists(feat_path):
            self.feature_cols = joblib.load(feat_path)

        # Load ML models
        for f in ['extratrees.pkl', 'randomforest.pkl', 'gradientboosting.pkl',
                   'xgboost.pkl', 'lightgbm.pkl']:
            path = os.path.join(ml_dir, f)
            if os.path.exists(path):
                name = f.replace('.pkl', '').title()
                self.models[name] = joblib.load(path)

        # Load DL models
        try:
            import torch
            from core.architectures import get_model
            for name in ['cnn', 'lstm', 'transformer', 'hybrid']:
                path = os.path.join(dl_dir, f'{name}.pt')
                if os.path.exists(path):
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    model = get_model(name, data['n_features'])
                    model.load_state_dict(data['model_state'])
                    model.eval()
                    display = {'cnn': 'CNN', 'lstm': 'LSTM',
                               'transformer': 'Transformer',
                               'hybrid': 'CNN+LSTM Hybrid'}[name]
                    self.dl_models[display] = model
        except Exception:
            pass

        # Load ensemble
        ens_path = os.path.join(ml_dir, 'stacking_ensemble.pkl')
        if os.path.exists(ens_path):
            from core.ensemble import StackingEnsemble
            self.ensemble = StackingEnsemble()
            self.ensemble.load(ens_path)

        # Load training results
        res_path = os.path.join(ml_dir, 'training_results.json')
        if os.path.exists(res_path):
            try:
                with open(res_path, 'r') as f:
                    data = json.load(f)
                    self.training_results = data.get('results', {})
                    self.best_model_name = data.get('best_model', None)
            except Exception as e:
                print(f"Warning: Could not load training_results.json: {e}")

    def _build_ui(self):
        """Build the main tabbed interface."""
        if CTK:
            self.tabview = ctk.CTkTabview(self.root)
            self.tabview.pack(fill='both', expand=True, padx=10, pady=10)

            tab_predict = self.tabview.add("🔮 Predict R2")
            tab_curves = self.tabview.add("📈 Stress-Strain")
            tab_training = self.tabview.add("⚙️ Training")
            tab_xai = self.tabview.add("🧠 XAI")
            tab_explorer = self.tabview.add("📊 Data Explorer")
            tab_metrics = self.tabview.add("📊 Detailed Metrics")
            tab_about = self.tabview.add("ℹ️ About")
        else:
            notebook = ttk.Notebook(self.root)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)

            tab_predict = ttk.Frame(notebook)
            tab_curves = ttk.Frame(notebook)
            tab_training = ttk.Frame(notebook)
            tab_xai = ttk.Frame(notebook)
            tab_explorer = ttk.Frame(notebook)
            tab_metrics = ttk.Frame(notebook)
            tab_about = ttk.Frame(notebook)

            notebook.add(tab_predict, text="Predict R2")
            notebook.add(tab_curves, text="Stress-Strain")
            notebook.add(tab_training, text="Training")
            notebook.add(tab_xai, text="XAI")
            notebook.add(tab_explorer, text="Data Explorer")
            notebook.add(tab_metrics, text="Detailed Metrics")
            notebook.add(tab_about, text="About")

        # Import and build tabs
        from gui.tab_predict import PredictTab
        from gui.tab_curves import CurvesTab
        from gui.tab_training import TrainingTab
        from gui.tab_xai import XAITab
        from gui.tab_explorer import ExplorerTab
        from gui.tab_metrics import MetricsTab

        self.predict_tab = PredictTab(tab_predict, self)
        self.curves_tab = CurvesTab(tab_curves, self)
        self.training_tab = TrainingTab(tab_training, self)
        self.xai_tab = XAITab(tab_xai, self)
        self.explorer_tab = ExplorerTab(tab_explorer, self)
        self.metrics_tab = MetricsTab(tab_metrics, self)
        self._build_about_tab(tab_about)

    def _build_about_tab(self, parent):
        """Build the About/Help tab."""
        if CTK:
            frame = ctk.CTkScrollableFrame(parent)
            frame.pack(fill='both', expand=True, padx=20, pady=20)

            ctk.CTkLabel(frame, text="M250 Maraging Steel Weld Repair Property Predictor",
                        font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(0, 5))
            ctk.CTkLabel(frame, text="Vikram Sarabhai Space Centre (VSSC/ISRO)",
                        font=ctk.CTkFont(size=16), text_color="#64b5f6").pack(pady=(0, 20))

            sections = [
                ("About This Tool",
                 "This application predicts the mechanical properties (UTS and Yield Strength) "
                 "of M250 Maraging Steel welds after a second repair cycle (R2), using measurements "
                 "from the original weld (R0) and optionally the first repair (R1).\n\n"
                 "M250 Maraging Steel is used in aerospace launch vehicle thin-shell structures "
                 "joined by TIG welding (LSLF — Longitudinal Seam Launch and Finish)."),
                ("Weld Repair Cycles",
                 "• R0 (Original): UTS ~1585–1650 MPa, YS ~1550–1596 MPa, KIC ~75 MPa√m\n"
                 "• R1 (First Repair): UTS ~1510–1610 MPa, YS ~1475–1565 MPa\n"
                 "• R2 (Second Repair): UTS ~1448 MPa, YS ~1366 MPa — prediction target"),
                ("Model Architecture",
                 "• Classical ML: Extra Trees, Random Forest, XGBoost, LightGBM, GBR\n"
                 "• Deep Learning: 1D-CNN, BiLSTM, Transformer, CNN+LSTM Hybrid\n"
                 "• Meta-Learner: Stacking Ensemble with Ridge Regression\n"
                 "• Curve Generator: LSTM for stress-strain curve prediction\n"
                 "• Uncertainty: Monte Carlo Dropout (50 passes)\n"
                 "• Explainability: SHAP (TreeExplainer / KernelExplainer)"),
                ("Typical Input Ranges",
                 "• R0 UTS: 1585–1650 MPa (S139/S200 specimens)\n"
                 "• R0 Yield Strength: 1550–1596 MPa\n"
                 "• R0 KIC: 60–100 MPa√m\n"
                 "• R1 UTS: 1510–1610 MPa (optional)\n"
                 "• R1 YS: 1475–1565 MPa (optional)"),
            ]

            for title, text in sections:
                ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=15, weight="bold"),
                            anchor="w").pack(fill='x', pady=(10, 2))
                ctk.CTkLabel(frame, text=text, font=ctk.CTkFont(size=12),
                            anchor="w", justify="left", wraplength=800).pack(fill='x', padx=10)
        else:
            import tkinter as tk
            text_widget = tk.Text(parent, wrap='word', font=('Arial', 11))
            text_widget.pack(fill='both', expand=True, padx=20, pady=20)
            text_widget.insert('1.0',
                "M250 Maraging Steel Weld Repair Property Predictor\n"
                "Vikram Sarabhai Space Centre (VSSC/ISRO)\n\n"
                "This application predicts R2 weld repair properties using ML/DL models.\n\n"
                "Weld Repair Cycles:\n"
                "  R0 (Original): UTS ~1585-1650 MPa\n"
                "  R1 (First Repair): UTS ~1510-1610 MPa\n"
                "  R2 (Second Repair): UTS ~1448 MPa (prediction target)\n")
            text_widget.config(state='disabled')

    def run(self):
        """Start the application."""
        self.root.mainloop()
