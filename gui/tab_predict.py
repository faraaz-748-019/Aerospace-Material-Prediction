"""
Tab 1: Predict R2 Properties
Input fields for R0/R1 properties, model selector, prediction results display.
"""

import os
import sys
import threading
import numpy as np

try:
    import customtkinter as ctk
    CTK = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk, messagebox
    CTK = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_all import TARGETS, R0_FEATURES, R1_FEATURES, CATEGORICAL_FEATURES

class PredictTab:
    """Prediction tab with R0/R1 inputs and results display."""

    # Typical ranges from dummy generated values, just for warnings
    VALID_RANGES = {
        'R0_UTS': (1400, 1800),
        'R0_YS': (1300, 1750),
        'R0_Elongation': (0, 25),
    }

    def __init__(self, parent, app):
        self.app = app
        self.inputs = {}
        self.r1_inputs = {}
        self.show_r1 = False
        self._build_ui(parent)

    def _build_ui(self, parent):
        if CTK:
            self._build_ctk(parent)
        else:
            self._build_tk(parent)

    def _build_ctk(self, parent):
        # Main container
        main = ctk.CTkFrame(parent, fg_color="transparent")
        main.pack(fill='both', expand=True, padx=10, pady=10)

        # Left panel: inputs (scrollable because of many inputs)
        left = ctk.CTkScrollableFrame(main, width=420)
        left.pack(side='left', fill='y', padx=(0, 10), pady=0)

        ctk.CTkLabel(left, text="Global Inputs",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(5, 5))
                    
        # Specimen Type
        g_frame = ctk.CTkFrame(left, fg_color="transparent")
        g_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(g_frame, text="Specimen Type", width=160, anchor='w').pack(side='left')
        self.specimen_var = ctk.StringVar(value="S139")
        ctk.CTkComboBox(g_frame, values=["S139", "S200"], variable=self.specimen_var, width=140).pack(side='right')

        # Repair Location
        l_frame = ctk.CTkFrame(left, fg_color="transparent")
        l_frame.pack(fill='x', padx=5, pady=2)
        ctk.CTkLabel(l_frame, text="Repair Location", width=160, anchor='w').pack(side='left')
        self.location_var = ctk.StringVar(value="Center")
        ctk.CTkComboBox(l_frame, values=["Center", "Edge", "Root", "Toe"], variable=self.location_var, width=140).pack(side='right')

        ctk.CTkLabel(left, text="R0 Input Properties",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))

        r0_fields = [
            ('R0_UTS', 'R0 UTS (MPa)', '1617'),
            ('R0_YS', 'R0 YS (MPa)', '1573'),
            ('R0_K1c', 'R0 K1c (MPa√m)', '75.0'),
            ('R0_Elongation', 'R0 Elongation (%)', '10.0'),
            ('R0_HAZ_Width', 'R0 HAZ Width (mm)', '5.0'),
            ('R0_Gauge_Depth', 'R0 Gauge Depth (mm)', '2.0'),
            ('R0_Austenite_Content', 'R0 Austenite (%)', '2.5'),
        ]

        for key, label, default in r0_fields:
            frame = ctk.CTkFrame(left, fg_color="transparent")
            frame.pack(fill='x', padx=5, pady=2)
            ctk.CTkLabel(frame, text=label, width=160, anchor='w',
                        font=ctk.CTkFont(size=12)).pack(side='left')
            entry = ctk.CTkEntry(frame, width=140, placeholder_text=default)
            entry.insert(0, default)
            entry.pack(side='right')
            self.inputs[key] = entry

        # R1 toggle
        self.r1_var = ctk.BooleanVar(value=False)
        r1_toggle = ctk.CTkSwitch(left, text="Include R1 Properties (Optional)",
                                   variable=self.r1_var, command=self._toggle_r1)
        r1_toggle.pack(pady=15)

        # R1 fields frame (initially hidden)
        self.r1_frame = ctk.CTkFrame(left, fg_color="transparent")

        r1_fields = [
            ('R1_UTS', 'R1 UTS (MPa)', '1560'),
            ('R1_YS', 'R1 YS (MPa)', '1520'),
            ('R1_K1c', 'R1 K1c (MPa√m)', '71.0'),
            ('R1_Elongation', 'R1 Elongation (%)', '8.0'),
            ('R1_HAZ_Width', 'R1 HAZ Width (mm)', '5.5'),
            ('R1_Gauge_Depth', 'R1 Gauge Depth (mm)', '2.1'),
            ('R1_Austenite_Content', 'R1 Austenite (%)', '3.5'),
        ]

        for key, label, default in r1_fields:
            frame = ctk.CTkFrame(self.r1_frame, fg_color="transparent")
            frame.pack(fill='x', padx=5, pady=2)
            ctk.CTkLabel(frame, text=label, width=160, anchor='w',
                        font=ctk.CTkFont(size=12)).pack(side='left')
            entry = ctk.CTkEntry(frame, width=140, placeholder_text=default)
            entry.insert(0, default)
            entry.pack(side='right')
            self.r1_inputs[key] = entry

        # Model selector
        ctk.CTkLabel(left, text="Model Selection",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        model_options = ["Auto-Best", "CNN", "LSTM", "Transformer", "CNN+LSTM Hybrid",
                         "Physics-Informed NN", "Stacking Ensemble", "ExtraTrees", "XGBoost", "LightGBM",
                         "RandomForest", "GradientBoosting"]
        self.model_var = ctk.StringVar(value="Auto-Best")
        self.model_dropdown = ctk.CTkComboBox(left, values=model_options,
                                               variable=self.model_var, width=200)
        self.model_dropdown.pack(pady=5)

        # Predict button
        self.predict_btn = ctk.CTkButton(left, text="🔬  PREDICT R2 PROPERTIES",
                                          font=ctk.CTkFont(size=16, weight="bold"),
                                          height=50, fg_color="#1565c0",
                                          hover_color="#0d47a1",
                                          command=self._predict_threaded)
        self.predict_btn.pack(pady=15, padx=20, fill='x')

        # Right panel: results
        right = ctk.CTkScrollableFrame(main)
        right.pack(side='right', fill='both', expand=True)

        ctk.CTkLabel(right, text="Prediction Results",
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 10))

        # Dynamic construction of 6 result frames
        self.result_frames = {}
        self.result_labels = {}
        self.result_cis = {}

        # Layout grids (2 columns)
        results_grid = ctk.CTkFrame(right, fg_color="transparent")
        results_grid.pack(fill='x', padx=5, pady=5)
        
        colors = ['#1a237e', '#1b5e20', '#b71c1c', '#e65100', '#4a148c', '#004d40']
        
        for idx, target in enumerate(TARGETS):
            prop_name = target.replace('R2_', '').replace('_', ' ')
            row, col = idx // 2, idx % 2
            
            frame = ctk.CTkFrame(results_grid, fg_color=colors[idx], corner_radius=10, width=200)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
            
            ctk.CTkLabel(frame, text=f"Predicted {prop_name}",
                        font=ctk.CTkFont(size=12), text_color="#e0e0e0").pack(pady=(8, 0))
            val_label = ctk.CTkLabel(frame, text="—", font=ctk.CTkFont(size=24, weight="bold"), text_color="white")
            val_label.pack(pady=(0, 4))
            ci_label = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=10), text_color="#b0bec5")
            ci_label.pack(pady=(0, 8))
            
            self.result_frames[target] = frame
            self.result_labels[target] = val_label
            self.result_cis[target] = ci_label
            
        # Comparison table
        ctk.CTkLabel(right, text="R0 → R1 → R2 Comparison",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        self.comparison_frame = ctk.CTkFrame(right)
        self.comparison_frame.pack(fill='x', padx=15, pady=5)

        # Export button
        self.export_btn = ctk.CTkButton(right, text="📄 Export PDF Report",
                                         command=self._export_pdf,
                                         fg_color="#4caf50", hover_color="#2e7d32")
        self.export_btn.pack(pady=15)

        # Status
        self.status_label = ctk.CTkLabel(right, text="Enter R0 values and click PREDICT",
                                          font=ctk.CTkFont(size=11), text_color="gray")
        self.status_label.pack(pady=5)

    def _build_tk(self, parent):
        """Fallback tkinter UI. Simplified for 6 targets."""
        import tkinter as tk
        tk.Label(parent, text="Predict R2 Properties (CTk highly recommended)", font=('Arial', 14)).pack()
        self.result_text = tk.Text(parent, height=20, font=('Courier', 11))
        self.result_text.pack(fill='both', expand=True, padx=20)

    def _toggle_r1(self):
        if self.r1_var.get():
            self.r1_frame.pack(fill='x', padx=0, pady=5)
        else:
            self.r1_frame.pack_forget()

    def _get_input_values(self):
        """Get and validate all input values."""
        values = {}
        warnings = []

        for key, entry in self.inputs.items():
            try:
                val = float(entry.get())
                values[key] = val
            except ValueError:
                raise ValueError(f"Invalid value for {key}")

        if self.show_r1 or (hasattr(self, 'r1_var') and self.r1_var.get()):
            for key, entry in self.r1_inputs.items():
                try:
                    val = float(entry.get())
                    values[key] = val
                except ValueError:
                    raise ValueError(f"Invalid value for {key}")

        return values, warnings
        
    def _predict_threaded(self):
        if CTK:
            self.status_label.configure(text="⏳ Predicting...")
        thread = threading.Thread(target=self._predict, daemon=True)
        thread.start()

    def _predict(self):
        """Run prediction."""
        try:
            values, warnings = self._get_input_values()

            if warnings and CTK:
                self.app.root.after(0, lambda: self.status_label.configure(text=f"⚠️ Warnings: {'; '.join(warnings)}"))

            # Build feature vector
            import torch
            from core.uncertainty import mc_dropout_predict

            feature_cols = getattr(self.app, 'feature_cols', None)
            if feature_cols is None:
                feature_cols = R0_FEATURES + CATEGORICAL_FEATURES
                if any(k.startswith('R1_') for k in values):
                    feature_cols = R0_FEATURES + R1_FEATURES + CATEGORICAL_FEATURES
                    
            # Extract categorical
            if hasattr(self, 'specimen_var'):
                spec = self.specimen_var.get()
                loc = self.location_var.get()
                values[f'Specimen_Type_{spec}'] = 1.0
                values[f'Repair_Location_{loc}'] = 1.0

            X = np.array([[values.get(c, 0.0) for c in feature_cols]], dtype=np.float32)

            selected_model = self.model_var.get() if CTK else "Auto-Best"
            
            if selected_model == "Auto-Best":
                selected_model = getattr(self.app, 'best_model_name', 'ExtraTrees')

            pred_mean = None
            pred_std = np.zeros(len(TARGETS))

            # Try DL model first
            dl_name_map = {'CNN': 'CNN', 'LSTM': 'LSTM', 'Transformer': 'Transformer',
                           'CNN+LSTM Hybrid': 'CNN+LSTM Hybrid',
                           'Physics-Informed NN': 'Physics-Informed NN'}
            ml_name_map = {'ExtraTrees': 'Extratrees', 'XGBoost': 'Xgboost',
                           'LightGBM': 'Lightgbm', 'RandomForest': 'Randomforest',
                           'GradientBoosting': 'Gradientboosting'}

            if selected_model in dl_name_map and selected_model in getattr(self.app, 'dl_models', {}):
                model = self.app.dl_models[selected_model]
                X_scaled = self.app.scaler.transform(X) if getattr(self.app, 'scaler', None) else X
                X_t = torch.FloatTensor(X_scaled)
                mean_pred, std_pred, _, min_guaranteed = mc_dropout_predict(model, X_t)
                pred_mean = mean_pred
                pred_std = std_pred * 2

            elif selected_model in ml_name_map and ml_name_map[selected_model] in getattr(self.app, 'models', {}):
                model = self.app.models[ml_name_map[selected_model]]
                pred_mean = model.predict(X)[0]

            elif selected_model == 'Stacking Ensemble' and getattr(self.app, 'ensemble', None):
                all_preds = {}
                import torch
                
                # Predict with ML models
                for name, m in self.app.models.items():
                    try:
                        p = m.predict(X)
                        if p.ndim == 1: p = p.reshape(-1, 1)
                        all_preds[name] = p
                    except Exception as e:
                        print(f"Ensemble ML pred error {name}: {e}")

                # Predict with DL models (ensure scaling)
                if getattr(self.app, 'dl_models', None):
                    dl_scaler = self.app.scaler.transform(X) if getattr(self.app, 'scaler', None) else X
                    dl_X = torch.FloatTensor(dl_scaler)
                    for name, m in self.app.dl_models.items():
                        try:
                            m.eval()
                            with torch.no_grad():
                                p = m(dl_X).cpu().numpy()
                            if p.ndim == 1: p = p.reshape(-1, 1)
                            all_preds[name] = p
                        except Exception as e:
                            print(f"Ensemble DL pred error {name}: {e}")

                if all_preds:
                    # Meta-learner expects specific base model names (case sensitive)
                    base_names = self.app.ensemble.base_model_names
                    normalized_preds = {}
                    
                    # Case-insensitive resolving of predictions for the meta-learner
                    lower_preds = {k.lower(): v for k, v in all_preds.items()}
                    for bn in base_names:
                        if bn in all_preds:
                            normalized_preds[bn] = all_preds[bn]
                        elif bn.lower() in lower_preds:
                            normalized_preds[bn] = lower_preds[bn.lower()]
                        else:
                            raise ValueError(f"Missing base model prediction: {bn}")
                            
                    pred_mean = self.app.ensemble.predict(normalized_preds)[0]
            else:
                for name, m in getattr(self.app, 'models', {}).items():
                    try:
                        pred_mean = m.predict(X)[0]
                        selected_model = name
                        break
                    except: pass

            if pred_mean is None:
                if CTK:
                    self.app.root.after(0, lambda: self.status_label.configure(text="❌ No trained models found. Train models first!"))
                return

            self.last_prediction = {
                'r0_inputs': {k: v for k, v in values.items() if k.startswith('R0_')},
                'model_name': selected_model,
            }
            
            def update_ui():
                try:
                    for i, target in enumerate(TARGETS):
                        v_pred = pred_mean[i]
                        v_conf = pred_std[i]
                        v_min = v_pred - v_conf
                        
                        # Degradation (for UTS, YS, Elongation, HAZ_Width, Gauge_Depth, Austenite)
                        r0_key = target.replace('R2_', 'R0_')
                        r0_val = values.get(r0_key, 1)
                        if r0_val == 0: r0_val = 1
                        deg = ((r0_val - v_pred) / r0_val * 100)
                        
                        # Assign to dict for PDF report mapping
                        short_name = target.replace('R2_', '').lower()
                        if short_name == 'austenite_content': short_name = 'austenite'
                        if short_name == 'haz_width': short_name = 'haz'
                        if short_name == 'gauge_depth': short_name = 'gauge'
                        self.last_prediction[f'pred_{short_name}'] = v_pred
                        self.last_prediction[f'conf_{short_name}'] = v_conf
                        self.last_prediction[f'min_{short_name}'] = v_min
                        self.last_prediction[f'deg_{short_name}'] = deg

                        if CTK:
                            self.result_labels[target].configure(text=f"{v_pred:.2f}")
                            self.result_cis[target].configure(text=f"± {v_conf:.2f} (95% CI)" if v_conf > 0 else "")

                    if CTK:
                        for w in self.comparison_frame.winfo_children():
                            w.destroy()

                        headers = ['Property', 'R0', 'R1', 'R2 (Predicted)']
                        for j, h in enumerate(headers):
                            ctk.CTkLabel(self.comparison_frame, text=h,
                                        font=ctk.CTkFont(size=11, weight="bold"),
                                        fg_color="#1565c0" if j > 0 else "#37474f",
                                        corner_radius=3, width=90).grid(row=0, column=j, padx=1, pady=1)

                        for i, target in enumerate(TARGETS):
                            r2_val = pred_mean[i]
                            r0_key = target.replace('R2_', 'R0_')
                            r1_key = target.replace('R2_', 'R1_')
                            
                            r0_v = values.get(r0_key, '—')
                            r1_v = values.get(r1_key, '—')
                            
                            ctk.CTkLabel(self.comparison_frame, text=target.replace('R2_', ''), font=ctk.CTkFont(size=11), width=90).grid(row=i+1, column=0, padx=1, pady=1)
                            ctk.CTkLabel(self.comparison_frame, text=f"{r0_v:.2f}" if isinstance(r0_v, float) else r0_v, font=ctk.CTkFont(size=11), width=90).grid(row=i+1, column=1, padx=1, pady=1)
                            ctk.CTkLabel(self.comparison_frame, text=f"{r1_v:.2f}" if isinstance(r1_v, float) else r1_v, font=ctk.CTkFont(size=11), width=90).grid(row=i+1, column=2, padx=1, pady=1)
                            ctk.CTkLabel(self.comparison_frame, text=f"{r2_val:.2f}", font=ctk.CTkFont(size=11), width=90).grid(row=i+1, column=3, padx=1, pady=1)

                        self.status_label.configure(text=f"✅ Prediction complete using {selected_model}")
                    else:
                        self.result_text.delete('1.0', 'end')
                        for i, tgt in enumerate(TARGETS):
                            self.result_text.insert('end', f"{tgt}: {pred_mean[i]:.2f}\\n")
                except Exception as ex:
                    print(ex)

            if CTK:
                self.app.root.after(0, update_ui)
            else:
                update_ui()

        except Exception as e:
            if CTK:
                self.app.root.after(0, lambda: self.status_label.configure(text=f"❌ Error: {str(e)}"))
            else:
                import traceback
                traceback.print_exc()

    def _export_pdf(self):
        """Export prediction results as PDF."""
        if not hasattr(self, 'last_prediction'):
            if CTK:
                self.status_label.configure(text="⚠️ Run a prediction first")
            return

        try:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension='.pdf',
                filetypes=[("PDF files", "*.pdf")],
                initialfile="R2_Prediction_Report.pdf"
            )
            if not path:
                return

            import matplotlib.pyplot as plt
            from core.report_generator import generate_report

            generate_report(path, self.last_prediction, curve_fig=None)

            if CTK:
                self.status_label.configure(text=f"✅ Saved to {os.path.basename(path)}")
        except Exception as e:
            if CTK:
                self.status_label.configure(text=f"❌ Export failed: {str(e)}")
