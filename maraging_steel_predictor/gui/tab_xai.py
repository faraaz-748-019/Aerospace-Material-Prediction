"""
Tab 4: XAI — Explainability (Multi-Target SHAP)
Per-target and cross-target feature importance visualization.

Fixed: MultiOutputRegressor handling, ensemble reference, DL model wrapping.
"""

import os
import sys
import threading

try:
    import customtkinter as ctk
    CTK = True
except ImportError:
    import tkinter as tk
    from tkinter import ttk
    CTK = False

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Target labels matching train_all.TARGETS order
TARGET_OPTIONS = [
    'UTS',
    'YS',
    'Elongation',
    'HAZ Width',
    'Gauge Depth',
    'Austenite Content',
    '── All Targets (Comparison) ──',
]

# Tooltip descriptions for each R0 feature
FEATURE_TOOLTIPS = {
    'R0 UTS': 'Ultimate Tensile Strength of original weld (R0), in MPa',
    'R0 YS': 'Yield Strength of original weld (R0), in MPa',
    'R0 K1c': 'Fracture Toughness of original weld (R0), in MPa√m',
    'R0 Elongation': 'Elongation at break of original weld (R0), in %',
    'R0 HAZ Width': 'Heat-Affected Zone width of original weld (R0), in mm',
    'R0 Gauge Depth': 'Gauge depth of original weld (R0), in mm',
    'R0 Austenite Content': 'Retained austenite content of original weld (R0), in %',
    'R1 UTS': 'Ultimate Tensile Strength after 1st repair (R1), in MPa',
    'R1 YS': 'Yield Strength after 1st repair (R1), in MPa',
    'R1 K1c': 'Fracture Toughness after 1st repair (R1), in MPa√m',
    'R1 Elongation': 'Elongation at break after 1st repair (R1), in %',
    'R1 HAZ Width': 'Heat-Affected Zone width after 1st repair (R1), in mm',
    'R1 Gauge Depth': 'Gauge depth after 1st repair (R1), in mm',
    'R1 Austenite Content': 'Retained austenite after 1st repair (R1), in %',
    'Specimen Type S139': 'Indicator: specimen is type S139',
    'Specimen Type S200': 'Indicator: specimen is type S200',
    'Repair Location Center': 'Indicator: repair at center location',
    'Repair Location Edge': 'Indicator: repair at edge location',
    'Repair Location Root': 'Indicator: repair at root location',
    'Repair Location Toe': 'Indicator: repair at toe location',
}


class XAITab:
    """SHAP explainability tab with multi-target support."""

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self._shap_cache = {}   # cache: (model_name, target_idx) -> importance
        self._build_ui(parent)

    def _build_ui(self, parent):
        if CTK:
            self._build_ctk(parent)
        else:
            self._build_tk(parent)

    # ──────────────────────────────────────────────────────────
    # CustomTkinter UI
    # ──────────────────────────────────────────────────────────
    def _build_ctk(self, parent):
        # ── Top control bar ──────────────────────────────────
        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.pack(fill='x', padx=10, pady=(8, 4))

        ctk.CTkLabel(ctrl, text="🧠 XAI — Feature Explainability",
                     font=ctk.CTkFont(size=17, weight="bold")).pack(side='left', padx=10)

        # Compute button (right side)
        ctk.CTkButton(ctrl, text="🔍 Compute SHAP", width=140,
                      command=self._compute_shap,
                      fg_color="#7b1fa2", hover_color="#9c27b0",
                      font=ctk.CTkFont(size=13, weight="bold")
                      ).pack(side='right', padx=5)

        # Target selector
        self.target_var = ctk.StringVar(value=TARGET_OPTIONS[0])
        ctk.CTkLabel(ctrl, text="Target:", font=ctk.CTkFont(size=12)).pack(side='right', padx=(10, 2))
        self.target_selector = ctk.CTkComboBox(
            ctrl, values=TARGET_OPTIONS, variable=self.target_var,
            width=200, font=ctk.CTkFont(size=12))
        self.target_selector.pack(side='right', padx=5)

        # Model selector
        model_names = self._get_model_names()
        self.model_var = ctk.StringVar(value=model_names[0] if model_names else '')
        ctk.CTkLabel(ctrl, text="Model:", font=ctk.CTkFont(size=12)).pack(side='right', padx=(10, 2))
        self.model_selector = ctk.CTkComboBox(
            ctrl, values=model_names, variable=self.model_var,
            width=180, font=ctk.CTkFont(size=12))
        self.model_selector.pack(side='right', padx=5)

        # ── Status bar ───────────────────────────────────────
        self.status = ctk.CTkLabel(parent, text="Select a model and target, then click Compute SHAP",
                                   font=ctk.CTkFont(size=11), text_color="#90caf9")
        self.status.pack(pady=(2, 4))

        # ── Progress bar ─────────────────────────────────────
        self.progress = ctk.CTkProgressBar(parent, width=400, height=6)
        self.progress.pack(pady=(0, 4))
        self.progress.set(0)

        # ── Chart area ───────────────────────────────────────
        chart_frame = ctk.CTkFrame(parent, fg_color="#1a1a2e", corner_radius=8)
        chart_frame.pack(fill='x', padx=10, pady=(0, 4))

        self.fig, self.ax = plt.subplots(figsize=(10, 3.5))
        self.fig.patch.set_facecolor('#1e1e2e')
        self.ax.set_facecolor('#1e1e2e')
        self._draw_placeholder()

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

        # ── Bottom panel: ranking + info ─────────────────────
        bottom = ctk.CTkFrame(parent, fg_color="transparent")
        bottom.pack(fill='both', expand=True, padx=10, pady=(0, 6))

        # Feature ranking scroll list
        self.ranking_frame = ctk.CTkScrollableFrame(bottom, width=280,
                                                     fg_color="#16213e",
                                                     corner_radius=6)
        self.ranking_frame.pack(side='left', fill='y', padx=(0, 5))

        # Info panel
        info_frame = ctk.CTkFrame(bottom, fg_color="#16213e",
                                   corner_radius=6)
        info_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        info_frame.pack_propagate(False)

        self.info_title_label = ctk.CTkLabel(info_frame, text="ℹ️  About SHAP",
                     font=ctk.CTkFont(size=14, weight="bold"))
        self.info_title_label.pack(pady=(8, 4), padx=8)
        
        info_text = (
            "SHAP (SHapley Additive exPlanations) "
            "assigns each feature an importance value "
            "for a particular prediction.\n\n"
            "• TreeExplainer: exact values for\n"
            "  tree-based models (fast)\n"
            "• KernelExplainer: model-agnostic\n"
            "  approximation for DL models (~30s)\n\n"
            "Higher |SHAP| = stronger influence\n"
            "on the predicted property.\n\n"
            "💡 Hover over feature names in the\n"
            "ranking list to see descriptions."
        )
        self.info_text_label = ctk.CTkLabel(info_frame, text=info_text, font=ctk.CTkFont(size=12),
                     text_color="#b0bec5", justify="left", wraplength=400,
                     anchor="nw")
        self.info_text_label.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self.info_text_label.bind("<Configure>", lambda e: self.info_text_label.configure(wraplength=e.width - 20) if e.width > 20 else None)

    # ──────────────────────────────────────────────────────────
    # Fallback tkinter UI
    # ──────────────────────────────────────────────────────────
    def _build_tk(self, parent):
        tk.Label(parent, text="XAI — Feature Explainability",
                 font=('Arial', 14, 'bold')).pack(pady=10)
        frame = tk.Frame(parent)
        frame.pack(fill='x', padx=10)
        tk.Label(frame, text="Model:").pack(side='left')
        self.model_var = tk.StringVar(value='ExtraTrees')
        ttk.Combobox(frame, textvariable=self.model_var, width=20).pack(side='left', padx=5)
        tk.Label(frame, text="Target:").pack(side='left')
        self.target_var = tk.StringVar(value='UTS')
        ttk.Combobox(frame, textvariable=self.target_var,
                     values=TARGET_OPTIONS, width=25).pack(side='left', padx=5)
        tk.Button(frame, text="Compute SHAP", command=self._compute_shap).pack(side='right')
        self.status = tk.Label(parent, text="")
        self.status.pack()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ranking_frame = tk.Frame(parent)
        self.ranking_frame.pack(fill='x')
        self.progress = None

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    def _get_model_names(self):
        names = list(self.app.models.keys()) + list(self.app.dl_models.keys())
        if self.app.ensemble:
            names.append('Stacking Ensemble')
        return names if names else ['No models loaded']

    def _draw_placeholder(self):
        """Draw a placeholder message on the chart."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Click "Compute SHAP" to analyze\nfeature importance',
                     ha='center', va='center', fontsize=16,
                     color='#546e7a', style='italic',
                     transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.fig.tight_layout()

    def _target_name_to_idx(self, name):
        """Convert display target name to index."""
        mapping = {
            'UTS': 0, 'YS': 1, 'Elongation': 2,
            'HAZ Width': 3, 'Gauge Depth': 4, 'Austenite Content': 5,
        }
        return mapping.get(name, None)

    # ──────────────────────────────────────────────────────────
    # SHAP Computation
    # ──────────────────────────────────────────────────────────
    def _compute_shap(self):
        """Launch SHAP computation in background thread."""
        if CTK:
            self.status.configure(text="⏳ Computing SHAP values… please wait")
            self.progress.set(0)
        thread = threading.Thread(target=self._run_shap, daemon=True)
        thread.start()

    def _run_shap(self):
        """Run SHAP computation (background thread)."""
        try:
            import numpy as np
            from core.shap_explainer import (compute_shap_values,
                                             compute_all_targets_importance,
                                             plot_shap_comparison,
                                             format_feature_name,
                                             TARGET_DISPLAY)

            selected_model = self.model_var.get() if CTK else 'Extratrees'
            selected_target = self.target_var.get() if CTK else 'UTS'
            is_comparison = 'All Targets' in selected_target

            # ── Load feature columns ────────────────────────
            feature_cols = getattr(self.app, 'feature_cols', None)
            if not feature_cols:
                self._update_status("❌ Feature columns not loaded. Train models first.")
                return

            # ── Load training data ──────────────────────────
            self._update_status("⏳ Loading training data…")
            if CTK and self.progress:
                self.parent.after(0, lambda: self.progress.set(0.1))

            data_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'data')
            from train_all import load_and_combine_data
            df = load_and_combine_data(data_dir)

            # Only use columns that exist in the dataframe
            available_cols = [c for c in feature_cols if c in df.columns]
            if not available_cols:
                self._update_status("❌ Feature columns not found in data.")
                return

            X = df[available_cols].values[:200].astype(np.float32)

            if CTK and self.progress:
                self.parent.after(0, lambda: self.progress.set(0.25))

            # ── Resolve model ───────────────────────────────
            model = None
            model_type = 'tree'

            if selected_model == 'Stacking Ensemble':
                # For ensemble, use the best individual ML model instead
                # since the meta-learner operates on stacked predictions, not raw features
                best_ml = None
                best_r2 = -999
                for name, m in self.app.models.items():
                    tr = self.app.training_results.get(name, {})
                    r2 = tr.get('R2', -999)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_ml = m
                        selected_model = f"Best ML ({name})"
                if best_ml is not None:
                    model = best_ml
                    model_type = 'tree'
                else:
                    self._update_status("⚠️ No ML models found for ensemble SHAP.")
                    return
            else:
                # Check ML models (case-insensitive)
                name_map = {v.lower(): v for v in self.app.models.keys()}
                if selected_model.lower() in name_map:
                    model = self.app.models[name_map[selected_model.lower()]]
                    model_type = 'tree'
                elif selected_model in self.app.models:
                    model = self.app.models[selected_model]
                    model_type = 'tree'
                elif selected_model in self.app.dl_models:
                    # DL models need special wrapping for SHAP
                    dl_model = self.app.dl_models[selected_model]
                    scaler = getattr(self.app, 'scaler', None)

                    # Wrap DL model as a callable that takes raw features
                    import torch

                    class DLWrapper:
                        def __init__(self, torch_model, scaler):
                            self.torch_model = torch_model
                            self.scaler = scaler

                        def predict(self, x):
                            if self.scaler is not None:
                                x = self.scaler.transform(x)
                            self.torch_model.eval()
                            with torch.no_grad():
                                t = torch.FloatTensor(x)
                                return self.torch_model(t).cpu().numpy()

                    model = DLWrapper(dl_model, scaler)
                    model_type = 'kernel'

            if model is None:
                self._update_status(f"⚠️ Model '{selected_model}' not found. Train models first.")
                return

            feature_names = [format_feature_name(f) for f in available_cols]

            if CTK and self.progress:
                self.parent.after(0, lambda: self.progress.set(0.4))

            # ── Compute ─────────────────────────────────────
            # Determine number of targets based on training results
            n_targets = 6
            tr_example = next(iter(self.app.training_results.values()), {}) if self.app.training_results else {}
            tm = tr_example.get('Target_Metrics', {})
            if tm:
                n_targets = len(tm)

            if is_comparison:
                self._update_status(f"⏳ Computing SHAP for all {n_targets} targets… (this may take a moment)")
                all_importance = compute_all_targets_importance(
                    model, X, feature_names=feature_names,
                    model_type=model_type, n_targets=n_targets)

                if not all_importance:
                    self._update_status("❌ SHAP computation failed for all targets")
                    return

                if CTK and self.progress:
                    self.parent.after(0, lambda: self.progress.set(0.85))

                # Cache results
                for t, imp in all_importance.items():
                    self._shap_cache[(selected_model, t)] = imp

                self._draw_comparison(all_importance, selected_model)
            else:
                target_idx = self._target_name_to_idx(selected_target)
                if target_idx is None:
                    target_idx = 0

                self._update_status(f"⏳ Computing SHAP for {selected_target}…")

                # Check cache first
                cache_key = (selected_model, target_idx)
                if cache_key in self._shap_cache:
                    importance = self._shap_cache[cache_key]
                else:
                    _, importance = compute_shap_values(
                        model, X, feature_names=feature_names,
                        model_type=model_type, target_idx=target_idx)

                    if importance is None:
                        self._update_status("❌ SHAP computation failed")
                        return

                    self._shap_cache[cache_key] = importance

                if CTK and self.progress:
                    self.parent.after(0, lambda: self.progress.set(0.85))

                self._draw_single_target(importance, selected_model, selected_target)

            if CTK and self.progress:
                self.parent.after(0, lambda: self.progress.set(1.0))

            self._update_status(f"✅ SHAP computed for {selected_model} — {selected_target}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_status(f"❌ Error: {str(e)}")

    # ──────────────────────────────────────────────────────────
    # Drawing functions (called on main thread via .after)
    # ──────────────────────────────────────────────────────────
    def _draw_single_target(self, importance, model_name, target_name):
        """Draw a single-target horizontal bar chart."""
        import numpy as np

        def update():
            self.ax.clear()
            names = list(importance.keys())[:15]
            values = list(importance.values())[:15]

            if not names:
                self.ax.text(0.5, 0.5, 'No SHAP data', ha='center', va='center',
                             fontsize=14, color='#546e7a', transform=self.ax.transAxes)
                self.ax.set_axis_off()
                self.canvas.draw()
                return

            colors = plt.cm.magma(np.linspace(0.25, 0.85, len(names)))
            self.ax.barh(range(len(names)), values, color=colors,
                         edgecolor='none', height=0.7)
            self.ax.set_yticks(range(len(names)))
            self.ax.set_yticklabels(names, fontsize=10, color='white')
            self.ax.set_xlabel('Mean |SHAP Value|', color='white', fontsize=11)
            self.ax.set_title(
                f'SHAP Feature Importance — {target_name}  ({model_name})',
                color='white', fontsize=13, fontweight='bold')
            self.ax.invert_yaxis()
            self.ax.tick_params(colors='white')
            self.ax.grid(axis='x', alpha=0.15, color='gray')
            for s in ['top', 'right']:
                self.ax.spines[s].set_visible(False)
            for s in ['bottom', 'left']:
                self.ax.spines[s].set_color('#444')
            self.fig.tight_layout()
            self.canvas.draw()

            # Update ranking list
            self._update_ranking(importance)

        self.parent.after(0, update)

    def _draw_comparison(self, all_importance, model_name):
        """Draw a grouped bar chart comparing all targets."""
        import numpy as np
        from core.shap_explainer import TARGET_DISPLAY

        def update():
            self.ax.clear()

            # Collect all features and compute average importance
            all_features = set()
            for imp in all_importance.values():
                all_features.update(imp.keys())

            avg_importance = {}
            for feat in all_features:
                vals = [all_importance[t].get(feat, 0.0) for t in all_importance]
                avg_importance[feat] = np.mean(vals)

            top_n = 10
            top_features = sorted(avg_importance, key=avg_importance.get,
                                  reverse=True)[:top_n]

            n_targets = len(all_importance)
            target_labels = [TARGET_DISPLAY.get(t, f'T{t}')
                             for t in sorted(all_importance.keys())]
            x = np.arange(len(top_features))
            bar_width = 0.8 / max(n_targets, 1)

            palette = ['#e91e63', '#2196f3', '#4caf50',
                       '#ff9800', '#9c27b0', '#00bcd4']

            for i, t in enumerate(sorted(all_importance.keys())):
                vals = [all_importance[t].get(f, 0.0) for f in top_features]
                self.ax.bar(x + i * bar_width, vals, bar_width,
                            label=target_labels[i],
                            color=palette[i % len(palette)],
                            edgecolor='none', alpha=0.88)

            self.ax.set_xticks(x + bar_width * (n_targets - 1) / 2)
            short_labels = []
            for f in top_features:
                lbl = f.replace('R0 ', '').replace('R1 ', '').replace('Specimen Type ', 'Spec ')
                if len(lbl) > 14:
                    lbl = lbl[:12] + '…'
                short_labels.append(lbl)
            self.ax.set_xticklabels(short_labels, fontsize=9, color='white',
                                    rotation=25, ha='right')
            self.ax.set_ylabel('Mean |SHAP Value|', fontsize=11, color='white')
            self.ax.set_title(
                f'Feature Importance Across All Targets — {model_name}',
                fontsize=13, fontweight='bold', color='white')
            self.ax.tick_params(colors='white')
            self.ax.legend(loc='upper right', fontsize=8, framealpha=0.6,
                           facecolor='#1e1e2e', edgecolor='gray',
                           labelcolor='white')
            self.ax.grid(axis='y', alpha=0.15, color='gray')
            for s in ['top', 'right']:
                self.ax.spines[s].set_visible(False)
            for s in ['bottom', 'left']:
                self.ax.spines[s].set_color('#444')
            self.fig.tight_layout()
            self.canvas.draw()

            # Show average ranking
            self._update_ranking(avg_importance, title="Average Across Targets")

        self.parent.after(0, update)

    def _update_ranking(self, importance, title="Feature Ranking"):
        """Update the bottom ranking list with tooltips."""
        if not CTK:
            return

        for w in self.ranking_frame.winfo_children():
            w.destroy()

        ctk.CTkLabel(self.ranking_frame, text=f"📋 {title}",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#64b5f6").pack(anchor='w', pady=(4, 2), padx=6)

        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        for i, (name, val) in enumerate(sorted_importance.items(), 1):
            # Color code by rank
            if i <= 3:
                color = "#e91e63"
                badge = "🥇🥈🥉"[i - 1]
            elif i <= 6:
                color = "#ff9800"
                badge = f"#{i}"
            else:
                color = "#78909c"
                badge = f"#{i}"

            row = ctk.CTkFrame(self.ranking_frame, fg_color="transparent", height=20)
            row.pack(fill='x', padx=6, pady=1)

            # Feature name with tooltip
            tooltip_text = FEATURE_TOOLTIPS.get(name, f"Feature: {name}")
            name_label = ctk.CTkLabel(row, text=f"  {badge}  {name}",
                         font=ctk.CTkFont(size=10), text_color=color,
                         anchor='w', width=220)
            name_label.pack(side='left')

            # Add tooltip on hover
            self._add_tooltip(name_label, tooltip_text)

            ctk.CTkLabel(row, text=f"{val:.6f}",
                         font=ctk.CTkFont(size=10, family="Consolas"),
                         text_color="#b0bec5", anchor='e').pack(side='right', padx=6)

        # Update text explanation panel
        if hasattr(self, 'info_title_label') and hasattr(self, 'info_text_label'):
            top_features = list(sorted_importance.keys())[:3]
            if len(top_features) >= 3:
                para1 = (
                    "To understand the core mechanisms behind our prediction, we use SHAP (SHapley Additive exPlanations). "
                    "This advanced XAI framework breaks down our complex machine learning models into understandable parts, "
                    "showing precisely how each input variable contributes to the final predicted material property."
                )
                
                para2 = (
                    "The graph visually represents these contributions. Each bar corresponds to a specific feature, "
                    "with its length indicating the magnitude of its impact on the prediction. Features with longer bars are the primary drivers "
                    "that strongly shift the model's output from the baseline expectation, whereas smaller bars have a more subtle effect."
                )
                
                para3 = (
                    f"Focusing on the current results, the most crucial drivers are clearly identified. "
                    f"First, '{top_features[0]}' ({FEATURE_TOOLTIPS.get(top_features[0], 'Unknown')}) has the strongest influence. "
                    f"It is followed by '{top_features[1]}' ({FEATURE_TOOLTIPS.get(top_features[1], 'Unknown')}) as a strong secondary factor, "
                    f"and '{top_features[2]}' ({FEATURE_TOOLTIPS.get(top_features[2], 'Unknown')}) which also notably shapes the outcome."
                )
                
                para4 = (
                    "Ultimately, this approach bridges the gap between raw data and actionable metallurgical insights. "
                    f"By continuously confirming that {top_features[0]} and other top-ranked features align with physical metallurgy principles, "
                    "we ensure that our predictive tool is both highly accurate and scientifically reliable for evaluating structural integrities."
                )
                explanation = f"{para1}\n\n{para2}\n\n{para3}\n\n{para4}"
                self.info_title_label.configure(text="📊 Detailed XAI Insight Explanation")
                self.info_text_label.configure(text=explanation)

    def _add_tooltip(self, widget, text):
        """Add a simple hover tooltip to a widget."""
        tooltip_window = [None]

        def enter(event):
            tw = ctk.CTkToplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
            tw.attributes('-topmost', True)
            label = ctk.CTkLabel(tw, text=text,
                                 font=ctk.CTkFont(size=10),
                                 fg_color="#37474f", corner_radius=4,
                                 text_color="#e0e0e0", wraplength=300)
            label.pack(padx=4, pady=2)
            tooltip_window[0] = tw

        def leave(event):
            if tooltip_window[0]:
                tooltip_window[0].destroy()
                tooltip_window[0] = None

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def _update_status(self, text):
        """Thread-safe status update."""
        if CTK:
            self.parent.after(0, lambda: self.status.configure(text=text))
        else:
            try:
                self.status.config(text=text)
            except Exception:
                pass
