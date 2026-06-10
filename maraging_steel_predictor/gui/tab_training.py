"""
Tab 3: Model Training & Performance
Train all models, view progress, and compare model metrics.
"""

import os
import sys
import threading
import json

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

from train_all import TARGETS

class TrainingTab:
    """Training and performance comparison tab."""

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.current_view = "Overall"
        self._build_ui(parent)
        self._load_existing_results()

    def _build_ui(self, parent):
        if CTK:
            # Top controls
            ctrl = ctk.CTkFrame(parent)
            ctrl.pack(fill='x', padx=10, pady=5)

            ctk.CTkLabel(ctrl, text="Model Training & Performance",
                        font=ctk.CTkFont(size=16, weight="bold")).pack(side='left', padx=10)

            self.train_btn = ctk.CTkButton(ctrl, text="🚀 Train All Models",
                                            command=self._start_training,
                                            fg_color="#1565c0", hover_color="#0d47a1",
                                            font=ctk.CTkFont(size=14, weight="bold"))
            self.train_btn.pack(side='right', padx=10)

            self.load_csv_btn = ctk.CTkButton(ctrl, text="📂 Load Custom CSV",
                                               command=self._load_custom_csv,
                                               fg_color="#4caf50")
            self.load_csv_btn.pack(side='right', padx=5)

            # Progress bar
            self.progress_var = ctk.DoubleVar(value=0)
            self.progress_bar = ctk.CTkProgressBar(parent, variable=self.progress_var)
            self.progress_bar.pack(fill='x', padx=10, pady=5)
            self.progress_bar.set(0)

            self.status_label = ctk.CTkLabel(parent, text="Ready to train",
                                              font=ctk.CTkFont(size=12))
            self.status_label.pack(pady=2)

            # Split: left = log, right = chart + table
            paned = ctk.CTkFrame(parent)
            paned.pack(fill='both', expand=True, padx=10, pady=5)

            # Log
            log_frame = ctk.CTkFrame(paned, width=400)
            log_frame.pack(side='left', fill='both', expand=False, padx=(0, 5))
            log_frame.pack_propagate(False)

            ctk.CTkLabel(log_frame, text="Training Log",
                        font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)
            self.log_text = ctk.CTkTextbox(log_frame, font=ctk.CTkFont(size=10))
            self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

            # Right: table + chart
            right = ctk.CTkFrame(paned)
            right.pack(side='right', fill='both', expand=True)

            # Controls above table
            tbl_ctrl = ctk.CTkFrame(right, fg_color="transparent")
            tbl_ctrl.pack(fill='x', padx=5, pady=5)
            ctk.CTkLabel(tbl_ctrl, text="Model Comparison",
                        font=ctk.CTkFont(size=13, weight="bold")).pack(side='left')
            
            view_opts = ["Overall"] + TARGETS
            self.view_var = ctk.StringVar(value="Overall")
            self.view_dropdown = ctk.CTkComboBox(tbl_ctrl, values=view_opts, variable=self.view_var, command=self._on_view_change, width=180)
            self.view_dropdown.pack(side='right')
            ctk.CTkLabel(tbl_ctrl, text="Metric View: ").pack(side='right', padx=5)

            # Results table
            self.table_frame = ctk.CTkScrollableFrame(right, height=200)
            self.table_frame.pack(fill='x', padx=5, pady=5)

            # Bar chart
            self.fig, self.ax = plt.subplots(figsize=(8, 3.5))
            self.fig.patch.set_facecolor('#1e1e2e')
            self.ax.set_facecolor('#1e1e2e')

            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        else:
            tk.Button(parent, text="Train All", command=self._start_training).pack(pady=10)
            self.log_text = tk.Text(parent, height=20)
            self.log_text.pack(fill='both', expand=True)

    def _start_training(self):
        """Start training in background thread."""
        if CTK:
            self.train_btn.configure(state='disabled', text="⏳ Training...")
            self.log_text.delete('0.0', 'end')

        thread = threading.Thread(target=self._run_training, daemon=True)
        thread.start()

    def _run_training(self):
        """Run the full training pipeline."""
        try:
            from train_all import run_full_training

            data_dir = getattr(self, '_custom_data_dir', None)

            def log_callback(msg):
                if CTK:
                    self.parent.after(0, lambda: self.log_text.insert('end', msg + '\n'))
                    self.parent.after(0, lambda: self.log_text.see('end'))
                    self.parent.after(0, lambda: self.status_label.configure(text=msg[:80]))

            def progress_callback(pct):
                if CTK:
                    self.parent.after(0, lambda: self.progress_bar.set(pct / 100))

            results, best_name, scaler = run_full_training(
                data_dir=data_dir, callback=log_callback,
                progress_callback=progress_callback)

            # Update app state
            self.app.training_results = results
            self.app.best_model_name = best_name
            self.app.scaler = scaler
            self.app._load_existing_models()

            # Update UI
            if CTK:
                self.parent.after(0, lambda: self._update_results(results, best_name))
                self.parent.after(0, lambda: self.train_btn.configure(
                    state='normal', text="🚀 Train All Models"))
                self.parent.after(0, lambda: self.status_label.configure(
                    text=f"✅ Training complete! Best: {best_name}"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            if CTK:
                self.parent.after(0, lambda: self.status_label.configure(
                    text=f"❌ Training error: {str(e)}"))
                self.parent.after(0, lambda: self.train_btn.configure(
                    state='normal', text="🚀 Train All Models"))

    def _on_view_change(self, choice):
        """Handle dropdown change to re-render table."""
        self.current_view = choice
        if getattr(self.app, 'training_results', None):
            self._update_results(self.app.training_results, getattr(self.app, 'best_model_name', 'Auto-Best'))

    def _update_results(self, results, best_name):
        """Update results table and bar chart based on current view."""
        # Clear table
        for w in self.table_frame.winfo_children():
            w.destroy()

        # Headers
        headers = ['Model', 'MAE', 'RMSE', 'R²', 'Accuracy%', 'Status']
        for j, h in enumerate(headers):
            ctk.CTkLabel(self.table_frame, text=h,
                        font=ctk.CTkFont(size=11, weight="bold"),
                        fg_color="#1565c0", corner_radius=3,
                        width=100 if j > 0 else 160).grid(row=0, column=j, padx=1, pady=1)

        # Extract values based on view
        display_results = []
        for name, metrics in results.items():
            if self.current_view == "Overall":
                mdict = metrics
            else:
                mdict = metrics.get('Target_Metrics', {}).get(self.current_view, {})
            display_results.append((name, mdict))

        # Sort by R2
        sorted_results = sorted(display_results, key=lambda x: x[1].get('R2', 0), reverse=True)

        for i, (name, metrics) in enumerate(sorted_results, 1):
            is_best = name == best_name
            fg = "#1b5e20" if is_best else "transparent"

            ctk.CTkLabel(self.table_frame, text=name,
                        font=ctk.CTkFont(size=10, weight="bold" if is_best else "normal"),
                        fg_color=fg, width=160).grid(row=i, column=0, padx=1, pady=1)

            for j, key in enumerate(['MAE', 'RMSE', 'R2', 'Accuracy%'], 1):
                val = metrics.get(key, 'N/A')
                # Format to 4 decimals if it's a number
                if isinstance(val, (int, float)): val = f"{val:.4f}"
                ctk.CTkLabel(self.table_frame, text=str(val),
                            font=ctk.CTkFont(size=10),
                            fg_color=fg, width=100).grid(row=i, column=j, padx=1, pady=1)

            status = "⭐ BEST" if is_best else "✓"
            ctk.CTkLabel(self.table_frame, text=status,
                        font=ctk.CTkFont(size=10), fg_color=fg,
                        width=100).grid(row=i, column=5, padx=1, pady=1)

        # Bar chart
        self.ax.clear()
        names = [n for n, _ in sorted_results]
        r2_scores = [m.get('R2', 0) for _, m in sorted_results]
        colors = ['#66bb6a' if n == best_name else '#42a5f5' for n in names]

        bars = self.ax.barh(range(len(names)), r2_scores, color=colors)
        self.ax.set_yticks(range(len(names)))
        self.ax.set_yticklabels(names, fontsize=9, color='white')
        self.ax.set_xlabel(f'R² Score ({self.current_view})', color='white', fontsize=11)
        self.ax.set_title(f'Model R² Comparison — {self.current_view}', color='white', fontsize=13, fontweight='bold')
        self.ax.tick_params(colors='white')
        self.ax.invert_yaxis()
        self.ax.grid(axis='x', alpha=0.2, color='gray')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('gray')
        self.ax.spines['left'].set_color('gray')
        self.fig.tight_layout()
        self.canvas.draw()

    def _load_existing_results(self):
        """Load and display existing training results."""
        if hasattr(self.app, 'training_results') and self.app.training_results:
            if CTK:
                self._update_results(self.app.training_results, getattr(self.app, 'best_model_name', 'Auto-Best'))

    def _load_custom_csv(self):
        """Load custom CSV data file for retraining."""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select dataset CSV file",
            filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
        )
        if file_path:
            self._custom_data_dir = file_path
            if CTK:
                import os
                self.status_label.configure(text=f"Custom data: {os.path.basename(file_path)}")
