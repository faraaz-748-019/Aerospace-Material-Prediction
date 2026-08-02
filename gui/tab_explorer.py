"""
Tab 5: Data Explorer
CSV browser, scrollable table, statistics, and scatter plots.
"""

import os
import sys

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

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExplorerTab:
    """Data exploration tab."""

    def __init__(self, parent, app):
        self.app = app
        self.parent = parent
        self.df = None
        self._build_ui(parent)
        self._auto_load_default()

    def _build_ui(self, parent):
        if CTK:
            # Controls
            ctrl = ctk.CTkFrame(parent)
            ctrl.pack(fill='x', padx=10, pady=5)

            ctk.CTkLabel(ctrl, text="Data Explorer",
                        font=ctk.CTkFont(size=16, weight="bold")).pack(side='left', padx=10)

            ctk.CTkButton(ctrl, text="📂 Load CSV",
                         command=self._load_csv,
                         fg_color="#1565c0").pack(side='right', padx=5)

            self.file_label = ctk.CTkLabel(ctrl, text="No file loaded",
                                            font=ctk.CTkFont(size=11), text_color="gray")
            self.file_label.pack(side='right', padx=10)

            # Main content area
            content = ctk.CTkFrame(parent)
            content.pack(fill='both', expand=True, padx=10, pady=5)

            # Left: table + stats
            left = ctk.CTkFrame(content, width=550)
            left.pack(side='left', fill='both', expand=True, padx=(0, 5))

            # Data table
            ctk.CTkLabel(left, text="Data Table",
                        font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)
            self.table_canvas_frame = ctk.CTkScrollableFrame(left, height=250)
            self.table_canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)

            # Statistics
            ctk.CTkLabel(left, text="Statistics",
                        font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(10, 5))
            self.stats_frame = ctk.CTkScrollableFrame(left, height=150)
            self.stats_frame.pack(fill='x', padx=5, pady=5)

            # Right: scatter plots
            right = ctk.CTkFrame(content, width=450)
            right.pack(side='right', fill='both', expand=False, padx=(5, 0))
            right.pack_propagate(False)

            ctk.CTkLabel(right, text="Scatter Plots",
                        font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)

            self.fig, self.axes = plt.subplots(1, 2, figsize=(9, 4))
            self.fig.patch.set_facecolor('#1e1e2e')
            for ax in self.axes:
                ax.set_facecolor('#1e1e2e')

            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        else:
            tk.Button(parent, text="Load CSV", command=self._load_csv).pack(pady=10)
            self.text = tk.Text(parent, height=20)
            self.text.pack(fill='both', expand=True)

    def _auto_load_default(self):
        """Auto-load the high-correlation dataset."""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'data')
        default_file = os.path.join(data_dir, 'synthetic_option_A_revised_high_correlation_3000.csv')
        if os.path.exists(default_file):
            self._load_file(default_file)

    def _load_csv(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if path:
            self._load_file(path)

    def _load_file(self, path):
        """Load a CSV file and display it."""
        try:
            self.df = pd.read_csv(path)
            if CTK:
                self.file_label.configure(text=f"{os.path.basename(path)} ({len(self.df)} rows)")
            self._display_table()
            self._display_stats()
            self._display_scatter()
        except Exception as e:
            if CTK:
                self.file_label.configure(text=f"Error: {e}")

    def _display_table(self):
        """Display data in a scrollable table."""
        if self.df is None or not CTK:
            return

        for w in self.table_canvas_frame.winfo_children():
            w.destroy()

        # Headers
        for j, col in enumerate(self.df.columns):
            ctk.CTkLabel(self.table_canvas_frame, text=col[:18],
                        font=ctk.CTkFont(size=9, weight="bold"),
                        fg_color="#1565c0", corner_radius=2,
                        width=90).grid(row=0, column=j, padx=1, pady=1)

        # Data rows (show first 100)
        for i in range(min(100, len(self.df))):
            for j, col in enumerate(self.df.columns):
                val = self.df.iloc[i, j]
                text = f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)
                ctk.CTkLabel(self.table_canvas_frame, text=text[:12],
                            font=ctk.CTkFont(size=9),
                            width=90).grid(row=i+1, column=j, padx=1, pady=0)

    def _display_stats(self):
        """Display summary statistics."""
        if self.df is None or not CTK:
            return

        for w in self.stats_frame.winfo_children():
            w.destroy()

        stats = self.df.describe()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Headers
        stat_labels = ['Mean', 'Std', 'Min', 'Max']
        ctk.CTkLabel(self.stats_frame, text="Column",
                    font=ctk.CTkFont(size=9, weight="bold"),
                    fg_color="#37474f", width=130).grid(row=0, column=0, padx=1, pady=1)
        for k, s in enumerate(stat_labels):
            ctk.CTkLabel(self.stats_frame, text=s,
                        font=ctk.CTkFont(size=9, weight="bold"),
                        fg_color="#37474f", width=90).grid(row=0, column=k+1, padx=1, pady=1)

        for i, col in enumerate(numeric_cols[:15]):
            ctk.CTkLabel(self.stats_frame, text=col[:20],
                        font=ctk.CTkFont(size=9), width=130).grid(row=i+1, column=0, padx=1)
            for k, stat in enumerate(['mean', 'std', 'min', 'max']):
                val = stats.loc[stat, col] if stat in stats.index else 0
                ctk.CTkLabel(self.stats_frame, text=f"{val:.2f}",
                            font=ctk.CTkFont(size=9), width=90).grid(row=i+1, column=k+1, padx=1)

    def _display_scatter(self):
        """Display scatter plots: R0 UTS vs R2 UTS, R0 YS vs R2 YS."""
        if self.df is None:
            return

        for ax in self.axes:
            ax.clear()

        plots = [
            ('R0_UTS_S139_MPa', 'Target_R2_UTS_MPa', 'R0 UTS vs R2 UTS'),
            ('R0_YS_S139_MPa', 'Target_R2_YS_MPa', 'R0 YS vs R2 YS'),
        ]

        for idx, (x_col, y_col, title) in enumerate(plots):
            if x_col in self.df.columns and y_col in self.df.columns:
                ax = self.axes[idx]
                ax.scatter(self.df[x_col], self.df[y_col],
                          alpha=0.4, s=8, color='#42a5f5')
                ax.set_xlabel(x_col.replace('_', ' ')[:20], fontsize=8, color='white')
                ax.set_ylabel(y_col.replace('_', ' ')[:20], fontsize=8, color='white')
                ax.set_title(title, fontsize=10, color='white', fontweight='bold')
                ax.tick_params(colors='white', labelsize=7)
                ax.grid(alpha=0.2, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('gray')
                ax.spines['left'].set_color('gray')

        self.fig.tight_layout()
        self.canvas.draw()
