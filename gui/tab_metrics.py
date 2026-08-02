"""
Tab 7: Detailed Metrics Dashboard
Comprehensive breakdown of all 6 target properties across models.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_all import TARGETS

class MetricsTab:
    """Tab showing detailed performance breakdown for all 6 targets."""

    def __init__(self, parent, app):
        self.app = app
        self._build_ui(parent)
        self._load_data()

    def _build_ui(self, parent):
        if not CTK:
            ctk.CTkLabel(parent, text="Metrics Tab requires CustomTkinter").pack()
            return

        # Main container
        self.main = ctk.CTkFrame(parent, fg_color="transparent")
        self.main.pack(fill='both', expand=True, padx=20, pady=20)

        # Header section
        header = ctk.CTkFrame(self.main, fg_color="transparent")
        header.pack(fill='x', pady=(0, 20))
        
        ctk.CTkLabel(header, text="Detailed Metrics Dashboard", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(side='left')
        
        # Model selector
        self.model_var = ctk.StringVar(value="Select Model")
        self.model_dropdown = ctk.CTkComboBox(header, values=["Select Model"], 
                                               command=self._on_model_change, 
                                               width=240, height=35)
        self.model_dropdown.pack(side='right', padx=10)
        ctk.CTkLabel(header, text="Choose Model:").pack(side='right')

        # Scrollable content area
        self.scroll = ctk.CTkScrollableFrame(self.main, fg_color="transparent")
        self.scroll.pack(fill='both', expand=True)

        # Initial message
        self.info_label = ctk.CTkLabel(self.scroll, text="Please select a model from the dropdown above to view detailed metrics.",
                                      font=ctk.CTkFont(size=14, slant="italic"), text_color="gray")
        self.info_label.pack(pady=100)

        # Placeholder for targets grid
        self.grid_frame = None

    def _on_model_change(self, choice):
        if choice == "Select Model" or not self.app.training_results:
            return
        
        if self.info_label:
            self.info_label.pack_forget()
            self.info_label = None

        if self.grid_frame:
            self.grid_frame.destroy()

        self.grid_frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.grid_frame.pack(fill='both', expand=True)

        metrics_all = self.app.training_results.get(choice, {})
        target_metrics = metrics_all.get('Target_Metrics', {})

        # 6 cards for 6 targets
        colors = ['#1a237e', '#1b5e20', '#b71c1c', '#e65100', '#4a148c', '#004d40']
        
        for i, target in enumerate(TARGETS):
            row, col = i // 2, i % 2
            data = target_metrics.get(target, {})
            
            card = ctk.CTkFrame(self.grid_frame, fg_color="#2c2c36", corner_radius=15, border_width=1, border_color=colors[i])
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            self.grid_frame.grid_columnconfigure(col, weight=1)

            # Property Title
            clean_name = target.replace('R2_', '').replace('_', ' ')
            title_box = ctk.CTkFrame(card, fg_color=colors[i], corner_radius=10)
            title_box.pack(fill='x', padx=10, pady=10)
            ctk.CTkLabel(title_box, text=clean_name, font=ctk.CTkFont(size=16, weight="bold"), text_color="white").pack(pady=5)

            # Metrics row
            metrics_grid = ctk.CTkFrame(card, fg_color="transparent")
            metrics_grid.pack(fill='both', expand=True, padx=15, pady=(0, 15))

            # Metric display boxes
            m_items = [
                ("Accuracy%", data.get('Accuracy%', 'N/A'), "%"),
                ("RMSE", data.get('RMSE', 'N/A'), ""),
                ("MAE", data.get('MAE', 'N/A'), ""),
                ("R² Score", data.get('R2', 'N/A'), "")
            ]

            for idx, (label, val, unit) in enumerate(m_items):
                r, c = idx // 2, idx % 2
                m_box = ctk.CTkFrame(metrics_grid, fg_color="#1e1e24", corner_radius=8)
                m_box.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
                metrics_grid.grid_columnconfigure(c, weight=1)

                ctk.CTkLabel(m_box, text=label, font=ctk.CTkFont(size=11), text_color="#90caf9").pack(pady=(5, 0))
                
                display_val = f"{val:.4f}" if isinstance(val, (float, int)) else str(val)
                if label == "Accuracy%": display_val = f"{val:.2f}"
                
                ctk.CTkLabel(m_box, text=f"{display_val}{unit}", font=ctk.CTkFont(size=18, weight="bold"), text_color="white").pack(pady=(0, 5))

    def _load_data(self):
        """Update dropdown with available models."""
        if hasattr(self.app, 'training_results') and self.app.training_results:
            models = list(self.app.training_results.keys())
            self.model_dropdown.configure(values=models)
            if self.app.best_model_name in models:
                self.model_dropdown.set(self.app.best_model_name)
                self._on_model_change(self.app.best_model_name)
