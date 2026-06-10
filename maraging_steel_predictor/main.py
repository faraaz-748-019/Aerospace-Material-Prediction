"""
M250 Maraging Steel — Weld Repair Property Predictor
Entry point: launches the GUI application.
On first run, automatically trains models if none exist.
"""

import os
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def check_dependencies():
    """Check and install required packages."""
    required = [
        'torch', 'sklearn', 'xgboost', 'lightgbm', 'shap',
        'pandas', 'numpy', 'matplotlib', 'scipy', 'joblib',
        'customtkinter', 'reportlab'
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            # Map package import names to pip install names
            pip_names = {
                'sklearn': 'scikit-learn',
                'torch': 'torch',
                'cv2': 'opencv-python',
            }
            missing.append(pip_names.get(pkg, pkg))

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet'
        ] + missing)
        print("Installation complete.")


def models_exist():
    """Check if trained models exist."""
    ml_dir = os.path.join(BASE_DIR, 'models', 'ml')
    dl_dir = os.path.join(BASE_DIR, 'models', 'dl')

    ml_exists = any(
        os.path.exists(os.path.join(ml_dir, f))
        for f in ['extratrees.pkl', 'randomforest.pkl', 'xgboost.pkl']
    )
    dl_exists = any(
        os.path.exists(os.path.join(dl_dir, f))
        for f in ['cnn.pt', 'lstm.pt', 'transformer.pt']
    )
    return ml_exists or dl_exists


def auto_train_if_needed(app):
    """Auto-train models on first run if none exist."""
    if not models_exist():
        safe_print("=" * 60)
        safe_print(" No trained models found. Starting automatic training...")
        safe_print("=" * 60)

        def train():
            try:
                from train_all import run_full_training
                results, best, scaler = run_full_training(
                    callback=lambda msg: safe_print(f"  {msg}"),
                    progress_callback=None
                )
                safe_print(f"\n[OK] Training complete! Best model: {best}")
                safe_print("   Reloading models into application...")
                app._load_existing_models()
                # Update training tab if it has results display
                if hasattr(app, 'training_tab'):
                    app.root.after(0, lambda: app.training_tab._load_existing_results())
            except Exception as e:
                safe_print(f"\n[ERROR] Auto-training failed: {e}")
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=train, daemon=True)
        thread.start()
    else:
        safe_print("[OK] Pre-trained models found. Ready for prediction.")


def safe_print(msg):
    """Print with fallback for unsupported characters on Windows console."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'))


def main():
    """Main entry point."""
    safe_print("=" * 60)
    safe_print(" M250 Maraging Steel -- Weld Repair Property Predictor")
    safe_print(" Vikram Sarabhai Space Centre (VSSC/ISRO)")
    safe_print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Load baseline model if available
    baseline_path = os.path.join(BASE_DIR, 'data', 'option_A_revised_best_model.pkl')
    if os.path.exists(baseline_path):
        safe_print(f"[BASELINE] Baseline model available: {baseline_path}")

    # Launch GUI
    from gui.app import MainApp
    app = MainApp()

    # Auto-train if needed (runs in background)
    auto_train_if_needed(app)

    app.run()


if __name__ == '__main__':
    main()
