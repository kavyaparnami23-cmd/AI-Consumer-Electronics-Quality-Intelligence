"""
start_mlflow_ui.py
───────────────────
Launches the MLflow tracking UI on localhost:5000.

Usage:
    python start_mlflow_ui.py           # default port 5000
    python start_mlflow_ui.py --port 5001

Opens the browser automatically if possible.
"""

import subprocess
import sys
import os
import argparse
import webbrowser
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.mlflow_utils.mlflow_config import MLFLOW_TRACKING_URI


def launch_ui(port: int = 5000, host: str = "127.0.0.1"):
    print("\n" + "=" * 60)
    print("  MLFLOW TRACKING UI")
    print("=" * 60)
    print(f"  Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"  UI Address   : http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")

    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", MLFLOW_TRACKING_URI,
        "--host", host,
        "--port", str(port),
    ]

    # Try to open browser after 2 seconds
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open(f"http://{host}:{port}")
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n  MLflow UI stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    launch_ui(args.port, args.host)
