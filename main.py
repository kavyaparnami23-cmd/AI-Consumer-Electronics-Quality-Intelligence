"""
main.py — CLI entry point for the AI Consumer Electronics
Quality Intelligence ML Pipeline.

Usage
-----
    cd src
    python ../main.py
"""

import sys
import os

# Make sure src/ is on the path so all imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline.training_pipeline import TrainingPipeline
from logger import logger


def main():
    print("\n" + "=" * 60)
    print("  AI Consumer Electronics Quality Intelligence")
    print("  Predictive Maintenance — ML Pipeline")
    print("=" * 60)

    pipeline = TrainingPipeline()
    artifact = pipeline.run()

    print("\nPipeline finished successfully.")
    print(f"Evaluation report → {artifact.evaluation_report_path}")


if __name__ == "__main__":
    main()
