"""
Main entry point for HengSeng Stock Sentiment & Price Prediction project.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.modelling.prediction_engine import generate_all_stock_predictions
from src.modelling.evaluate_metrics import calculate_metrics

def main():
    print("=" * 80)
    print("HengSeng Stock Sentiment & Price Prediction System")
    print("=" * 80)
    print("\n1. Calculating Comprehensive Model Evaluation Metrics...")
    calculate_metrics()
    print("\n2. Generating Latest Stock Predictions & Agent Synthesis...")
    generate_all_stock_predictions()
    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()
