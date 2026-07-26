"""
app.py — Flask REST API for the AI Consumer Electronics
Quality Intelligence system.

Endpoints
---------
GET  /            → health check
POST /predict     → predict machine failure from JSON payload
POST /train       → trigger full training pipeline
"""

import sys
import os
import json
import traceback

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify

from pipeline.prediction_pipeline import PredictionPipelineRunner
from logger import logger

app = Flask(__name__)

# ------------------------------------------------------------------
# Lazy-loaded prediction runner (loaded once, shared across requests)
# ------------------------------------------------------------------
_predictor: PredictionPipelineRunner | None = None


def get_predictor() -> PredictionPipelineRunner:
    global _predictor
    if _predictor is None:
        _predictor = PredictionPipelineRunner()
    return _predictor


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "AI Consumer Electronics Quality Intelligence",
        "description": "Predictive Maintenance ML API",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict machine failure.

    Expected JSON body (single sample)
    ------------------------------------
    {
        "Type": 1,
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0
    }

    Response
    --------
    {
        "prediction": [0],
        "probability": [0.03],
        "label": ["No Failure"]
    }
    """
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "No JSON body provided"}), 400

        logger.info(f"/predict called with : {data}")

        predictor = get_predictor()
        result    = predictor.predict(data)

        return jsonify(result), 200

    except FileNotFoundError as fnf:
        msg = (
            "Model or preprocessor not found. "
            "Please run the training pipeline first via POST /train"
        )
        logger.error(f"FileNotFoundError: {fnf}")
        return jsonify({"error": msg}), 503

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train():
    """Trigger the full training pipeline."""
    try:
        logger.info("/train endpoint called")
        print("\n[API] Training pipeline triggered ...")

        from pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        artifact = pipeline.run()

        # Reset cached predictor so it picks up the new model
        global _predictor
        _predictor = None

        return jsonify({
            "status":          "success",
            "message":         "Training pipeline completed",
            "best_model":      "see evaluation report",
            "f1_score":         artifact.f1_score,
            "roc_auc":          artifact.roc_auc,
            "model_accepted":   artifact.model_accepted,
            "evaluation_report": artifact.evaluation_report_path,
        }), 200

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Consumer Electronics Quality Intelligence — API")
    print("  Running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
