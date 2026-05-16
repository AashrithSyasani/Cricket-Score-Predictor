from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "model.pkl"

app = Flask(__name__)

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        runs = float(request.form.get("current_score"))
        wickets = int(request.form.get("wickets"))
        overs = float(request.form.get("overs"))
        last_5 = float(request.form.get("last_5"))

        # Innings already over?
        if overs >= 20 or wickets >= 10:
            return render_template("result.html",
                                   prediction=int(runs),
                                   lower=int(runs),
                                   upper=int(runs))

        # Prepare input
        X = pd.DataFrame([{
            "current_score": runs,
            "wickets": wickets,
            "overs": overs,
            "last_5": last_5
        }])

        predicted_total = model.predict(X)[0]

        # Apply penalty for wickets lost
        if wickets > 5:
            penalty_percent = 0.05 * (wickets - 5)
            predicted_total *= (1 - penalty_percent)

        # Ensure prediction not less than current runs
        if predicted_total <= runs:
            predicted_total = runs + 10

        # Remaining overs
        remaining_overs = 20.0 - overs
        if remaining_overs <= 0:
            final_score = runs
        else:
            remaining_runs = predicted_total - runs
            rate_per_over = remaining_runs / remaining_overs
            predicted_future_runs = rate_per_over * remaining_overs
            final_score = runs + predicted_future_runs

        # Round and add buffer range
        final_score = int(round(final_score))
        low, high = max(runs, final_score - 10), final_score + 10

        return render_template("result.html",
                               prediction=final_score,
                               lower=low,
                               upper=high)
    except Exception as e:
        return f"Error: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
