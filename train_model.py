import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from pathlib import Path

# Load dataset
df = pd.read_csv("cricket_data.csv")

# Features & target
X = df[["current_score", "wickets", "overs", "last_5"]]
y = df["final_score"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
MODEL_PATH = Path("model/model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model/model.pkl")
