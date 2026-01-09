import pandas as pd
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("output", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# Preprocessing (Scaling)
# -----------------------------
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "output/model.pkl")

# -----------------------------
# Save Metrics
# -----------------------------
results = {
    "MSE": mse,
    "R2": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Training completed successfully.")


