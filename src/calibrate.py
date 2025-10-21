import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, f1_score, accuracy_score

# === Load latest trained model ===
latest_model = sorted(os.listdir("models"))[-1]
model = joblib.load(os.path.join("models", latest_model))

# === Load test data ===
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# === Calibrate model ===
calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated_model.fit(X_test, y_test)

# === Predictions before and after calibration ===
y_pred_uncal = model.predict(X_test)
y_prob_uncal = model.predict_proba(X_test)[:, 1]

y_pred_cal = calibrated_model.predict(X_test)
y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

# === Compute metrics ===
brier_uncal = brier_score_loss(y_test, y_prob_uncal)
brier_cal = brier_score_loss(y_test, y_prob_cal)

f1_uncal = f1_score(y_test, y_pred_uncal)
f1_cal = f1_score(y_test, y_pred_cal)

acc_uncal = accuracy_score(y_test, y_pred_uncal)
acc_cal = accuracy_score(y_test, y_pred_cal)

# === Save calibrated model ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
calibrated_path = f"models/calibrated_model_{timestamp}.pkl"
joblib.dump(calibrated_model, calibrated_path)

# === Write calibration report ===
os.makedirs("metrics", exist_ok=True)
with open("metrics/calibration_metrics.txt", "w") as f:
    f.write(f"Calibrated from model: {latest_model}\n\n")
    f.write(f"Before calibration:\n  Brier Score = {brier_uncal:.4f}\n  F1 = {f1_uncal:.4f}\n  Accuracy = {acc_uncal:.4f}\n")
    f.write(f"\nAfter calibration:\n  Brier Score = {brier_cal:.4f}\n  F1 = {f1_cal:.4f}\n  Accuracy = {acc_cal:.4f}\n")

print(f"✅ Model calibrated and saved as {calibrated_path}")
print(f"📈 Calibration results saved to metrics/calibration_metrics.txt")
