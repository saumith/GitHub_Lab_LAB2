import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === Step 1: Check for models directory ===
if not os.path.exists("models") or len(os.listdir("models")) == 0:
    sys.exit("❌ No trained models found in /models. Please run train_model.py first.")

# === Step 2: Load the latest trained model ===
latest_model = sorted(os.listdir("models"))[-1]
model_path = os.path.join("models", latest_model)
model = joblib.load(model_path)
print(f"✅ Loaded model: {latest_model}")

# === Step 3: Load test data ===
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()
print(f"✅ Test data loaded: {X_test.shape}")

# === Step 4: Evaluate model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# === Step 5: Save metrics ===
os.makedirs("metrics", exist_ok=True)
metrics_path = "metrics/metrics.txt"

with open(metrics_path, "w") as f:
    f.write(f"Model: {latest_model}\n")
    f.write(f"Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

print(f"✅ Evaluation complete. Metrics saved to {metrics_path}")
print(f"📊 Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
