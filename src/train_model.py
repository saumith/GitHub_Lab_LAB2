import os
import pandas as pd
import joblib
from urllib.request import urlretrieve
from zipfile import ZipFile
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# === Step 1: Download the dataset ZIP ===
base_url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
zip_path = "bank_marketing.zip"

if not os.path.exists(zip_path):
    print("📦 Downloading dataset...")
    urlretrieve(base_url, zip_path)

# === Step 2: Extract it ===
extract_dir = "data_raw"
os.makedirs(extract_dir, exist_ok=True)

with ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

# === Step 3: Extract the inner 'bank.zip' ===
bank_zip_path = os.path.join(extract_dir, "bank.zip")
with ZipFile(bank_zip_path, "r") as z:
    z.extractall(extract_dir)

# === Step 4: Load the CSV ===
csv_path = os.path.join(extract_dir, "bank.csv")
df = pd.read_csv(csv_path, sep=';')
print("✅ Dataset loaded successfully:", df.shape)

# === Step 5: Preprocess categorical columns ===
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# === Step 6: Split into features and target ===
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split done:", X_train.shape, X_test.shape)

# === Step 7: Train a Random Forest Classifier ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === Step 8: Save model and test data ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

model_path = f"models/model_{timestamp}.pkl"
joblib.dump(model, model_path)

X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print(f"✅ Model trained and saved as {model_path}")
print("📊 Test data stored for evaluation in /data folder.")

# === Step 9: Optional cleanup ===
# Remove large nested zips to keep repo light
os.remove(zip_path)
os.remove(bank_zip_path)
print("🧹 Cleaned up ZIP files.")
