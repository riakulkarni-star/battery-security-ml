import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


INPUT_FILE = "data/processed/battery_sensor_processed.csv"
MODEL_FILE = "models/battery_thermal_model.pkl"


# Load processed dataset
df = pd.read_csv(INPUT_FILE)

# Create a simple thermal index for the BMS use case
df["thermal_index"] = (
    0.6 * df["temperature"]
    + 0.2 * df["current"]
    - 0.1 * df["coolant_flow"]
)

# Features and target
features = [
    "temperature",
    "voltage",
    "current",
    "coolant_flow"
]

X = df[features]
y = df["thermal_index"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Training Complete")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save model
joblib.dump(model, MODEL_FILE)

print(f"Model saved to: {MODEL_FILE}")