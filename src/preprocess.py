import pandas as pd
from sklearn.preprocessing import MinMaxScaler


INPUT_FILE = "data/raw/battery_sensor_data.csv"
OUTPUT_FILE = "data/processed/battery_sensor_processed.csv"

FEATURES = [
    "temperature",
    "voltage",
    "current",
    "coolant_flow"
]


# Load raw dataset
df = pd.read_csv(INPUT_FILE)

print("Original Dataset:")
print(df)
print("\nMissing Values:")
print(df[FEATURES].isnull().sum())


# Handle missing values using column mean
for column in FEATURES:
    df[column] = df[column].fillna(df[column].mean())


print("\nMissing Values After Handling:")
print(df[FEATURES].isnull().sum())


# Normalize sensor features
scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])


# Save processed dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessed Dataset:")
print(df)

print("\nProcessed dataset saved to:")
print(OUTPUT_FILE)