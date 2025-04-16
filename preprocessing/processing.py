import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import ipywidgets as widgets
from IPython.display import display
from datetime import datetime

df = pd.read_csv("../data/raw/LOG.csv")
df = df[~df['Timestamp'].astype(str).str.lower().str.contains("timestamp")]
df['Distance_mm'] = pd.to_numeric(df['Distance_mm'], errors='coerce')
df = df.dropna(subset=['Distance_mm'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df = df[(df['Timestamp'].dt.month == 4) & (df['Distance_mm'] >= 700)]


df.reset_index(drop=True, inplace=True)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

activity = df.resample('10min').size().rename("Activity_Count").reset_index()

activity['mod7'] = np.arange(len(activity)) % 7

activity['Hour'] = activity['Timestamp'].dt.hour
activity['Minute'] = activity['Timestamp'].dt.minute
activity['DayOfWeek'] = activity['Timestamp'].dt.dayofweek
activity['Weekend'] = activity['DayOfWeek'] >= 5

activity.to_csv("full_aggregated_data.csv", index=False)
print("Exported 'full_aggregated_data.csv' with all rows, new features, and a 'mod7' column.")

features = ['Hour', 'Minute', 'DayOfWeek', 'Weekend']
X = activity[features]
y = activity['Activity_Count']

indices = np.arange(len(X))
val_inds = indices[::7]
test_inds = indices[1::7]
both = np.union1d(val_inds, test_inds)
train_inds = np.setdiff1d(indices, both)

print(f"Total rows:      {len(X)}")
print(f"Train rows:      {len(train_inds)}  (~{100*len(train_inds)/len(X):.1f}%)")
print(f"Validation rows: {len(val_inds)}    (~{100*len(val_inds)/len(X):.1f}%)")
print(f"Test rows:       {len(test_inds)}    (~{100*len(test_inds)/len(X):.1f}%)")

X_train = X.iloc[train_inds]
y_train = y.iloc[train_inds]

X_val   = X.iloc[val_inds]
y_val   = y.iloc[val_inds]

X_test  = X.iloc[test_inds]
y_test  = y.iloc[test_inds]

# Export subsets
train_df = activity.iloc[train_inds].copy()
val_df   = activity.iloc[val_inds].copy()
test_df  = activity.iloc[test_inds].copy()

train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
print("\nExported 'train_data.csv', 'val_data.csv', 'test_data.csv'.")

# Train model
model = HistGradientBoostingRegressor(loss="poisson")  
model.fit(X_train, y_train)

# val
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2  = r2_score(y_val, y_val_pred)

print("\nValidation Metrics (Approx 15% of data):")
print(f"MAE: {val_mae:.2f}")
print(f"MSE: {val_mse:.2f}")
print(f"R² : {val_r2:.2f}")

# test
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2  = r2_score(y_test, y_test_pred)

print("\nTest Metrics (Approx 15% of data):")
print(f"MAE: {test_mae:.2f}")
print(f"MSE: {test_mse:.2f}")
print(f"R² : {test_r2:.2f}")

# Plot
timestamps_test = activity['Timestamp'].iloc[test_inds]

plt.figure(figsize=(12, 5))
plt.plot(timestamps_test, y_test.values, label='Actual', marker='o', linewidth=1)
plt.plot(timestamps_test, y_test_pred, label='Predicted', marker='x', linewidth=1)
plt.xlabel("Time")
plt.ylabel("Activity Count")
plt.title("Test Set: Predicted vs Actual Activity Count (Every 7th row offset=1)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

joblib.dump(model, "model/output_model.pkl")
print("\nModel saved to 'output_model.pkl'.")
