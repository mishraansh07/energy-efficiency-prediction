"""
Practice model prep and evaluation for college project,
Btech final year project, and self-learning.
Energy Efficiency Prediction using Random Forest Regressor
Dataset: UCI Energy Efficiency (ID: 242 via ucimlrepo)
Target: Heating Load
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# Use Heating Load (first column) as the target
y_heating = y.iloc[:, 0]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_heating, test_size=0.2, random_state=42
)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Show some predictions
print("\nFirst 10 Predictions vs Actual:")
print("Predicted:", y_pred[:10])
print("Actual:   ", y_test[:10].values)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="k")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Heating Load")
plt.ylabel("Predicted Heating Load")
plt.title("Predicted vs Actual Heating Load")
plt.show()