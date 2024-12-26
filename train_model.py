import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset (replace 'house_prices.csv' with your dataset)
data = pd.read_csv('Housing.csv')

# Select features and target (e.g., square footage as a single feature)
X = data[['area']]  # Feature: square footage
y = data['price']          # Target: price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'simple_linear_model.pkl')