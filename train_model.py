# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# --- Data Loading and Preprocessing ---
try:
    df = pd.read_csv('loan_approval_dataset.csv')
except FileNotFoundError:
    print("Error: 'loan_approval_dataset.csv' not found. Please download from the Kaggle link and place it in this folder.")
    exit()

# Strip any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Drop the loan_id column
df = df.drop('loan_id', axis=1)

# Encode categorical features
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
df['education'] = le_education.fit_transform(df['education'])
df['self_employed'] = le_self_employed.fit_transform(df['self_employed'])

# The target variable is 'loan_status'
le_loan_status = LabelEncoder()
df['loan_status'] = le_loan_status.fit_transform(df['loan_status'])

# --- Model Training ---
# Define features (X) and target (y)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# --- Save the Model and Scaler ---
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# Also save the encoders to use in the app
with open('models/le_education.pkl', 'wb') as f:
    pickle.dump(le_education, f)
with open('models/le_self_employed.pkl', 'wb') as f:
    pickle.dump(le_self_employed, f)

print("Model, scaler, and encoders for the new dataset have been saved successfully!")