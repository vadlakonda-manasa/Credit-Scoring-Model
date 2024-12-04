Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun  6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create a synthetic dataset for Credit Scoring
np.random.seed(42)

# Simulating some data
n_samples = 1000
age = np.random.randint(18, 70, size=n_samples)
income = np.random.randint(20000, 120000, size=n_samples)
loan_amount = np.random.randint(1000, 50000, size=n_samples)
credit_history = np.random.randint(0, 2, size=n_samples)  # 0 = bad, 1 = good
defaulted = np.random.randint(0, 2, size=n_samples)  # 0 = no, 1 = yes (target variable)

# Construct a DataFrame
data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Loan_Amount': loan_amount,
    'Credit_History': credit_history,
    'Defaulted': defaulted
})

# Data Exploration (Optional)
... print(data.head())
... print(data.describe())
... 
... # Feature and Target variables
... X = data.drop(columns=['Defaulted'])  # Features
... y = data['Defaulted']  # Target variable (Defaulted: 0 = No, 1 = Yes)
... 
... # Train-Test Split (70% train, 30% test)
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
... 
... # Feature Scaling
... scaler = StandardScaler()
... X_train = scaler.fit_transform(X_train)
... X_test = scaler.transform(X_test)
... 
... # Model Selection: Random Forest Classifier
... rf = RandomForestClassifier(n_estimators=100, random_state=42)
... rf.fit(X_train, y_train)
... 
... # Predictions and Evaluation
... y_pred = rf.predict(X_test)
... 
... # Classification Report and Confusion Matrix
... print("Classification Report:\n", classification_report(y_test, y_pred))
... print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
... 
... # ROC-AUC Score
... roc_auc = roc_auc_score(y_test, y_pred)
... print(f"ROC-AUC Score: {roc_auc:.4f}")
... 
... # Plotting Confusion Matrix
... sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
...             xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
... plt.title('Confusion Matrix')
... plt.xlabel('Predicted')
... plt.ylabel('Actual')
