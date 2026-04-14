# Purchase Prediction Based on Social Network Advertisements
# Objectives:
    #To analyze customer data from social network advertisements
    #To preprocess and prepare the dataset for modeling
    #To build a machine learning model to predict purchase behavior
    #To evaluate the model using standard performance metrics
    #To identify key factors influencing customer decisions
    #To develop a simple web interface for real-time prediction

# Imorting libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,recall_score, f1_score, roc_auc_score)

# Load dataset
df = pd.read_csv(r'C:\Users\Akshaya G\OneDrive\Attachments\2ndYr\ML\major_project\Social_Network_Ads.csv')

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Features and target
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)
print("Confusion Matrix:\n", cm)

# Graphs
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.show()

import numpy as np

features = ['Gender', 'Age', 'EstimatedSalary']
importances = model.feature_importances_

# Sort features by importance
indices = np.argsort(importances)

plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()












