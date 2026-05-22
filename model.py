import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# CREATE DATASET
# -----------------------------

np.random.seed(42)

data_size = 1000

ages = np.random.randint(18, 60, data_size)

income = np.random.choice(
    ["Low", "Medium", "High"],
    data_size,
    p=[0.3, 0.4, 0.3]
)

rating = np.round(np.random.uniform(1, 5, data_size), 1)

satisfaction = np.random.choice(
    ["Low", "Medium", "High"],
    data_size,
    p=[0.25, 0.35, 0.4]
)

ads = np.random.choice(
    ["Low", "Medium", "High"],
    data_size,
    p=[0.3, 0.3, 0.4]
)

decision = np.random.randint(1, 72, data_size)

# -----------------------------
# TARGET CREATION LOGIC
# -----------------------------

purchased = []

for i in range(data_size):

    score = 0

    # income
    if income[i] == "High":
        score += 2
    elif income[i] == "Medium":
        score += 1

    # rating
    if rating[i] >= 4:
        score += 2
    elif rating[i] >= 3:
        score += 1

    # satisfaction
    if satisfaction[i] == "High":
        score += 2
    elif satisfaction[i] == "Medium":
        score += 1

    # ads
    if ads[i] == "High":
        score += 2
    elif ads[i] == "Medium":
        score += 1

    # decision time
    if decision[i] <= 5:
        score += 2
    elif decision[i] <= 24:
        score += 1

    # age
    if 20 <= ages[i] <= 35:
        score += 1

    # final target
    if score >= 7:
        purchased.append(1)
    else:
        purchased.append(0)

# -----------------------------
# DATAFRAME
# -----------------------------

df = pd.DataFrame({
    "Age": ages,
    "Income": income,
    "Rating": rating,
    "Satisfaction": satisfaction,
    "Ads": ads,
    "Decision": decision,
    "Purchased": purchased
})

# -----------------------------
# ENCODING
# -----------------------------

income_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

satisfaction_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

ads_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["Income"] = df["Income"].map(income_map)
df["Satisfaction"] = df["Satisfaction"].map(satisfaction_map)
df["Ads"] = df["Ads"].map(ads_map)

# -----------------------------
# FEATURES & TARGET
# -----------------------------

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# -----------------------------
# SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# MODEL
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# -----------------------------
# ACCURACY
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClass Distribution:")
print(df["Purchased"].value_counts())

# -----------------------------
# SAVE MODEL
# -----------------------------

pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved successfully!")

# -----------------------------
# FEATURE IMPORTANCE GRAPH
# -----------------------------

importance = model.feature_importances_

feature_names = X.columns

plt.figure(figsize=(8,5))

plt.bar(feature_names, importance)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("static/feature_importance.png")

plt.close()

# -----------------------------
# CLASS DISTRIBUTION GRAPH
# -----------------------------

counts = df["Purchased"].value_counts()

labels = ["Not Purchase", "Purchase"]

plt.figure(figsize=(6,6))

plt.pie(
    counts,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title("Class Distribution")

plt.savefig("static/class_distribution.png")

plt.close()

print("\nGraphs generated successfully!")