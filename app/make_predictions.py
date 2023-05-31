import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
import json
from joblib import load

df = pd.read_excel("data/row_dataset.xlsx")

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = load("models/model.joblib")

y_pred = model.predict(X_test)

score = f1_score(y_test, y_pred, average="macro")

with open("metrics/metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"f1_score: {score}")

