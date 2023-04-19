from joblib import load
import json
from pathlib import Path

from sklearn.metrics import accuracy_score

with open("json_data_1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = load("model.joblib")


X = [[x] for x in data["first"]]
y = data["second"]

accuracy = model.score(X, y)
metrics = {"accuracy": accuracy}


with open("accuracy.json", "w", encoding="utf-8") as f:
    data = json.dumps(metrics)

