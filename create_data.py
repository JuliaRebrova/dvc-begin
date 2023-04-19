from this import d
import numpy as np
import pandas as pd
import json
from joblib import dump


from sklearn.linear_model import LinearRegression


with open("json_data_1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X = [[x] for x in data["first"]]
y = data["second"]

reg = LinearRegression(n_jobs=-1).fit(X, y)

dump(reg, "model.joblib")

print(reg.score(X, y))