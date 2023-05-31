import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
import json
from joblib import dump

df = pd.read_excel("data/row_dataset.xlsx")

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = PassiveAggressiveClassifier()

clf.fit(X_train, y_train)

dump(clf, "models/model.joblib")

# y_pred = clf.predict(X_test)

# f_score = f1_score(y_test, y_pred, average='weighted')
# print(f_score)



