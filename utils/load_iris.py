import pandas as pd
from sklearn.datasets import load_iris


data = load_iris()

df = pd.DataFrame(data = data.data)
df["y"] = data.target

df = df[:100]

df.to_excel("../data/row_dataset.xlsx", index=False)
