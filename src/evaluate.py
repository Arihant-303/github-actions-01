import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv("data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# load model
model = joblib.load("model/model.pkl")

# predict
pred = model.predict(X)

acc = accuracy_score(y, pred)

print("Accuracy:", acc)

# fail pipeline if accuracy low
if acc < 0.65:
    raise Exception("Accuracy below threshold")