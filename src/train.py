import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib, os

X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("Model trained and saved to models/model.pkl")
