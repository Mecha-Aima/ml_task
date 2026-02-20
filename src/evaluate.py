import pandas as pd
from sklearn.metrics import accuracy_score
import joblib, os

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

model = joblib.load('models/model.pkl')
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc:.4f}")

os.makedirs('results', exist_ok=True)
with open('results/accuracy.txt', 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\n")
print("Results saved to results/accuracy.txt")
