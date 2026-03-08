# cyber_attack_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import warnings

warnings.filterwarnings("ignore")

# Veri yükleme
df = pd.read_csv("7-cyber_attack_data.csv")

# Özellik ve hedef değişken
X = df.drop("attack_type", axis=1)
y = df["attack_type"]

# Eğitim ve test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

# Baseline Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Baseline Logistic Regression")
print("Accuracy Score:", accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_pred, y_test))
print("-" * 50)

# Grid Search Hyperparameter Tuning
penalty = ["l1", "l2", "elasticnet"]
c_values = [100, 10, 1, 0.1, 0.01]
solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga", "newton-cholesky"]

params = dict(penalty=penalty, C=c_values, solver=solver)

cv = StratifiedKFold()
grid = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print("Grid Search Best Params:", grid.best_params_)
print("Grid Search Best Score:", grid.best_score_)

y_pred_grid = grid.predict(X_test)
print("Accuracy Score after Grid Search:", accuracy_score(y_pred_grid, y_test))
print(classification_report(y_pred_grid, y_test))
print("Confusion Matrix after Grid Search:\n", confusion_matrix(y_pred_grid, y_test))
print("-" * 50)

# One-vs-One Classifier
onevsonemodel = OneVsOneClassifier(LogisticRegression(max_iter=1000))
onevsonemodel.fit(X_train, y_train)
y_pred_ovo = onevsonemodel.predict(X_test)
print("One-vs-One Classifier")
print("Accuracy Score:", accuracy_score(y_pred_ovo, y_test))
print(classification_report(y_pred_ovo, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_pred_ovo, y_test))
print("-" * 50)

# One-vs-Rest Classifier
onevsrestmodel = OneVsRestClassifier(LogisticRegression(max_iter=1000))
onevsrestmodel.fit(X_train, y_train)
y_pred_ovr = onevsrestmodel.predict(X_test)
print("One-vs-Rest Classifier")
print("Accuracy Score:", accuracy_score(y_pred_ovr, y_test))
print(classification_report(y_pred_ovr, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_pred_ovr, y_test))