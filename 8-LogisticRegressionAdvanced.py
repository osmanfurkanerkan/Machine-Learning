import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("8-fraud_detection.csv")
print(df.head())
print(df.info())
print(df["is_fraud"].value_counts())

# Features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Scatter plot of the data
sns.scatterplot(x=X["transaction_amount"], y=X["transaction_risk_score"], hue=y)
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Logistic Regression with hyperparameter tuning
model = LogisticRegression()

# Hyperparameters
penalty = ['l1', 'l2', 'elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
class_weight = [{0: w, 1: y} for w in [1, 10, 50, 100] for y in [1, 10, 50, 100]]

params = dict(penalty=penalty, C=c_values, solver=solver, class_weight=class_weight)

# Grid search with stratified k-fold
cv = StratifiedKFold()
grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=cv)
grid.fit(X_train, y_train)

print("Best Hyperparameters:", grid.best_params_)

# Predictions
y_pred = grid.predict(X_test)
print("Accuracy:", accuracy_score(y_pred, y_test))
print("Classification Report:\n", classification_report(y_pred, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_pred, y_test))

# ROC Curve & AUC
model_prob = grid.predict_proba(X_test)[:, 1]  # Probabilities for positive class
model_auc = roc_auc_score(y_test, model_prob)
print("ROC AUC:", model_auc)

model_fpr, model_tpr, thresholds = roc_curve(y_test, model_prob)

plt.figure(figsize=(10,6))
plt.plot(model_fpr, model_tpr, marker='.', label=f'Logistic (AUC={model_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# Custom threshold example
custom_threshold = 0.2
y_pred_custom = (model_prob >= custom_threshold).astype(int)
print(f"\nEvaluation using custom threshold = {custom_threshold}")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))

# Precision-Recall vs Threshold plot
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, model_prob)
plt.figure(figsize=(10,6))
plt.plot(pr_thresholds, precisions[:-1], label='Precision')
plt.plot(pr_thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision & Recall vs. Threshold')
plt.legend()
plt.grid()
plt.show()