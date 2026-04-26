import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DATASET
df = pd.read_csv("14-income_evaluation.csv")

# COLUMN NAMES
col_names = ['age', 'workclass', 'finalweight', 'education', 'education_num',
             'marital_status', 'occupation', 'relationship', 'race', 'sex',
             'capital_gain', 'capital_loss', 'hours_per_week',
             'native_country', 'income']
df.columns = col_names

# CLEAN '?' VALUES
df['workclass'] = df['workclass'].replace(' ?', np.nan)
df['occupation'] = df['occupation'].replace(' ?', np.nan)
df['native_country'] = df['native_country'].replace(' ?', np.nan)

# FEATURES / TARGET
X = df.drop('income', axis=1)
y = df['income']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# CATEGORICAL COLUMNS
categorical = [col for col in X_train.columns if X_train[col].dtype == 'O']

# FILL MISSING VALUES (MODE)
for dataset in [X_train, X_test]:
    dataset['workclass'] = dataset['workclass'].fillna(X_train['workclass'].mode()[0])
    dataset['occupation'] = dataset['occupation'].fillna(X_train['occupation'].mode()[0])
    dataset['native_country'] = dataset['native_country'].fillna(X_train['native_country'].mode()[0])

# TARGET ENCODING (native_country)
y_train_binary = y_train.apply(lambda x: 1 if x.strip() == '>50K' else 0)

target_means = y_train_binary.groupby(X_train['native_country']).mean()

X_train['native_country_encoded'] = X_train['native_country'].map(target_means)
X_test['native_country_encoded'] = X_test['native_country'].map(target_means)

X_train['native_country_encoded'].fillna(y_train_binary.mean(), inplace=True)
X_test['native_country_encoded'].fillna(y_train_binary.mean(), inplace=True)

X_train.drop("native_country", axis=1, inplace=True)
X_test.drop("native_country", axis=1, inplace=True)

# ONE HOT ENCODING
one_hot_categories = ['workclass', 'education', 'marital_status',
                      'occupation', 'relationship', 'race', 'sex']

encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_categories)
    ],
    remainder='passthrough'
)

X_train_enc = encoder.fit_transform(X_train)
X_test_enc = encoder.transform(X_test)

columns = encoder.get_feature_names_out()

X_train = pd.DataFrame(X_train_enc, columns=columns)
X_test = pd.DataFrame(X_test_enc, columns=columns)

# SCALING
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL
rfc = RandomForestClassifier(n_estimators=100, random_state=15)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# HYPERPARAMETER TUNING
rf_params = {
    "max_depth": [5, 8, 15, None, 10],
    "max_features": [5, 7, "sqrt", 8],
    "min_samples_split": [2, 8, 15, 20],
    "n_estimators": [100, 200, 500]
}

rscv = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=rf_params,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1
)

rscv.fit(X_train, y_train)

y_pred = rscv.predict(X_test)

print("\nTUNED MODEL")
print("Best Params:", rscv.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))