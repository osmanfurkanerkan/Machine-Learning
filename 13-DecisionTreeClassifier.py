# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# DATASET AÇIKLAMASI
# =========================
# Car Evaluation Dataset:
# Amaç: Arabanın kabul edilebilirliğini (class) tahmin etmek
# Feature'lar:
# buying, maint, doors, persons, lug_boot, safety
# Target:
# class (unacc, acc, good, vgood)

# =========================
# VERİYİ OKUMA
# =========================
df = pd.read_csv("13-car_evaluation.csv")

# Kolon isimlerini düzeltme
col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df.columns = col_names

# =========================
# VERİ ANALİZİ
# =========================
print(df.head())
print(df.shape)
print(df.info())

# Her sütundaki dağılım
for col in df.columns:
    print(df[col].value_counts())

# Null değer kontrolü
print(df.isnull().sum())

# =========================
# VERİ TEMİZLEME
# =========================

# doors -> '5more' -> 5
df['doors'] = df['doors'].replace('5more', '5')
df['doors'] = df['doors'].astype(int)

# persons -> 'more' -> 5
df['persons'] = df['persons'].replace('more', '5')
df['persons'] = df['persons'].astype(int)

print(df.info())

# =========================
# GÖRSELLEŞTİRME
# =========================
sns.scatterplot(x=df["buying"], y=df["maint"], hue=df["class"])
plt.show()

sns.barplot(x=df["buying"], hue=df["class"])
plt.show()

sns.scatterplot(x=df["lug_boot"], y=df["safety"], hue=df["class"])
plt.show()

# =========================
# FEATURE / TARGET AYIRMA
# =========================
X = df.drop('class', axis=1)
y = df["class"]

# =========================
# TRAIN - TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

print(X_train.shape)

# =========================
# ENCODING (Ordinal)
# =========================
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

categorical_cols = ["buying", "maint", "lug_boot", "safety"]
numerical_cols = ["doors", "persons"]

# Sıralı encoding (önemli!)
ordinal_encoder = OrdinalEncoder(categories=[
    ["low", "med", "high", "vhigh"],   # buying
    ["low", "med", "high", "vhigh"],   # maint
    ["small", "med", "big"],           # lug_boot
    ["low", "med", "high"]             # safety
])

preprocessor = ColumnTransformer(
    transformers=[
        ('encoder', ordinal_encoder, categorical_cols)
    ],
    remainder="passthrough"
)

# Transform işlemleri
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# =========================
# DECISION TREE (BASE MODEL)
# =========================
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=0
)

tree_model.fit(X_train_transformed, y_train)

# Tahmin
y_pred = tree_model.predict(X_test_transformed)

# Performans
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("BASE MODEL")
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# TREE GÖRSELİ
# =========================
from sklearn import tree

column_names = categorical_cols + numerical_cols

plt.figure(figsize=(12, 8))
tree.plot_tree(tree_model, feature_names=column_names)
plt.show()

# =========================
# HYPERPARAMETER TUNING
# =========================
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

param = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [1, 2, 3, 4, 5, 15, None],
    "max_features": ["sqrt", "log2", None]
}

grid = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=param,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_transformed, y_train)

print("BEST PARAMS:", grid.best_params_)

# =========================
# EN İYİ MODEL
# =========================
y_pred = grid.predict(X_test_transformed)

print("TUNED MODEL")
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# FINAL MODEL (MANUAL)
# =========================
tree_model_new = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    max_features=None,
    splitter="best"
)

tree_model_new.fit(X_train_transformed, y_train)

y_pred = tree_model_new.predict(X_test_transformed)

print("FINAL MODEL")
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# FINAL TREE
# =========================
plt.figure(figsize=(14, 10))
tree.plot_tree(tree_model_new, feature_names=column_names)
plt.show()

# =========================
# NOTLAR
# =========================
# - max_depth = None -> ağaç aşırı büyür (overfitting riski)
# - küçük max_depth -> underfitting olabilir
# - bu yüzden pruning (budama) önemlidir