# Gerekli kütüphaneleri import ediyoruz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataseti okuyoruz
df = pd.read_csv("10-diamonds.csv")

# İlk 5 veriyi göster
print(df.head())

# Dataset boyutu
print(df.shape)

# Genel bilgi
print(df.info())

# Gereksiz kolonu siliyoruz
df = df.drop(["Unnamed: 0"], axis=1)

# İstatistiksel özet
print(df.describe())

# x, y, z = 0 olan hatalı verileri temizliyoruz
df = df.drop(df[df["x"] == 0].index)
df = df.drop(df[df["y"] == 0].index)
df = df.drop(df[df["z"] == 0].index)

print(df.describe())

# Veri dağılımını incelemek için pairplot
sns.pairplot(df)
plt.show()

# Fiyat ile x ilişkisi
sns.scatterplot(x=df["x"], y=df["price"])
plt.show()

# Fiyat ile y ilişkisi (outlier var)
sns.scatterplot(x=df["y"], y=df["price"])
plt.show()

# Fiyat ile z ilişkisi (outlier var)
sns.scatterplot(x=df["z"], y=df["price"])
plt.show()

# table ile price
sns.scatterplot(x=df["table"], y=df["price"])
plt.show()

# depth ile price
sns.scatterplot(x=df["depth"], y=df["price"])
plt.show()

# Outlier temizleme işlemleri
df = df[(df["depth"] < 75) & (df["depth"] > 45)]
df = df[(df["table"] < 80) & (df["table"] > 40)]
df = df[(df["y"] < 30)]
df = df[(df["z"] < 30) & (df["z"] > 2)]

print(df.describe())

# Temizlenmiş veri sonrası tekrar kontrol
sns.scatterplot(x=df["z"], y=df["price"])
plt.show()

print(df.head())

# Kategorik değişkenleri inceleme
print(df['cut'].value_counts())
print(df['color'].value_counts())
print(df['clarity'].value_counts())

# Bağımsız ve bağımlı değişkenleri ayırıyoruz
X = df.drop(["price"], axis=1)
y = df["price"]

# Train-test split (data leakage olmaması için önce bölüyoruz)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)

# Kategorik verileri sayısal hale getiriyoruz (Label Encoding)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for col in ['cut', 'color', 'clarity']:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

print(X_train.head())
print(X_train.describe())
print(X_train.info())

# Verileri ölçeklendirme (StandardScaler)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- LINEAR REGRESSION --------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

y_pred = linreg.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print("Linear Regression MAE:", mae)
print("Linear Regression R2:", score)

plt.scatter(y_test, y_pred)
plt.show()

# -------- SVR (SVM REGRESSION) --------
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train_scaled, y_train)

y_pred = svr.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print("SVR MAE:", mae)
print("SVR R2:", score)

plt.scatter(y_test, y_pred)
plt.show()

# -------- GRID SEARCH (HYPERPARAMETER TUNING) --------
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(
    SVR(),
    param_grid,
    refit=True,
    verbose=3,
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

# En iyi parametreler
print("Best Params:", grid.best_params_)

# Test verisi ile tahmin
y_pred = grid.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print("GridSearch SVR MAE:", mae)
print("GridSearch SVR R2:", score)

plt.scatter(y_test, y_pred)
plt.show()