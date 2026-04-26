import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Veri Setini Yükleme
df = pd.read_csv("15-gym_crowdedness.csv")

# 2. Veri Keşfi (PyCharm'da görebilmek için print fonksiyonu eklendi)
print("--- İlk 5 Satır ---")
print(df.head())
print("\n--- Boyut ---")
print(df.shape)
print("\n--- Sütunlar ---")
print(df.columns)
print("\n--- Bilgi ---")
df.info()
print("\n--- İstatistiksel Özet ---")
print(df.describe())
print("\n--- Eksik Değerler ---")
print(df.isnull().sum())

# 3. Veri Ön İşleme
df['date'] = pd.to_datetime(df['date'], utc=True)
print("\n--- Tarih Dönüşümü Sonrası Bilgi ---")
df.info()

print("\n--- Yıllar ---")
print(df['year'].unique())

df.drop('date', axis=1, inplace=True)
print("\n--- Date Sütunu Düşürüldükten Sonra İlk 5 Satır ---")
print(df.head())

# 4. Görselleştirmeler
# Not: PyCharm'da grafikler sırayla açılır. Bir grafiği kapattığınızda kod çalışmaya devam edip diğerini açar.
sns.lineplot(data=df, x="hour", y="number_people", errorbar=None)
plt.title("Average People per Hour")
plt.show()

sns.barplot(data=df, x="day_of_week", y="number_people")
plt.title("People per Day")
plt.show()

sns.regplot(data=df, x="temperature", y="number_people")
plt.title("Temperature vs People")
plt.show()

sns.boxplot(data=df, x="is_holiday", y="number_people")
plt.title("People on Holidays")
plt.show()

sns.boxplot(data=df, x="is_start_of_semester", y="number_people")
plt.title("People on Start of Semesters")
plt.show()

sns.boxplot(data=df, x="is_during_semester", y="number_people")
plt.title("People During Semesters")
plt.show()

# Sadece sayısal verilerle korelasyon hesaplama
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

print("\n--- Korelasyon Matrisi ---")
print(df.corr(numeric_only=True))

# 5. Makine Öğrenmesi Hazırlık
df.drop('timestamp', axis=1, inplace=True)

# Bağımlı & Bağımsız Değişken Ayrımı
X = df.drop('number_people', axis=1)
y = df['number_people']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Modellerin Kurulması ve Metriklerin Hesaplanması
def calculate_model_metrics(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor()
}

print("\n--- Tüm Modellerin Temel Eğitimi ---")
for name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae, model_train_rmse, model_train_r2 = calculate_model_metrics(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = calculate_model_metrics(y_test, y_test_pred)

    print(f"\nModel: {name}")
    print("Evaluation for Training Set")
    print("RMSE :", model_train_rmse)
    print("Mean Absolute Error :", model_train_mae)
    print("R2 Score :", model_train_r2)
    print("-----------------------------")
    print("Evaluation for Test Set")
    print("RMSE :", model_test_rmse)
    print("Mean Absolute Error :", model_test_mae)
    print("R2 Score :", model_test_r2)
    print("-----------------------------")

# 7. Hiperparametre Optimizasyonu (Tuning)
knn_params = {"n_neighbors": [2, 3, 10, 20, 40, 50]}
rf_params = {
    "max_depth": [5, 8, 10, 15, None],
    "max_features": ["sqrt", "log2", 5, 7, 10],
    "min_samples_split": [2, 8, 12, 20],
    "n_estimators": [100, 200, 500, 1000]
}

randomcv_models = [
    ("KNN", KNeighborsRegressor(), knn_params),
    ("RF", RandomForestRegressor(), rf_params)
]

print("\n--- Hiperparametre Arama Süreci Başlıyor (RandomizedSearchCV) ---")
for name, model, params in randomcv_models:
    randomcv = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=3, n_jobs=-1)
    randomcv.fit(X_train, y_train)
    print("best params for :", name, randomcv.best_params_)

# 8. En İyi Parametrelerle Yeniden Eğitim
best_models = {
    "K-Neighbors Regressor": KNeighborsRegressor(n_neighbors=2),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=500,
        min_samples_split=2,
        max_features=7,
        max_depth=None
    )
}

print("\n--- En İyi Parametrelerle Modellerin Test Edilmesi ---")
for name, model in best_models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae, model_train_rmse, model_train_r2 = calculate_model_metrics(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = calculate_model_metrics(y_test, y_test_pred)

    print(f"\nTuned Model: {name}")
    print("Evaluation for Training Set")
    print("RMSE :", model_train_rmse)
    print("Mean Absolute Error :", model_train_mae)
    print("R2 Score :", model_train_r2)
    print("-----------------------------")
    print("Evaluation for Test Set")
    print("RMSE :", model_test_rmse)
    print("Mean Absolute Error :", model_test_mae)
    print("R2 Score :", model_test_r2)
    print("-----------------------------")