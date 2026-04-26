# Gerekli kütüphaneleri import ediyoruz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===================== CLASSIFICATION (KNN) =====================

# Veri setini yüklüyoruz
df = pd.read_csv("12-health_risk_classification.csv")

# Veri hakkında genel bilgi
df.info()

# İstatistiksel özet
df.describe()

# İlk 5 satırı göster
df.head()

# Scatter plot (renk: risk durumu)
sns.scatterplot(x=df['blood_pressure_variation'],
                y=df['activity_level_index'],
                hue=df['high_risk_flag'])
plt.show()

# Sınıf dağılımı
df['high_risk_flag'].value_counts()

# Feature ve target ayırımı
X = df.drop('high_risk_flag', axis=1)
y = df['high_risk_flag']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Boxenplot ile veri dağılımını gözlemleme
sns.boxenplot(df)
plt.show()

# Veriler zaten scale edilmiş gibi ama pratik için scaler uyguluyoruz
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN classifier oluşturma (k=5)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto')

# Model eğitimi
classifier.fit(X_train_scaled, y_train)

# Tahmin
y_pred = classifier.predict(X_test_scaled)

# Performans metrikleri
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print("confusion matrix: \n ", confusion_matrix(y_pred, y_test))
print("accuracy score: ", accuracy_score(y_pred, y_test))
print("classification report: ", classification_report(y_pred, y_test))

# KD-Tree algoritması ile tekrar deneme (performans değişmez, hız değişebilir)
classifier = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

print("confusion matrix: \n ", confusion_matrix(y_pred, y_test))
print("accuracy score: ", accuracy_score(y_pred, y_test))
print("classification report: ", classification_report(y_pred, y_test))

# K değerini değiştirerek performans test etme (k=3)
classifier = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

print("confusion matrix: \n ", confusion_matrix(y_pred, y_test))
print("accuracy score: ", accuracy_score(y_pred, y_test))
print("classification report: ", classification_report(y_pred, y_test))

# ===================== REGRESSION (KNN) =====================

# Veri setini yükle
df_reg = pd.read_csv("12-house_energy_regression.csv")

# Veri hakkında bilgi
df_reg.info()

# İlk 5 satır
df_reg.head()

# İstatistiksel özet
df_reg.describe()

# Nem ile enerji tüketimi ilişkisi
sns.scatterplot(x=df_reg['outdoor_humidity_level'],
                y=df_reg['daily_energy_consumption_kwh'])
plt.show()

# Sıcaklık değişimi ile enerji tüketimi ilişkisi
sns.scatterplot(x=df_reg['avg_indoor_temp_change'],
                y=df_reg['daily_energy_consumption_kwh'])
plt.show()

# Korelasyon matrisi
df_reg.corr()

# Feature ve target ayırımı
X = df_reg.drop('daily_energy_consumption_kwh', axis=1)
y = df_reg['daily_energy_consumption_kwh']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=15)

# Scaling işlemi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN regressor (k=5)
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5, algorithm='auto')

# Model eğitimi
regressor.fit(X_train_scaled, y_train)

# Tahmin
y_pred = regressor.predict(X_test_scaled)

# Performans metrikleri
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("r2 score: ", r2_score(y_test, y_pred))
print("mean absolute error: ", mean_absolute_error(y_test, y_pred))
print("mean squared error: ", mean_squared_error(y_test, y_pred))

# Gerçek vs tahmin scatter plot
plt.scatter(y_test, y_pred)
plt.show()

# k=7 ile deneme
regressor = KNeighborsRegressor(n_neighbors=7, algorithm='auto')
regressor.fit(X_train_scaled, y_train)

y_pred = regressor.predict(X_test_scaled)

print("r2 score: ", r2_score(y_test, y_pred))
print("mean absolute error: ", mean_absolute_error(y_test, y_pred))
print("mean squared error: ", mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.show()

# k=25 ile deneme (çok büyük k performansı düşürür)
regressor = KNeighborsRegressor(n_neighbors=25, algorithm='auto')
regressor.fit(X_train_scaled, y_train)

y_pred = regressor.predict(X_test_scaled)

print("r2 score: ", r2_score(y_test, y_pred))
print("mean absolute error: ", mean_absolute_error(y_test, y_pred))
print("mean squared error: ", mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.show()