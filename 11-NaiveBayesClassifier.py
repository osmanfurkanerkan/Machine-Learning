# Gerekli kütüphaneleri import ediyoruz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini okuma (aynı klasörde olmalı)
df = pd.read_csv("11-iris.csv")

# İlk 5 veriyi görüntüleme
print(df.head())

# Veri hakkında genel bilgi
print(df.info())

# Species sütunundaki sınıf dağılımı
print(df["Species"].value_counts())

# Sayısal sütunların istatistiksel özeti
print(df.describe())

# Tüm değişkenlerin pairplot grafiği
sns.pairplot(df)
plt.show()

# Sütun isimlerini kontrol etme
print(df.columns)

# Sepal uzunluk-genişlik grafiği (renkler türlere göre)
sns.scatterplot(x=df["SepalLengthCm"], y=df["SepalWidthCm"], hue=df["Species"])
plt.show()

# Petal uzunluk-genişlik grafiği
sns.scatterplot(x=df["PetalLengthCm"], y=df["PetalWidthCm"], hue=df["Species"])
plt.show()

# Id sütununu kaldırıyoruz (gereksiz)
df = df.drop("Id", axis=1)

# Kategorik veriyi sayısal veriye çeviriyoruz (Label Encoding)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])

# İlk ve son verileri kontrol edelim
print(df.head())
print(df.tail())

# Encode sonrası sınıf dağılımı
print(df["Species"].value_counts())

# Bağımsız (X) ve bağımlı (y) değişkenleri ayırıyoruz
X = df.drop("Species", axis=1)
y = df["Species"]

# Eğitim ve test verisine ayırma (%75 train - %25 test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Verileri ölçeklendirme (Standardizasyon)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gaussian Naive Bayes modeli oluşturma ve eğitme
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# Test verisi ile tahmin yapma
y_pred = gnb.predict(X_test_scaled)

# Model performansını ölçme
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("confusion matrix: \n", confusion_matrix(y_pred, y_test))
print("accuracy score: ", accuracy_score(y_pred, y_test))
print("classification report: \n", classification_report(y_pred, y_test))




from sklearn.linear_model import LogisticRegression

# Modeli oluşturma ve eğitme
# Not: Çok sınıflı (multiclass) veri setlerinde varsayılan ayarlar genellikle iyi çalışır.
log_model = LogisticRegression(random_state=15)
log_model.fit(X_train_scaled, y_train)

# Test verisi ile tahmin yapma
y_pred_log = log_model.predict(X_test_scaled)

# Lojistik Regresyon performansını ölçme

print("--- LOJİSTİK REGRESYON MODELİ SONUÇLARI ---")

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_log))
print("\nAccuracy Score: ", accuracy_score(y_test, y_pred_log))
print("\nClassification Report: \n", classification_report(y_test, y_pred_log))

from sklearn.svm import SVC
svc_model = SVC(random_state=15)
svc_model.fit(X_train_scaled, y_train)

# Test verisi ile tahmin yapma
y_pred_svc = svc_model.predict(X_test_scaled)

# SVC performansını ölçme

print("--- SVC MODELİ SONUÇLARI ---")

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_svc))
print("\nAccuracy Score: ", accuracy_score(y_test, y_pred_svc))
print("\nClassification Report: \n", classification_report(y_test, y_pred_svc))

