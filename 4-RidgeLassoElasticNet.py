import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("4-Algerian_forest_fires_dataset.csv")
print(df.head())
print(df.info()) #bu bize veri setimizin genel bilgilerini verir. bu şekilde veri setimizin hangi sütunlardan oluştuğunu, her bir sütunun veri tipini ve kaç tane eksik değer olduğunu görebiliriz.
print(df.isnull().sum()) #bu bize her bir sütunda kaç tane eksik değer olduğunu gösterir. bu şekilde eksik değerleri görebiliriz ve bunları nasıl dolduracağımıza karar verebiliriz.
print(df[df.isnull().any(axis=1)]) #bu bize eksik değer içeren satırları gösterir. bu şekilde hangi satırların eksik değer içerdiğini görebiliriz ve bu satırları nasıl işleyeceğimize karar verebiliriz.

df.drop(122,inplace=True) #bu bize 122. satırı siler. bu şekilde eksik değer içeren satırları silebiliriz.
df.loc[:123,"Region"]=0
df.loc[123:,"Region"]=1

df=df.dropna().reset_index(drop=True) #bu bize eksik değer içeren satırları siler ve indexleri yeniden düzenler. bu şekilde eksik değer içeren satırları silebiliriz ve indexleri yeniden düzenleyebiliriz.
print(df.shape) #bu bize veri setimizin kaç satır ve kaç sütundan oluştuğunu gösterir. bu şekilde veri setimizin boyutunu görebiliriz.
print(df.info()) #bu bize veri setimizin genel bilgilerini verir. bu şekilde veri setimizin hangi sütunlardan oluştuğunu, her bir sütunun veri tipini ve kaç tane eksik değer olduğunu görebiliriz.
df.columns=df.columns.str.strip() #bu bize sütun isimlerindeki boşlukları temizler. bu şekilde sütun isimlerindeki boşlukları temizleyebiliriz ve sütun isimlerini daha okunabilir hale getirebiliriz.

df = df[df["day"] != "day"].copy() #tekrar eden başlık satırını temizler.
df[["day","month","year","Temperature","RH","Ws"]]=df[["day","month","year","Temperature","RH","Ws"]].astype(int) #bu bize belirtilen sütunları int veri tipine dönüştürür. bu şekilde belirtilen sütunları float veri tipine dönüştürebiliriz ve bu sütunlarda matematiksel işlemler yapabiliriz.
print(df.info()) #bu bize veri setimizin genel bilgilerini verir. bu şekilde veri setimizin hangi sütunlardan oluştuğunu, her bir sütunun veri tipini ve kaç tane eksik değer olduğunu görebiliriz.
df["Classes"]= np.where(df["Classes"].str.contains("not fire"),0,1) #bu bize "Classes" sütunundaki değerleri "not fire" içerenleri 0, diğerlerini 1 yapar. bu şekilde "Classes" sütunundaki değerleri ikili hale getirebiliriz ve bu sütunu modelleme için kullanabiliriz.
print(df["Classes"].value_counts()) #bu bize "Classes" sütunundaki her bir değerin kaç kez tekrar ettiğini gösterir. bu şekilde "Classes" sütunundaki değerlerin dağılımını görebiliriz ve bu sütunu modelleme için kullanabiliriz.
df.drop(["day","month","year"],axis=1,inplace=True) #bu bize "day", "month" ve "year" sütunlarını siler. bu şekilde "day", "month" ve "year" sütunlarını modelleme için kullanmayabiliriz ve bu sütunları silebiliriz.


#dependent ve independent değişkenlerimizi tanımlayalım
X=df.drop("FWI",axis=1) #bu bize "FWI" sütunu hariç tüm sütunları verir. bu şekilde bağımsız değişkenlerimizi tanımlayabiliriz.
y=df["FWI"] #bu bize "FWI" sütununu verir. bu şekilde bağımlı değişkenimizi tanımlayabiliriz.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15 ) #bu bize bağımsız değişkenlerimizi ve bağımlı değişkenimizi eğitim ve test setlerine böler. bu şekilde modelimizi eğitmek ve test etmek için veri setimizi bölebiliriz.

print(X_train.shape) #bu bize eğitim setimizde kaç satır ve kaç sütun olduğunu gösterir. bu şekilde
print(X_train.corr()) #bu bize eğitim setimizdeki bağımsız değişkenler arasındaki korelasyonu gösterir. bu şekilde bağımsız değişkenler arasındaki ilişkileri görebiliriz ve bu ilişkileri modelleme için kullanabiliriz.


#redundacy ,multiconnitary ve overfitting

def correlation_for_dropping(df,threshold):
    corr=df.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
           print(j)
 

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train) #bu bize eğitim setimizdeki bağımsız değişkenleri standartlaştırır. bu şekilde bağımsız değişkenlerimizi standartlaştırarak modelleme için daha uygun hale getirebiliriz.
X_test_scaled=scaler.transform(X_test) #bu bize test setimizdeki bağımsız değişkenleri standartlaştırır. bu şekilde bağımsız değişkenlerimizi standartlaştırarak modelleme için daha uygun hale getirebiliriz.


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
lr=LinearRegression()
lr.fit(X_train_scaled,y_train) #bu bize eğitim setimizdeki bağımsız değişkenler ve bağımlı değişken arasındaki ilişkiyi öğrenir. bu şekilde modelimizi eğitebiliriz.
mae=mean_absolute_error(y_test,lr.predict(X_test_scaled)) #bu bize test setimizdeki bağımlı değişken ve modelimizin tahminleri arasındaki ortalama mutlak hatayı verir. bu şekilde modelimizin performansını değerlendirebiliriz.
mse=mean_squared_error(y_test,lr.predict(X_test_scaled)) #bu bize test
score=r2_score(y_test,lr.predict(X_test_scaled)) #bu bize test setimizdeki bağımlı değişken ve modelimizin tahminleri arasındaki R^2 skorunu verir. bu şekilde modelimizin performansını değerlendirebiliriz.
print("MAE:",mae)
print("MSE:",mse)
print("R^2 Score:",score)