import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df=pd.read_csv("2-multiplegradesdataset.csv")
print(df.head())

df.isnull().sum()  #bu bize her bir sütunda kaç tane eksik değer olduğunu gösterir. bu şekilde eksik değerleri görebiliriz ve bunları nasıl dolduracağımıza karar verebiliriz.

sns.pairplot(df)  #bu bize her bir sütunun birbirleriyle olan ilişkisini gösterir. bu şekilde hangi sütunların birbirleriyle ilişkili olduğunu görebiliriz ve bu ilişkileri modelimize dahil edebiliriz.
plt.show()

df.corr()  #bu bize her bir sütunun birbirleriyle olan korelasyonunu gösterir. bu şekilde hangi sütunların birbirleriyle yüksek korelasyona sahip olduğunu görebiliriz ve bu sütunları modelimize dahil edebiliriz.



X=df[["Study Hours","Sleep Hours","Attendance Rate","Social Media Hours"]]
y=df["Exam Score"]


#x=df.iloc[:,:-1]  #bu da aynı şekilde X'i oluşturur
#y=df.iloc[:,-1]  #bu da aynı şekilde y'yi oluşturur


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=15)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)

print(regression.coef_)  #bu bize her bir özelliğin katsayısını verir. bu katsayılar bize her bir özelliğin sınav puanına etkisini gösterir. örneğin, Study Hours'ın katsayısı 10 ise, bu bize 1 saat çalışmanın sınav puanına 10 puanlık bir etkisi olduğunu gösterir.
print(regression.intercept_)  #bu bize sabit terimi verir. bu sabit terim bize hiç çalışmadan alınacak sınav puanını gösterir. örneğin, intercept 50 ise, bu bize hiç çalışmadan alınacak sınav puanının 50 olduğunu gösterir.
print(regression.predict(scaler.transform([[5,8,90,2]])))  #bu bize 20 saat çalışmanın, 8 saat uyumanın, %90 katılımın ve 2 saat sosyal medya kullanımının sınav puanına etkisini gösterir. bu şekilde modelimizi kullanarak yeni verilerle tahminler yapabiliriz.


