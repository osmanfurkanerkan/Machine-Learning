import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("1-studyhours.csv")
df.head()

print(df.describe())

plt.scatter(df["Study Hours"], df["Exam Score"])
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")
plt.show()
print(df.columns)


X=df[["Study Hours"]]
y=df["Exam Score"]

print(type(X))
print(type(y))

#test train sprits

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
print(X_train.shape)
print(X_test.shape)


#standardize the data set
#yapma sebeimiz de balanced  feature  values  elde etmek istiyoruz. 0-1 arasında değerler elde etmek istiyoruz.  bu şekilde modelimiz daha iyi öğrenir.
#efficent learning  için  feature  scaling  yapmamız gerekiyor.  bu şekilde modelimiz daha hızlı öğrenir ve daha iyi sonuçlar verir.
#l1 ve l2 regularization  yaparken de feature scaling  yapmamız gerekiyor.  bu şekilde modelimiz daha iyi sonuçlar verir.
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)  #veri sızmasını önlemek için aslında fit kullanmadık burda. fit sadece train seti üzerinde yapılır. test seti üzerinde transform yapılır. bu şekilde veri sızmasını önleriz.
#bunun sebebi de test seti üzerinde fit yaparsak, test setindeki verilerin ortalamasını ve standart sapmasını öğreniriz. bu da modelimizin test setine aşırı uyum sağlamasına neden olur. bu da modelimizin genelleme yeteneğini azaltır. bu yüzden test seti üzerinde fit yapmıyoruz. sadece transform yapıyoruz. bu şekilde modelimiz test setine aşırı uyum sağlamaz ve genelleme yeteneği artar.


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)   #bu 2 sini kullanarak bir eğitim yapıyoruz.  X_train ve y_train kullanarak modelimizi eğitiyoruz.

print("coefficent :" + str(regression.coef_))  #bu bize katsayıyı verir. bu katsayı bize 1 saat çalışmanın sınav puanına etkisini gösterir.   
print("intercept : " + str(regression.intercept_))  #bu bize sabit terimi verir. bu sabit terim bize hiç çalışmadan alınacak sınav puanını gösterir.

#y=76.91+16.17860223x 

plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,regression.predict(X_train),"r")
plt.show()

#x=20 y=?

regression.predict([[20]])   #bu yanlış bir kullanım. çünkü biz X_train'i standardize ettik. bu yüzden 20'yi de standardize etmemiz gerekiyor. 20'yi standardize etmek için önce 20'den X_train'in ortalamasını çıkarıp, sonra da standart sapmasına bölmemiz gerekiyor. bu şekilde 20'yi standardize ederiz ve modelimize verebiliriz.
#20'yi standardize etmek için önce 20'den X_train'in ortalamasını çıkarıp, sonra da standart sapmasına bölmemiz gerekiyor. bu şekilde 20'yi standardize ederiz ve modelimize verebiliriz.
scaler.transform([[20]])  #bu şekilde 20'yi standardize ederiz. bu bize 20'nin standardize edilmiş halini verir. bu değeri modelimize verebiliriz.
regression.predict(scaler.transform([[20]]))  #bu şekilde 20'yi standardize edip modelimize verebiliriz. bu bize 20 saat çalışmanın sınav puanına etkisini gösterir. bu bize 20 saat çalışmanın sınav puanına etkisini verir. bu bize 20 saat çalışmanın sınav puanına etkisini verir. bu bize 20 saat çalışmanın sınav pu


#yeni bir featur eklerken adj r2 score kullanırız. adj r2 score 1'e ne kadar yakınsa modelimiz o kadar iyi demektir. adj r2 score 0'a ne kadar yakınsa modelimiz o kadar kötü demektir. adj r2 score negatifse modelimiz çok kötü demektir. adj r2 score 0.5 ise modelimiz orta seviyede demektir. adj r2 score 0.8 ise modelimiz iyi demektir. adj r2 score 0.9 ise modelimiz çok iyi demektir. adj r2 score 1 ise modelimiz mükemmel demektir. adj r2 score 0 ise modelimiz hiç iyi değil demektir. adj r2 score -1 ise modelimiz çok kötü demektir. adj r2 score -0.5 ise modelimiz kötü demektir. adj r2 score -0.8 ise modelimiz çok kötü demektir. adj r2 score -0.9 ise modelimiz çok kötü demektir. adj r2 score -1 ise modelimiz çok kötü demektir.

y_pred_test=regression.predict(X_test)
plt.scatter(y_pred_test,y_test,color="blue")
plt.show()



from sklearn.metrics import mean_absolute_error ,mean_squared_error ,r2_score
mse =mean_squared_error(y_test,y_pred_test)
mae=mean_absolute_error(y_test,y_pred_test)
r2=r2_score(y_test,y_pred_test)
print("Mean Absolute Error : " + str(mae))
print("Mean Squared Error : " + str(mse))
print("R2 Score : " + str(r2))
 