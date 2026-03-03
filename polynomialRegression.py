import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df=pd.read_csv("3-customersatisfaction.csv")

print(df.head())

df.drop("Unnamed: 0",axis=1,inplace=True)
print(df.head())

plt.scatter(df["Customer Satisfaction"], df["Incentive"])
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")

plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(x="Customer Satisfaction", y="Incentive", data=df)
plt.show()

#dependent ve independent değişkenlerimizi tanımlayalım
X=df[["Customer Satisfaction"]]
y=df["Incentive"] 

#veri setimizi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=15)
#verilerimizi ölçeklendirelim
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#polynomial regression modelimizi oluşturalım

regression=LinearRegression()
regression.fit(X_train, y_train)

#prediction
y_pred =    regression.predict(X_test)

score =r2_score(y_test, y_pred)
print(score)



poly =PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regression.fit(X_poly, y_train)
y_pred = regression.predict(X_test_poly)
