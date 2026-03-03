import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("weatherHistory.csv")
print(df.head())
print(df.info()) 
print(df.isnull().sum())
print(df[df.isnull().any(axis=1)])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler





df["Precip Type"] = df["Precip Type"].fillna("unknown")

df["Formatted Date"] = pd.to_datetime(df["Formatted Date"],utc=True)

df["year"] = df["Formatted Date"].dt.year
df["month"] = df["Formatted Date"].dt.month
df["day"] = df["Formatted Date"].dt.day
df["hour"] = df["Formatted Date"].dt.hour

df.drop("Formatted Date", axis=1, inplace=True)

df.drop("Daily Summary", axis=1, inplace=True)

df = pd.get_dummies(df, columns=["Summary", "Precip Type"], drop_first=True)

df.drop("Apparent Temperature (C)", axis=1, inplace=True)

y = df["Temperature (C)"]
X = df.drop("Temperature (C)", axis=1)

print(X.columns)
print(df.info())


model=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

