import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("6-bank_customers.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())


X=df.drop("subcribed",axis=1)
y=df["subcribed"] 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=15)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
score=accuracy_score(y_test,y_pred)
print("Accuracy Score:", score)
print("Classification Report:\n", classification_report(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred)    )