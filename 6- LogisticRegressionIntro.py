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

#hyperparameter tuning
model=LogisticRegression()
penalty=['l1','l2',"elasticnet"]
solver=["liblinear","saga","newton-cg","lbfgs","newton-cholesky","sag"]
c_values=[100,10,1,0.1,0.01]

params=dict(penalty=penalty,solver=solver,C=c_values)
from sklearn.model_selection import GridSearchCV,StratifiedKFold
cv=StratifiedKFold()
grid= GridSearchCV(estimator=model,param_grid=params,cv=cv,scoring="accuracy",n_jobs=-1)

grid.fit(X_train,y_train)
print("Best Hyperparameters:", grid.best_params_)
y_pred_grid=grid.predict(X_test)

score=accuracy_score(y_pred,y_test)
print("Accuracy Score after Hyperparameter Tuning:", score)
print("Confusion Matrix after Hyperparameter Tuning:\n",confusion_matrix(y_pred,y_test))


#random search cv
from sklearn.model_selection import RandomizedSearchCV
model=LogisticRegression()
randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,cv=cv,scoring="accuracy",n_jobs=-1,n_iter=10,scoring="accuracy")
randomcv.fit(X_train,y_train)


y_pred=randomcv.predict(X_test)
score=accuracy_score(y_pred,y_test)
print("Best Hyperparameters from Random Search:", randomcv.best_params_)
print("Best Score from Random Search:", randomcv.best_score_)
