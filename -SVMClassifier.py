import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

#email classification yapıtğımız alan
#subject_formality score - sender relationgship_score
#email_type -> 0 = personal , 1 = work email
print("\n===== EMAIL CLASSIFICATION =====")

df = pd.read_csv("9-email_classification_svm.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull.sum())
sns.scatterplot(
    x=df['subject_formality_score'],
    y=df['sender_relationship_score'],
    hue=df['email_type']
)
plt.title("Email Data")
plt.show()

X = df.drop('email_type', axis=1)
y = df['email_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Linear
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
print(svc.coef_)
print(svc.score(X_test, y_test))
y_pred = svc.predict(X_test)

print("\n--- Linear ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# kernel linear yerine rbf yaparsak ne olur?
rbf = SVC(kernel='rbf')
rbf.fit(X_train, y_train)
y_pred = rbf.predict(X_test)

print("\n--- RBF ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#veri nerdeyse aynı olduğu için fark oluşmadı



print("\n===== LOAN RISK  CSV=====")

df = pd.read_csv("9-loan_risk_svm.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull.sum())

sns.scatterplot(
    x=df['credit_score_fluctuation'],
    y=df['recent_transaction_volume'],
    hue=df['loan_risk']
)
#artık daha karışık bir veri elimize geldi
plt.title("Loan Risk Data")
plt.show()

X = df.drop('loan_risk', axis=1)
y = df['loan_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Kernelslerin değişimini görmek için for döngüsü oluşturdum.
for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n--- {kernel.upper()} ---")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# GridSearch (RBF)
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest Params:", grid.best_params_)

y_pred = grid.predict(X_test)
print("\n--- Tuned RBF ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


print("\n===== SEISMIC DATA =====")

df = pd.read_csv("9-seismic_activity_svm.csv")

sns.scatterplot(
    x=df['underground_wave_energy'],
    y=df['vibration_axis_variation'],
    hue=df['seismic_event_detected']
)
plt.title("Seismic Data")
plt.show()
#2 boyutlu çözülemeyecek bir sorun karşımıza çıktı burda 3 boyutlu düşünmemiz lazım.

# ---- Manual Feature Engineering (RBF mantığı) ----
df['energy_sq'] = df['underground_wave_energy'] ** 2 #karelerini aldık
df['vibration_sq'] = df['vibration_axis_variation'] ** 2  #karelerini aldık
df['interaction'] = df['underground_wave_energy'] * df['vibration_axis_variation']

X = df.drop('seismic_event_detected', axis=1)
y = df['seismic_event_detected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

linear = SVC(kernel='linear')
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

print("\n--- Manual Feature + Linear ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



df = pd.read_csv("9-seismic_activity_svm.csv")

X = df.drop('seismic_event_detected', axis=1)
y = df['seismic_event_detected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)


linear = SVC(kernel='linear')
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

print("\n--- Raw Linear ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


rbf = SVC(kernel='rbf')
rbf.fit(X_train, y_train)
y_pred = rbf.predict(X_test)

print("\n--- RBF ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))