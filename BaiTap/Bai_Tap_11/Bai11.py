import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
data_link = './data/Social_Network_Ads.csv'
data = pd.read_csv(data_link)
print (data.head)

X = np.array(data[['Age', 'EstimatedSalary']])
y = np.array(data['Purchased'])
     
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

listKernel = ['linear', 'poly', 'rbf', 'sigmoid']
listC = [0.01, 0.1, 1, 10, 100, 1000]
listGamma = [1, 0.1, 0.01, 0.001, 0.0001]
result = GridSearchCV(estimator=SVC(),param_grid={'C': listC, 'kernel': listKernel, 'gamma': listGamma})

result.fit(X_train, y_train)

print("Best param: ",result.best_params_) 
print("Best estimatior: ",result.best_estimator_)

best_C = result.best_params_['C']
best_gamma = result.best_params_['gamma']
best_kernel = result.best_params_['kernel']

model = SVC(C = best_C, gamma = best_gamma, kernel = best_kernel)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



