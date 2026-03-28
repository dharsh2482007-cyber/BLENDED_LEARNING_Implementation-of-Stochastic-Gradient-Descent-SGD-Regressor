# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset, then preprocess the data by removing unnecessary columns and converting categorical variables into numerical form.


2.Normalize the feature values using StandardScaler and split the dataset into training and testing sets.

3.Train the SGD Regressor model using the training data.


4.Predict values using the test data and evaluate the model using metrics such as MSE, R², and MAE.


## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)
X=data.drop('price',axis=1)
y=data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1,1))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

sgd_model.fit(X_train,y_train)
y_pred=sgd_model.predict(X_test)

print('Name: Priyadharshini P ')
print('Reg. No: 212225220076')

print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")

print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.grid(True)
plt.show()

````

## Output:

<img width="624" height="544" alt="Screenshot 2026-03-28 211603" src="https://github.com/user-attachments/assets/700fafce-c341-409f-baae-59d7ad0e1ebb" />
<img width="420" height="598" alt="Screenshot 2026-03-28 211616" src="https://github.com/user-attachments/assets/db854fb4-60d0-4a15-9967-e6089bbec492" />
<img width="617" height="199" alt="Screenshot 2026-03-28 211630" src="https://github.com/user-attachments/assets/7bc5577a-583a-48a9-ba73-5c5277234d1b" />
<img width="627" height="225" alt="Screenshot 2026-03-28 211642" src="https://github.com/user-attachments/assets/79c056f2-5f86-4ee3-bcc9-bcdeaa3654b0" />
<img width="594" height="429" alt="Screenshot 2026-03-28 211651" src="https://github.com/user-attachments/assets/808906f8-76df-4dea-980a-e3c7ff8591a9" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
