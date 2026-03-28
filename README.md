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

print('Name: KRITHIKAA P ')
print('Reg. No: 212225040193')

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

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
