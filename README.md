# EXPERIMENT : 4
## SGD Regressor for Multivariate Linear Regression
### NAME: PAVITHRAN S
### REG NO: 212223240113
## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Start Step
2.Data Preparation
3.Hypothesis Definition
4.Cost Function 
5.Parameter Update Rule 
6.Iterative Training 
7.Model Evaluation 
8.End
## Program:
```
#Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
#Developed by: AVINASH T
#RegisterNumber: 212223230026 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
print(df.tail())

 
df.info()
x=data.data[:,:3]

x=df.drop(columns=['AveOccup','target'])
x.info()

y=df[['AveOccup','target']]
y.info()

x.head()

scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

print(X_train)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])


```

## Output:
![image](https://github.com/user-attachments/assets/f605181f-fcf3-401d-aef4-3c4127448257)

![image](https://github.com/user-attachments/assets/803c1e9f-d845-4b4f-904f-d89799ea9f56)

![image](https://github.com/user-attachments/assets/d52dee0d-8f14-429d-a33f-bdd6b031a2c4)

![image](https://github.com/user-attachments/assets/edcecbb4-6715-492d-b331-9fdd9ede4dfa)

![image](https://github.com/user-attachments/assets/1d0e672b-c612-433d-a4c7-7347b868ae26)

![image](https://github.com/user-attachments/assets/51fb3ad4-3e96-4856-84af-60d295ab3179)

![image](https://github.com/user-attachments/assets/553b9ed6-affd-466a-9617-7749bb9d4fce)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
