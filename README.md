# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
```
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: rakesh s
RegisterNumber:  212225240114
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## SALARY DATA
<img width="384" height="263" alt="image" src="https://github.com/user-attachments/assets/77ae2274-cfbb-4fb4-9dbf-48e896eeda2a" />

## ACTUAL vs PREDICTED
<img width="452" height="238" alt="image" src="https://github.com/user-attachments/assets/1f2e2d97-ac96-4626-b1f8-1253ad15fa97" />

## ACCURACY
<img width="389" height="66" alt="image" src="https://github.com/user-attachments/assets/7c762a18-a3ab-430a-a09a-6e2420a96fdd" />

## DECISION TREE
<img width="885" height="593" alt="image" src="https://github.com/user-attachments/assets/8eea9b73-c588-465b-a4c9-9234cc629449" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
