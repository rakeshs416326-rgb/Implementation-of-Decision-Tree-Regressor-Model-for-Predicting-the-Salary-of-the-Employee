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
```
<img width="411" height="254" alt="image" src="https://github.com/user-attachments/assets/b4b0d8be-ae15-4cfb-acad-a83eb5fffa70" />
```
```
<img width="450" height="238" alt="image" src="https://github.com/user-attachments/assets/6be350c7-6600-4c16-a6f8-f4f94691c0a2" />
```
```
<img width="204" height="227" alt="image" src="https://github.com/user-attachments/assets/1a72fcac-1c2a-4767-aff5-976d22916fbd" />
```
```
<img width="342" height="256" alt="image" src="https://github.com/user-attachments/assets/d35f932d-6793-4f47-b277-f271fc14ae57" />
```
```
<img width="196" height="58" alt="image" src="https://github.com/user-attachments/assets/4312f0ad-4868-4185-ba2b-d6dcbeed7bb6" />
```
```
<img width="259" height="53" alt="image" src="https://github.com/user-attachments/assets/a41e2133-ed94-4e8d-a170-82288da74a82" />
```
```
<img width="244" height="44" alt="image" src="https://github.com/user-attachments/assets/11826a88-5c70-46f1-a449-2715425c5474" />
```

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
