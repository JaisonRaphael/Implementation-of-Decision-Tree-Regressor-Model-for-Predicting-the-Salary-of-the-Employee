# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jaison Raphael V
RegisterNumber:  212221230038
*/
```
~~~
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
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
~~~

## Output:
Data Head:

![1](https://user-images.githubusercontent.com/94165957/174469880-fc44a1a4-6914-42eb-af20-52b893824f4d.png)

Data Info:

![2](https://user-images.githubusercontent.com/94165957/174469889-a86bb2b1-3cff-40c5-bf0e-aa1a04a05296.png)

Data Isnull:

![3](https://user-images.githubusercontent.com/94165957/174469893-60d3f660-b096-4afc-96db-66789836f731.png)

Data Head:

![4](https://user-images.githubusercontent.com/94165957/174469903-d869d65a-81fa-4935-9fed-4835868e82f8.png)

dt.fit:

![5](https://user-images.githubusercontent.com/94165957/174469916-10466bae-590a-4564-ac9f-b4649102c3f6.png)

MSE:

![6](https://user-images.githubusercontent.com/94165957/174469930-83f31c1f-5acc-4c44-b04a-be12ef48c4f7.png)

RE2:

![7](https://user-images.githubusercontent.com/94165957/174469938-4196a1ba-c80b-497a-a5c3-cc34b30b4d1b.png)

Predicted Value:

![8](https://user-images.githubusercontent.com/94165957/174469945-3ea9a5ef-2061-4396-a78e-79fdb18ead87.png)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
