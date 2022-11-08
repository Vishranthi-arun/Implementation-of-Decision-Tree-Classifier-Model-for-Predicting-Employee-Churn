# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Read the data set.
3. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4. Determine training and test data set.
5. Apply decision tree Classifier and get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vishranthi A
RegisterNumber:  212221230124
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![1](https://user-images.githubusercontent.com/93427278/200617343-4b606eaf-6918-46b1-a7ef-91a49bf4e4fc.png)


![2](https://user-images.githubusercontent.com/93427278/200617442-2d8a2866-06b5-4717-952b-c75a55b73886.png)


![3](https://user-images.githubusercontent.com/93427278/200617478-f81bd031-ccdd-4d22-8295-758384740aa5.png)


![4](https://user-images.githubusercontent.com/93427278/200617600-5af99405-5076-49df-b9e7-e3ea8696d097.png)


![5](https://user-images.githubusercontent.com/93427278/200617634-e069520e-33b3-4015-8a77-4cfe9c47b53c.png)


![6](https://user-images.githubusercontent.com/93427278/200617666-daff2eab-0306-453c-94c6-4456beb9565d.png)


![7](https://user-images.githubusercontent.com/93427278/200617788-622864f2-ae41-4327-b5c9-776ebcad87b7.png)


![8](https://user-images.githubusercontent.com/93427278/200617741-207d4daa-627a-4ddd-99b0-4bdfd63d5365.png)


![9](https://user-images.githubusercontent.com/93427278/200617830-127c32a2-fb2a-48b7-87ad-391705a0f589.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
