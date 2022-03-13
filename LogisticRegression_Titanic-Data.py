# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:14:54 2020

@author: User
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree

data_titanic = pd.read_excel(r"D:\AanshFolder\datasets\titanic-data.xlsx")
print(data_titanic.head())
print("Total:",str(len(data_titanic)))
sexwise_list = data_titanic['sex'].tolist()
print("Male:",sexwise_list.count('male'))
print("Female:",sexwise_list.count('female'))

fig,ax = plt.subplots(3,2)
sns.countplot(x="survived",data=data_titanic,ax=ax[0][0])
sns.countplot(x="survived",hue="sex",data=data_titanic,ax=ax[0][1])
sns.countplot(x="survived",hue="pclass",data=data_titanic,ax=ax[1][0])
sns.countplot(x="survived",hue="embarked",data=data_titanic,ax=ax[1][1])
fig.show()

print(data_titanic.isnull())
print(data_titanic.isnull().sum())

sns.heatmap(data_titanic.isnull(),yticklabels='false',cmap='viridis',ax=ax[2][0])

data_titanic.drop('body',axis=1,inplace=True)
data_titanic.drop('cabin',axis=1,inplace=True)
data_titanic.drop('boat',axis=1,inplace=True)
data_titanic.drop('home.dest',axis=1,inplace=True)
data_titanic.drop('age',axis=1,inplace=True)
data_titanic.drop('fare',axis=1,inplace=True)
sns.heatmap(data_titanic.isnull(),yticklabels='false',cmap='viridis',ax=ax[2][1])
print(data_titanic.isnull().sum())

sex_categorical = pd.get_dummies(data_titanic['sex'],drop_first=True)
print(sex_categorical)
embarked_categorical = pd.get_dummies(data_titanic['embarked'],drop_first=True)
print(embarked_categorical)
pclass_categorical = pd.get_dummies(data_titanic['pclass'],drop_first=True)
print(pclass_categorical)

data_titanic = pd.concat([data_titanic,sex_categorical,embarked_categorical,pclass_categorical],axis=1)
data_titanic.drop(['sex','embarked','pclass','name'],axis=1,inplace=True)
print(data_titanic.head())

y=data_titanic['survived']
x=data_titanic.drop(['survived','ticket'],axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.33,random_state=1)
X_train.fillna(X_train.mean(),inplace=True)
Y_train.fillna(Y_train.mean(),inplace=True)

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)

predictions = logmodel.predict(X_test)
print(classification_report(Y_test,predictions))
print(accuracy_score(Y_test,predictions))




model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)

prediction = model.predict(X_test)
#df = pd.DataFrame(prediction,test_Y)
#print(df)
print("Decision Tree=",metrics.accuracy_score(prediction,Y_test))









