# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:25:17 2020

@author: PAK
"""

import pandas as pd

dataset=pd.read_csv("train.csv")

print (dataset.head())
print (dataset.info())
print (dataset.describe())
print (dataset.columns)

dataset1=dataset.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)

print (dataset1.columns)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

dataset1["Sex"]=le.fit_transform(dataset["Sex"])
dataset1["Embarked"]=le.fit_transform(dataset["Embarked"])

print (dataset1.head())

dataset2 = dataset1.dropna()
dataset3=dataset2.drop_duplicates()

print (dataset3.columns)

from sklearn import neighbors
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


x=dataset3.drop(["Survived"],axis=1)
y=dataset3["Survived"]

x1=dataset3.drop(["Pclass"],axis=1)
y1=dataset3["Pclass"]

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=9)

train_x1,test_x1,train_y1,test_y1=train_test_split(x1,y1,test_size=0.3,random_state=9)

#knn =  neighbors.KNeighborsRegressor()

print ("With Survived")

score = []
for i in range(1,268):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    score.append(accuracy_score(knn.predict(test_x),test_y, normalize=True))

import matplotlib.pyplot as plt
plt.plot(range(1,268),score)

print ("With Pclass")

score = []
for i in range(1,268):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x1,train_y1)
    score.append(accuracy_score(knn.predict(test_x1),test_y1, normalize=True))

import matplotlib.pyplot as plt
plt.plot(range(1,268),score)