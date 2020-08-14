# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:46:34 2020

@author: PAK
"""

import pandas as pd

dataset=pd.read_csv("train.csv")

print ("Dataset")
print (dataset.head())

print (dataset.columns)

dataset1=dataset.drop(["PassengerId","Name","Ticket", "Cabin"],axis=1)

print ("Dataset1")
print (dataset1.head())
print (dataset1.columns)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

dataset1["Sex"]=le.fit_transform(dataset1["Sex"])
dataset1["Embarked"]=le.fit_transform(dataset1["Embarked"])

print ("After Preprocessing:")

print (dataset1.head())

dataset2=dataset1.dropna()
dataset3=dataset2.drop_duplicates()

print ("Final Dataset- dataset3")
print (dataset3.head())

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

X=dataset3.drop(["Pclass"],axis=1)
Y=dataset3["Pclass"]

train_x,test_x,train_y,test_y=train_test_split(X,Y,train_size=0.35,random_state=9)

from sklearn import svm

clf=svm.SVC(gamma=0.1,C=100)

clf.fit(train_x,train_y)

y_predict=clf.predict(test_x)


print (accuracy_score(test_y,y_predict,normalize=True))

def svm_model(dataset3):
    columns_dataset3=dataset3.drop(["Age","Fare"],axis=1)
    score=[]
    for i in columns_dataset3:
        
        x=dataset3.drop([i],axis=1)
        y=dataset3[i]
        
        train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.25,random_state=42)
        
        clf=svm.SVC(gamma=0.1,C=100)

        clf.fit(train_x,train_y)

        y_predict=clf.predict(test_x)
        print (f"--{i}")
        print("=========================================================================")
        print(f"Accuracy score is {accuracy_score(test_y,clf.predict(test_x), normalize=True)}")
        print("=========================================================================")
        print(f"Confusion matrix: \n{confusion_matrix(clf.predict(test_x), test_y)}")
        print("=========================================================================")
        #print (accuracy_score(test_y,y_predict,normalize=True))
        score.append(accuracy_score(test_y,clf.predict(test_x), normalize=True))
    import matplotlib.pyplot as plt
    plt.plot(range(0,len(score)),score) 
    pass

svm_model(dataset3)