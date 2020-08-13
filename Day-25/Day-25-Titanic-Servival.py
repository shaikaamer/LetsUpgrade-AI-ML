# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:57:04 2020

@author: PAK
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("train.csv")

print (dataset.info())
print (dataset.columns)
print (dataset.describe())
print (dataset.head())

dataset1=dataset.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)

print (dataset1.columns)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

dataset1["Sex"]=le.fit_transform(dataset1["Sex"])
dataset1["Embarked"]=le.fit_transform(dataset1["Embarked"])

dataset2=dataset1.dropna()
dataset3=dataset2.drop_duplicates()
Column_Names=[]
Column_Names=dataset1.columns


print (dataset3.head())
print (type(Column_Names))
print (Column_Names[0])
print (len(Column_Names))

k=0
def test_train_test_split_Funtion(dataset3,Column_Names):
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import  BernoulliNB
    from sklearn.metrics import accuracy_score,confusion_matrix
    Column_Names=dataset1.drop(["Age","Fare"],axis=1)
    result_list=[]
    j=0
    '''x=dataset3.drop([Column_Names],axis=1)
    y=dataset3[Column_Names]
    #print (x.columns)
   # print (y.head())
        #
        
        #print (y)
    train_x,test_x,train_y,test_y=train_test_split(x, y,test_size=0.2,random_state=5)
    model=BernoulliNB()
    model.fit(train_x, train_y)
    ypred=model.predict(test_x)
    print (model.score(x,y))'''

    for i in Column_Names:
        print (i)
        x=dataset3.drop([i],axis=1)
        y=dataset3[i]
        print (x.columns)
        train_x,test_x,train_y,test_y=train_test_split(x, y,test_size=0.2,random_state=5)
        model=BernoulliNB()
        model.fit(train_x, train_y)
        ypred=model.predict(test_x)
        print("==================================================================")
        score=accuracy_score(test_y, ypred)
        print (f"Score {score}")
        print (f"{i} model.score(x,y) is {model.score(x,y)} " )
        print(f"Accuracy score is {accuracy_score(test_y, ypred)}")
        print("==================================================================")
        print(f"Confusion Matrix: \n{confusion_matrix(test_y, ypred)}")
        print("==================================================================")
        print (j)
        #print (result_list[j])
        result_list.append(score)
        j+=1
    k=j-1
    print (k,j)
    plt.plot(range(0,j), result_list)
    return result_list
    pass

def with_train_data(dataset3):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import accuracy_score,confusion_matrix
    test=pd.read_csv("test.csv")
    print ("Train Data Head Function")
    print (test.head())
    from sklearn.naive_bayes import  BernoulliNB
    from sklearn.metrics import accuracy_score,confusion_matrix
    test1=test.drop(["PassengerId","Name","Ticket" ],axis=1)
    test1["Sex"]=le.fit_transform(test1["Sex"])
    test1["Embarked"]=le.fit_transform(test1["Embarked"])
    test1.dropna()
    test1.drop_duplicates()
    print (dataset3.columns)
    train_x=dataset3.drop(["Survived"],axis=1)
    train_y=dataset3["Survived"]
    test_model=BernoulliNB()
    test_model.fit(train_x,train_y)
    print (test_model.score(train_x,train_y))
    print (test.columns)
    y_predit=test_model.predict(test1)
    Column_Names=dataset3.drop(["Age","Fare"],axis=1)
    '''for i in Column_Names:
        print (i)
        x=dataset3.drop([i],axis=1)
        y=dataset3[i]
        print (x.columns)
        #train_x,test_x,train_y,test_y=train_test_split(x, y,test_size=0.2,random_state=5)
        model=BernoulliNB()
        model.fit(x, y)
        ypred=model.predict(test1)
        print (ypred)
        print("==================================================================")
        #score=accuracy_score(x, ypred)
        #print (f"Score {score}")
        print (f"{i} model.score(x,y) is {model.score(x,y)} " )
        #print(f"Accuracy score is {accuracy_score(test_y, ypred)}")
        print("==================================================================")
        print(f"Confusion Matrix: \n{confusion_matrix(test1, ypred)}")
        print("==================================================================")
        print (j)
        #print (result_list[j])
        result_list.append(score)
        j+=1'''
    return
    #pass
#Column_Names=dataset1.drop(["Age","Fare"],axis=1)

#train_split_result=

result_list=test_train_test_split_Funtion(dataset3,Column_Names)
#plt.plot(range(0,5), result_list)
#test_train_test_split_Funtion(dataset3,Column_Names)

print ("with_train_data")
with_train_data(dataset3)

