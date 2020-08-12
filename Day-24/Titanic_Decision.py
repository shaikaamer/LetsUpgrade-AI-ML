# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:41:31 2020

@author: PAK
"""

import pandas as pd

dataset=pd.read_csv("train.csv")

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

print (dataset.head())

dataset["Sex"]=le.fit_transform(dataset["Sex"])

print(dataset.columns)


dataset1=dataset.drop(['PassengerId', 'Name', 'Ticket',    'Cabin'],axis=1)

'''dataset1=pd.DataFrame(dataset[[  'Survived', 'Pclass',  'Sex', 'Age',  
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]).T'''


feature=[   'Pclass',   'Sex', 'Age',  
       'Parch']

dataset2=dataset1.dropna()

dataset3=dataset2.drop_duplicates()

print (dataset3.columns)

from sklearn.ensemble import RandomForestClassifier

rf_Titanic=RandomForestClassifier(n_estimators=1000,max_depth=2,oob_score=True)

rf_Titanic.fit(dataset3[feature], dataset3["Survived"])

print ("oob Score")
print (rf_Titanic.oob_score_)

features=[   'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
for feature,imp in zip(features,rf_Titanic.feature_importances_):
    print (feature,imp)

from sklearn import  tree

Titatic_tree=tree.DecisionTreeClassifier(max_depth=8,max_leaf_nodes=16)

predictors=pd.DataFrame([dataset3["Pclass"],dataset3["Sex"],dataset3["Fare"]]).T

Titatic_tree.fit(predictors,dataset3["Survived"])


print (Titatic_tree.score(predictors,dataset3["Survived"]))

Train_Titanic=pd.read_csv("train.csv")

Train_Titanic_predictors=pd.DataFrame([Train_Titanic["Pclass"],Train_Titanic["Sex"],Train_Titanic["Age"]]).T

Train_Titanic["Sex"]=le.fit_transform(Train_Titanic["Sex"])

predictors=pd.DataFrame([Train_Titanic["Pclass"],Train_Titanic["Sex"],dataset3["Fare"]]).T

test_x = Train_Titanic[['Sex','Age','Fare']]

y_pres=Titatic_tree.predict(test_x)

#print (y_pres)

Train_Titanic["PassengerId"]=y_pres

Train_Titanic.to_csv("Train_OutPut.csv")

#print (Train_Titanic_predictors.score(Train_Titanic[feature],Train_Titanic["Survived"]))

