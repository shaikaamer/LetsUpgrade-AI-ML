# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:33:39 2020

@author: PAK
"""

import pandas as pd

dataset=pd.read_csv("general_data.csv")

print (dataset.columns)

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

dataset["Attrition"]=le.fit_transform(dataset["Attrition"])
dataset["BusinessTravel"]=le.fit_transform(dataset["BusinessTravel"])
dataset["Department"]=le.fit_transform(dataset["Department"])
dataset["EducationField"]=le.fit_transform(dataset["EducationField"])
dataset["Gender"]=le.fit_transform(dataset["Gender"])
dataset["MaritalStatus"]=le.fit_transform(dataset["MaritalStatus"])
dataset["JobRole"]=le.fit_transform(dataset["JobRole"])

dataset1=dataset.drop(["EmployeeCount","EmployeeID","Over18","StandardHours"],axis=1)

print ("After Droping few Columns from the main Dataframe \n")
print(dataset1.columns)

dataset2=dataset1.dropna()

dataset3=dataset2.drop_duplicates()

from sklearn.ensemble import RandomForestClassifier

rf_model=RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=True)

features=['Age',  'BusinessTravel', 'Department', 'DistanceFromHome',
       'Education', 'EducationField', 'Gender', 'JobLevel', 'JobRole',
       'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

rf_model.fit(dataset3[features],dataset3["Attrition"])


print ("Oob Score")
print (rf_model.oob_score_)

print()

for feature,imp in zip(features,rf_model.feature_importances_):
    print (feature,imp)
    
from sklearn import tree

tree_model=tree.DecisionTreeClassifier(max_depth=6,max_leaf_nodes=12)


predictors=pd.DataFrame([dataset3["Age"],dataset3["MonthlyIncome"],dataset3["TotalWorkingYears"]]).T

tree_model.fit(predictors,dataset3["Attrition"])

with open("AttritionDTree.dot","w") as f:
    f=tree.export_graphviz(tree_model, feature_names=["Age","MonthlyIncome","TotalWorkingYears"], out_file=f)