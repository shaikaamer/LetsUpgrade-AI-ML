# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:25:42 2020

@author: PAK
"""


import pandas as pd

dataset=pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)

print(dataset.columns)

dataset1=dataset.drop(["ID","ZIP Code"],axis=1)

dataset2=dataset1.dropna()

dataset3=dataset2.drop_duplicates()

from sklearn.ensemble import RandomForestClassifier

import numpy as np

dataset3["CCAvg"]=np.round(dataset3["CCAvg"])

# Initialize the Random Forest
rf_model=RandomForestClassifier(n_estimators=1000,max_depth=2,oob_score=True)

features=[ 'Age', 'Experience', 'Income',  'Family', 'CCAvg','Education', 'Mortgage', 'Securities Account','CD Account', 'Online', 'CreditCard']

rf_model.fit(dataset3[features],dataset3["Personal Loan"])

print ("OOB Accuracy")
print (rf_model.oob_score_)

print ()
for feature,imp in zip(features,rf_model.feature_importances_):
    print (feature,imp);

# Income,CCAvg, Education - Has the Heigher Importancew with the help of Random Forest, now with these 
# Important Variables will do the Decision Tree Algorithm

from sklearn import tree

# Initialize the Model Tree
tree_model=tree.DecisionTreeClassifier(max_depth=6,max_leaf_nodes=12)

#tree_mode.fit

predictors=pd.DataFrame([dataset3["Education"],dataset3["CCAvg"],dataset3["Income"]]).T

# Here we are fitmenting the model with IV,DV
tree_model.fit(predictors,dataset3["Personal Loan"]) 

# The above tree model genrated the tree, and this tree we are opening in a 
# file which should be written in the File as follows
with open("BankLoanDtree.dot","w") as f:
    f=tree.export_graphviz(tree_model,feature_names=["Education","CCAvg","Income"], out_file=f)

# from the above file "http://www.webgraphviz.com/" got this link
# and then paste the data which is present in file
