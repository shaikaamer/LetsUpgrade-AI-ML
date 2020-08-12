# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:57:10 2020

@author: PAK
"""

import pandas as pd

dataset=pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)

print (dataset.head())

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

print (dataset.info())

import numpy as np

dataset["CCAvg"]=np.round(dataset["CCAvg"])


dataset1=dataset.drop(["ID","ZIP Code"],axis=1)

#dataset["CCAvg"]=le.fit_transform(dataset["CCAvg"].round())

dataset2=dataset1.dropna()

dataset3 = dataset2.drop_duplicates()

print (dataset3.head())

print ("dataset3")
print (dataset3.columns)

print (dataset3.info())
Y=dataset3["Personal Loan"]

X=dataset3[[ "Age", "Experience", "Income" ,"Family", "CCAvg","Education","Mortgage",
           "Securities Account","CD Account", "Online", "CreditCard"]]

import statsmodels.api as sm

X1=sm.add_constant(X)


Bank_loan=sm.Logit(Y,X1)

result=Bank_loan.fit()
print (result.summary())

'''                           Logit Regression Results                           
==============================================================================
Dep. Variable:          Personal Loan   No. Observations:                 4984
Model:                          Logit   Df Residuals:                     4972
Method:                           MLE   Df Model:                           11
Date:                Wed, 12 Aug 2020   Pseudo R-squ.:                  0.5930
Time:                        18:01:37   Log-Likelihood:                -642.82
converged:                       True   LL-Null:                       -1579.4
Covariance Type:            nonrobust   LLR p-value:                     0.000
======================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------
const                -12.1417      1.645     -7.380      0.000     -15.366      -8.917
Age                   -0.0537      0.061     -0.874      0.382      -0.174       0.067
Experience             0.0634      0.061      1.039      0.299      -0.056       0.183
Income                 0.0548      0.003     20.959      0.000       0.050       0.060
Family                 0.6946      0.074      9.357      0.000       0.549       0.840
CCAvg                  0.1078      0.038      2.802      0.005       0.032       0.183
Education              1.7278      0.115     15.076      0.000       1.503       1.952
Mortgage               0.0005      0.001      0.816      0.415      -0.001       0.002
Securities Account    -0.9327      0.285     -3.267      0.001      -1.492      -0.373
CD Account             3.8184      0.324     11.801      0.000       3.184       4.453
Online                -0.6699      0.157     -4.267      0.000      -0.978      -0.362
CreditCard            -1.1170      0.205     -5.451      0.000      -1.519      -0.715
======================================================================================'''