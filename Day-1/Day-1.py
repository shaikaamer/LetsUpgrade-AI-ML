# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:48:24 2020

@author: PAK
"""


import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat# mATLAB fILE rEADING 

# Import Pyod Packages & the Methods

from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD

from pyod.models.iforest import IForest
from pyod.models.feature_bagging import FeatureBagging

# Import Metrics Packages - to Evaluate the performance of those MOdels

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores # Is the Measure to Identify the Performance Model

from sklearn.metrics import roc_auc_score

from time import time

# Define Data File and read X and Y

mat_file_list=[]# copy all the file names

# Define Data File and Read X, and y
# All the File names in to a list which we can Iterate with this file and can load the file and deploy all the algorithms

mat_file_list =["arrhythmia.mat","cardio.mat","glass.mat","ionosphere.mat","letter.mat",
  "lympho.mat","mnist.mat", "musk.mat","optdigits.mat","pendigits.mat",
  "pima.mat","satellite.mat","satimage-2.mat","shuttle.mat","vertebral.mat",
  "vowels.mat","wbc.mat"]


# To Load the .mat File
data=loadmat("data/cardio.mat")# give the file path
print ("Data:Pritning")
print (data)
print ("Length of data:")
print (len(data))# This will display the total number of Keys Count
print ("data:Keys")
print (data.keys())# will print all the keys
print ("data:values")
print (data.values())

# To know the type  of the File
print ("To know the type of x")
print (type(data["X"]))
print ("To know the type of y")
print (type(data["y"]))

# Input Feature Shape in Mat file Format

# To know the shape of the File
print ("To know the shape of the File Data - X")
print ( data["X"].shape)
print ("To know the shape of the File for Data - y")
print ( data["y"].shape)


df_columns = ['Data','#Samples','# Dimensions','Outlier Perc',
              'ABOD','CBLOF','FB','HBOS','IForest','KNN','LOF','MCD'
              ,'OCSVM','PCA']# we are going to create one DataFrame
#df_columns=pd.DataFrame(columns=df_columns)# this will the set of Columns
print (df_columns)

# ROC Performance Evolution 
# ROC - Region of Caracteristics
print ("ROC Performance Evolution Table:")
roc_df=pd.DataFrame(columns=df_columns)
print (roc_df)

# precision_n_score - Performace evolution table
print ("Performace evolution table:")
prn_df=pd.DataFrame(columns=df_columns)
print (prn_df)
table_df=pd.DataFrame(columns=df_columns)


# Time Dataframe
print ("Time Dataframe:")
time_df=pd.DataFrame(columns=df_columns)
print (time_df.columns)

# Exploring all the Mat File
print ("Exploring all the Mat File")

random_state=np.random.RandomState(42)

for mat_file in mat_file_list:
    print ("\n ..Processing",mat_file)
    mat=loadmat(os.path.join("data",mat_file))
    
    x=mat["X"]
    y=mat["y"].ravel() # ravel is used to convert the 2D into 1D
    
    outlier_fraction=np.count_nonzero(y)/len(y)
    outlier_percentage=round(outlier_fraction*100, ndigits=4)
    
    # Construct Containers for saving Results
    
    roc_list=[mat_file[:-4],x.shape[0],x.shape[1],outlier_percentage]
    prn_list=[mat_file[:-4],x.shape[0],x.shape[1],outlier_percentage]
    time_list=[mat_file[:-4],x.shape[0],x.shape[1],outlier_percentage]
    
    # 60% Data is for Training, and 40% Data is for Testing
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=random_state)
    
    # Standardizing Data for Processing- also called as Normalizing the Data
    
    x_train_norm,x_test_norm=standardizer(x_train,x_test)
    
    # Classifiers is Dictionary which will be having all the Pyod Algorithm Test Values
    
    classifiers={
        "Angle-Based Outlier Detection (ABOD)":ABOD(contamination=outlier_fraction),
        "Cluster-based Local Outlier Factor":CBLOF(contamination=outlier_fraction,check_estimator=False, random_state=random_state),
        "Feature Bagging":FeatureBagging(contamination=outlier_fraction,random_state=random_state),
        "Histogram-base Outlier Detection (HBOS)": HBOS(contamination=outlier_fraction),
        "Isolation Forest": IForest(contamination=outlier_fraction,random_state=random_state),
        "K Nearest Neighbours (KNN)":KNN(contamination=outlier_fraction),
        "Local Outlier Factor (LOF)":LOF(contamination=outlier_fraction),
        "Minimum Coveriance Dtermination (MCD)":MCD(contamination=outlier_fraction,random_state=random_state),
        "One-class SVM(OCSVM)":OCSVM(contamination=outlier_fraction),
        "Principal Component Analysis (PCA)":PCA(contamination=outlier_fraction,random_state=random_state)
        }
    
    
    for clf_name,clf in classifiers.items():
        t0=time()
        clf.fit(x_train_norm)
        test_score=clf.decision_function(x_test_norm)
        t1=time()
        duration=round(t1-t0,ndigits=4)
        time_list.append(duration)
        roc=round(roc_auc_score(y_test,test_score),ndigits=4)
        prn=round(precision_n_scores(y_test,test_score),ndigits=4)
        
        print ("{clf_name} ROC:{roc},precision @ rank n:{prn} execution time:{duration}s".
               format(clf_name=clf_name,roc=roc,prn=prn,duration=duration))
        
        roc_list.append(roc)
        prn_list.append(prn)   
        
    # df_columns=pd.DataFrame(columns=df_columns)
    # roc_df=pd.DataFrame(columns=df_columns)
   # print ("From First")
    #print (df_columns)
    #print (time_list)
    
    temp_df = pd.DataFrame(time_list).transpose()
    
    temp_df.columns = df_columns

   # print (temp_df.columns)
    time_df = pd.concat([time_df, temp_df], axis=0)
    #print (time_df.columns)
    #print ("Time,roc,prn")
    #print (time_df)

    temp_df = pd.DataFrame(roc_list).transpose() 
    temp_df.columns = df_columns 
    roc_df = pd.concat([roc_df, temp_df], axis=0)
   # print (roc_df)

    temp_df = pd.DataFrame(prn_list).transpose() 
    temp_df.columns = df_columns 
    prn_df = pd.concat([prn_df, temp_df], axis=0)
   # print (prn_df)
    
print ("Final-Results")
roc_df.to_csv("roc_df.csv",index=False)
prn_df.to_csv("prn_df.csv",index=False)
time_df.to_csv("time_df.csv",index=False)
print (roc_df)    
print (prn_df)
print (time_df)