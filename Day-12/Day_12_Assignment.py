# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 20:41:03 2020

@author: PAK
"""

import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

dataset=pd.read_csv("general_data.csv")
dataset["Attrition"].replace(to_replace=("No","Yes"),value=(0,1),inplace=True)
dataset["Gender"].replace(to_replace=("Male","Female"),value=(0,1),inplace=True)
dataset["BusinessTravel"].replace(to_replace=("Non-Travel", "Travel_Rarely", "Travel_Frequently"),value=(0,1,2),inplace=True)
print (dataset.head())
left = dataset[dataset['Attrition']==1]
print (left.head())
working = dataset[dataset['Attrition']==0]
print (working.head())

# mannwhitneyu Test
# H0: There is no significant difference between the DistanceFromHome from Home w.r.t Attrition
# H1: There is significant difference between the DistanceFromHome from Home w.r.t Attrition

stats,p=mannwhitneyu(working.DistanceFromHome, left.DistanceFromHome)
print ("There is no significant difference between the DistanceFromHome from Home w.r.t Attrition",stats,p)


# 2-sample separate T-Test
# H0: There is no significant difference between the MonthlyIncome  w.r.t Attrition
# H1: There is significant difference between the MonthlyIncome  w.r.t Attrition
stats,p=ttest_ind(left.MonthlyIncome,working.MonthlyIncome) # 0.03842748490605113
print ("There is significant difference between the MonthlyIncome  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the BusinessTravel  w.r.t Attrition
# H1: There is significant difference between the BusinessTravel  w.r.t Attrition
stats,p=ttest_ind(left.BusinessTravel,working.BusinessTravel) # 2.5366396530230266e-17
print ("There is no significant difference between the BusinessTravel  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the Distance from Home w.r.t Attrition
# H1: There is significant difference between the Distance from Home w.r.t Attrition
stats,p=ttest_ind(left.DistanceFromHome,working.DistanceFromHome) # 0.518286042805572
print ("There is no significant difference between the Distance  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the Education  w.r.t Attrition
# H1: There is significant difference between the Education  w.r.t Attrition
stats,p=ttest_ind(left.Education,working.Education) # 0.3157293177122392
print ("There is no significant difference between the Education  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the JobLevel  w.r.t Attrition
# H1: There is significant difference between the JobLevel  w.r.t Attrition
stats,p=ttest_ind(left.JobLevel,working.JobLevel) # 0.4945171727187496
print ("There is no significant difference between the JobLevel  w.r.t Attrition",stats,p)


# H0: There is no significant difference between the PercentSalaryHike  w.r.t Attrition
# H1: There is significant difference between the PercentSalaryHike  w.r.t Attrition
stats,p=ttest_ind(left.PercentSalaryHike,working.PercentSalaryHike) # 0.03074338643339195
print ("There is significant difference between the PercentSalaryHike  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the TrainingTimesLastYear  w.r.t Attrition
# H1: There is significant difference between the TrainingTimesLastYear  w.r.t Attrition
stats,p=ttest_ind(left.TrainingTimesLastYear,working.TrainingTimesLastYear) # 0.0010247061915374478
print ("There is significant difference between the TrainingTimesLastYear  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the YearsAtCompany  w.r.t Attrition
# H1: There is significant difference between the YearsAtCompany  w.r.t Attrition
stats,p=ttest_ind(left.YearsAtCompany,working.YearsAtCompany) # 3.163883122491456e-19
print ("There is no significant difference between the YearsAtCompany  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the YearsSinceLastPromotion  w.r.t Attrition
# H1: There is significant difference between the YearsSinceLastPromotion  w.r.t Attrition
stats,p=ttest_ind(left.YearsSinceLastPromotion,working.YearsSinceLastPromotion) # 0.028330336189428353
print ("There is significant difference between the YearsSinceLastPromotion  w.r.t Attrition",stats,p)

# H0: There is no significant difference between the YearsWithCurrManager  w.r.t Attrition
# H1: There is significant difference between the YearsWithCurrManager  w.r.t Attrition
stats,p=ttest_ind(left.YearsWithCurrManager,working.YearsWithCurrManager) # 1.7339322652918153e-25
print ("There is no significant difference between the YearsWithCurrManager  w.r.t Attrition",stats,p)

