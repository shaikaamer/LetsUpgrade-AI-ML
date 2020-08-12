# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:56:21 2020

@author: PAK



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("Linear Regression.xlsx")

data.head()
 
data.describe()
 
data.hist()
 
data.hist()
 
data.corr()
 
sns.pairplot(data)
 
sns.pairplot(data)
 
data.info()
 
sns.scatterplot(data['price'],data['sqft_living'])
 
sns.scatterplot(data['price'],data['bedrooms'])
 
sns.scatterplot(data['price'],data['bathrooms'])
 
sns.scatterplot(data['price'],data['floors'])
 
data.boxplot()
 
sns.boxplot(data["sqft_living"])
 
sns.boxplot(data["price"])
 
x = data.drop(['price','sqft_living','bedrooms','floors'], axis=1)
y = data.drop(['sqft_living','bedrooms','bathrooms','floors'], axis=1)

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y,random_state = 42,test_size=0.25)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_x, train_y)
 
plt.scatter(train_x, train_y, color = 'red')
plt.plot(train_x, lr.predict(train_x), color = 'green')
plt.show()

ypred = lr.predict(test_x)

print(ypred)

plt.scatter(test_x, test_y, color = 'red')
plt.plot(test_x, lr.predict(test_x), color = 'green')
plt.show()

from sklearn.metrics import r2_score

print(r2_score(test_y, ypred))
 

plt.scatter(test_x, test_y, color = 'red')
plt.plot(test_x, lr.predict(test_x), color = 'green')
plt.show()

unseen_pred=lr.predict(np.array([[3234]]))

unseen_pred
 
x.head()
 
y.head()
 
unseen_pred=lr.predict(np.array([[323486]]))

print(unseen_pred)
 
#Price vs Sq.Ft.ipynb

x = data.drop(['price','bedrooms','bathrooms','floors'], axis=1)
y = data.drop(['sqft_living','bedrooms','bathrooms','floors'], axis=1)

x.head()
 
y.head()
 
train_x,test_x,train_y,test_y = train_test_split(x,y,random_state = 24,test_size=0.35)

lr_sqft=LinearRegression()

lr_sqft.fit(train_x,train_y)
 
plt.scatter(train_x, train_y, color = 'red')
plt.plot(train_x, lr.predict(train_x), color = 'green')
plt.xlabel('Sqrft')
plt.ylabel('price')
plt.show()

plt.scatter(train_x, train_y, color = 'red')
plt.plot(train_x, lr.predict(train_x), color = 'green')
plt.xlabel('price')
plt.ylabel('Sqrft')
plt.show()

 
ypred=lr_sqft.predict(test_x)

r2_score(test_y, ypred)
 
plt.scatter(test_x, test_y, color = 'red')
plt.plot(test_x, lr.predict(test_x), color = 'green')
plt.xlabel('price')
plt.ylabel('Sqrft')
plt.show()

# to Check to move accuracy

train_x,test_x,train_y,test_y = train_test_split(x,y,random_state = 24,test_size=0.20)

lr_sqft1=LinearRegression()

 
lr_sqft1.fit(train_x,train_y)
 
plt.scatter(train_x, train_y, color = 'red')
plt.plot(train_x, lr.predict(train_x), color = 'green')
plt.xlabel('price')
plt.ylabel('Sqrft')
plt.show()

ypred=lr_sqft1.predict(test_x)
print(ypred)
plt.scatter(train_x, train_y, color = 'red')
plt.plot(train_x, lr_sqft1.predict(train_x), color = 'green')
plt.xlabel('price')
plt.ylabel('Sqrft')
plt.show()

plt.scatter(train_x, train_y, color = 'red')
plt.plot(train_x, lr_sqft.predict(train_x), color = 'green')
plt.xlabel('price')
plt.ylabel('Sqrft')
plt.show()

ypred=lr_sqft1.predict(test_x)
print(ypred)
print(r2_score(test_y, ypred))
 
 
 # Price with BedRoom



x = data.drop(['price','sqft_living','bathrooms','floors'], axis=1)
y = data.drop(['sqft_living','bedrooms','bathrooms','floors'], axis=1)

train_x,test_x,train_y,test_y = train_test_split(x,y,random_state = 42,test_size=0.25)

lr_dedroom=LinearRegression()

lr_dedroom.fit(train_x,train_y)
 
plt.scatter(train_x, train_y, color = 'blue')
plt.plot(train_x, lr_dedroom.predict(train_x), color = 'black')
plt.xlabel('price')
plt.ylabel('bedrooms')
plt.show()

ypred = lr_dedroom.predict(test_x)

r2_score(test_y,ypred)
print(ypred)
print(r2_score(test_y, ypred))

plt.scatter(test_x, test_y, color = 'red')
plt.plot(test_x, lr_dedroom.predict(test_x), color = 'green')
plt.xlabel('price')
plt.ylabel('bedrooms')
plt.show()

# Proce vs Floor

x = data.drop(['price','sqft_living','bedrooms','bathrooms'], axis=1)
y = data.drop(['sqft_living','bedrooms','bathrooms','floors'], axis=1)

train_x,test_x,train_y,test_y = train_test_split(x,y,random_state = 42,test_size=0.25)

lr_floor=LinearRegression()

lr_floor.fit(train_x,train_y)
 
plt.scatter(train_x, train_y, color = 'blue')
plt.plot(train_x, lr_floor.predict(train_x), color = 'black')
plt.xlabel('price')
plt.ylabel('lr_floor')
plt.show()

floor_predict=lr_floor.predict(test_x)

 
r2_score(test_y,floor_predict)
print(ypred)
print(r2_score(test_y, floor_predict))
 

 
plt.scatter(test_x, test_y, color = 'red')
plt.plot(test_x, lr_floor.predict(test_x), color = 'green')
plt.xlabel('price')
plt.ylabel('lr_floor')
plt.show()
