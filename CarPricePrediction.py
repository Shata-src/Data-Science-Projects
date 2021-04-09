# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 02:56:28 2020

@author: Shatadru
"""

import pandas as pd  #for dataframe
import numpy as np   #for numerical operation
import seaborn as sns  #for data visualization

# Seting Dimensions for Plot
sns.set(rc={'figure.figsize': (8,5)})



## Reading CSV file
cars_data=pd.read_csv("cars_sampled.csv")

cars=cars_data.copy() # Creating deep copy

cars.info() # Structure of dataset


## Set the display of float upto 3 decimal points
cars.describe() #display of Int valued columns
pd.set_option("display.float_format", lambda x: '%.3f' % x)
cars.describe()

pd.set_option('display.max_columns',100) #set a display of columns upto max 100 columns
cars.describe()

## Dropping unwanted columns
col=['name','dateCrawled','postalCode','lastSeen','dateCreated']
cars=cars.drop(columns=col, axis=1)
cars.info()



### Remove duplicates
cars.drop_duplicates(keep='first',inplace=True)

## No. of missing values in each column
cars.isnull().sum()


## Analysis on variable- YearOfREgistration

yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
# Working range 1950-2018
sns.regplot(x='yearOfRegistration', y='price',scatter=True, fit_reg=False,data=cars)


## Analysis on variable- price

price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])

sum(cars['price']<1)        #1447
sum(cars['price']<10)       #1628
sum(cars['price']<100)      #1780
sum(cars['price']<1000)     #11129
sum(cars['price']>1000)     #38073
sum(cars['price']>10000)    #8103
sum(cars['price']>100000)   #58
sum(cars['price']>150000)   #34
sum(cars['price']>1000000)  #5
#working range 100-150000



## Analysis on variable- powerPS

power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS', y='price',scatter=True, fit_reg=False,data=cars)

sum(cars['powerPS']<10)    #5621
sum(cars['powerPS']>500)   #115
sum(cars['powerPS']>1000)  #40
#working range 10-500


### Cleaning data as per working range set

cars= cars[(cars.yearOfRegistration<=2018)\
           & (cars.yearOfRegistration>=1950)\
           & (cars.price>=100)\
           & (cars.price<=150000)\
           & (cars.powerPS>=10)\
           & (cars.powerPS<=500)]
# 6700 records dropped



## Further simplification - Combine yearOfRegistration and monthOfRegistration

cars['monthOfRegistration']/=12 #converting months to years


# Creating new variable Age
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()



##Dropping the columns yearOfRegistration and monthOfRegistration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'], axis=1)


## Visualizing Parameters

#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#Price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

# Age Vs price
sns.regplot(x='Age', y='price',scatter=True, fit_reg=False,data=cars)
#Car price is disproportional to Age, however exceptions are there

# powerPS Vs price
sns.regplot(x='powerPS', y='price',scatter=True, fit_reg=False,data=cars)
# Car price is proportional to PowerPS



## Analysis on variable- seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],'count', normalize=True)
sns.countplot(x='seller', data=cars)
#only one seller is Commercial, insignificant


## Analysis on variable- offerType
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'],'count', normalize=True)
sns.countplot(x='offerType', data=cars)
#All cars have offer, insignificant feature


## Analysis on variable- abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],'count', normalize=True)
sns.countplot(x='abtest', data=cars)
#Almost equally distributed
sns.boxplot(x='abtest',y='price', data=cars)



## Analysis on variable- vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],'count', normalize=True)
sns.countplot(x='vehicleType', data=cars)
sns.boxplot(x='vehicleType',y='price', data=cars)


## Analysis on variable- gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],'count', normalize=True)
sns.countplot(x='gearbox', data=cars)
sns.boxplot(x='gearbox',y='price', data=cars)


## Analysis on variable- model
cars['model'].value_counts()
pd.crosstab(cars['model'],'count', normalize=True)
sns.countplot(x='model', data=cars)
sns.boxplot(x='model',y='price', data=cars)


## Analysis on variable- kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],'count', normalize=True)
sns.countplot(x='kilometer', data=cars)
sns.boxplot(x='kilometer',y='price', data=cars)
sns.regplot(x='kilometer', y='price',scatter=True, fit_reg=False,data=cars)
#Price going down as kilometer travelled increases, except for the first 5000


## Analysis on variable- fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count', normalize=True)
sns.countplot(x='fuelType', data=cars)
sns.boxplot(x='fuelType',y='price', data=cars)
#Fuel type affects price


## Analysis on variable- brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count', normalize=True)
sns.countplot(x='brand', data=cars)
sns.boxplot(x='brand',y='price', data=cars)


## Analysis on variable- notRepairedDamage
#Yes=damage not repaired
#No= Damage repaired
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count', normalize=True)
sns.countplot(x='notRepairedDamage', data=cars)
sns.boxplot(x='notRepairedDamage',y='price', data=cars)
#cars for which damage was not repaired has been priced lower



## removing insignificant variables
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()



## Correlation

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


## Omitting missing values
cars_omit=cars.dropna(axis=0)

#Convert categorical variables to dummy variables
cars_omit=pd.get_dummies(cars_omit, drop_first=True)


##Importing necessary libraries for learnign algorithm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


x1=cars_omit.drop(['price'],axis=1, inplace=False)
y1=cars_omit['price']


# Plotting the variable price
prices= pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()

y1=np.log(y1)


#Splitting the data into train and testsets
train_x, test_x, train_y, test_y = train_test_split(x1,y1,test_size=0.3, random_state=3) #random_state to choose the random set
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


## Baseline model for Omitted data

# Finding the mean of the test data
base_pred=np.mean(test_y)
print(base_pred)


# Repeating same value till length of test data
base_pred=np.repeat(base_pred, len(test_y))

#Find the RMSE
base_rmse=np.sqrt(mean_squared_error(test_y, base_pred))
print(base_rmse)


# =============================================================================
# Linear Rgression Model
# =============================================================================

#Setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#Model
model_lin1=lgr.fit(train_x, train_y)


#predicting model on test set
cars_pred_lin1=lgr.predict(test_x)

#Computing MSE and RMSE
lin_mse1= mean_squared_error (test_y, cars_pred_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)


# R squared value
r2_lin_test1=model_lin1.score(test_x, test_y)
r2_lin_train1=model_lin1.score(train_x, train_y)
print(r2_lin_test1, r2_lin_train1)

## Regression diagnostics - Residual plot analysis
residuals1 = test_y - cars_pred_lin1
sns.regplot(x=cars_pred_lin1, y=residuals1,scatter=True, fit_reg=False)
residuals1.describe()
# difference should be as near to zero as possible