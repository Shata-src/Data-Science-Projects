# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:01:35 2020

@author: Shatadru
"""

import pandas as pd  #for dataframe
import numpy as np   #for numerical operation
import seaborn as sns  #for data visualization
from sklearn.model_selection import train_test_split #to partition data
from sklearn.linear_model import LogisticRegression  #for logistic regression
from sklearn.metrics import accuracy_score, confusion_matrix #for performance metrics
import matplotlib.pyplot as plt

data_income=pd.read_csv("income.csv", sep='\t')

data=data_income.copy()

print(data.info()) # detailed information

data.isnull().sum() #Sum of null variables

summary_num=data.describe() #summary of numerical data type
print(summary_num)

summary_cate=data.describe(include='O') #summary of object datatype
print(summary_cate)

data["JobType"].value_counts() #Count of each job type
data["occupation"].value_counts() #Count of each occupation type

print(np.unique(data['JobType'])) #list of unique job type
print(np.unique(data['occupation'])) #List of unique occupation type

# There exists ' ?' instead of NaN(blank)
#Read the data again, including '?' values as NaN

data=pd.read_csv('income.csv', sep='\t', na_values=[" ?"])
data.isnull().sum()

missing=data[data.isnull().any(axis=1)]

#Missing values in JobType          1809
#Missing values in occupation       1816
#7 rows in JobType have value= Never-worked, rest all are nan
#All missing rows of occupation are nan

data2=data.dropna(axis=0) #drp the missing valued rows

correlation=data2.corr() #relationship b/w independent variables, no two variables are found to be correlated

data2.columns #Extract column names


gender=pd.crosstab(index=data2["gender"], columns='count', normalize=True)
print(gender)

gender_salstat= pd.crosstab(index=data['gender'], columns=data2['SalStat'], margins=True, normalize='index')
print(gender_salstat)

SalStat=sns.countplot(data2['SalStat'])

plt.hist(data2['age'], bins=10, edgecolor='white')

sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('SalStat')['age'].median()

#plt.bar(['JobType','SalStat'],color=['red','blue'])
#plt.show()

sns.countplot(y='SalStat',data=data2, hue_order='Automatic')


#  Logistic Regression

## reindexing Salary status names to 0,1

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names
columns_list=list(new_data.columns)
print(columns_list)


# Separating the input names from data
features=list(set(columns_list)- set(['SalStat']))
print(features)


# Storing the output values t Y
y=new_data['SalStat'].values
print(y)

#Storing the values from input features
x=new_data[features].values
print(x)


#Splitting the data into train and testsets
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state=0) #random_state to choose the random set


#Make an instatnce of model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_


## Prediction from test data
prediction=logistic.predict(test_x)
print(prediction)


# Confusion matrix
confusion_matrix1=confusion_matrix(test_y, prediction)
print(confusion_matrix1)

accuracy=accuracy_score(test_y,prediction)
print(accuracy)


#print the misclassifications
print("Misclassified samples: ",(test_y!=prediction).sum())

#####################################################################################
## Removing insignificant variables
data2=data.dropna(axis=0) #drop the missing valued rows
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names
columns_list=list(new_data.columns)
print(columns_list)


# Separating the input names from data
features=list(set(columns_list)- set(['SalStat']))
print(features)


# Storing the output values t Y
y=new_data['SalStat'].values
print(y)

#Storing the values from input features
x=new_data[features].values
print(x)


#Splitting the data into train and testsets
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state=0) #random_state to choose the random set


#Make an instatnce of model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_


## Prediction from test data
prediction=logistic.predict(test_x)
print(prediction)


# Confusion matrix
confusion_matrix2=confusion_matrix(test_y, prediction)
print(confusion_matrix2)

accuracy=accuracy_score(test_y,prediction)
print(accuracy)

#print the misclassifications
print("Misclassified samples: ",(test_y!=prediction).sum())




# =============================================================================
# KNN Model
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

KNN_Classifier = KNeighborsClassifier(n_neighbors=5)
KNN_Classifier.fit(train_x, train_y)

KNNPrediction=KNN_Classifier.predict(test_x)

KNNconfusion_matrix=confusion_matrix(test_y, KNNPrediction)
print(KNNconfusion_matrix)


KNNaccuracy=accuracy_score(test_y,KNNPrediction)
print(KNNaccuracy)


print("Misclassified samples: ",(test_y!=KNNPrediction).sum())


## Calculating error for K values between 1-20

Misclassified_sample=[]
for i in range (1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)

plt.bar(range(1,21), Misclassified_sample)    
plt.show()