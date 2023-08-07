#!/usr/bin/env python
# coding: utf-8

# ## Collect and Clean the data from 'diabetes_data_upload.csv'

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv("D:\data_analyst\diabetes_data_upload.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


# replace the value in data frame with 0 and 1 (no and yes)
df = df.replace("No",0)
df = df.replace("Yes",1)
df


# In[8]:


df = df.replace("Positive",1)
df = df.replace("Negative",0)

#replace male with 1 and female with 0
df = df.replace("Male",1)
df = df.replace("Female",0)
df


# In[9]:


# check for missing values

df.isnull().sum()


# In[10]:


#check the dtypes of the columns
#as long as you get int or float, its all good
df.dtypes


# In[11]:


replace = {"Gender":"ismale"}
df = df.rename(columns=replace)

# convert all columns into lower cases

df.columns = df.columns.str.lower()
df


# In[13]:


# export dataframe to csv
df.to_csv("diabetes_data_clean.csv", index=None)


# In[14]:


pd.read_csv("diabetes_data_clean.csv")


# Summary of part 1
# 1. Colect the data from UCI Repository
# 2. Replaced string to 1s and 0s
# 3. Replaced changed column name
# 4. Lowercased everything in columns
# 5. Exported the clean DataFrame to a new CSV

# ##  Statistical Analysis of the data and visualising the data using visualization mathods

# In[20]:


# import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import stats libraries
from scipy.stats import chi2_contingency
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.weightstats import ztest


# In[21]:


# read dataframe 
df = pd.read_csv("diabetes_data_clean.csv")
df


# In[22]:


# examine age with a histogram
plt.hist(df['age'])


# In[23]:


df['age'].mean()


# In[24]:


df['age'].median()


# In[28]:


# create a countplot for ismalr
sns.countplot(df['ismale'])
plt.title('ismale')
sns.despine()


# In[29]:


columns = df.columns[1:]
columns


# In[30]:


# iteratively plot countplot for each columns
for column in columns:
    sns.countplot(df[column])
    plt.title(column)
    sns.despine()
    plt.show()


# ### Questions:
# 1. Is obesity related to diabetes status?
# 2. Is age related to diabetes status?

# In[33]:


obesity_diabetes_crosstab = pd.crosstab(df['class'],df['obesity'])
obesity_diabetes_crosstab


# In[34]:


chi2_contingency(obesity_diabetes_crosstab)


# In[35]:


ismale_diabetes_crosstab = pd.crosstab(df['class'],df['ismale'])
ismale_diabetes_crosstab


# In[36]:


chi2_contingency(ismale_diabetes_crosstab)


# Suggestions:
# 1. polyunia vs class
# 2.  ismale vs polyunia

# In[37]:


# is there a relationship between age and diabetic satus?
sns.boxplot(df['class'],df['age'])


# In[38]:


no_diabetes = df[df['class']==0]
no_diabetes['age'].median()


# In[39]:


diabetes = df[df['class']==1]
diabetes['age'].median()


# In[41]:


# qqplot
qqplot(df['age'],fit=True,line="s")
plt.show()


# In[42]:


# conduct z test of difference
ztest(diabetes['age'],no_diabetes['age'])


# In[43]:


# get a correlation plot 
df.corr()


# In[44]:


sns.heatmap(df.corr())


# Summary of part 2
# 1. Looked at the single columns(univariate analysia)
# 2. Looked at the relationship between twio columns (bivariste analysis)
# 3. Conducted a statistical test of difference between ages of non-diabetic and diabetic patients
# 3. Plotted a correlation heatmap

# ##  Training a Machine Learning Model to predict Diabeties Patient based on symptoms

# In[47]:


# training a machine learning model to predict whether a patient have a diabetes or not based on symptoms
# import machine learning libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report


# In[53]:


# prepare our independent and dependent varibles
df = pd.read_csv("diabetes_data_upload.csv")
df
X = df.drop('class',axis=1)
Y = df['class']


# In[54]:


# class is missing
X


# In[55]:


# class is there
Y


# In[56]:


# split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y)


# In[58]:


# beign our model training
# start with DummyClassifier to establish baseline
dummy = DummyClassifier()
dummy.fit(X_train,Y_train)
dummy_pred = dummy.predict(X_test)


# In[59]:


confusion_matrix(Y_test,dummy_pred)


# In[60]:


# use a classification report
print(classification_report(Y_test,dummy_pred))


# In[65]:


# start with logistic regression
logr = LogisticRegression(max_iter=10000)
logr.fit(X_train,Y_train)
logr_pred = logr.predict(X_test)


# In[66]:


confusion_matrix(Y_test,logr_pred)


# In[67]:


print(classification_report(Y_test,logr_pred))


# In[69]:


# try decision tree
tree = DecisionTreeClassifier()
tree.fit(X_train,Y_train)
tree_pred = tree.predict(X_test)


# In[71]:


confusion_matrix(Y_test,tree_pred)


# In[72]:


print(classification_report(Y_test,tree_pred))


# In[73]:


# try RandomForest
forest = RandomForestClassifier()
forest.fit(X_train,Y_train)
forest_pred = forest.predict(X_test)


# In[75]:


confusion_matrix(Y_test,forest_pred)


# In[76]:


print(classification_report(Y_test,forest_pred))


# In[77]:


# getting model features importance
forest.feature_importances_


# In[78]:


X.columns


# In[80]:


pd.DataFrame({'feature': X.columns,
             'importance':forest.feature_importances_}).sort_values('importance',ascending=False)


# Summary of part 3:
# 1. Trained a baseline model
# 2. Trained three different models - logistic regression, decision tree and random forest
# 3. Identified the important features in the best performing model
# 

# In[ ]:




