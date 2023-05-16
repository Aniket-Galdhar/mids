#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("tested.csv")
titanic.head()


# In[4]:


titanic.dtypes


# In[5]:


titanic.isna().sum()


# In[6]:


titanic["Age"].mean()


# In[7]:


titanic["Age"].median()


# In[8]:


titanic["Age"].fillna(titanic["Age"].mean(), inplace= True)


# In[9]:


titanic.isna().sum()


# In[10]:


titanic["Cabin"].unique()


# In[11]:


titanic.drop(['Cabin','PassengerId','Name','Ticket'],axis = 1, inplace = True)


# In[12]:


titanic.head()


# In[13]:


titanic['Embarked'].unique()


# In[14]:


gender = pd.get_dummies(titanic['Sex'])


# In[15]:


gender.drop('female',axis = 1,inplace = True)


# In[16]:


gender


# In[17]:


Embarked = pd.get_dummies(titanic['Embarked'])


# In[18]:


Embarked


# In[19]:


titanic.drop(['Sex','Embarked'],axis = 1,inplace = True)


# In[20]:


titanic["Gender"] = gender
titanic = pd.concat([titanic,Embarked],axis = 1)


# In[21]:


titanic


# In[ ]:




