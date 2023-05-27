#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# In[21]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#in order to ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[22]:


df=pd.read_csv('C:/Users/lakav/Downloads/Iris.csv')


# In[23]:


df=pd.DataFrame(df)
df.head()


# In[24]:


df.drop(["Id"],axis=1,inplace=True)
# removing unwanted columns --i.e,Id
df.head()


# In[25]:


df.shape
# shows the no.of columns and no.of rows 
# dataset contains 150 rows and 5 columns


# In[26]:


df.info()


# In[27]:


df.describe()


# In[28]:


print(df['Species'].unique())
# number of target classes


# In[29]:


df.isnull().sum()
# no null values were present in dataset


# VISUALISATION

# In[40]:


plt.scatter(df['PetalWidthCm'],df['PetalLengthCm'])
plt.show()


# In[41]:


plt.scatter(df['SepalWidthCm'],df['SepalLengthCm'])
plt.show()


# In[42]:


sns.violinplot(data=df[['PetalWidthCm','PetalLengthCm','SepalWidthCm','SepalLengthCm']])
ata=plt.show()


# In[53]:


sns.boxplot(data=df[['PetalWidthCm','PetalLengthCm','SepalWidthCm','SepalLengthCm']])
plt.show()


# In[55]:


sns.pairplot(df,hue="Species")
plt.show()


# SPLITTING THE DATASET INTO TRAINING & TESTING DATASETS

# In[57]:


x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['Species']


# In[32]:


x


# In[33]:


y


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=42)


# In[35]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


IMPORTING MODELS


# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# LOGISTIC REGRESSION

# In[37]:


# creating
model=LogisticRegression(solver='liblinear',max_iter=1000)
#Training the model
model.fit(x_train,y_train)
# make predictions on the test data
y_prd=model.predict(x_test)
from sklearn.metrics import accuracy_score
#performance
accuracy=accuracy_score(y_test,y_prd)
print("Accuracy:",accuracy)


# (SVM)SUPPORT VECTOR MACHINES

# In[38]:


# Creating
clf=SVC()
#Training the model
clf.fit(x_train,y_train)
#make predictions on the data 
y_prd=clf.predict(x_test)
# Performace
accuracy=accuracy_score(y_test,y_prd)
print("Accuracy:",accuracy)


# DECISION TREE CLASSIFIER

# In[39]:


# Creating
classifier = DecisionTreeClassifier()
# Training the model
classifier.fit(x_train,y_train)
# Make predictions on the testing data
y_prediction=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
# performance
accuracy=accuracy_score(y_test,y_prediction)
print('Accuracy:', accuracy)


# --> COMPLETED <--

# In[ ]:




