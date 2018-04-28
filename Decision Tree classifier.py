
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[2]:


import os
os.chdir("c://users/sujith/desktop")
iris=pd.read_csv("iris.csv")
iris.head()


# In[3]:


print(iris.dtypes)


# In[4]:


iris.shape


# In[9]:


feature_cols=['sepal_length','petal_length','sepal_width','petal_width']
x=iris[feature_cols]
y=iris.species
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[10]:


classifier=DecisionTreeClassifier(max_depth=5)
classifier.fit(x_train,y_train)


# In[11]:


predictions=classifier.predict(x_test)


# In[12]:


from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(y_test,predictions)


# In[13]:


conf_matrix


# In[14]:


print(metrics.accuracy_score(y_test,predictions))

