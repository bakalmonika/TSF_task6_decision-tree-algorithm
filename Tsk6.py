#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 
# ## INTERN NAME: MONIKA BAKAL
# 
# ### TASK 6: Prediction using Decision Tree
# 

# In[100]:


# IMPORTING REQUIRED LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# ### IMPORTING DATASET

# In[101]:


# HERE I HAVE SAVED THE DATASET , YOU CAN DIRECTLY IMPORT BY GIVEN URL

dataset=pd.read_csv(r'C:\Users\HP\Desktop\tsf_decisiontree.csv')
X = dataset.iloc[:, -5:-1].values
y = dataset.iloc[:, -1].values


# In[102]:


print(X)


# In[103]:


print(y)


# In[104]:


dataset.head()


# In[105]:


dataset.describe()


# ### Splitting the dataset into the Training set and Test set

# In[106]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)


# In[107]:


print(y_train)


# In[108]:


print(X_test)


# In[109]:


print(y_test)


# ### Feature Scaling

# In[110]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


# ### Training the Decision Tree Classification model on the Training set

# In[111]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# ### Predicting a new result

# In[113]:


print(classifier.predict(sc.transform([[4.6,2,0.23,0.34]])))


# ### Predicting the Test set results

# In[117]:



y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ### Making the Confusion Matrix

# In[115]:



from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# ### Visualising decision tree

# In[118]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['iris-setosa', 'iris-versicolor', 'iris-virginica']


fig, axes = plt.subplots(nrows = 0,ncols =0,figsize = (20,25), )

tree.plot_tree(classifier.fit(X_train, y_train),
           feature_names = fn, 
           class_names=cn,
           filled = True);

fig.savefig('imagename.png')

