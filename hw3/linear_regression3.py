#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from random import randint
get_ipython().run_line_magic('matplotlib', 'inline')
#read data
df = pd.read_csv('Concrete_Data.csv') 


# In[2]:


X = [df['Cement (component 1)(kg in a m^3 mixture)'].values,
     df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'].values,
     df['Fly Ash (component 3)(kg in a m^3 mixture)'].values,
     df['Water  (component 4)(kg in a m^3 mixture)'].values,
     df['Superplasticizer (component 5)(kg in a m^3 mixture)'].values,
     df['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'].values,
     df['Fine Aggregate (component 7)(kg in a m^3 mixture)'].values,
     df['Age (day)'].values]
y = df['Concrete compressive strength(MPa, megapascals) '].values
X = np.array(X)
X = X.transpose()
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

features = X_train.shape[1]
w = np.empty(features)
w.fill(0)
b=0
h=lambda x : np.matmul(w,x) + b
learn=0.000001
N=len(X_train)


# In[3]:


for i in range (1000):
    sum_b = 0
    sum_w = np.empty(features)
    sum_w.fill(0)
    for j in range (N):
        sum_b += h(X_train[j])-y_train[j]
        for k in range (features):
            sum_w[k] += (h(X_train[j])-y_train[j]) * X_train[j][k]
    sum_b, sum_w = sum_b/N, sum_w/N
    b = b - (0.1 * sum_b)
    w = w - (learn * sum_w)
    
y_pred = np.empty(len(X_test))
for i in range (len(X_test)):
    y_pred[i] = h(X_test[i])

print("Weight: {}".format(w))
print("Bias: {}".format(b))
print("r2_score: {}".format(r2_score(y_test, y_pred)))
print("mean_squared_error: {}".format(mean_squared_error(y_test, y_pred)))


# In[4]:


#only update one random weight per iteration
for i in range (1000):
    sum_b = 0
    sum_w = np.empty(features)
    sum_w.fill(0)
    for j in range (N):
        sum_b += h(X_train[j])-y_train[j]
        for k in range (features):
            sum_w[k] += (h(X_train[j])-y_train[j]) * X_train[j][k]
    sum_b, sum_w = sum_b/N, sum_w/N
    rd = randint(0,features-1)
    b = b - (0.1 * sum_b)
    w[rd] = w[rd] - (learn * sum_w[rd])
    
y_pred = np.empty(len(X_test))
for i in range (len(X_test)):
    y_pred[i] = h(X_test[i])

print("Weight: {}".format(w))
print("Bias: {}".format(b))
print("r2_score: {}".format(r2_score(y_test, y_pred)))
print("mean_squared_error: {}".format(mean_squared_error(y_test, y_pred)))


# In[ ]:




