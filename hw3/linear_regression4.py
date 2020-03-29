#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from decimal import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from random import randint
get_ipython().run_line_magic('matplotlib', 'inline')
#read data
df = pd.read_csv('Concrete_Data.csv') 
print(getcontext())
getcontext().prec = 50


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
X = np.array(X) #many arrays to one high dimention array
X = X.transpose()
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train_shaped = y_train.reshape((824, 1))
y_test_shaped = y_test.reshape((len(y_test), 1))

features = X_train.shape[1]
w = np.empty(165) # 8->164
w.fill(0)
h=lambda x : sum_w*x
learn = (10e-18)*4
N=len(X_train) #824 row numbers

# train: make 8 to 164 features
X_train_new = np.zeros((N, 165))
for j in range(N):
    
    tmp = np.empty(8)
    tmp = X_train[j] 
    
    c = 0
    X_train_new[j][c] = 1
    c = c+1
    for m in range(8): # 0 to 7
        for n in range(m, 8): # m to 7
            for s in range(n, 8):
                X_train_new[j][c] = tmp[m]*tmp[n]*tmp[s]
                c = c+1
    for m in range(8): # 0 to 7
        for n in range(m, 8): # m to 7
            X_train_new[j][c] = tmp[m]*tmp[n]
            c = c+1
    for m in range(8):
        X_train_new[j][c] = tmp[m]
        c=c+1
S = np.zeros(165)
for col in range(165):
        G = X_train_new[:,col]
        S[col] = mean_squared_error(G, y_train) 
N=6
top_N_index = sorted(range(len(S)), key=lambda i:S[i])[-N:]
print(top_N_index)    

# test: make 8 to 45 features
X_test_new = np.zeros((len(X_test), 165))
for i in range(len(X_test)):
    
    tmp3 = np.empty(8)
    tmp3 = X_test[i] 
    c = 0
    X_test_new[i][c] = 1
    c = c+1
    for m in range(8): # 0 to 7
        for n in range(m, 8): # m to 7
            for s in range(n, 8):
                X_test_new[i][c] = tmp3[m]*tmp3[n]*tmp3[s]
                c = c+1
    for m in range(8): # 0 to 7
        for n in range(m, 8): # m to 7
            X_test_new[i][c] = tmp3[m]*tmp3[n]
            c = c+1
    for m in range(8):
        X_test_new[i][c] = tmp3[m]
        c=c+1
#xprint(X_test_new)

#for i in range(165): 
    X_train_new[:,111] = 0
    X_test_new[:,111] = 0
    X_train_new[:,112] = 0
    X_test_new[:,112] = 0
    X_train_new[:,114] = 0
    X_test_new[:,114] = 0
    X_train_new[:,117] = 0
    X_test_new[:,117] = 0
    X_train_new[:,32] = 0
    X_test_new[:,32] = 0
    X_train_new[:,31] = 0
    X_test_new[:,31] = 0
   

sum_w = np.empty([165, 1]) 
sum_w.fill(0)


# In[3]:


#training
for i in range (100000):
    pred = 0
    grad = 0
    st=0
    for j in range (165):
        pred = np.dot(X_train_new, sum_w) 
        grad = np.dot((y_train_shaped - pred).T, X_train_new[:, j])/X_train_new.shape[0] 
        np.nan_to_num(grad)
        sum_w[j, 0] += (learn*grad)
        #print(grad, end=" ")
        st+=grad**2
        sum_w[j, 0] -= learn/np.sqrt(st+1e-6)*grad
        #print(grad/np.sqrt(st))

#testing
#for j in range (165):
    y_pred = np.dot(X_test_new, sum_w) 
 #   grad = np.dot((y_test_shaped - y_pred).T, X_test_new[:, j])/X_test_new.shape[0] 
  #  sum_w[j, 0] += (10e-22*grad)
    if i%100000 ==0:
        print("Weight: {}".format(sum_w))
        print("r2_score: {:.2f}".format(r2_score(y_test, y_pred)))

    
print("Weight: {}".format(sum_w))
print("r2_score: {:.2f}".format(r2_score(y_test, y_pred)))
#print("Bias: {}".format(b))
n =len(X_test)
r2 = r2_score(y_test, y_pred)
r2_adj =1- (1-r2)*(n-1)/(n-(165+1))
#print(r2_adj)

print("mean_squared_error: {}".format(mean_squared_error(y_test, y_pred)))


# In[ ]:




