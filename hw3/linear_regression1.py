#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#read data
df = pd.read_csv('Concrete_Data.csv') 


# In[22]:


#draw the plot for each feature 
for i in range (0,8):
    plt.scatter(df.iloc[:,[i]], df.iloc[:,[8]], c='r', s=1)
    plt.xlabel(df.columns.values[i])
    plt.ylabel(df.columns.values[8])
    print("feature",i,":target")
    plt.show()


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#linrear regression with feature0
X = df[['Cement (component 1)(kg in a m^3 mixture)']]
y = df[['Concrete compressive strength(MPa, megapascals) ']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
y_pred = regression_model.predict(X_test)

weight = regression_model.coef_[0][0]
bias = regression_model.intercept_[0]

print("Weight: {}".format(weight))
print("Bias: {}".format(bias))
print("r2_score: {}".format(r2_score(y_test, y_pred)))

plt.scatter(X_train, y_train, c='r', s=1)
plt.plot(X_test,y_pred,c='b')
plt.xlabel(df.columns.values[0])
plt.ylabel(df.columns.values[8])


# In[26]:


#linrear regression with feature0-7
X = df.iloc[:,0:8]
y = df[['Concrete compressive strength(MPa, megapascals) ']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
y_pred = regression_model.predict(X_test)

weight = regression_model.coef_[0]
bias = regression_model.intercept_[0]

print("Weight: {}".format(weight))
print("Bias: {}".format(bias))
print("r2_score: {}".format(r2_score(y_test, y_pred)))


# In[ ]:




