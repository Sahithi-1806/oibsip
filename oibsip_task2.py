#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[58]:


df=pd.read_csv("C:/Users/lakshmi sahithi/Downloads/sales oasis.csv")


# In[59]:


df.head()


# In[60]:


df.shape


# In[61]:


df.columns.values.tolist()


# In[62]:


df.info()


# In[63]:


df.describe()


# In[64]:


df.isnull().sum()


# In[65]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


fig,axs=plt.subplots(3,figsize=(5,5))
plt1=sns.boxplot(df['TV'],ax=axs[0])
plt2=sns.boxplot(df['Newspaper'],ax=axs[1])
plt3=sns.boxplot(df['Radio'],ax=axs[2])
plt.tight_layout()


# In[67]:


sns.distplot(df['Newspaper'])


# In[12]:


iqr = df.Newspaper.quantile(0.75) - df.Newspaper.quantile(0.25)

lower_bridge = df["Newspaper"].quantile(0.25) - (iqr*1.5)
upper_bridge = df["Newspaper"].quantile(0.75) + (iqr*1.5)
print(lower_bridge)
print(upper_bridge)

# In[68]:


lower_bridge = df["Newspaper"].quantile(0.25) - (iqr*1.5)
upper_bridge = df["Newspaper"].quantile(0.75) + (iqr*1.5)
print(lower_bridge)
print(upper_bridge)


# In[69]:


data=df.copy()


# In[70]:


data.loc[data['Newspaper']>=93,'Newspaper']=93


# In[71]:


sns.boxplot(data['Newspaper'])


# In[72]:


sns.boxplot(data['Sales'])


# In[73]:


sns.pairplot(data,x_vars=['TV','Newspaper','Radio'],
            y_vars='Sales',height=4,aspect=1,kind='scatter')
plt.show()


# In[74]:


sns.heatmap(data.corr(),cmap="YlGnBu",annot=True)
plt.show()


# In[75]:


important_features=list(df.corr()['Sales'][(df.corr()['Sales']>+0.5)|(df.corr()['Sales']<-0.5)].index)


# In[76]:


print(important_features)


# In[77]:


x=data['TV']
y=data['Sales']


# In[78]:


print(x)
print(y)


# In[79]:


x=x.values.reshape(-1,1)


# In[25]:


print(x)


# In[80]:


print(x.shape,y.shape)


# In[81]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# In[82]:


print(x_train.shape,y_train.shape)


# In[83]:


from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[84]:


knn=KNeighborsRegressor().fit(x_train,y_train)
knn


# In[85]:


knn_train_pred=knn.predict(x_train)


# In[86]:


knn_test_pred=knn.predict(x_test)


# In[87]:


print(knn_train_pred,knn_test_pred)


# In[88]:


Results = pd.DataFrame(columns=["Model","Train R2","Test R2","Test RMSE","Variance"])


# In[89]:


r2=r2_score(y_test,knn_test_pred)
r2_train=r2_score(y_train,knn_train_pred)
rmse=np.sqrt(mean_squared_error(y_test,knn_test_pred))
variance=r2_train - r2
Results=Results.append({"Model":"K-Nearest Neighbors","Train R2":r2_train,"Test R2":r2,"Test RMSE":rmse,"Variance":variance},ignore_index=True)
print("R2:",r2)
print("RMSE:",rmse)


# In[90]:


Results.head()


# In[91]:


svr=SVR().fit(x_train,y_train)
svr


# In[92]:


svr_train_pred=svr.predict(x_train)
svr_test_pred=svr.predict(x_test)


# In[93]:


print(svr_train_pred,svr_test_pred)


# In[94]:


r2=r2_score(y_test,svr_test_pred)
r2_train=r2_score(y_train,svr_train_pred)
rmse=np.sqrt(mean_squared_error(y_test,svr_test_pred))
variance=r2_train-r2
Results=Results.append({"Model":"Support Vector Machine","Train R2":r2_train,"Test R2":r2,"Test RMSE":rmse,"Variance":variance},ignore_index=True)
print("R2:",r2)
print("RMSE:",rmse)


# In[95]:


Results.head()


# In[96]:


import statsmodels.api as sm


# In[97]:


x_train_constant=sm.add_constant(x_train)


# In[98]:


model=sm.OLS(y_train,x_train_constant).fit()


# In[99]:


model.params


# In[100]:


print(model.summary())


# In[101]:


plt.scatter(x_train,y_train)
plt.plot(x_train,6.995+0.0541*x_train,'y')


# In[102]:


y_train_pred=model.predict(x_train_constant)
res=(y_train - y_train_pred)
res


# In[103]:


fig=plt.figure()
sns.distplot(res,bins=15)
fig.suptitle('Error Terms',fontsize=15)
plt.xlabel('difference between y_train and y_train_pred',fontsize=15)
plt.show()


# In[104]:


plt.scatter(x_train,res)
plt.show()


# In[105]:


x_test_constant=sm.add_constant(x_test)
y_pred=model.predict(x_test_constant)


# In[106]:


y_pred


# In[107]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[108]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[109]:


r2=r2_score(y_test,y_pred)
r2


# In[110]:


plt.scatter(x_test,y_test)
plt.plot(x_test,6.995+0.0541*x_test,'y')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




