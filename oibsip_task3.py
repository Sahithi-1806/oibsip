#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[18]:


data=pd.read_csv("C:/Users/lakshmi sahithi/Downloads/unemployment oasis.csv")


# In[19]:


print(data.head())


# In[20]:


data.info


# In[21]:


print(data.describe)


# In[22]:


print(data.isnull().sum())


# In[23]:


data.columns=["Region","Date","Frequency","Estimated Unemployment Rate (%)","Estimated Employed", "Estimated Labour Participation Rate (%)","Area"]


# In[24]:


print(data)


# In[25]:


plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(12,10))
sns.heatmap(data.corr())
plt.show()


# In[26]:


data.columns=["Region","Date","Frequency","Estimated Unemployment Rate (%)","Estimated Employed", "Estimated Labour Participation Rate (%)","Area"]
plt.title("Indian unmeployment")
sns.histplot(x="Estimated Employed",hue="Region",data=data)
plt.show()


# In[27]:


plt.figure(figsize=(12,10))
plt.title("Indian Unemployement")
sns.histplot(x="Estimated Unemployment Rate (%)",hue="Region",data=data)
plt.show()


# In[28]:


un=data[["Region","Area","Estimated Unemployment Rate (%)"]]
print(un)


# In[ ]:




