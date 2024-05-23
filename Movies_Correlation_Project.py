#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# In[2]:


# Read the data

df= pd.read_csv(r"C:\Users\singa\Downloads\movies.csv\movies.csv")


# In[3]:


df


# In[4]:


df.info()


# In[10]:


df.dropna(subset=['budget', 'gross'], how = 'any', inplace = True)


# In[11]:


df.info()


# In[12]:


df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')


# In[13]:


df.info()


# In[14]:


df.sort_values(by = ['gross'], inplace =  True, ascending =  False)


# In[15]:


df


# In[16]:


# Drop Duplicates

df.drop_duplicates(inplace = True)


# In[18]:


df.info()


# In[20]:


#Correlation using Scatter Plot

plt.scatter(x= df['budget'], y= df['gross'])
plt.title('Budget v/s Gross Collection')
plt.xlabel('Gross Collection')
plt.ylabel('Budget')

plt.show()


# In[23]:


# Regression Plot

sns.regplot(x= 'budget', y= 'gross', data = df,  scatter_kws= {'color': 'red'}, line_kws={'color': 'blue'})


# In[24]:


# Let's start looking at the Correlation
# Pearson(default), Kendall, Spearson


# In[26]:


df.corr(numeric_only = True)


# In[31]:


correaltion_matrix = df.corr(numeric_only = True)

sns.heatmap(correaltion_matrix, annot = True)

plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[33]:


# Changing the datatype of non-numeric columns to random numbers

df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized
        


# In[34]:


df_numerized.corr()


# In[35]:


correaltion_numerized_matrix = df_numerized.corr()
sns.heatmap(correaltion_numerized_matrix, annot = True)

plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[36]:


#Unstacking

corr_pairs = correaltion_numerized_matrix.unstack()

corr_pairs


# In[38]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[39]:


sorted_pairs[(sorted_pairs) > 0.5]


# ## Conclusion
# 
# ##### Budget and Gross has the highest correlation to the gross collection.
# 
# ##### Company has low correlation with the gross collection of their films.
# 
