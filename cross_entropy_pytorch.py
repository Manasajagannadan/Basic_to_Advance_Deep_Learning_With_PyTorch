#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Impliment Cross entropy function for Decress the loss and incresa the propablility to obtain best model
print("\nJ MANASA")
import numpy as np


# In[11]:


i = np.array([1, 1, 0])
print(i)
p = np.array([0.8, 0.6, 0.1])
print(p)


# In[13]:


def cross_entropy(i,p):
    i = np.float_(i)
    p = np.float_(p)
    return -np.sum(i * np.log(p) + (1 - i) * np.log(1 - p))
print(cross_entropy(i,p))
print("\nJ MANASA")


# In[ ]:




