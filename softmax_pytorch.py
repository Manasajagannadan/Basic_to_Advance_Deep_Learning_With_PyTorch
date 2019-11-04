#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Impliment softmax formula
print("\nJ MANASA")
import numpy as np


# In[18]:


x = 10
def softmax(x):
    exponential = np.exp(x)
    return np.divide(exponential, exponential.sum())
#scores = [3.0, 1.0, 0.2]
print(softmax(x))

print("\nJ MANASA")


# In[ ]:




