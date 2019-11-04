#!/usr/bin/env python
# coding: utf-8

# In[218]:


import numpy as np
import torch


# In[219]:


torch.manual_seed(7)
x = torch.randn((1,5), requires_grad=True)
w = torch.randn_like((x), requires_grad=True) 
b = torch.randn((1,1), requires_grad=True)
y_op = torch.randn((1,6), requires_grad=True)

print(x)
print(y)

print(w)
print(b)


# In[220]:


learn_rate = 0.5
for i in range(7):
    y_pre = torch.sum(x * w.view(5,1)) + b
    #mean square of value
    msq = torch.mean((y_pre - y_op) ** 2)   #loss
    d = msq.backward()
print(msq)
    #print('w:', w)
    #print('b:', b)
    #print('w.grad:', w.grad)
    #print('b.grad:', b.grad)
    


# In[221]:


print(w)
print(w.grad)

#If a gradient element is positive:
#increasing the element’s value slightly will increase the loss.
#decreasing the element’s value slightly will decrease the loss.

#If a gradient element is negative:
#increasing the element’s value slightly will decrease the loss.
#decreasing the element’s value slightly will increase the loss.


# In[222]:


w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)


# In[223]:


learn_rate = 0.5
for i in range(7):
    y_pre = torch.sum(x * w.view(5,1)) + b
    #mean square of value
    msq = torch.mean((y_pre - y_op) ** 2)   #loss
    d = msq.backward()
print(msq)

print(w.grad)
print(b.grad)


# In[224]:


#Calculate the difference between the two matrices (preds and targets).
#Square all elements of the difference matrix to remove negative values.
#Calculate the average of the elements in the resulting matrix.

with torch.no_grad():
        w = w - learn_rate * w.grad
        b = b - learn_rate * b.grad
w.requires_grad = True
b.requires_grad = True


# In[225]:


#new weights
print(w)
print(b)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




