#!/usr/bin/env python
# coding: utf-8

# In[20]:


#Take your data
#Pick a random model
#Calculate the error
#Minimize the error, and obtain a better model
print("\nJ MANASA\n")


#Sigmoid activation function  : σ(x)=1/1+e−x

#Output (prediction) formula : ˆy =σ(w1x1+w2x2+b)

#Error function Error(y,ˆy) = −ylog(ˆy)−(1−y)log(1−ˆy)
#y^ ​	 =σ(Wx+b)

#The function that updates the weights  :  wi⟶ wi+α(y−ˆy)xi
#                                          b⟶  b+α(y− ˆy )


# In[21]:


import numpy as np
import torch

torch.manual_seed(7)
inputs = torch.randn((1,5), requires_grad = True)
weights = torch.randn_like(inputs)
bias = torch.randn((1,1))
y = torch.randn((5,1))

print("input values : ",inputs)
print("weights : ",weights)
print("bias : ",bias)
print("output value : ",y)


# In[22]:


def sigmoid(inputs):
    return 1 / (1 + torch.exp(-inputs))
#print(sigmoid(inputs))

y1 = sigmoid(torch.sum(inputs * weights.view(5,1)) + bias)

print(y1)


# In[23]:


def softmax(inputs):
    return torch.exp(inputs) / torch.sum(torch.exp(inputs), dim = 1).view(1, -1)
print(softmax(inputs))

propabilities = softmax(y1)
print(propabilities.shape)

print(propabilities.sum(dim = 1))


# In[24]:


#order to calculate the derivative of this error with respect to the weights,
#∂w j​	 ∂​	  y^

y1 = sigmoid(torch.sum(inputs * weights.view(5,1)) + bias)
xi = y1.backward()
print(inputs.grad)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




