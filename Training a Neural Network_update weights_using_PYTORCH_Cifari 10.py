#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[4]:


from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
print(transform)


# In[5]:


trainset = datasets.CIFAR10('CIFAR10_data/', download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)


# In[6]:


#a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images 
#containing one of 10 object classes, with 6000 images per class.

print(len(trainset))
indexs = list(range(len(trainset)))
#print(len(indexs))
print(indexs[0:2])


# In[7]:


dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)


# In[8]:


print(images[0].shape)


# In[10]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[11]:


import torch.nn as nn
import torch.nn.functional as F

#having 10 classes
#32x32 pixels
#


# In[24]:


#Build a fed-forward method
model = nn.Sequential(nn.Linear(3072, 100), #input,hidden
                      nn.ReLU(),
                      nn.Linear(100, 64), 
                      nn.ReLU(),
                      nn.Linear(64, 10), #hidden, output
                      nn.LogSoftmax(dim = 1)) 

#criterion =  nn.CrossEntropyLoss(); orrrrr
criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)

logits = model(images)

loss = criterion(logits, labels)

#print(loss)


# In[25]:


print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


# In[27]:


#using the above gradients need to intialize the weights / update
from torch import optim

optimizer = optim.SGD(model.parameters(), lr = 0.01)


# In[30]:


print('Initialize weights : ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 3072)
optimizer.zero_grad()
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('gradeint : - ', model[0].weight.grad)


# In[32]:


optimizer.step()
print('Update weights : -', model[0].weight)


# In[ ]:





# In[ ]:





# In[ ]:




