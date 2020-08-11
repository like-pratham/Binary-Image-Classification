#!/usr/bin/env python
# coding: utf-8

# In[4]:





# In[2]:





# In[5]:


pwd


# In[6]:


cd Anaconda packages


# In[7]:


cd Myfiles


# In[12]:


import os
import cv2
import numpy
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
import torch.optim as optim
from tqdm import tqdm 


# In[9]:


pwd


# In[27]:


Rebuild_data=False
class CatvsDog:
    img_size=50
    cat="PetImages\Cat"
    dog="PetImages\Dog"
    test="PetImages\Testing"
    labels={cat:0,dog:1}
    training_data=[]
    dogcnt=0
    catcnt=0
    def make_training_data (self): #making the training set
        for label in self.labels:
            print (label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path=os.path.join(label,f) # join cat with its images and dog with its
                        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                        img=cv2.resize(img,(self.img_size,self.img_size))
                        self.training_data.append([np.array(img),np.eye(2)[self.labels[label]]])
                        #in training set append image with its one hot vector, dog with [0 1] , cat with [1 0]
                        if label==self.cat:
                            self.catcnt+=1
                        elif label==self.dog:
                            self.dogcnt+=1
                    except Exception as e:
                        pass
        
        # make training data
        # grascale , size, append with one hot vector
        # shuffle
        # balance
        np.random.shuffle(self.training_data)
        #save training data
        np.save("training_data.npy",self.training_data)
        print("Cats",CatvsDog.catcnt)
        print("Dogs",CatvsDog.dogcnt)

if Rebuild_data:
    class_obj=CatvsDog()
    class_obj.make_training_data()
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))
            
    
    


# In[35]:


X=torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X/=255.0
Y=torch.Tensor([i[1] for i in training_data])
# X has images , Y has one hot vector


# In[46]:


print (training_data.shape)
print (X.shape)
print(X[0].shape)
print(Y[0].shape)
x = torch.randn(50,50).view(-1,1,50,50)
print(x)
print(x.shape)


# In[47]:



#Building the neural network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,5)# 1 image ip , 32 o/ps, 5*5 conv
        self.conv2=nn.Conv2d(32,64,5)
        self.conv3=nn.Conv2d(64,128,5)
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out bc we're doing 2 classes (dog vs cat).
    
    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
        
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)
    """
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return F.log_softmax(x,dim=1) 
    """
    
net=Net()
print(net)


"""
#send an X as input and print output
X=torch.rand(28,28)
#print(X)
out=net(X.view(1,28*28))
print(out)   
"""


# In[50]:


#initialize loss fn and optimizer

# Training the Model

  
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=0.001)#Adaptive Momentum is optim


loss_function = nn.MSELoss() # mean  squared error beacuse of one hot vectors
optimizer = optim.Adam(net.parameters(), lr=0.001)

X=torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X/=255.0
y=torch.Tensor([i[1] for i in training_data])
# X has images , Y has one hot vector


#Separating the testing, cross validation and training data
VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]
print(len(train_X), len(test_X))


# In[52]:


BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        #print(f"{i}:{i+BATCH_SIZE}")
        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update

    print(f"Epoch: {epoch}. Loss: {loss}")
   


# In[54]:



correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list, 
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total*100, 3))


# In[ ]:




