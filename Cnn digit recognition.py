# -*- coding: utf-8 -*-
"""
Created on Sat May 19 22:07:56 2018

@author: jaylee
"""
import torch
import numpy as np 
import pandas as pd
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import data_processed
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 7
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = True

train_data_set = data_processed.torch_dataset #(42000, 1, 28, 28) and (42000, 1)
train_loader = Data.DataLoader(dataset=train_data_set, batch_size=BATCH_SIZE, shuffle=True)

test_data = data_processed.test_data_tensor #(28000, 28, 28)
test_data = torch.unsqueeze(test_data, dim=1)  #convert (28000, 28, 28) to (28000, 1, 28, 28)
test_x = Variable(test_data)                # value in range(0,1)

#%% Lee-net
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
cnn1 = Net()
print(cnn1)

#%% my self net 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization

cnn2 = CNN()
print(cnn2)  # net architecture

#%%
#choose cnn net
cnn = cnn1

#%%
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
loss_list=[]
accuracy_list=[]
# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y
        output = cnn(b_x)               # cnn output
        pred_y = torch.max(output, 1)[1].data.squeeze()
        accuracy = sum(pred_y == b_y).numpy() / pred_y.size(0)
        loss = loss_func(output, b_y)   # cross entropy loss
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        print('Epoch: ', epoch, '| step: %d' % step, '| train loss: %.4f' % loss.data[0], '| train accuracy: %.4f' % accuracy)

#%%predictions from test data
#save the model 
torch.save(cnn, './model/cnn1_digit_recognizer.pkl')
model=torch.load('./model/cnn1_digit_recognizer.pkl')
test_output = model(test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
sample_submission = pd.read_csv('./data/standard_sample_submission.csv')
sample_submission.ix[:,'Label'] = pred_y
sample_submission.to_csv('./data/sample_submission.csv')

#%%Evaluation the model
plt.plot(np.array(loss_list),label="train-loss")
plt.plot(np.array(accuracy_list),label="train-accuracy")
plt.legend(loc='upper left')
