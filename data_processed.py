# -*- coding: utf-8 -*-
"""
Created on Sat May 19 22:07:56 2018

@author: jaylee
"""
import torch
import pandas as pd
import torch.utils.data as Data
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def Convert_Tensor(data):
    n_samples = data.shape[0]
    tensor_data = torch.zeros(n_samples, 28, 28)
    for i in range(n_samples):
        tensor_data[i] = torch.from_numpy(data.ix[i,:].values.reshape(28,28)).long()
    return tensor_data

train_data_set = pd.read_csv('.\data\\train.csv')
plt.imshow(train_data_set.ix[7,1:].reshape(28,28), cmap='gray')
test_data = pd.read_csv('.\data\\test.csv')
train_data = train_data_set.ix[:,1:]
train_data_label = train_data_set.ix[:,0]#第一列为标签列
train_data_tensor = Convert_Tensor(train_data)
train_data_tensor = torch.unsqueeze(train_data_tensor, dim=1)
train_data_label = torch.from_numpy(train_data_label.values).long()
torch_dataset = Data.TensorDataset(train_data_tensor, train_data_label)
test_data_tensor = Convert_Tensor(test_data)
