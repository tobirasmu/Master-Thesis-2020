#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:17:27 2020

@author: Tobias

A collection of network structures 

"""

# %% The one from Kaggle with many convolutions and high performance but takes
# forever to run

num_classes = 10
channels, height, width = fullData.x_train.shape[1:]

num_filters = (32, 32, 32, 64, 64, 64)
kernel_size = (3 , 3 , 5 , 3 , 3 , 5 )
stride = (1, 1, 2, 1, 1, 2)
padding = (1, 1, 2, 1, 1, 2)
num_l1 = 128

def compute_conv_dim(dim_size, kernel_size, stride_size, padding_size):
    return int((dim_size - kernel_size + 2 * padding_size) / stride_size + 1)

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # Conv1
        i = 0
        self.conv1 = Conv2d(in_channels = channels, out_channels=num_filters[i], 
                            kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
        self.conv1_height = compute_conv_dim(height, kernel_size[i], stride[i], padding[i])
        self.conv1_width = compute_conv_dim(width, kernel_size[i], stride[i], padding[i])
        # Conv2
        i = 1
        self.conv2 = Conv2d(in_channels = num_filters[i-1], out_channels=num_filters[i], 
                            kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
        self.conv2_height = compute_conv_dim(self.conv1_height, kernel_size[i], stride[i], padding[i])
        self.conv2_width = compute_conv_dim(self.conv1_width, kernel_size[i], stride[i], padding[i])
        # Conv3
        i = 2
        self.conv3 = Conv2d(in_channels = num_filters[i-1], out_channels=num_filters[i], 
                            kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
        self.conv3_height = compute_conv_dim(self.conv2_height, kernel_size[i], stride[i], padding[i])
        self.conv3_width = compute_conv_dim(self.conv2_width, kernel_size[i], stride[i], padding[i])
        # Conv4
        i = 3
        self.conv4 = Conv2d(in_channels = num_filters[i-1], out_channels=num_filters[i], 
                            kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
        self.conv4_height = compute_conv_dim(self.conv3_height, kernel_size[i], stride[i], padding[i])
        self.conv4_width = compute_conv_dim(self.conv3_width, kernel_size[i], stride[i], padding[i])
        # Conv5
        i = 4
        self.conv5 = Conv2d(in_channels = num_filters[i-1], out_channels=num_filters[i], 
                            kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
        self.conv5_height = compute_conv_dim(self.conv4_height, kernel_size[i], stride[i], padding[i])
        self.conv5_width = compute_conv_dim(self.conv4_width, kernel_size[i], stride[i], padding[i])
        # Conv6
        self.conv6 = Conv2d(in_channels = num_filters[i-1], out_channels=num_filters[i], 
                            kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
        self.conv6_height = compute_conv_dim(self.conv5_height, kernel_size[i], stride[i], padding[i])
        self.conv6_width = compute_conv_dim(self.conv5_width, kernel_size[i], stride[i], padding[i])
        
        # Dropout function
        self.dropout2 = Dropout2d(p = 0.4)
        self.dropout = Dropout(p = 0.4)
        
        # Batch Normalization
        self.norm1 = BatchNorm2d(num_filters[0])
        self.norm2 = BatchNorm2d(num_filters[1])
        self.norm3 = BatchNorm2d(num_filters[2])
        self.norm4 = BatchNorm2d(num_filters[3])
        self.norm5 = BatchNorm2d(num_filters[4])
        self.norm6 = BatchNorm2d(num_filters[5])
        
        self.norm = BatchNorm1d(num_l1)
        
        # Hidden layer
        self.l1_in_features = self.conv6_height * self.conv6_width * num_filters[5]
        self.l1 = Linear(in_features = self.l1_in_features, out_features = num_l1, bias = True)
        self.l_out = Linear(in_features = num_l1, out_features = num_classes, bias = False)
        
    def forward(self,x):
        # First convolutional layer
        x = self.dropout2(self.norm1(relu(self.conv1(x))))
        # 2nd
        x = self.dropout2(self.norm2(relu(self.conv2(x))))
        # 3rd
        x = self.dropout2(self.norm3(relu(self.conv3(x))))
        # 4th
        x = self.dropout2(self.norm4(relu(self.conv4(x))))
        # 5th
        x = self.dropout2(self.norm5(relu(self.conv5(x))))
        # 6th
        x = self.dropout2(self.norm6(relu(self.conv6(x))))
        
        # Reshaping for hidden layer
        x = x.view(-1,self.l1_in_features)
        x = self.dropout(self.norm(relu(self.l1(x))))
        
        return softmax(self.l_out(x), dim = 1)
        
net = Net()