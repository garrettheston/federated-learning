#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class MNISTCNN(nn.Module):
    def __init__(self, params):
        super(MNISTCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 10, kernel_size=5) # 1 = params.num_channels
        self.layer2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout_layer = nn.Dropout2d()
        self.fc_layer1 = nn.Linear(320, 50)
        self.fc_layer2 = nn.Linear(50, 10) # 10 = params.num_classes

    def forward(self, input_data):
        data = F.relu(F.max_pool2d(self.layer1(input_data), 2))
        data = F.relu(F.max_pool2d(self.dropout_layer(self.layer2(data)), 2))
        data = data.view(data.size(0), -1)  # Flattening
        data = F.relu(self.fc_layer1(data))
        data = F.dropout(data, training=self.training)
        output = self.fc_layer2(data)
        return output

class CIFARCNN(nn.Module):
    def __init__(self, params):
        super(CIFARCNN, self).__init__()
        self.first_conv = nn.Conv2d(3, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.second_conv = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc_first = nn.Linear(64 * 5 * 5, 256)
        self.fc_second = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, params.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_data):
        data = self.pool_layer(F.relu(self.bn1(self.first_conv(input_data))))
        data = self.pool_layer(F.relu(self.bn2(self.second_conv(data))))
        data = data.view(data.size(0), -1)  # Flattening
        data = self.dropout(F.relu(self.fc_first(data)))
        data = self.dropout(F.relu(self.fc_second(data)))
        output = self.output_layer(data)
        return output