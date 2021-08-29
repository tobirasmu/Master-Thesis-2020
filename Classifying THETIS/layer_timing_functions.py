import torch.nn as nn
from torch.nn import Linear, Conv3d, Conv2d

"""
Has the different types of networks corresponding to a single layer with given architecture
"""


class conv_layer_timing(nn.Module):

    def __init__(self, ch, out, kernel_size, padding, stride=(1, 1, 1), bias=True, dimensions=3):
        super(conv_layer_timing, self).__init__()
        self.layer = Conv3d(in_channels=ch, out_channels=out, kernel_size=kernel_size, stride=stride, padding=padding,
                            bias=bias) if dimensions == 3 else Conv2d(in_channels=ch, out_channels=out,
                                                                      kernel_size=kernel_size, stride=stride,
                                                                      padding=padding, bias=bias)

    def forward(self, x):
        return self.layer(x)


class lin_layer_timing(nn.Module):

    def __init__(self, in_neurons, out_neurons, bias=True):
        super(lin_layer_timing, self).__init__()

        self.layer = Linear(in_features=in_neurons, out_features=out_neurons, bias=bias)

    def forward(self, x):
        return self.layer(x)
