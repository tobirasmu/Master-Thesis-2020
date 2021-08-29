import numpy as np
import torch as tc
from timeit import repeat
import torch.nn as nn
from torch.nn import Linear, Conv3d, Conv2d

from sklearn.metrics import accuracy_score
import tensorly as tl
from layer_timing_functions import conv_layer_timing, lin_layer_timing


from tensorly.decomposition import partial_tucker
from tools.VBMF import EVBMF
import matplotlib.pyplot as plt
import cv2
import os













# %% Timing a single layer of different types
def time_conv(num_obs, input_size, in_ch, out_ch, kernel, padding, bias=True, sample_size=10, num_dim=3):
    """
    Timing a convolutional layer with the given structure.
    INPUT:
            num_obs    : how many observations are to be pushed through
            input_size : the size of the video (frames, height, width)
            in_ch      : input channels
            out_ch     : output channels
            kernel     : the given kernel
            padding    : the given padding
            bias       : if bias is needed
            number     : how many times it should be timed
            num_dim    : number of dimensions (3 for video, 2 for picture)
    OUTPUT:
            the time in seconds
    """
    burn_in = sample_size // 10
    input_shape = (num_obs, in_ch, *input_size)
    net = conv_layer_timing(in_ch, out_ch, kernel, stride=(1, 1, 1), padding=padding, bias=bias, dimensions=num_dim)
    if tc.cuda.is_available():
        net = net.cuda()
    x = get_variable(Variable(tc.rand(input_shape)))
    times = tc.tensor(repeat("net(x)", globals=locals(), number=1, repeat=(sample_size + burn_in))[burn_in:])
    return tc.mean(times), tc.std(times), times


def time_lin(num_obs, in_neurons, out_neurons, bias=True, sample_size=10):
    """
        Timing the linear forward push with the given structure. Repeats number times and reports the mean, standard
        deviation, and the list of times.
    INPUT:
            num_obs     : how many observations to be pushed forward
            in_neurons  : how many input neurons in the layer
            out_neurons : how many output neurons
            bias        : if bias is also timed
            number      : how many times it should be timed
    OUTPUT:
            mean times, std time, list times
    """
    burn_in = sample_size // 10
    input_shape = (num_obs, in_neurons)
    net = lin_layer_timing(in_neurons, out_neurons, bias)
    if tc.cuda.is_available():
        net = net.cuda()
    x = get_variable(Variable(tc.rand(input_shape)))
    times = tc.tensor(repeat("net(x)", globals=locals(), number=1, repeat=(sample_size + burn_in))[burn_in:])
    return tc.mean(times), tc.std(times), times
