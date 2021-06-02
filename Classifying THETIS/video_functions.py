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










def numParams(net):
    """
    Returns the number of parameters in the entire network.
    """
    return sum(np.prod(p.size()) for p in net.parameters())


# %% Functions for calculating the theoretical speed-up
# Two matrices of size [N1 x N2] and [N2 x N3] respectively excluding N1 x N3 biases
def linearFLOPs(out_features, in_features):
    """
    Returns the number of FLOPs needed to perform forward push through a linear layer with the given in- and output
    features.
    Based on the paper "Pruning CNNs for resource efficiency" by Molchanov P, Tyree S, Karras T, et al.
    """
    return (2 * in_features - 1) * out_features


def conv_dims(dims, kernels, strides, paddings):
    """
    Computes the resulting output dimensions when performing a convolution with the given kernel, stride and padding.
    INPUT:
            dims - the input dimensions
            kernels - kernel width for each dimension
            strides - strides for each dimension
            paddings - for each dimension
    OUTPUT:
            list of resulting dimensions
    """
    dimensions = len(dims)
    new_dims = tc.empty(dimensions)
    for i in range(dimensions):
        new_dims[i] = int((dims[i] - kernels[i] + 2 * paddings[i]) / strides[i] + 1)
    return new_dims


def convFLOPs(kernel_shape, output_shape):
    """
    The FLOPs needed to perform a convolution.
    INPUT:
            kernel_shape = (out_channels, in_channels, (d_f), d_h, d_w)
            input_shape = ((f), h, w)
    """
    C_out, C_in = kernel_shape[0:2]
    filter_shape = kernel_shape[2:]
    return C_out * tc.prod(output_shape.long()) * (2 * C_in * tc.prod(filter_shape.long()) - 1)


def numFLOPsPerPush(net, input_shape, paddings=None, pooling=None, pool_kernels=None):
    """
    Returns the number of floating point operations needed to make one forward push of each of the layers,
    in a given network. Padding is a list of the layer number that has padding in it (assumed full padding), pooling is
    a list of the layers that have pooling just after them. Pool_kernels are the corresponding pooling kernels for each
    layer that have pooling in them
    """
    FLOPs = []
    layer = 0
    paddings = [] if paddings is None else paddings
    pooling = [] if pooling is None else pooling
    wasConv = False
    output_shape = input_shape
    for weights in list(net.parameters()):
        kernel_shape = tc.tensor(weights.shape)
        if len(kernel_shape) == 2:
            layer += 1
            FLOPs.append(linearFLOPs(kernel_shape[0], kernel_shape[1]))
            wasConv = False
        elif len(kernel_shape) > 2:
            wasConv = True
            layer += 1
            this_padding = kernel_shape[2:] // 2 if layer in paddings else (0, 0, 0)
            output_shape = conv_dims(input_shape, kernel_shape[2:], strides=(1, 1, 1), paddings=this_padding)
            FLOPs.append(convFLOPs(kernel_shape, output_shape))
            if layer in pooling:
                this_kernel = pool_kernels.pop(0)
                input_shape = conv_dims(output_shape, this_kernel, strides=this_kernel, paddings=(0, 0, 0))
            else:
                input_shape = output_shape
        else:
            # Bias term requires number of additions equal to the amount of output values
            if wasConv:
                FLOPs[-1] += tc.prod(output_shape.long())
            else:
                FLOPs[-1] += kernel_shape[0]
    return tc.tensor(FLOPs)


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
