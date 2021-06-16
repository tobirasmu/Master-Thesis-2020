import numpy as np
import torch as tc
import torch.nn as nn
from torch.nn import Conv3d, MaxPool3d, Linear, Dropout3d, Dropout
from torch.nn.functional import relu, softmax


# First convolution
c1_channels = 6
c1_kernel = (5, 11, 11)
c1_stride = (1, 1, 1)
c1_padding = (2, 5, 5)
# Second convolution
c2_channels = (6, 16)
c2_kernel = (3, 5, 5)
c2_stride = (1, 1, 1)
c2_padding = (0, 0, 0)
# Pooling layer
pool_kernel = (2, 4, 4)
pool_stride = (2, 4, 4)
pool_padding = (0, 0, 0)
# Linear layers
l1_features = 128
l2_features = 84
l_out_features = 2


# The CNN for the THETIS dataset
class Net(nn.Module):

    def __init__(self, channels, frames, height, width):
        super(Net, self).__init__()

        # Adding the convolutional layers
        self.c1 = Conv3d(in_channels=channels, out_channels=c1_channels, kernel_size=c1_kernel,
                         stride=c1_stride, padding=c1_padding)
        dim1s = conv_dims((frames, height, width), kernels=c1_kernel, strides=c1_stride, paddings=c1_padding)
        dim1sP = conv_dims(dim1s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        self.c2 = Conv3d(in_channels=c2_channels[0], out_channels=c2_channels[1], kernel_size=c2_kernel,
                         stride=c2_stride, padding=c2_padding)
        dim2s = conv_dims(dim1sP, kernels=c2_kernel, strides=c2_stride, paddings=c2_padding)
        dim2sP = conv_dims(dim2s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        # The pooling layer
        self.pool3d = MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        self.dropout = Dropout3d(0.4)

        # Features into the linear layers
        self.lin_feats_in = int(16 * tc.prod(tc.tensor(dim2sP)))
        # Adding the linear layers
        self.l1 = Linear(in_features=self.lin_feats_in, out_features=l1_features)
        self.l2 = Linear(in_features=l1_features, out_features=l2_features)
        self.l_out = Linear(in_features=l2_features, out_features=l_out_features)

    def forward(self, x):
        x = relu(self.c1(x))
        x = self.pool3d(x)
        x = self.dropout(x)

        x = relu(self.c2(x))
        x = self.pool3d(x)
        x = self.dropout(x)

        x = tc.flatten(x, 1)

        x = relu(self.l1(x))
        x = relu(self.l2(x))

        return softmax(self.l_out(x), dim=1)
    


# The CNN for the THETIS dataset
class Net2(nn.Module):

    def __init__(self, channels, frames, height, width):
        super(Net2, self).__init__()

        # Adding the convolutional layers
        self.c1 = Conv3d(in_channels=channels, out_channels=c1_channels, kernel_size=c1_kernel,
                         stride=c1_stride, padding=c1_padding)
        dim1s = conv_dims((frames, height, width), kernels=c1_kernel, strides=c1_stride, paddings=c1_padding)
        dim1sP = conv_dims(dim1s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        self.c2 = Conv3d(in_channels=c2_channels[0], out_channels=c2_channels[1], kernel_size=c2_kernel,
                         stride=c2_stride, padding=c2_padding)
        dim2s = conv_dims(dim1sP, kernels=c2_kernel, strides=c2_stride, paddings=c2_padding)
        dim2sP = conv_dims(dim2s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        # The pooling layer
        self.pool3d = MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        self.dropout3 = Dropout3d(0.4)
        self.dropout = Dropout(0.4)

        # Features into the linear layers
        self.lin_feats_in = int(16 * tc.prod(tc.tensor(dim2sP)))
        # Adding the linear layers
        self.l1 = Conv3d(in_channels=c2_channels[1], out_channels=l1_features, kernel_size=dim2sP)
        self.l2 = Linear(in_features=l1_features, out_features=l2_features)
        self.l_out = Linear(in_features=l2_features, out_features=l_out_features)

    def forward(self, x):
        x = relu(self.c1(x))
        x = self.pool3d(x)
        x = self.dropout3(x)

        x = relu(self.c2(x))
        x = self.pool3d(x)
        x = self.dropout3(x)
        
        x = relu(self.l1(x)[:, :, 0, 0, 0])
        x = self.dropout(x)
        
        x = relu(self.l2(x))
        x = self.dropout(x)

        return softmax(self.l_out(x), dim=1)


def numParams(net):
    """
    Returns the number of parameters in the entire network.
    """
    return sum(np.prod(p.size()) for p in net.parameters())
# Model evaluation functions


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
    new_dims = []
    for i in range(dimensions):
        new_dims.append(int((dims[i] - kernels[i] + 2 * paddings[i]) / strides[i] + 1))
    return tuple(new_dims)


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