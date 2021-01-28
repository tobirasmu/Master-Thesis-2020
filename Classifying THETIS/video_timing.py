"""
    Timing the networks both fully and layer-wise. The time is reported as the mean and standard deviation of SAMPLE_SIZE
    number of pushes. Before this a BURN_IN number of pushes is carried out and discarded.
"""

HPC = False

import os
path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
from video_networks import compressNet
from video_functions import numFLOPsPerPush, get_variable
from torch.autograd import Variable
from torch.nn import Conv3d, Linear, MaxPool3d, Sequential
from torch.nn.functional import relu, softmax
import torch.nn as nn
import numpy as np
from time import process_time

NUM_OBS = 2
SAMPLE_SIZE = 5
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((NUM_OBS, 4, 28, 120, 160))))


# %% Defining the timed network
def conv_dims(dims, kernels, strides, paddings):
    dimensions = len(dims)
    new_dims = tc.empty(dimensions)
    for i in range(dimensions):
        new_dims[i] = int((dims[i] - kernels[i] + 2 * paddings[i]) / strides[i] + 1)
    return new_dims


# First convolution
c1_channels = 6
c1_kernel = (5, 11, 11)
c1_stride = (1, 1, 1)
c1_padding = (2, 5, 5)
# Second convolution
c2_channels = (6, 16)
c2_kernel = (5, 11, 11)
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

# (start, conv1, conv2, lin1, lin2, lout)
timing = np.zeros((SAMPLE_SIZE + BURN_IN, 6))


# The CNN for the THETIS dataset
class Net_timed(nn.Module):

    def __init__(self, channels, frames, height, width):
        super(Net_timed, self).__init__()

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

        # Features into the linear layers
        self.lin_feats_in = int(16 * tc.prod(dim2sP))
        # Adding the linear layers
        self.l1 = Linear(in_features=self.lin_feats_in, out_features=l1_features)
        self.l2 = Linear(in_features=l1_features, out_features=l2_features)
        self.l_out = Linear(in_features=l2_features, out_features=l_out_features)

    def forward(self, x, sample_num):
        timing[sample_num, 0] = process_time()
        x = relu(self.c1(x))
        x = self.pool3d(x)
        timing[sample_num, 1] = process_time()
        x = relu(self.c2(x))
        x = self.pool3d(x)
        timing[sample_num, 2] = process_time()
        x = tc.flatten(x, 1)

        x = relu(self.l1(x))
        timing[sample_num, 3] = process_time()
        x = relu(self.l2(x))
        timing[sample_num, 4] = process_time()
        x = softmax(self.l_out(x), dim=1)
        timing[sample_num, 5] = process_time()
        return x


# %% Timing the network
print("Based on {} samples".format(SAMPLE_SIZE))

net = Net_timed(4, 28, 120, 160)

# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/THETIS_network_92.pt"))
else:
    net.load_state_dict(
        tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Trained networks/THETIS_network_92.pt"))

if tc.cuda.is_available():
    print("Using CUDA")
    net = net.cuda()

for i in range(SAMPLE_SIZE + BURN_IN):
    net(test, i)

# Calculating the layer times
full_time = timing[BURN_IN:, -1] - timing[BURN_IN:, 0]
for i in range(5, 0, -1):
    timing[:, i] -= timing[:, i - 1]
timing = timing[BURN_IN:, 1:]
times_m, times_s = np.mean(timing, axis=0), np.std(timing, axis=0)
full_time_m, full_time_s = np.mean(full_time), np.std(full_time)


# %% Timing of the decomposed network
# Converting to cpu in order to decompose
if tc.cuda.is_available():
    net = net.cpu()

# Compressing (and converting back to GPU)
netDec = compressNet(net)
if tc.cuda.is_available():
    netDec = netDec.cuda()
    net = net.cuda()

# Ranks of the decomposition
r1_c1 = netDec.c1[0].out_channels
r2_c1 = netDec.c1[1].out_channels
r1_c2 = netDec.c2[0].out_channels
r2_c2 = netDec.c2[1].out_channels
r1_l1 = netDec.l1[0].out_features
r2_l1 = netDec.l1[1].out_features
r_l2 = netDec.l2[0].out_features

# (start, conv1, conv2, lin1, lin2, lout)
timing_dec = np.zeros((SAMPLE_SIZE + BURN_IN, 6))


# The timed decomposed network
class NetDec_timed(nn.Module):

    def __init__(self, channels, frames, height, width):
        super(NetDec_timed, self).__init__()
        # First layer
        c1_1 = Conv3d(in_channels=channels, out_channels=r1_c1, kernel_size=(1, 1, 1), bias=False)
        c1_2 = Conv3d(in_channels=r1_c1, out_channels=r2_c1, kernel_size=c1_kernel, padding=c1_padding,
                      stride=c1_stride, bias=False)
        c1_3 = Conv3d(in_channels=r2_c1, out_channels=c1_channels, kernel_size=(1, 1, 1), bias=True)
        self.c1 = Sequential(c1_1, c1_2, c1_3)
        dim1s = conv_dims((frames, height, width), kernels=c1_kernel, strides=c1_stride, paddings=c1_padding)
        dim1sP = conv_dims(dim1s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        # Second layer
        c2_1 = Conv3d(in_channels=c2_channels[0], out_channels=r1_c2, kernel_size=(1, 1, 1), bias=False)
        c2_2 = Conv3d(in_channels=r1_c2, out_channels=r2_c2, kernel_size=c2_kernel, stride=c2_stride,
                      padding=c2_padding, bias=False)
        c2_3 = Conv3d(in_channels=r2_c2, out_channels=c2_channels[1], kernel_size=(1, 1, 1), bias=True)
        self.c2 = Sequential(c2_1, c2_2, c2_3)
        dim2s = conv_dims(dim1sP, kernels=c2_kernel, strides=c2_stride, paddings=c2_padding)
        dim2sP = conv_dims(dim2s, kernels=pool_kernel, strides=pool_stride, paddings=pool_padding)

        # The pooling layer
        self.pool3d = MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        # Features into the linear layers
        self.lin_feats_in = int(c2_channels[1] * tc.prod(dim2sP))
        # First linear layer
        l1_1 = Linear(in_features=self.lin_feats_in, out_features=r1_l1, bias=False)
        l1_2 = Linear(in_features=r1_l1, out_features=r2_l1, bias=False)
        l1_3 = Linear(in_features=r2_l1, out_features=l1_features, bias=True)
        self.l1 = Sequential(l1_1, l1_2, l1_3)

        # Second linear layer
        l2_1 = Linear(in_features=l1_features, out_features=r_l2, bias=False)
        l2_2 = Linear(in_features=r_l2, out_features=l2_features, bias=True)
        self.l2 = Sequential(l2_1, l2_2)

        # Output layer
        self.l_out = Linear(in_features=l2_features, out_features=l_out_features, bias=True)

    def forward(self, x, sample_num):
        timing_dec[sample_num, 0] = process_time()
        x = relu(self.c1(x))
        x = self.pool3d(x)
        timing_dec[sample_num, 1] = process_time()
        x = relu(self.c2(x))
        x = self.pool3d(x)
        timing_dec[sample_num, 2] = process_time()
        x = tc.flatten(x, 1)

        x = relu(self.l1(x))
        timing_dec[sample_num, 3] = process_time()
        x = relu(self.l2(x))
        timing_dec[sample_num, 4] = process_time()
        x = softmax(self.l_out(x), dim=1)
        timing_dec[sample_num, 5] = process_time()
        return x


# %% Timing the decomposed network
netDec = NetDec_timed(4, 28, 120, 160)

if tc.cuda.is_available():
    netDec = netDec.cuda()

for i in range(SAMPLE_SIZE + BURN_IN):
    netDec(test, i)

# Calculating the layer times
full_time_dec = timing_dec[BURN_IN:, -1] - timing_dec[BURN_IN:, 0]
for i in range(5, 0, -1):
    timing_dec[:, i] -= timing_dec[:, i - 1]
timing_dec = timing_dec[BURN_IN:, 1:]
times_dec_m, times_dec_s = np.mean(timing_dec, axis=0), np.std(timing_dec, axis=0)
full_time_dec_m, full_time_dec_s = np.mean(full_time_dec), np.std(full_time_dec)

# %% Theoretical speed-ups based on the number of FLOPs (floating point operations (+, -, *, %))

input_shape = (28, 120, 160)
FLOPs_orig = numFLOPsPerPush(net, input_shape, paddings=[1], pooling=[1, 2], pool_kernels=[(2, 4, 4), (2, 4, 4)])
dcmp_layer_wise = numFLOPsPerPush(netDec, input_shape, paddings=[2], pooling=[3, 6], pool_kernels=[(2, 4, 4), (2, 4, 4)])
FLOPs_dcmp = tc.tensor([tc.sum(dcmp_layer_wise[0:3]), tc.sum(dcmp_layer_wise[3:6]), tc.sum(dcmp_layer_wise[6:9]),
                        tc.sum(dcmp_layer_wise[9:11]), dcmp_layer_wise[11]])

# Calculating layer-wise speed-ups
theoretical_SP_layer = FLOPs_orig / FLOPs_dcmp
observed_SP_layer = times_m / times_dec_m

print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the networks ", ''))
print("{: <11}{: ^16}{: ^16}{: ^16}\n{:-^60s}".format("Layer", "Theoretical", "Observed", "Accounts for", ''))

layer_names = ["Conv1", "Conv2", "Lin1", "Lin2", "Lin3"]
for i in range(len(FLOPs_orig)):
    print("{: <11s}{: ^16.4f}{: ^16.4f}{: >7.5f} / {: <7.5f}".format(layer_names[i], theoretical_SP_layer[i],
                                                                     observed_SP_layer[i],
                                                                     FLOPs_orig[i] / tc.sum(FLOPs_orig),
                                                                     times_m[i] / np.sum(times_m)))
print("{:-^60s}\n{: <11s}{: ^16.4f}{: ^16.4f}".format('', "Total", tc.sum(FLOPs_orig) / tc.sum(FLOPs_dcmp),
                                                      full_time_m / full_time_dec_m))

print("FLOPS orig: ", FLOPs_orig)
print("FLOPS dcmp: ", FLOPs_dcmp)
print("Time orig: ", times_m * 1000, times_s * 1000)
print("Layer time dcmp: ", times_dec_m * 1000, times_dec_s * 1000)
