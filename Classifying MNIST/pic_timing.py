"""
    Timing the networks both fully and layer-wise. The time is reported as the mean and standard deviation of SAMPLE_SIZE
    number of pushes. Before this a BURN_IN number of pushes is carried out and discarded.
"""
HPC = True
import os
path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying MNIST/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)

import torch as tc
from pic_functions import numFLOPsPerPush, numParams
from pic_networks import get_VGG16, compressNetwork
from timeit import repeat
from torch.autograd import Variable
from pic_functions import get_variable
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.functional import relu, softmax
import torch.nn as nn
import numpy as np
from time import process_time

NUM_OBS = 100
SAMPLE_SIZE = 1000
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((NUM_OBS, 1, 28, 28))))


# %% Testing something new
def conv_dim(dim, kernel, stride, padding):
    return int((dim - kernel + 2 * padding) / stride + 1)


c1_channels = 6
c1_kernel = 5
c1_padding = 2
c1_stride = 1
c2_channels = (6, 16)
c2_kernel = 5
c2_padding = 0
c2_stride = 1
# Pooling layer
pool_kernel = 2
pool_stride = 2
pool_padding = 0
# Linear layers
l1_features = 120
l2_features = 84
l_out_features = 10

# (start, conv1, conv2, lin1, lin2, lout)
timing = np.zeros((SAMPLE_SIZE + BURN_IN, 6))


class Net_timed(nn.Module):

    def __init__(self, channels, height):  # We only need height since the pictures are square
        super(Net_timed, self).__init__()

        # The convolutions
        self.conv1 = Conv2d(in_channels=channels, out_channels=c1_channels, kernel_size=c1_kernel, padding=c1_padding,
                            stride=c1_stride)
        dim1 = conv_dim(height, kernel=c1_kernel, padding=c1_padding, stride=c1_stride)
        dim1P = conv_dim(dim1, kernel=pool_kernel, padding=pool_padding, stride=pool_stride)
        self.conv2 = Conv2d(in_channels=c2_channels[0], out_channels=c2_channels[1], kernel_size=c2_kernel,
                            padding=c2_padding, stride=c2_stride)
        dim2 = conv_dim(dim1P, kernel=c2_kernel, padding=c2_padding, stride=c2_stride)
        dim2P = conv_dim(dim2, kernel=pool_kernel, padding=pool_padding, stride=pool_stride)

        # The average pooling
        self.pool = MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        self.lin_in_feats = c2_channels[1] * (dim2P ** 2)
        # The linear layers
        self.l1 = Linear(in_features=self.lin_in_feats, out_features=l1_features, bias=True)
        self.l2 = Linear(in_features=l1_features, out_features=l2_features, bias=True)
        self.l_out = Linear(in_features=l2_features, out_features=l_out_features, bias=True)

    def forward(self, x, sample_num):
        timing[sample_num, 0] = process_time()
        # Conv 1
        x = relu(self.conv1(x))
        x = self.pool(x)
        timing[sample_num, 1] = process_time()
        # Conv 2
        x = relu(self.conv2(x))
        x = self.pool(x)

        x = tc.flatten(x, 1)
        # Lin 1
        timing[sample_num, 2] = process_time()
        x = relu(self.l1(x))
        # Lin 2
        timing[sample_num, 3] = process_time()
        x = relu(self.l2(x))
        # Lin out
        timing[sample_num, 4] = process_time()
        x = softmax(self.l_out(x), dim=1)

        timing[sample_num, 5] = process_time()
        return x


# %% Testing the function

net = Net_timed(1, 28)
if tc.cuda.is_available():
    print(" - Cuda enabled - ")
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


# %% Decomposing the network

if tc.cuda.is_available():
    net = net.cpu()

if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/MNIST_network_9866_acc.pt"))
else:
    net.load_state_dict(tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Trained "
                                "networks/MNIST_network_9866_acc.pt"))

netDec = compressNetwork(net)

if tc.cuda.is_available():
    netDec = netDec.cuda()
    net = net.cuda()

# The ranks of the decomposition
r_c1 = netDec.conv1[0].out_channels
r1_c2 = netDec.conv2[0].out_channels
r2_c2 = netDec.conv2[1].out_channels
r1_l1 = netDec.l1[0].out_features
r2_l1 = netDec.l1[1].out_features
r_l2 = netDec.l2[0].out_features

# (start, conv1, conv2, lin1, lin2, lout)
timing_dec = np.zeros((SAMPLE_SIZE + BURN_IN, 12))


class NetDec_timed(nn.Module):

    def __init__(self, channels, height):
        super(NetDec_timed, self).__init__()
        # First layer
        self.c1_1 = Conv2d(in_channels=channels, out_channels=r_c1, kernel_size=c1_kernel, padding=c1_padding, bias=False)
        self.c1_2 = Conv2d(in_channels=r_c1, out_channels=c1_channels, kernel_size=(1, 1), bias=True)
        # self.conv1 = Sequential(c1_1, c1_2)
        dim1 = conv_dim(height, kernel=c1_kernel, padding=c1_padding, stride=c1_stride)
        dim1P = conv_dim(dim1, kernel=pool_kernel, padding=pool_padding, stride=pool_stride)

        # Second layer
        self.c2_1 = Conv2d(in_channels=c2_channels[0], out_channels=r1_c2, kernel_size=(1, 1), bias=False)
        self.c2_2 = Conv2d(in_channels=r1_c2, out_channels=r2_c2, kernel_size=c2_kernel, padding=c2_padding, bias=False)
        self.c2_3 = Conv2d(in_channels=r2_c2, out_channels=c2_channels[1], kernel_size=(1, 1), bias=True)
        # self.conv2 = Sequential(c2_1, c2_2, c2_3)
        dim2 = conv_dim(dim1P, kernel=c2_kernel, padding=c2_padding, stride=c2_stride)
        dim2P = conv_dim(dim2, kernel=pool_kernel, padding=pool_padding, stride=pool_stride)

        # Pooling layer
        self.pool = MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

        self.lin_in_feats = c2_channels[1] * (dim2P ** 2)
        # First linear layer
        self.l1_1 = Linear(in_features=self.lin_in_feats, out_features=r1_l1, bias=False)
        self.l1_2 = Linear(in_features=r1_l1, out_features=r2_l1, bias=False)
        self.l1_3 = Linear(in_features=r2_l1, out_features=l1_features, bias=True)
        # self.l1 = Sequential(l1_1, l1_2, l1_3)

        # Second linear layer
        self.l2_1 = Linear(in_features=l1_features, out_features=r_l2, bias=False)
        self.l2_2 = Linear(in_features=r_l2, out_features=l2_features, bias=True)
        # self.l2 = Sequential(l2_1, l2_2)

        # L_out
        self.l_out = Linear(in_features=l2_features, out_features=l_out_features, bias=True)

    def forward(self, x, sample_num):
        # Conv 1
        timing_dec[sample_num, 0] = process_time()
        x = relu(self.c1_1(x))
        timing_dec[sample_num, 1] = process_time()
        x = relu(self.c1_2(x))
        x = self.pool(x)
        # Conv 2
        timing_dec[sample_num, 2] = process_time()
        x = relu(self.c2_1(x))
        timing_dec[sample_num, 3] = process_time()
        x = relu(self.c2_2(x))
        timing_dec[sample_num, 4] = process_time()
        x = relu(self.c2_3(x))
        x = self.pool(x)
        # Lin 1
        timing_dec[sample_num, 5] = process_time()
        x = tc.flatten(x, 1)
        x = relu(self.l1_1(x))
        timing_dec[sample_num, 6] = process_time()
        x = relu(self.l1_2(x))
        timing_dec[sample_num, 7] = process_time()
        x = relu(self.l1_3(x))
        # Lin 2
        timing_dec[sample_num, 8] = process_time()
        x = relu(self.l2_1(x))
        timing_dec[sample_num, 9] = process_time()
        x = relu(self.l2_2(x))
        # Lin 3
        timing_dec[sample_num, 10] = process_time()
        x = softmax(self.l_out(x), dim=1)
        timing_dec[sample_num, 11] = process_time()
        return x


# %% Timing the decomposed network
netDec = NetDec_timed(1, 28)
if tc.cuda.is_available():
    netDec = netDec.cuda()

for i in range(SAMPLE_SIZE + BURN_IN):
    netDec(test, i)

# Calculating the layer times
full_time_dec = timing_dec[BURN_IN:, -1] - timing_dec[BURN_IN:, 0]
for i in range(11, 0, -1):
    timing_dec[:, i] -= timing_dec[:, i - 1]
timing_dec = timing_dec[BURN_IN:, 1:]
times_dec_m, times_dec_s = np.mean(timing_dec, axis=0), np.std(timing_dec, axis=0)
full_time_dec_m, full_time_dec_s = np.mean(full_time_dec), np.std(full_time_dec)
# To compare
comp_c1 = np.sum(timing_dec[:, 0:2], axis=1)
comp_c2 = np.sum(timing_dec[:, 2:5], axis=1)
comp_lin1 = np.sum(timing_dec[:, 5:8], axis=1)
comp_lin2 = np.sum(timing_dec[:, 8:10], axis=1)
comp_lin3 = timing_dec[:, 10]
comp_m = np.array([np.mean(comp_c1), np.mean(comp_c2), np.mean(comp_lin1), np.mean(comp_lin2), np.mean(comp_lin3)])

# %% Calculating the theoretical speed-ups
input_shape = (28, 28)
FLOPs_orig = numFLOPsPerPush(net, input_shape, paddings=[1], pooling=[1, 2], pool_kernels=[(2, 2), (2, 2)])
dcmp_layer_wise = numFLOPsPerPush(netDec, input_shape, paddings=[1], pooling=[3, 6], pool_kernels=[(2, 2), (2, 2)])
FLOPs_dcmp = tc.tensor([tc.sum(dcmp_layer_wise[0:2]), tc.sum(dcmp_layer_wise[2:5]), tc.sum(dcmp_layer_wise[5:8]),
                             tc.sum(dcmp_layer_wise[8:10]), dcmp_layer_wise[10]])

# Calculating layer-wise speed-ups
theoretical_SP_layer = FLOPs_orig / FLOPs_dcmp
observed_SP_layer = times_m / comp_m

print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the networks ", ''))
print("{: <11}{: ^16}{: ^16}{: ^16}\n{:-^60s}".format("Layer", "Theoretical", "Observed", "Accounts for", ''))

layer_names = ["Conv1", "Conv2", "Lin1", "Lin2", "Lin3"]
for i in range(len(FLOPs_orig)):
    print("{: <11s}{: ^16.4f}{: ^16.4f}{: >7.5f} / {: <7.5f}".format(layer_names[i], theoretical_SP_layer[i],
                                                                     observed_SP_layer[i],
                                                                     FLOPs_orig[i] / tc.sum(FLOPs_orig),
                                                                     times_m[i] / full_time_m))
print("{:-^60s}\n{: <11s}{: ^16.4f}{: ^16.4f}".format('', "Total", tc.sum(FLOPs_orig) / tc.sum(FLOPs_dcmp),
                                                      full_time_m / full_time_dec_m))
print("\nThe number of parameters:\nOriginal: {:d}    Compressed:  {:d}    Ratio:  {:.3f}".format(numParams(net),
                                                                                                  numParams(netDec),
                                                                                                  numParams(netDec) /
                                                                                                  numParams(net)))
print("Based on {} samples and {} observations pushed forward".format(SAMPLE_SIZE, NUM_OBS))
print("FLOPS orig: ", FLOPs_orig)
print("FLOPS dcmp: ", FLOPs_dcmp)
print("FLOPs dcmp 2: ", dcmp_layer_wise)
print("Full time orig {} +- {} and compressed {} +- {}".format(full_time_m * 1000, full_time_s * 1000, full_time_dec_m * 1000, full_time_dec_s * 1000))
print("Time orig: ", times_m * 1000, times_s * 1000)
print("Layer time dcmp: ", times_dec_m * 1000, times_dec_s * 1000)
print("Layer time dcmp comp: ", comp_m * 1000, times_dec_s * 1000)

# %% Investigating whether it works for bigger architectures
print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the VGG-16 network ", ''))
NUM_OBS = 10
# Loading the networks and compressing using the algorithm
vgg16 = get_VGG16()
vgg16_dec = get_VGG16(compressed=True)

if tc.cuda.is_available():
    vgg16 = vgg16.cuda()
    vgg16_dec = vgg16_dec.cuda()
    print(" -- Using GPU -- ")

test = get_variable(Variable(tc.rand((NUM_OBS, 3, 224, 224))))

# Timing VGG16 - both original and compressed
fullTime_VGG16 = tc.tensor(repeat('vgg16(test)', globals=locals(), number=1, repeat=(SAMPLE_SIZE +
                                                                                     BURN_IN))[BURN_IN:])
fullTime_VGG16_dec = tc.tensor(repeat('vgg16_dec(test)', globals=locals(), number=1, repeat=(SAMPLE_SIZE +
                                                                                             BURN_IN))[BURN_IN:])

# %% Calculating the theoretical speed-up using FLOPs
FLOPs_vgg16 = numFLOPsPerPush(vgg16, (224, 224), paddings=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                              pooling=[2, 4, 7, 10, 13], pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
FLOPs_vgg16_dcmp = numFLOPsPerPush(vgg16_dec, (224, 224),
                                   paddings=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37],
                                   pooling=[5, 11, 20, 29, 38],
                                   pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
FLOPs_vgg16_comp = tc.tensor([tc.sum(FLOPs_vgg16_dcmp[0:2]), tc.sum(FLOPs_vgg16_dcmp[2:5]),
                              tc.sum(FLOPs_vgg16_dcmp[5:8]), tc.sum(FLOPs_vgg16_dcmp[8:11]),
                              tc.sum(FLOPs_vgg16_dcmp[11:14]), tc.sum(FLOPs_vgg16_dcmp[14:17]),
                              tc.sum(FLOPs_vgg16_dcmp[17:20]), tc.sum(FLOPs_vgg16_dcmp[20:23]),
                              tc.sum(FLOPs_vgg16_dcmp[23:26]), tc.sum(FLOPs_vgg16_dcmp[26:29]),
                              tc.sum(FLOPs_vgg16_dcmp[29:32]), tc.sum(FLOPs_vgg16_dcmp[32:35]),
                              tc.sum(FLOPs_vgg16_dcmp[35:38]), tc.sum(FLOPs_vgg16_dcmp[38:41]),
                              tc.sum(FLOPs_vgg16_dcmp[41:43]), tc.sum(FLOPs_vgg16_dcmp[43:44])])
# Overall theoretical speed-up
print("Time for the VGG-16 network was {} +- {} second, while the decomposed version took {} +- {} seconds".format(
    tc.mean(fullTime_VGG16), tc.std(fullTime_VGG16), tc.mean(fullTime_VGG16_dec), tc.std(fullTime_VGG16_dec)))
print("Theoretical speed-up is {:.4f} while the observed speed-up is {:.2f}".format(
    sum(FLOPs_vgg16) / sum(FLOPs_vgg16_dcmp), tc.mean(fullTime_VGG16) / tc.mean(fullTime_VGG16_dec)))
print("FLOPs: ", sum(FLOPs_vgg16), sum(FLOPs_vgg16_dcmp))
print("Weights: ", numParams(vgg16), numParams(vgg16_dec))
