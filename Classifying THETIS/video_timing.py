"""
    Timing the networks both fully and layer-wise. The time is reported as the mean and standard deviation of SAMPLE_SIZE
    number of pushes. Before this a BURN_IN number of pushes is carried out and discarded.
"""

HPC = True

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/home/tenra/PycharmProjects/Master-Thesis-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
from tools.decomp import compressNet
from tools.models import numFLOPsPerPush, numParamsByLayer, Specs, Net2, conv_dims, numParams
from tools.trainer import get_variable
from torch.autograd import Variable
from torch.nn import Conv3d, Linear, MaxPool3d, Dropout, Dropout3d
from torch.nn.functional import relu, softmax
import torch.nn as nn
import numpy as np
from time import time

SAMPLE_SIZE = 1000
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((2, 4, 28, 120, 160))))


# %% Defining the timed network
specs = Specs()

# (start, conv1, conv2, lin1, lin2, lout)
timing = np.zeros((SAMPLE_SIZE + BURN_IN, 6))


# The CNN for the THETIS dataset
class Net_timed(Net2):

    def __init__(self, channels, frames, height, width):
        super(Net_timed, self).__init__(channels, frames, height, width)

    def forward(self, x, sample_num):
        tc.cuda.synchronize()
        timing[sample_num, 0] = time()
        x = relu(self.c1(x))
        x = self.pool3d(x)
        x = self.dropout3(x)

        tc.cuda.synchronize()
        timing[sample_num, 1] = time()
        x = relu(self.c2(x))
        x = self.pool3d(x)
        x = self.dropout3(x)

        tc.cuda.synchronize()
        timing[sample_num, 2] = time()
        x = relu(self.l1(x)[:, :, 0, 0, 0])
        x = self.dropout(x)

        tc.cuda.synchronize()
        timing[sample_num, 3] = time()
        x = relu(self.l2(x))
        x = self.dropout(x)

        tc.cuda.synchronize()
        timing[sample_num, 4] = time()
        x = softmax(self.l_out(x), dim=1)
        tc.cuda.synchronize()
        timing[sample_num, 5] = time()
        return x


# %% Timing the network

net = Net_timed(4, 28, 120, 160)

# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/THETIS_new.pt"))
else:
    net.load_state_dict(
        tc.load("/home/tenra/PycharmProjects/Master-Thesis-2020/Trained networks/THETIS_new.pt"))

# %% Timing of the decomposed network
# Compressing (and converting back to GPU)
netDec = compressNet(net)

# Ranks of the decomposition
r1_c1 = netDec.c1[0].out_channels
r2_c1 = netDec.c1[1].out_channels
r1_c2 = netDec.c2[0].out_channels
r2_c2 = netDec.c2[1].out_channels
r1_l1 = netDec.l1[0].out_channels
r2_l1 = netDec.l1[1].out_channels
r_l2 = netDec.l2[0].out_features

# (start, conv1, conv2, lin1, lin2, lout)
timing_dec = np.zeros((SAMPLE_SIZE + BURN_IN, 13))
specs = Specs()


# The timed decomposed network
class NetDec_timed(nn.Module):

    def __init__(self, channels, frames, height, width):
        super(NetDec_timed, self).__init__()
        # First layer
        self.c1_1 = Conv3d(in_channels=channels, out_channels=r1_c1, kernel_size=(1, 1, 1), bias=False)
        self.c1_2 = Conv3d(in_channels=r1_c1, out_channels=r2_c1, kernel_size=specs.c1_kernel, padding=specs.c1_padding,
                           stride=specs.c1_stride, bias=False)
        self.c1_3 = Conv3d(in_channels=r2_c1, out_channels=specs.c1_channels, kernel_size=(1, 1, 1), bias=True)
        # self.c1 = Sequential(c1_1, c1_2, c1_3)
        dim1s = conv_dims((frames, height, width), kernels=specs.c1_kernel, strides=specs.c1_stride,
                          paddings=specs.c1_padding)
        dim1sP = conv_dims(dim1s, kernels=specs.pool_kernel, strides=specs.pool_stride, paddings=specs.pool_padding)

        # Second layer
        self.c2_1 = Conv3d(in_channels=specs.c2_channels[0], out_channels=r1_c2, kernel_size=(1, 1, 1), bias=False)
        self.c2_2 = Conv3d(in_channels=r1_c2, out_channels=r2_c2, kernel_size=specs.c2_kernel, stride=specs.c2_stride,
                           padding=specs.c2_padding, bias=False)
        self.c2_3 = Conv3d(in_channels=r2_c2, out_channels=specs.c2_channels[1], kernel_size=(1, 1, 1), bias=True)
        # self.c2 = Sequential(c2_1, c2_2, c2_3)
        dim2s = conv_dims(dim1sP, kernels=specs.c2_kernel, strides=specs.c2_stride, paddings=specs.c2_padding)
        dim2sP = conv_dims(dim2s, kernels=specs.pool_kernel, strides=specs.pool_stride, paddings=specs.pool_padding)

        # The pooling layer
        self.pool3d = MaxPool3d(kernel_size=specs.pool_kernel, stride=specs.pool_stride, padding=specs.pool_padding)
        self.dropout3 = Dropout3d(0.2)
        self.dropout = Dropout(0.2)

        # Features into the linear layers
        self.lin_feats_in = int(specs.c2_channels[1] * tc.prod(tc.tensor(dim2sP)))
        # First linear layer

        self.l1_1 = Conv3d(in_channels=specs.c2_channels[1], out_channels=r1_l1, kernel_size=(1, 1, 1), bias=False)
        self.l1_2 = Conv3d(in_channels=r1_l1, out_channels=r2_l1, kernel_size=dim2sP, bias=False)
        self.l1_3 = Conv3d(in_channels=r2_l1, out_channels=specs.l1_features, kernel_size=(1, 1, 1), bias=True)
        # self.l1 = Sequential(l1_1, l1_2, l1_3)

        # Second linear layer
        self.l2_1 = Linear(in_features=specs.l1_features, out_features=r_l2, bias=False)
        self.l2_2 = Linear(in_features=r_l2, out_features=specs.l2_features, bias=True)
        # self.l2 = Sequential(l2_1, l2_2)

        # Output layer
        self.l_out = Linear(in_features=specs.l2_features, out_features=specs.l_out_features, bias=True)

    def forward(self, x, sample_num):
        # Conv 1
        tc.cuda.synchronize()
        timing_dec[sample_num, 0] = time()
        x = relu(self.c1_1(x))
        tc.cuda.synchronize()
        timing_dec[sample_num, 1] = time()
        x = relu(self.c1_2(x))
        tc.cuda.synchronize()
        timing_dec[sample_num, 2] = time()
        x = relu(self.c1_3(x))
        x = self.pool3d(x)
        x = self.dropout3(x)

        # Conv 2
        tc.cuda.synchronize()
        timing_dec[sample_num, 3] = time()
        x = relu(self.c2_1(x))
        tc.cuda.synchronize()
        timing_dec[sample_num, 4] = time()
        x = relu(self.c2_2(x))
        tc.cuda.synchronize()
        timing_dec[sample_num, 5] = time()
        x = relu(self.c2_3(x))
        x = self.pool3d(x)
        x = self.dropout3(x)

        tc.cuda.synchronize()
        timing_dec[sample_num, 6] = time()
        x = relu(self.l1_1(x))
        tc.cuda.synchronize()
        timing_dec[sample_num, 7] = time()
        x = relu(self.l1_2(x))
        tc.cuda.synchronize()
        timing_dec[sample_num, 8] = time()
        x = relu(self.l1_3(x))
        x = self.dropout(x[:, :, 0, 0, 0])

        tc.cuda.synchronize()
        timing_dec[sample_num, 9] = time()
        x = relu(self.l2_1(x))
        tc.cuda.synchronize()
        timing_dec[sample_num, 10] = time()
        x = relu(self.l2_2(x))
        x = self.dropout(x)

        tc.cuda.synchronize()
        timing_dec[sample_num, 11] = time()
        x = softmax(self.l_out(x), dim=1)
        tc.cuda.synchronize()
        timing_dec[sample_num, 12] = time()

        return x


netDec = NetDec_timed(4, 28, 120, 160)

# %% Assessing the theoretical improvements of the compressed network compared to the original
# Number of parameters by layers
orig_parms = numParamsByLayer(net, which=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)])
comp_parms = numParamsByLayer(netDec, which=[(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14), (15, 16)])
layer_names = ["Conv 1", "Conv 2", "Linear 1", "Linear 2", "Linear 3"]

print(f"{'':-^60s}\n{' Number of parameters ':-^60s}\n{'':-^60s}\n{'Layer': ^15s}{'Original': ^15s}{'Compressed': ^15s}"
      f"{'Speed-up': ^15s}")
for i in range(len(orig_parms)):
    _, orig = orig_parms[i]
    _, comp = comp_parms[i]
    print(f"{layer_names[i]: ^15s}{orig: ^15d}{comp: ^15d}{orig / comp: ^15.2f}")
print(f"{'':-^60s}\n{'Total': ^15s}{numParams(net): ^15d}{numParams(netDec): ^15d}"
      f"{numParams(net) / numParams(netDec): ^15.2f}\n")


input_shape = (28, 120, 160)
FLOPs_orig = numFLOPsPerPush(net, input_shape, paddings=[1], pooling=[1, 2], pool_kernels=[(2, 4, 4), (2, 4, 4)])
dcmp_layer_wise = numFLOPsPerPush(netDec, input_shape, paddings=[2], pooling=[3, 6],
                                  pool_kernels=[(2, 4, 4), (2, 4, 4)])
FLOPs_dcmp = tc.tensor([tc.sum(dcmp_layer_wise[0:3]), tc.sum(dcmp_layer_wise[3:6]), tc.sum(dcmp_layer_wise[6:9]),
                        tc.sum(dcmp_layer_wise[9:11]), dcmp_layer_wise[11]])

# Calculating layer-wise speed-ups
theoretical_SP_layer = FLOPs_orig / FLOPs_dcmp
print(f"{'':-^60s}\n{' Number of FLOPs ':-^60s}\n{'':-^60s}\n{'Layer': ^15s}{'Original': ^15s}{'Compressed': ^15s}"
      f"{'Speed-up': ^15s}")
for i in range(len(FLOPs_orig)):
    print(f"{layer_names[i]: ^15s}{FLOPs_orig[i]: ^15d}{FLOPs_dcmp[i]: ^15d}{FLOPs_orig[i] / FLOPs_dcmp[i]: ^15.2f}")
print(f"{'':-^60s}\n{'Total': ^15s}{sum(FLOPs_orig): ^15d}{sum(FLOPs_dcmp): ^15d}"
      f"{sum(FLOPs_orig) / sum(FLOPs_dcmp): ^15.2f}")

# %% Timing the decomposed network

if tc.cuda.is_available():
    print("Using CUDA")
    net = net.cuda()

for i in range(SAMPLE_SIZE + BURN_IN):
    net(test, i)
full_time = timing[BURN_IN:, -1] - timing[BURN_IN:, 0]
for i in range(5, 0, -1):
    timing[:, i] -= timing[:, i - 1]

# Calculating the layer times

timing = timing[BURN_IN:, 1:]
times_m, times_s = np.mean(timing, axis=0), np.std(timing, axis=0)
full_time_m, full_time_s = np.mean(full_time), np.std(full_time)





if tc.cuda.is_available():
    netDec = netDec.cuda()

for i in range(SAMPLE_SIZE + BURN_IN):
    netDec(test, i)
full_time_dec = timing_dec[BURN_IN:, -1] - timing_dec[BURN_IN:, 0]    
for i in range(12, 0, -1):
    timing_dec[:, i] -= timing_dec[:, i - 1]

# Calculating the layer times

timing_dec = timing_dec[BURN_IN:, 1:]
times_dec_m, times_dec_s = np.mean(timing_dec, axis=0), np.std(timing_dec, axis=0)
full_time_dec_m, full_time_dec_s = np.mean(full_time_dec), np.std(full_time_dec)
# To compare
comp_c1 = np.sum(timing_dec[:, 0:3], axis=1)
comp_c2 = np.sum(timing_dec[:, 3:6], axis=1)
comp_lin1 = np.sum(timing_dec[:, 6:9], axis=1)
comp_lin2 = np.sum(timing_dec[:, 9:11], axis=1)
comp_lin3 = timing_dec[:, 11]
comp_m = np.array([np.mean(comp_c1), np.mean(comp_c2), np.mean(comp_lin1), np.mean(comp_lin2), np.mean(comp_lin3)])

# %% Theoretical speed-ups based on the number of FLOPs (floating point operations (+, -, *, %))

observed_SP_layer = times_m / comp_m

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

print("Based on {} samples and {} observations pushed forward".format(SAMPLE_SIZE, 1))
print("FLOPS orig: ", FLOPs_orig)
print("FLOPS dcmp: ", FLOPs_dcmp)
print("FLOPs dcmp 2: ", dcmp_layer_wise)
print("Full time orig {} +- {} and compressed {} +- {}".format(full_time_m * 1000, full_time_s * 1000,
                                                               full_time_dec_m * 1000, full_time_dec_s * 1000))
print("Time orig: ", times_m * 1000, times_s * 1000)
print("Layer time dcmp: ", times_dec_m * 1000, times_dec_s * 1000)
print("Layer time dcmp comp: ", comp_m * 1000, times_dec_s * 1000)
