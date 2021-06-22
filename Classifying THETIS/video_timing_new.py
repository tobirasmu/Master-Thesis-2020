"""
    Timing the networks both fully and layer-wise. The time is reported as the mean and standard deviation of SAMPLE_SIZE
    number of pushes. Before this a BURN_IN number of pushes is carried out and discarded.
"""

HPC = False

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/home/tenra/PycharmProjects/Master-Thesis-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
from tools.decomp import compressNet
from tools.models import numFLOPsPerPush, numParamsByLayer, Net2, numParams
from tools.trainer import get_variable
from torch.autograd import Variable

SAMPLE_SIZE = 100
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((1, 4, 28, 120, 160))))


# %% Define the networks
net = Net2(4, 28, 120, 160)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/THETIS_new.pt"))
else:
    net.load_state_dict(
        tc.load("/home/tenra/PycharmProjects/Master-Thesis-2020/Trained networks/THETIS_new.pt"))
netDec = compressNet(net)
netDec.eval()
net.eval()

# %% Timing the compressed network using the pytorch profiler
with tc.autograd.profiler.profile() as prof:
    for _ in range(SAMPLE_SIZE + BURN_IN):
        netDec(test)

# Taking out the individual times in order to make statistics:
dcmp_times = tc.empty((SAMPLE_SIZE + BURN_IN, 12))

i, j = 0, 0
for event in prof.function_events:
    if event.name == "aten::conv3d" or event.name == "aten::linear":
        dcmp_times[i, j] = event.cpu_time_total
        if j == 11:
            j = 0
            i += 1
        else:
            j += 1

dcmp_times = dcmp_times[BURN_IN:]
dcmp_means = tc.mean(dcmp_times, dim=0) / 1000
print(tc.mean(dcmp_times, dim=0) / 1000)

# %% Timing the original network using the pytorch profiler
with tc.autograd.profiler.profile() as prof:
    for _ in range(SAMPLE_SIZE + BURN_IN):
        net(test)

# Taking out the individual times in order to make statistics:
orig_times = tc.empty((SAMPLE_SIZE + BURN_IN, 5))

i, j = 0, 0
for event in prof.function_events:
    if event.name == "aten::conv3d" or event.name == "aten::linear":
        orig_times[i, j] = event.cpu_time_total
        if j == 4:
            j = 0
            i += 1
        else:
            j += 1

orig_times = orig_times[BURN_IN:]
orig_means = tc.mean(orig_times, dim=0) / 1000


# %% Assessing the theoretical speed-ups
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

# %% Printing the observed speed-ups
layer_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]]
layer_names = ["Conv 1", "Conv 2", "Linear 1", "Linear 2", "Linear 3"]
print(f"{'':-^100}\n{'  Mean time  ':-^100s}\n{'':-^100}")
print(f"\n{'Layer': ^20s}{'Original': ^20s}{'Compressed': ^40s}{'Speed-up': ^20s}\n{'':-^100s}")
for i, orig in enumerate(orig_means):
    dcmp_comp = tc.sum(dcmp_means[layer_groups[i]])
    sub = 0
    if len(layer_groups[i]) == 3:
        print(f"{'': ^60s}{dcmp_means[layer_groups[i][sub]]: ^20.4f}")
        sub += 1
    print(f"{layer_names[i]: ^20s}{orig: ^20.4f}{dcmp_comp: ^20.4f}"
          f"{dcmp_means[layer_groups[i][sub]]: ^20.4f}{orig / dcmp_comp: ^20.4f}")
    sub += 1
    if len(layer_groups[i]) > 1:
        print(f"{'': ^60s}{dcmp_means[layer_groups[i][sub]]: ^20.4f}")
    print(f"{'':-^100s}")

orig_sds = tc.std(orig_times / 1000, dim=0)
dcmp_sds = tc.std(dcmp_times / 1000, dim=0)
print(f"\n\n{'':-^100}\n{'  Standard deviation  ':-^100s}\n{'':-^100}")
print(f"\n{'Layer': ^25s}{'Original': ^25s}{'Compressed': ^50s}\n{'':-^100s}")
for i, orig in enumerate(orig_sds):
    dcmp_comp = tc.sqrt(tc.sum(dcmp_sds[layer_groups[i]] ** 2))
    sub = 0
    if len(layer_groups[i]) == 3:
        print(f"{'': ^75s}{dcmp_sds[layer_groups[i][sub]]: ^25.4f}")
        sub += 1
    print(f"{layer_names[i]: ^25s}{orig: ^25.4f}{dcmp_comp: ^25.4f}"
          f"{dcmp_sds[layer_groups[i][sub]]: ^25.4f}")
    sub += 1
    if len(layer_groups[i]) > 1:
        print(f"{'': ^75s}{dcmp_sds[layer_groups[i][sub]]: ^25.4f}")
    print(f"{'':-^100s}")