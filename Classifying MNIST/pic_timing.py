import torch as tc
from pic_functions import time_conv, time_lin, numFLOPsPerPush, numParams
from pic_networks import get_VGG16, Net, compressNetwork
from timeit import repeat
from torch.autograd import Variable
from pic_functions import get_variable

HPC = False

NUM_OBS = 100
SAMPLE_SIZE = 1000
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((NUM_OBS, 1, 28, 28))))

# %% Timing the original network
net = Net(1, 28)
if tc.cuda.is_available():
    net = net.cuda()

fullTime = tc.tensor(repeat("net(test)", globals=locals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[BURN_IN:])

# Layer-wise timing
layer_time_m, layer_time_s = tc.zeros(5), tc.zeros(5)
layer_time_m[0], layer_time_s[0], _ = time_conv(NUM_OBS, (28, 28), 1, 6, 5, 2, bias=True, sample_size=SAMPLE_SIZE)
layer_time_m[1], layer_time_s[1], _ = time_conv(NUM_OBS, (14, 14), 6, 16, 5, 0, bias=True, sample_size=SAMPLE_SIZE)
layer_time_m[2], layer_time_s[2], _ = time_lin(NUM_OBS, 400, 120, bias=True, sample_size=SAMPLE_SIZE)
layer_time_m[3], layer_time_s[3], _ = time_lin(NUM_OBS, 120, 84, bias=True, sample_size=SAMPLE_SIZE)
layer_time_m[4], layer_time_s[4], _ = time_lin(NUM_OBS, 84, 10, bias=True, sample_size=SAMPLE_SIZE)

print("The mean time for a {:d} forward pushes was {} seconds, while the sum of the layers was {} seconds".format(
    NUM_OBS, tc.mean(fullTime), tc.sum(layer_time_m)))

# %% Timing the decomposed network
if tc.cuda.is_available():
    net = net.cpu()
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/MNIST_network.pt"))
else:
    net.load_state_dict(tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Trained "
                                "networks/MNIST_network.pt"))
netDec = compressNetwork(net)

if tc.cuda.is_available():
    netDec = netDec.cuda()
    net = net.cuda()

fullTime_dec = tc.tensor(repeat("netDec(test)", globals=locals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[BURN_IN:])

# The ranks of the decomposition
r_c1 = netDec.conv1[0].out_channels
r1_c2 = netDec.conv2[0].out_channels
r2_c2 = netDec.conv2[1].out_channels
r1_l1 = netDec.l1[0].out_features
r2_l1 = netDec.l1[1].out_features
r_l2 = netDec.l2[0].out_features

# Layer-wise timing
layer_time_dec_m, layer_time_dec_s = tc.zeros(11), tc.zeros(11)
layer_time_dec_m[0], layer_time_dec_s[0], _ = time_conv(NUM_OBS, (28, 28), 1, r_c1, 5, 2, bias=False,
                                                        sample_size=SAMPLE_SIZE)
layer_time_dec_m[1], layer_time_dec_s[1], _ = time_conv(NUM_OBS, (28, 28), r_c1, 6, 5, 0, bias=True,
                                                        sample_size=SAMPLE_SIZE)
layer_time_dec_m[2], layer_time_dec_s[2], _ = time_conv(NUM_OBS, (14, 14), 6, r1_c2, 1, 0, bias=False,
                                                        sample_size=SAMPLE_SIZE)
layer_time_dec_m[3], layer_time_dec_s[3], _ = time_conv(NUM_OBS, (14, 14), r1_c2, r2_c2, 5, 0, bias=False,
                                                        sample_size=SAMPLE_SIZE)
layer_time_dec_m[4], layer_time_dec_s[4], _ = time_conv(NUM_OBS, (10, 10), r2_c2, 16, 1, 0, bias=True,
                                                        sample_size=SAMPLE_SIZE)
layer_time_dec_m[5], layer_time_dec_s[5], _ = time_lin(NUM_OBS, 400, r1_l1, bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[6], layer_time_dec_s[6], _ = time_lin(NUM_OBS, r1_l1, r2_l1, bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[7], layer_time_dec_s[7], _ = time_lin(NUM_OBS, r2_l1, 120, bias=True, sample_size=SAMPLE_SIZE)
layer_time_dec_m[8], layer_time_dec_s[8], _ = time_lin(NUM_OBS, 120, r_l2, bias=False, sample_size=SAMPLE_SIZE)
layer_time_dec_m[9], layer_time_dec_s[9], _ = time_lin(NUM_OBS, r_l2, 84, bias=True, sample_size=SAMPLE_SIZE)
layer_time_dec_m[10], layer_time_dec_s[10], _ = time_lin(NUM_OBS, 84, 10, bias=True, sample_size=SAMPLE_SIZE)

print("The mean time for 100 forward pushes was {} seconds, while the sum of the layers was {} seconds".format(
    tc.mean(fullTime_dec), tc.sum(layer_time_dec_m)))

layer_time_dec_comp = tc.tensor([tc.sum(layer_time_dec_m[0:2]), tc.sum(layer_time_dec_m[2:5]),
                                 tc.sum(layer_time_dec_m[5:8]), tc.sum(layer_time_dec_m[8:10]), layer_time_dec_m[10]])
# %% Calculating the theoretical speed-ups
input_shape = (28, 28)
FLOPs_orig = numFLOPsPerPush(net, input_shape, paddings=[1], pooling=[1, 2], pool_kernels=[(2, 2), (2, 2)])
FLOPs_dcmp = numFLOPsPerPush(netDec, input_shape, paddings=[1], pooling=[3, 6], pool_kernels=[(2, 2), (2, 2)])
dcmp_layer_wise = tc.tensor([tc.sum(FLOPs_dcmp[0:2]), tc.sum(FLOPs_dcmp[2:5]), tc.sum(FLOPs_dcmp[5:8]),
                             tc.sum(FLOPs_dcmp[8:10]), FLOPs_dcmp[10]])

# Calculating layer-wise speed-ups
theoretical_SP_layer = FLOPs_orig / dcmp_layer_wise
observed_SP_layer = layer_time_m / layer_time_dec_comp

print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the networks ", ''))
print("{: <11}{: ^16}{: ^16}{: ^16}\n{:-^60s}".format("Layer", "Theoretical", "Observed", "Accounts for", ''))

layer_names = ["Conv1", "Conv2", "Lin1", "Lin2", "Lin3"]
for i in range(len(FLOPs_orig)):
    print("{: <11s}{: ^16.4f}{: ^16.4f}{: >7.5f} / {: <7.5f}".format(layer_names[i], theoretical_SP_layer[i],
                                                                     observed_SP_layer[i],
                                                                     FLOPs_orig[i] / tc.sum(FLOPs_orig),
                                                                     layer_time_m[i] / tc.sum(layer_time_m)))
print("{:-^60s}\n{: <11s}{: ^16.4f}{: ^16.4f}".format('', "Total", tc.sum(FLOPs_orig) / tc.sum(FLOPs_dcmp),
                                                      tc.mean(fullTime) / tc.mean(fullTime_dec)))
print("\nThe number of parameters:\nOriginal: {:d}    Compressed:  {:d}    Ratio:  {:.3f}".format(numParams(net),
                                                                                                  numParams(netDec),
                                                                                                  numParams(netDec) /
                                                                                                  numParams(net)))

# %% Investigating whether it works for bigger architectures
print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the VGG-16 network ", ''))

# Loading the networks and compressing using the algorithm
vgg16 = get_VGG16()
vgg16_dec = get_VGG16(compressed=True)

if tc.cuda.is_available():
    vgg16 = vgg16.cuda()
    vgg16_dec = vgg16_dec.cuda()
    print(" -- Using GPU -- ")

NUM_OBS_VGG = 10
SAMPLE_SIZE_VGG = 10
BURN_IN_VGG = SAMPLE_SIZE // 10

test = get_variable(Variable(tc.rand((NUM_OBS_VGG, 3, 224, 224))))
# Timing VGG16 - both original and compressed
fullTime_VGG16 = tc.tensor(repeat('vgg16(test)', globals=locals(), number=1, repeat=(SAMPLE_SIZE_VGG +
                                                                                     BURN_IN_VGG))[BURN_IN_VGG:])
fullTime_VGG16_dec = tc.tensor(repeat('vgg16_dec(test)', globals=locals(), number=1, repeat=(SAMPLE_SIZE_VGG +
                                                                                             BURN_IN_VGG))[BURN_IN_VGG:])

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
print("Theoretical speed-up is {:.4f} while the observed speed-up is {:.2f}".format(
    sum(FLOPs_vgg16) / sum(FLOPs_vgg16_dcmp), tc.mean(fullTime_VGG16) / tc.mean(fullTime_VGG16_dec)))

# %% Observed layer-wise speed-up

# For the original
layer_time_vgg16_m, layer_time_vgg16_s = tc.zeros(16), tc.zeros(16)
layer_time_vgg16_m[0], layer_time_vgg16_s[0], _ = time_conv(NUM_OBS, (224, 224), 3, 64, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[1], layer_time_vgg16_s[1], _ = time_conv(NUM_OBS, (224, 224), 64, 64, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[2], layer_time_vgg16_s[2], _ = time_conv(NUM_OBS, (112, 112), 64, 128, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[3], layer_time_vgg16_s[3], _ = time_conv(NUM_OBS, (112, 112), 128, 128, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[4], layer_time_vgg16_s[4], _ = time_conv(NUM_OBS, (56, 56), 128, 256, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[5], layer_time_vgg16_s[5], _ = time_conv(NUM_OBS, (56, 56), 256, 256, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[6], layer_time_vgg16_s[6], _ = time_conv(NUM_OBS, (56, 56), 256, 256, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[7], layer_time_vgg16_s[7], _ = time_conv(NUM_OBS, (28, 28), 256, 512, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[8], layer_time_vgg16_s[8], _ = time_conv(NUM_OBS, (28, 28), 512, 512, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[9], layer_time_vgg16_s[9], _ = time_conv(NUM_OBS, (28, 28), 512, 512, 3, 1, bias=True,
                                                            sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[10], layer_time_vgg16_s[10], _ = time_conv(NUM_OBS, (14, 14), 512, 512, 3, 1, bias=True,
                                                              sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[11], layer_time_vgg16_s[11], _ = time_conv(NUM_OBS, (14, 14), 512, 512, 3, 1, bias=True,
                                                              sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[12], layer_time_vgg16_s[12], _ = time_conv(NUM_OBS, (14, 14), 512, 512, 3, 1, bias=True,
                                                              sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[13], layer_time_vgg16_s[13], _ = time_lin(NUM_OBS, 25088, 4096, bias=True,
                                                             sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[14], layer_time_vgg16_s[14], _ = time_lin(NUM_OBS, 4096, 4096, bias=True,
                                                             sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_m[15], layer_time_vgg16_s[15], _ = time_lin(NUM_OBS, 4096, 1000, bias=True,
                                                             sample_size=SAMPLE_SIZE_VGG)

# For the decomposed version
ranks_vgg16 = []
for i in range(len(vgg16.features)):
    if type(vgg16.features[i]) is tc.nn.modules.conv.Conv2d:
        decomposition = vgg16_dec.features[i]
        if len(decomposition) == 3:
            ranks_vgg16.append((decomposition[0].out_channels, decomposition[1].out_channels))
        else:
            ranks_vgg16.append(decomposition[0].out_channels)
for i in range(len(vgg16.classifier)):
    if type(vgg16.classifier[i]) is tc.nn.modules.linear.Linear:
        decomposition = vgg16_dec.classifier[i]
        if len(decomposition) == 3:
            ranks_vgg16.append((decomposition[0].out_features, decomposition[1].out_features))
        else:
            ranks_vgg16.append(decomposition[0].out_features)

layer_time_vgg16_dec_m, layer_time_vgg16_dec_s = tc.zeros(45), tc.zeros(45)
r = ranks_vgg16[0]
layer_time_vgg16_dec_m[0], layer_time_vgg16_dec_s[0], _ = time_conv(NUM_OBS_VGG, (224, 224), 3, r, 3, 1, bias=False,
                                                                    sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[1], layer_time_vgg16_dec_s[1], _ = time_conv(NUM_OBS_VGG, (224, 224), r, 64, 1, 0, bias=True,
                                                                    sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[1]
layer_time_vgg16_dec_m[2], layer_time_vgg16_dec_s[2], _ = time_conv(NUM_OBS_VGG, (224, 224), 64, r_1, 1, 0, bias=False,
                                                                    sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[3], layer_time_vgg16_dec_s[3], _ = time_conv(NUM_OBS_VGG, (224, 224), r_1, r_2, 3, 1, bias=False,
                                                                    sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[4], layer_time_vgg16_dec_s[4], _ = time_conv(NUM_OBS_VGG, (224, 224), r_2, 64, 1, 0, bias=True,
                                                                    sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[2]
layer_time_vgg16_dec_m[5], layer_time_vgg16_dec_s[5], _ = time_conv(NUM_OBS_VGG, (112, 112), 64, r_1, 1, 0, bias=False,
                                                                    sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[6], layer_time_vgg16_dec_s[6], _ = time_conv(NUM_OBS_VGG, (112, 112), r_1, r_2, 3, 1, bias=False,
                                                                    sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[7], layer_time_vgg16_dec_s[7], _ = time_conv(NUM_OBS_VGG, (112, 112), r_2, 128, 1, 0, bias=True,
                                                                    sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[3]
layer_time_vgg16_dec_m[8], layer_time_vgg16_dec_s[8], _ = time_conv(NUM_OBS_VGG, (112, 112), 128, r_1, 1, 0, bias=False,
                                                                    sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[9], layer_time_vgg16_dec_s[9], _ = time_conv(NUM_OBS_VGG, (112, 112), r_1, r_2, 3, 1, bias=False,
                                                                    sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[10], layer_time_vgg16_dec_s[10], _ = time_conv(NUM_OBS_VGG, (112, 112), r_2, 128, 1, 0,
                                                                      bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[4]
layer_time_vgg16_dec_m[11], layer_time_vgg16_dec_s[11], _ = time_conv(NUM_OBS_VGG, (56, 56), 128, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[12], layer_time_vgg16_dec_s[12], _ = time_conv(NUM_OBS_VGG, (56, 56), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[13], layer_time_vgg16_dec_s[13], _ = time_conv(NUM_OBS_VGG, (56, 56), r_2, 256, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[5]
layer_time_vgg16_dec_m[14], layer_time_vgg16_dec_s[14], _ = time_conv(NUM_OBS_VGG, (56, 56), 256, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[15], layer_time_vgg16_dec_s[15], _ = time_conv(NUM_OBS_VGG, (56, 56), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[16], layer_time_vgg16_dec_s[16], _ = time_conv(NUM_OBS_VGG, (56, 56), r_2, 256, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[6]
layer_time_vgg16_dec_m[17], layer_time_vgg16_dec_s[17], _ = time_conv(NUM_OBS_VGG, (56, 56), 256, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[18], layer_time_vgg16_dec_s[18], _ = time_conv(NUM_OBS_VGG, (56, 56), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[19], layer_time_vgg16_dec_s[19], _ = time_conv(NUM_OBS_VGG, (56, 56), r_2, 256, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[7]
layer_time_vgg16_dec_m[20], layer_time_vgg16_dec_s[20], _ = time_conv(NUM_OBS_VGG, (28, 28), 256, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[21], layer_time_vgg16_dec_s[21], _ = time_conv(NUM_OBS_VGG, (28, 28), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[22], layer_time_vgg16_dec_s[22], _ = time_conv(NUM_OBS_VGG, (28, 28), r_2, 512, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[8]
layer_time_vgg16_dec_m[23], layer_time_vgg16_dec_s[23], _ = time_conv(NUM_OBS_VGG, (28, 28), 512, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[24], layer_time_vgg16_dec_s[24], _ = time_conv(NUM_OBS_VGG, (28, 28), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[25], layer_time_vgg16_dec_s[25], _ = time_conv(NUM_OBS_VGG, (28, 28), r_2, 512, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[9]
layer_time_vgg16_dec_m[26], layer_time_vgg16_dec_s[26], _ = time_conv(NUM_OBS_VGG, (28, 28), 512, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[27], layer_time_vgg16_dec_s[27], _ = time_conv(NUM_OBS_VGG, (28, 28), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[28], layer_time_vgg16_dec_s[28], _ = time_conv(NUM_OBS_VGG, (28, 28), r_2, 512, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[10]
layer_time_vgg16_dec_m[29], layer_time_vgg16_dec_s[29], _ = time_conv(NUM_OBS_VGG, (14, 14), 512, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[30], layer_time_vgg16_dec_s[30], _ = time_conv(NUM_OBS_VGG, (14, 14), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[31], layer_time_vgg16_dec_s[31], _ = time_conv(NUM_OBS_VGG, (14, 14), r_2, 512, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[11]
layer_time_vgg16_dec_m[32], layer_time_vgg16_dec_s[32], _ = time_conv(NUM_OBS_VGG, (14, 14), 512, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[33], layer_time_vgg16_dec_s[33], _ = time_conv(NUM_OBS_VGG, (14, 14), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[34], layer_time_vgg16_dec_s[34], _ = time_conv(NUM_OBS_VGG, (14, 14), r_2, 512, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[12]
layer_time_vgg16_dec_m[35], layer_time_vgg16_dec_s[35], _ = time_conv(NUM_OBS_VGG, (14, 14), 512, r_1, 1, 0, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[36], layer_time_vgg16_dec_s[36], _ = time_conv(NUM_OBS_VGG, (14, 14), r_1, r_2, 3, 1, bias=False,
                                                                      sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[37], layer_time_vgg16_dec_s[37], _ = time_conv(NUM_OBS_VGG, (14, 14), r_2, 512, 1, 0, bias=True,
                                                                      sample_size=SAMPLE_SIZE_VGG)
r_1, r_2 = ranks_vgg16[13]
layer_time_vgg16_dec_m[38], layer_time_vgg16_dec_s[38], _ = time_lin(NUM_OBS_VGG, 25088, r_1, bias=False,
                                                                     sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[39], layer_time_vgg16_dec_s[39], _ = time_lin(NUM_OBS_VGG, r_1, r_2, bias=False,
                                                                     sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[40], layer_time_vgg16_dec_s[40], _ = time_lin(NUM_OBS_VGG, r_2, 4096, bias=True,
                                                                     sample_size=SAMPLE_SIZE_VGG)
r = ranks_vgg16[14]
layer_time_vgg16_dec_m[41], layer_time_vgg16_dec_s[41], _ = time_lin(NUM_OBS_VGG, 4096, r, bias=False,
                                                                     sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[42], layer_time_vgg16_dec_s[42], _ = time_lin(NUM_OBS_VGG, r, 4096, bias=True,
                                                                     sample_size=SAMPLE_SIZE_VGG)
r = ranks_vgg16[15]
layer_time_vgg16_dec_m[43], layer_time_vgg16_dec_s[43], _ = time_lin(NUM_OBS_VGG, 4096, r, bias=False,
                                                                     sample_size=SAMPLE_SIZE_VGG)
layer_time_vgg16_dec_m[44], layer_time_vgg16_dec_s[44], _ = time_lin(NUM_OBS_VGG, r, 1000, bias=True,
                                                                     sample_size=SAMPLE_SIZE_VGG)

layer_time_vgg16_dec_comp = tc.tensor([tc.sum(layer_time_vgg16_dec_m[0:2]), tc.sum(layer_time_vgg16_dec_m[2:5]),
                                       tc.sum(layer_time_vgg16_dec_m[5:8]), tc.sum(layer_time_vgg16_dec_m[8:11]),
                                       tc.sum(layer_time_vgg16_dec_m[11:14]), tc.sum(layer_time_vgg16_dec_m[14:17]),
                                       tc.sum(layer_time_vgg16_dec_m[17:20]), tc.sum(layer_time_vgg16_dec_m[20:23]),
                                       tc.sum(layer_time_vgg16_dec_m[23:26]), tc.sum(layer_time_vgg16_dec_m[26:29]),
                                       tc.sum(layer_time_vgg16_dec_m[29:32]), tc.sum(layer_time_vgg16_dec_m[32:35]),
                                       tc.sum(layer_time_vgg16_dec_m[35:38]), tc.sum(layer_time_vgg16_dec_m[38:41]),
                                       tc.sum(layer_time_vgg16_dec_m[41:43]), tc.sum(layer_time_vgg16_dec_m[43:44])])

theoretical_SP_layer_vgg16 = FLOPs_vgg16 / FLOPs_vgg16_comp
observed_SP_layer_vgg16 = layer_time_vgg16_m / layer_time_vgg16_dec_comp

# Printing the result layer-wise
layer_names_vgg16 = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "conv10",
                     "conv11", "conv12", "conv13", "Lin1", "Lin2", "Lin3"]
for i in range(len(FLOPs_vgg16)):
    print("{: <11s}{: ^16.4f}{: ^16.4f}{: >7.5f} / {: <7.5f}".format(layer_names_vgg16[i], theoretical_SP_layer_vgg16[i],
                                                                     observed_SP_layer_vgg16[i],
                                                                     FLOPs_vgg16[i] / tc.sum(FLOPs_vgg16),
                                                                     layer_time_vgg16_m[i] / tc.sum(layer_time_vgg16_m)))
print("{:-^60s}\n{: <11s}{: ^16.4f}{: ^16.4f}".format('', "Total", tc.sum(FLOPs_vgg16) / tc.sum(FLOPs_vgg16_dcmp),
                                                      tc.mean(fullTime_VGG16) / tc.mean(fullTime_VGG16_dec)))
print("\nThe number of parameters:\nOriginal: {:d}    Compressed:  {:d}    Ratio:  {:.3f}".format(numParams(vgg16),
                                                                                                  numParams(vgg16_dec),
                                                                                                  numParams(vgg16_dec) /
                                                                                                  numParams(vgg16)))