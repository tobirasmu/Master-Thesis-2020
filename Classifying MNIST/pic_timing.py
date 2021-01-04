import cv2
import torch as tc
from pic_functions import time_conv, time_lin, numFLOPsPerPush
from pic_networks import get_VGG16, Net, compressNetwork
from timeit import repeat
from torch.autograd import Variable
from pic_functions import get_variable

HPC = False

NUM_OBS = 1000
SAMPLE_SIZE = 100
test = get_variable(Variable(tc.rand((NUM_OBS, 1, 28, 28))))

# %% Timing the original network
net = Net(1, 28)
if tc.cuda.is_available():
    net = net.cuda()

fullTime = tc.tensor(repeat("net(test)", globals=locals(), number=1, repeat=SAMPLE_SIZE))

# Layer-wise timing
layer_time_m, layer_time_s = tc.zeros(5), tc.zeros(5)
layer_time_m[0], layer_time_s[0], _ = time_conv(NUM_OBS, (28, 28), 1, 6, 5, 2, bias=True, number=SAMPLE_SIZE)
layer_time_m[1], layer_time_s[1], _ = time_conv(NUM_OBS, (14, 14), 6, 16, 5, 0, bias=True, number=SAMPLE_SIZE)
layer_time_m[2], layer_time_s[2], _ = time_lin(NUM_OBS, 400, 120, bias=True, number=SAMPLE_SIZE)
layer_time_m[3], layer_time_s[3], _ = time_lin(NUM_OBS, 120, 84, bias=True, number=SAMPLE_SIZE)
layer_time_m[4], layer_time_s[4], _ = time_lin(NUM_OBS, 84, 10, bias=True, number=SAMPLE_SIZE)

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

fullTime_dec = tc.tensor(repeat("netDec(test)", globals=locals(), number=1, repeat=SAMPLE_SIZE))

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
                                                        number=SAMPLE_SIZE)
layer_time_dec_m[1], layer_time_dec_s[1], _ = time_conv(NUM_OBS, (28, 28), r_c1, 6, 5, 0, bias=True,
                                                        number=SAMPLE_SIZE)
layer_time_dec_m[2], layer_time_dec_s[2], _ = time_conv(NUM_OBS, (14, 14), 6, r1_c2, 1, 0, bias=False,
                                                        number=SAMPLE_SIZE)
layer_time_dec_m[3], layer_time_dec_s[3], _ = time_conv(NUM_OBS, (14, 14), r1_c2, r2_c2, 5, 0, bias=False,
                                                        number=SAMPLE_SIZE)
layer_time_dec_m[4], layer_time_dec_s[4], _ = time_conv(NUM_OBS, (10, 10), r2_c2, 16, 1, 0, bias=True,
                                                        number=SAMPLE_SIZE)
layer_time_dec_m[5], layer_time_dec_s[5], _ = time_lin(NUM_OBS, 400, r1_l1, bias=False, number=SAMPLE_SIZE)
layer_time_dec_m[6], layer_time_dec_s[6], _ = time_lin(NUM_OBS, r1_l1, r2_l1, bias=False, number=SAMPLE_SIZE)
layer_time_dec_m[7], layer_time_dec_s[7], _ = time_lin(NUM_OBS, r2_l1, 120, bias=True, number=SAMPLE_SIZE)
layer_time_dec_m[8], layer_time_dec_s[8], _ = time_lin(NUM_OBS, 120, r_l2, bias=False, number=SAMPLE_SIZE)
layer_time_dec_m[9], layer_time_dec_s[9], _ = time_lin(NUM_OBS, r_l2, 84, bias=True, number=SAMPLE_SIZE)
layer_time_dec_m[10], layer_time_dec_s[10], _ = time_lin(NUM_OBS, 84, 10, bias=True, number=SAMPLE_SIZE)

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


# %% Investigating whether it works for bigger architectures
print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the VGG-16 network ", ''))
# Loading the networks and compressing using the algorithm
vgg16 = get_VGG16()
vgg16_dec = get_VGG16(compressed=True)

# Timing




# Calculating the theoretical speed-up using FLOPs
FLOPs_vgg16 = numFLOPsPerPush(vgg16, (224, 224), paddings=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                              pooling=[2, 4, 7, 10, 13], pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
FLOPs_vgg16_dcmp = numFLOPsPerPush(vgg16_dec, (224, 224),
                                   paddings=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37],
                                   pooling=[5, 11, 20, 29, 38],
                                   pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])

# Overall theoretical speed-up
print("Overall ratio is {:.4f} hence the speed-up should be of around {:.2f} times".format(
    sum(FLOPs_vgg16_dcmp) / sum(FLOPs_vgg16), sum(FLOPs_vgg16) / sum(FLOPs_vgg16_dcmp)))

# Actual speed-up
test_cat = tc.tensor(np.moveaxis(cv2.cvtColor(cv2.imread(directory + "cat.png"), cv2.COLOR_BGR2RGB), -1, 0),
                     dtype=tc.float).unsqueeze(0) / 255
test_ball = tc.tensor(np.moveaxis(cv2.cvtColor(cv2.imread(directory + "ball.png"), cv2.COLOR_BGR2RGB), -1, 0),
                      dtype=tc.float).unsqueeze(0) / 255
test_vgg16 = Variable(get_variable(tc.cat((test_cat, test_ball), 0)))

t = process_time()
for i in range(1000):
    vgg16(test_vgg16)
timeOrig = process_time() - t

t = process_time()
for i in range(1000):
    vgg16_dec(test_vgg16)
timeNew = process_time() - t
print("Actual speed-up was {} times based on {} seconds for the original and {} for the decomposed.".format(
    timeOrig / timeNew, timeOrig / 2000, timeNew / 2000))

# VGG 16 network header and number of params
print("\n{:-^60}\n{:-^60}\n{:-^60}".format('', " The VGG-16 network ", ''))
# Converting to cuda if possible
if tc.cuda.is_available():
    vgg16 = vgg16.cuda()
    vgg16_dec = vgg16_dec.cuda()
    print("\n-- Using GPU for VGG-16 --\n")

print(
    "Number of parameters:\n{: <30s}{: >12d}\n{: <30s}{: >12d}\n{: <30s}{: >12f}".format("Original:", numParams(vgg16),
                                                                                         "Decomposed:",
                                                                                         numParams(vgg16_dec), "Ratio:",
                                                                                         numParams(
                                                                                             vgg16_dec) / numParams(
                                                                                             vgg16)))
