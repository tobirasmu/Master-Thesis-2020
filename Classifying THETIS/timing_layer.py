import torch as tc
from video_networks import Net, compressNet, get_VGG16
from video_functions import numFLOPsPerPush, numParams, time_conv, time_lin
import timeit

HPC = False

NUMBER = 10
NUM_OBS = 10
# %% Full timing of the network including layer-wise
code = """
import torch as tc
from video_networks import Net
from torch.autograd import Variable
from video_functions import get_variable
net = Net(4, 28, 120, 160)
if tc.cuda.is_available():
    net = net.cuda()
test = get_variable(Variable(tc.rand((10, 4, 28, 120, 160))))
"""
fullTime = timeit.timeit("net(test)", setup=code, number=NUMBER) / (10 * NUMBER)
# Layer-wise :
layer_time = tc.tensor([time_conv(NUM_OBS, (28, 120, 160), 4, 6, (5, 11, 11), padding=(2, 5, 5)) / NUM_OBS,
                        time_conv(NUM_OBS, (14, 30, 40), 6, 16, (5, 11, 11), padding=(0, 0, 0)) / NUM_OBS,
                        time_lin(1000, 2800, 128) / 1000,
                        time_lin(1000, 128, 84) / 1000,
                        time_lin(1000, 84, 2) / 1000])
print("Full time was: {:.3} seconds, while sum of layers was {:.3} seconds".format(fullTime, tc.sum(layer_time)))

# %% Timing of the decomposed network
code = """
import torch as tc
from video_networks import Net, compressNet
from torch.autograd import Variable
from video_functions import get_variable
net = Net(4, 28, 120, 160)
HPC = """ + str(HPC) + """
# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Results_hpc/trained_network_92.pt"))
else:
    net.load_state_dict(
        tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Results_hpc/trained_network_92.pt"))
netDec = compressNet(net)
if tc.cuda.is_available():
    netDec = netDec.cuda()
test = get_variable(Variable(tc.rand((10, 4, 28, 120, 160))))
"""
fullTime_dec = timeit.timeit("netDec(test)", setup=code, number=NUMBER) / (NUM_OBS * NUMBER)
layer_time_dec = [[time_conv(NUM_OBS, (28, 120, 160), 4, 2, (1, 1, 1), padding=(0, 0, 0), bias=False) / NUM_OBS,
                   time_conv(NUM_OBS, (28, 120, 160), 2, 2, (5, 11, 11), padding=(2, 5, 5), bias=False) / NUM_OBS,
                   time_conv(NUM_OBS, (28, 120, 160), 2, 6, (1, 1, 1), padding=(0, 0, 0)) / NUM_OBS],
                  [time_conv(NUM_OBS, (14, 30, 40), 6, 2, (1, 1, 1), padding=(0, 0, 0), bias=False) / NUM_OBS,
                   time_conv(NUM_OBS, (14, 30, 40), 2, 3, (5, 11, 11), padding=(0, 0, 0), bias=False) / NUM_OBS,
                   time_conv(NUM_OBS, (14, 30, 40), 3, 16, (1, 1, 1), padding=(0, 0, 0)) / NUM_OBS],
                  [time_lin(1000, 2800, 5, bias=False) / 1000,
                   time_lin(1000, 5, 2, bias=False) / 1000,
                   time_lin(1000, 2, 128) / 1000],
                  [time_lin(1000, 128, 1, bias=False) / 1000,
                   time_lin(1000, 1, 84) / 1000, 0],
                  [time_lin(1000, 84, 2) / 1000, 0, 0]]
layer_time_dec_compare = tc.sum(tc.tensor(layer_time_dec, dtype=tc.double), dim=1)

# %% Theoretical speed-ups based on the number of FLOPs (floating point operations (+, -, *, %))
net = Net(4, 28, 120, 160)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Results_hpc/trained_network_92.pt"))
else:
    net.load_state_dict(
        tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Results_hpc/trained_network_92.pt"))
netDec = compressNet(net)

# Calculating the FLOPs for theoretical speed-up
input_shape = (28, 120, 160)
FLOPs_orig = numFLOPsPerPush(net, input_shape, paddings=[1], pooling=[1, 2], pool_kernels=[(2, 4, 4), (2, 4, 4)])
FLOPs_dcmp = numFLOPsPerPush(netDec, input_shape, paddings=[1], pooling=[3, 6], pool_kernels=[(2, 4, 4), (2, 4, 4)])
dcmp_layer_wise = tc.tensor([tc.sum(FLOPs_dcmp[0:3]), tc.sum(FLOPs_dcmp[3:6]), tc.sum(FLOPs_dcmp[6:9]),
                             tc.sum(FLOPs_dcmp[9:11]), FLOPs_dcmp[11]])
# Calculating layer-wise speed-ups
theoretical_SP_layer = FLOPs_orig / dcmp_layer_wise
observed_SP_layer = layer_time / layer_time_dec_compare

print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the networks ", ''))
print("{: <11}{: ^16}{: ^16}{: ^16}\n{:-^60s}".format("Layer", "Theoretical", "Observed", "Accounts for", ''))

layer_names = ["Conv1", "Conv2", "Lin1", "Lin2", "Lin3"]
for i in range(len(FLOPs_orig)):
    print("{: <11s}{: ^16.4f}{: ^16.4f}{: >7.5f} / {: <7.5f}".format(layer_names[i], theoretical_SP_layer[i],
                                                                     observed_SP_layer[i],
                                                                     FLOPs_orig[i] / tc.sum(FLOPs_orig),
                                                                     layer_time[i] / tc.sum(layer_time)))
print("{:-^60s}\n{: <11s}{: ^16.4f}{: ^16.4f}".format('', "Total", tc.sum(FLOPs_orig) / tc.sum(FLOPs_dcmp),
                                                      fullTime / fullTime_dec))

# %% Calculating the theoretical and actual speed-ups
vgg16 = get_VGG16()
vgg16_dec = get_VGG16(compressed=True)

FLOPs_vgg16 = numFLOPsPerPush(vgg16, (224, 224), paddings=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                              pooling=[2, 4, 7, 10, 13], pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
FLOPs_vgg16_dcmp = numFLOPsPerPush(vgg16_dec, (224, 224),
                                   paddings=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37],
                                   pooling=[5, 11, 20, 29, 38],
                                   pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
# Distribution of time used:

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
