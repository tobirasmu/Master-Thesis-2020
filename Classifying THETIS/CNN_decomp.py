# True if using the
HPC = False

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
import cv2
import numpy as np
import torch.nn as nn
from torch.nn import Linear, Conv3d, MaxPool3d
from torch.nn.functional import relu, softmax
from torch.autograd import Variable
import torchvision.models as models
import tensorly as tl
from tensorly.decomposition import partial_tucker
from video_functions import showFrame, loadTHETIS, conv_to_tucker1_3d, conv_to_tucker2_3d, conv_to_tucker1, \
    conv_to_tucker2, lin_to_tucker1, lin_to_tucker2, numParams, get_variable, train_epoch, eval_epoch, plotAccs, \
    numFLOPsPerPush, numFLOPsPerPush_mul
import torch.optim as optim
from sklearn.model_selection import KFold
from copy import deepcopy
from time import process_time_ns, process_time

tl.set_backend('pytorch')

# %% Loading the data
data_loaded = True

LENGTH = 1.5
RESOLUTION = 0.25

t = process_time()
directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/Users/Tobias/Desktop/Data/"
if data_loaded:
    X, Y = tc.load(directory + "data.pt")
else:
    if HPC:
        inputForehand = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/forehand_filenames_adapted.csv"
        inputBackhand = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/backhand_filenames_adapted.csv"
        X, Y = loadTHETIS((0, 1), (inputForehand, inputBackhand), ([10, 45], [0]), directory,
                          out_directory=directory + "data.pt", length=LENGTH, resolution=RESOLUTION)
    else:
        # Forehands
        inputForehand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying " \
                        "THETIS/forehand_filenames_adapted.csv "
        # Backhands
        inputBackhand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying " \
                        "THETIS/backhand_filenames_adapted.csv "
        X, Y = loadTHETIS((0, 1), (inputForehand, inputBackhand), ([10, 45], [0]), directory,
                          out_directory="/Users/Tobias/Desktop/Data/data.pt",
                          length=LENGTH, resolution=RESOLUTION)

print("Took {:.2f} seconds to load the data".format(process_time() - t))

# %% The network that we are working with:
_, channels, frames, height, width = X.shape


def conv_dims(dims, kernels, strides, paddings):
    dimensions = len(dims)
    new_dims = tc.empty(dimensions)
    for i in range(dimensions):
        new_dims[i] = int((dims[i] - kernels[i] + 2 * paddings[i]) / strides[i] + 1)
    return new_dims


# First convolution
c1_channels = (channels, 6)
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


# The CNN for the THETIS dataset
class Net_timed(nn.Module):

    def __init__(self):
        super(Net_timed, self).__init__()

        # Adding the convolutional layers
        self.c1 = Conv3d(in_channels=c1_channels[0], out_channels=c1_channels[1], kernel_size=c1_kernel,
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

    def forward(self, x, timed=False):
        layer_time = []
        t = process_time()
        x = relu(self.c1(x))
        layer_time.append(process_time() - t)

        x = self.pool3d(x)

        t = process_time()
        x = relu(self.c2(x))
        layer_time.append(process_time() - t)

        x = self.pool3d(x)

        x = tc.flatten(x, 1)

        t = process_time()
        x = relu(self.l1(x))
        layer_time.append(process_time() - t)

        t = process_time()
        x = relu(self.l2(x))
        layer_time.append(process_time() - t)

        t = process_time()
        output = softmax(self.l_out(x), dim=1)
        layer_time.append(process_time() - t)
        if timed:
            return output, tc.tensor(layer_time)
        return output


# Initializing the CNN
net = Net_timed()

# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Results_hpc/trained_network_92.pt"))
else:
    net.load_state_dict(
        tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Results_hpc/trained_network_92.pt"))

# %% The decomposition functions:
netDec = deepcopy(net)
netDec.c1 = conv_to_tucker2_3d(net.c1)
netDec.c2 = conv_to_tucker2_3d(net.c2)
netDec.l1 = lin_to_tucker2(net.l1, ranks=[2, 5])
netDec.l2 = lin_to_tucker1(net.l2)
print("The decomposed network has the following architecture\n", netDec)
print("Parameters:\nOriginal: {}  Decomposed: {}  Ratio: {:.3f}\n".format(numParams(net), numParams(netDec),
                                                                          numParams(netDec) / numParams(net)))

if tc.cuda.is_available():
    net = net.cuda()
    print("-- USING GPU --")
    netDec = netDec.cuda()

# %% Training the decomposed network
BATCH_SIZE = 10
NUM_FOLDS = 5
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
nTrain = int(0.85 * X.shape[0])


def train(this_net, X_train, y_train, X_test, y_test):
    optimizer = optim.SGD(this_net.parameters(), lr=LEARNING_RATE, momentum=0.5, weight_decay=0.01)
    train_accs, val_accs, test_accs = tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS)
    kf = list(KFold(NUM_FOLDS).split(X_train))
    epoch, interrupted = 0, False
    while epoch < NUM_EPOCHS:
        epoch += 1
        print("{:-^60s}".format(" EPOCH {:3d} ".format(epoch)))
        fold_loss = tc.empty(NUM_FOLDS)
        fold_train_accs = tc.empty(NUM_FOLDS)
        fold_val_accs = tc.empty(NUM_FOLDS)
        for i, (train_inds, val_inds) in enumerate(kf):
            try:
                fold_loss[i], fold_train_accs[i] = train_epoch(this_net, X_train[train_inds], y_train[train_inds],
                                                               optimizer=optimizer, batch_size=BATCH_SIZE)
                fold_val_accs[i] = eval_epoch(this_net, X_train[val_inds], y_train[val_inds])
            except KeyboardInterrupt:
                print('\nKeyboardInterrupt')
                interrupted = True
                break
        if interrupted is True:
            break
        this_loss, this_train_acc, this_val_acc = tc.mean(fold_loss), tc.mean(fold_train_accs), tc.mean(fold_val_accs)
        train_accs[epoch - 1], val_accs[epoch - 1] = this_train_acc, this_val_acc
        # Doing the testing evaluation
        test_accs[epoch - 1] = eval_epoch(this_net, X_test, y_test)
        print("{: ^15}{: ^15}{: ^15}{: ^15}".format("Loss:", "Train acc.:", "Val acc.:", "Test acc.:"))
        print("{: ^15.4f}{: ^15.4f}{: ^15.4f}{: ^15.4f}".format(this_loss, this_train_acc, this_val_acc,
                                                                test_accs[epoch - 1]))
    saveAt = "/zhome/2a/c/108156/Outputs/accuracies_decomp.png" if HPC else "/Users/Tobias/Desktop/accuracies_decomp.png"
    plotAccs(train_accs, val_accs, saveName=saveAt)
    print("{:-^60}\nFinished".format(""))


print("{:-^60s}".format(" Training details "))
print("{: ^20}{: ^20}{: ^20}".format("Learning rate:", "Batch size:", "Number of folds"))
print("{: ^20.4f}{: ^20d}{: ^20d}\n{:-^60}".format(LEARNING_RATE, BATCH_SIZE, NUM_FOLDS, ''))
train(netDec, X[:nTrain], Y[:nTrain], X[nTrain:], Y[nTrain:])

# %% Observed time both total and layer-wise
num_samples, num_runs = 10, 1000

# Original network
net.eval()
timeTest = Variable(get_variable(X[0:num_samples]))
t = process_time()
layer_time_orig = tc.zeros(5)
for i in range(num_runs):
    _, this_time = net(timeTest, timed=True)
    layer_time_orig += this_time
timeOrig = (process_time() - t) / (num_runs * num_samples)
layer_time_orig /= (num_runs * num_samples)

# Decomposed network
netDec.eval()
t = process_time()
layer_time_dcmp = tc.zeros(5)
for i in range(num_runs):
    _, this_time = netDec(timeTest, timed=True)
    layer_time_dcmp += this_time
timeNew = (process_time() - t) / (num_runs * num_samples)
layer_time_dcmp /= (num_runs * num_samples)

# %% Theoretical speed-ups based on the number of FLOPs (floating point operations (+, -, *, %))
input_shape = (frames, height, width)

FLOPs_orig = numFLOPsPerPush(net, input_shape, paddings=[1], pooling=[1, 2], pool_kernels=[pool_kernel, pool_kernel])

FLOPs_dcmp = numFLOPsPerPush(netDec, input_shape, paddings=[1], pooling=[3, 6], pool_kernels=[pool_kernel, pool_kernel])
dcmp_layer_wise = tc.tensor([tc.sum(FLOPs_dcmp[0:3]), tc.sum(FLOPs_dcmp[3:6]), tc.sum(FLOPs_dcmp[6:9]),
                             tc.sum(FLOPs_dcmp[9:11]), FLOPs_dcmp[11]])

theoretical_SP_layer = FLOPs_orig / dcmp_layer_wise
observed_SP_layer = layer_time_orig / layer_time_dcmp

print("\n\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format('', " Timing the networks ", ''))
print("{: <11}{: ^16}{: ^16}{: ^16}\n{:-^60s}".format("Layer", "Theoretical", "Observed", "Accounts for", ''))

layer_names = ["Conv1", "Conv2", "Lin1", "Lin2", "Lin3"]
for i in range(len(FLOPs_orig)):
    print("{: <11s}{: ^16.4f}{: ^16.4f}{: >7.5f} / {: <7.5f}".format(layer_names[i], theoretical_SP_layer[i],
                                                                     observed_SP_layer[i],
                                                                     FLOPs_orig[i] / tc.sum(FLOPs_orig),
                                                                     layer_time_orig[i] / tc.sum(layer_time_orig)))
print("{:-^60s}\n{: <11s}{: ^16.4f}{: ^16.4f}".format('', "Total", tc.sum(FLOPs_orig) / tc.sum(FLOPs_dcmp),
                                                      timeOrig / timeNew))

# %% How about VGG-16 ? Is that compressable?
vgg16 = models.vgg16(pretrained=True)

vgg16_dec = deepcopy(vgg16)
vgg16_dec.features[0] = conv_to_tucker1(vgg16.features[0])
vgg16_dec.features[2] = conv_to_tucker2(vgg16.features[2])
vgg16_dec.features[5] = conv_to_tucker2(vgg16.features[5])
vgg16_dec.features[7] = conv_to_tucker2(vgg16.features[7])
vgg16_dec.features[10] = conv_to_tucker2(vgg16.features[10])
vgg16_dec.features[12] = conv_to_tucker2(vgg16.features[12])
vgg16_dec.features[14] = conv_to_tucker2(vgg16.features[14])
vgg16_dec.features[17] = conv_to_tucker2(vgg16.features[17])
vgg16_dec.features[19] = conv_to_tucker2(vgg16.features[19])
vgg16_dec.features[21] = conv_to_tucker2(vgg16.features[21])
vgg16_dec.features[24] = conv_to_tucker2(vgg16.features[24])
vgg16_dec.features[26] = conv_to_tucker2(vgg16.features[26])
vgg16_dec.features[28] = conv_to_tucker2(vgg16.features[28])
vgg16_dec.classifier[0] = lin_to_tucker2(vgg16.classifier[0], ranks=[50, 10])   # Takes LONG to decompose
vgg16_dec.classifier[3] = lin_to_tucker1(vgg16.classifier[3])
vgg16_dec.classifier[6] = lin_to_tucker1(vgg16.classifier[6])

print("\n{:-^60}\n{:-^60}\n{:-^60}".format('', " The VGG-16 network ", ''))
#Converting to cuda if possible
if tc.cuda.is_available():
    vgg16 = vgg16.cuda()
    vgg16_dec = vgg16_dec.cuda()
    print("\n-- Using GPU --\n")

print("Number of parameters:\n{: <30s}{: >12d}\n{: <30s}{: >12d}\n{: <30s}{: >12f}".format("Original:",numParams(vgg16),
                                                                                           "Decomposed:",
                                                                                           numParams(vgg16_dec), "Ratio:",
                                                                                           numParams(vgg16_dec) / numParams(vgg16)))
# %% Calculating the theoretical and actual speed-ups
FLOPs_vgg16 = numFLOPsPerPush(vgg16, (224, 224), paddings=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                  pooling=[2, 4, 7, 10, 13], pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
FLOPs_vgg16_dcmp = numFLOPsPerPush_mul(vgg16_dec, (224, 224), paddings=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37],
                                       pooling=[5, 11, 20, 29, 38], pool_kernels=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
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
print("Actual speed-up was {} times based on {} seconds for the original and {} for the decomposed.".format(timeOrig / timeNew, timeOrig/2000, timeNew/2000))