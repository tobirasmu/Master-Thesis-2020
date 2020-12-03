# True if using the
HPC = True

import os
path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
       "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
import torch.nn as nn
from torch.nn import Linear, Conv3d, MaxPool3d
from torch.nn.functional import relu, softmax
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import partial_tucker
from video_functions import showFrame, loadTHETIS, conv_to_tucker1_3d, conv_to_tucker2_3d, lin_to_tucker1, \
    lin_to_tucker2, numParams, get_variable, train_epoch, eval_epoch, plotAccs
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

if data_loaded:
    X, Y = tc.load("/zhome/2a/c/108156/Data_MSc/data.pt") if HPC else tc.load("/Users/Tobias/Desktop/Data/data.pt")
else:
    if HPC:
        directory = "/zhome/2a/c/108156/Data_MSc/"
        inputForehand = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/forehand_filenames_adapted.csv"
        inputBackhand = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/backhand_filenames_adapted.csv"
        X, Y = loadTHETIS((0, 1), (inputForehand, inputBackhand), ([10, 45], [0]), directory, 
                          out_directory = directory + "data.pt", length=LENGTH, resolution=RESOLUTION)
    else:
        directory = "/Users/Tobias/Desktop/Data/"
        # Forehands
        inputForehand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/forehand_filenames_adapted.csv"
        # Backhands
        inputBackhand = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/backhand_filenames_adapted.csv"
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
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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

    def forward(self, x):
        x = relu(self.c1(x))
        x = self.pool3d(x)

        x = relu(self.c2(x))
        x = self.pool3d(x)

        x = tc.flatten(x, 1)

        x = relu(self.l1(x))
        x = relu(self.l2(x))
        return softmax(self.l_out(x), dim=1)


# Initializing the CNN
net = Net()

# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Results_hpc/trained_network_92.pt"))
else:
    net.load_state_dict(tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Results_hpc/trained_network.pt"))

# %% The decomposition functions:
netDec = deepcopy(net)
netDec.c1 = conv_to_tucker1_3d(net.c1)
netDec.c2 = conv_to_tucker2_3d(net.c2)
netDec.l1 = lin_to_tucker2(net.l1, ranks=[2, 5])
netDec.l2 = lin_to_tucker1(net.l2)
print("The decomposed network has the following architecture\n",netDec)
print("Parameters:\nOriginal: {}  Decomposed: {}  Ratio: {:.3f}\n".format(numParams(net), numParams(netDec), numParams(netDec)/numParams(net)))

if tc.cuda.is_available():
    net = net.cuda()
    netDec = netDec.cuda()
    print("-- USING GPU --")

# %% Training the decomposed network
BATCH_SIZE = 10
NUM_FOLDS = 5
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
nTrain = int(0.85*X.shape[0])


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
print("{: ^20}{: ^20}{: ^20}".format("Learning rate:","Batch size:", "Number of folds"))
print("{: ^20.4f}{: ^20d}{: ^20d}\n{:-^60}".format(LEARNING_RATE, BATCH_SIZE, NUM_FOLDS,''))
train(netDec, X[:nTrain], Y[:nTrain], X[nTrain:], Y[nTrain:])

# %% Time to compute ten forward pushes:
net.eval()
timeTest = Variable(get_variable(X[0:100]))
t = process_time_ns()
for i in range(10):
    net(timeTest)
timeOrig = process_time_ns() - t
print("Mean time for 10 forward pushes using original net was {}".format(timeOrig/1000))

netDec.eval()
t = process_time_ns()
for i in range(10):
    netDec(timeTest)
timeNew = process_time_ns() - t
print("Mean time for 10 forward pushes using decomposed net was {}".format(timeNew/(1000)))
print("Which is a speed-up ratio of {:.2f}".format(timeNew/timeOrig))