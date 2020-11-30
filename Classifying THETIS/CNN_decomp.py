import torch as tc
import torch.nn as nn
from torch.nn import Linear, Conv3d, MaxPool3d
from torch.nn.functional import relu, softmax
from torch.autograd import Variable
import tensorly as tl
from tensorly.decomposition import partial_tucker
from video_functions import showFrame, loadTHETIS, conv_to_tucker1_3d, conv_to_tucker2_3d, lin_to_tucker1, \
    lin_to_tucker2, numParams
from copy import deepcopy
from time import process_time

tl.set_backend('pytorch')

# %% Loading the data
data_loaded = True

LENGTH = 1.5
RESOLUTION = 0.25

t = process_time()

if data_loaded:
    X, Y = tc.load("/Users/Tobias/Desktop/Data/data.pt")
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
l1_features = 120
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
net.load_state_dict(tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Results_hpc/trained_network.pt"))

# %% The decomposition functions:
netDec = deepcopy(net)
netDec.c1 = conv_to_tucker1_3d(net.c1)
netDec.c2 = conv_to_tucker2_3d(net.c2)
netDec.l1 = lin_to_tucker2(net.l1, ranks=[20, 50])
netDec.l2 = lin_to_tucker1(net.l2)
print("Parameters:\nOriginal: {}  Decomposed: {}  Ratio: {:.3f}".format(numParams(net), numParams(netDec), numParams(netDec)/numParams(net)))

# %% Time to compute two forward pushes:
t = process_time()
out = net(Variable(X[0:10]))
timeOrig = process_time() - t
print("Time for 2 forward pushes using original net was {:.3} seconds".format(timeOrig))

t = process_time()
out = netDec(Variable(X[0:10]))
timeNew = process_time() - t
print("Time for 2 forward pushes using decomposed net was {:.3} seconds".format(timeNew))
print("Which is a speed-up ratio of {:.2f}".format(timeNew/timeOrig))