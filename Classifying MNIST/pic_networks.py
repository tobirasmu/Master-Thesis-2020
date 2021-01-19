import torchvision.models as models
from copy import deepcopy
from pic_functions import conv_to_tucker2, conv_to_tucker1, lin_to_tucker1, lin_to_tucker2
import torch as tc
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Linear
from torch.nn.functional import relu, softmax


def conv_dim(dim, kernel, stride, padding):
    return int((dim - kernel + 2 * padding) / stride + 1)


# First convolution
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


class Net(nn.Module):
    def __init__(self, channels, height):  # We only need height since the pictures are square
        super(Net, self).__init__()

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

    def forward(self, x):
        # Conv 1
        x = relu(self.conv1(x))
        x = self.pool(x)

        # Conv 2
        x = relu(self.conv2(x))
        x = self.pool(x)

        x = tc.flatten(x, 1)
        # Lin 1
        x = relu(self.l1(x))
        # Lin 2
        x = relu(self.l2(x))

        return softmax(relu(self.l_out(x)), dim=1)


def compressNetwork(net):
    """
        Function that compresses and returns the network given above.
    """
    netDec = deepcopy(net)
    # Decomposing
    netDec.conv1 = conv_to_tucker1(netDec.conv1)
    netDec.conv2 = conv_to_tucker2(netDec.conv2)
    netDec.l1 = lin_to_tucker2(netDec.l1)
    netDec.l2 = lin_to_tucker1(netDec.l2)
    return netDec


def get_VGG16(compressed=False):
    """
        Returns the VGG-16 network either compressed or not
    """
    vgg16 = models.vgg16(pretrained=True)

    if compressed:
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
        vgg16_dec.classifier[0] = lin_to_tucker2(vgg16.classifier[0], ranks=[50, 10])  # Takes LONG to decompose
        vgg16_dec.classifier[3] = lin_to_tucker1(vgg16.classifier[3])
        vgg16_dec.classifier[6] = lin_to_tucker1(vgg16.classifier[6])
        return vgg16_dec
    else:
        return vgg16
