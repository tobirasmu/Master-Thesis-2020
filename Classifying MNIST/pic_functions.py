#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:32:27 2020

@author: Tobias
"""
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import partial_tucker
from mnist import MNIST
from sklearn.metrics import accuracy_score

# For using pytorch nn framework
import torch.nn as nn
from torch.nn import Linear, Conv2d
from torch.autograd import Variable
from VBMF import EVBMF
from timeit import repeat


# %% Functions for loading the data
def loadMNIST(p=1 / 3, normalize=True):
    """ For loading the MNIST data set and making an instance of the data-class 
        p percent of the training data being validation """
    mndata = MNIST()

    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()

    # Making into stack of images
    x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('int32')
    x_test, y_test = np.array(x_test).astype('float32'), np.array(y_test).astype('int32')

    channels, rows, cols = 1, 28, 28  # 1 channel since BW and 28x28 pics
    x_train = x_train.reshape((-1, channels, rows, cols))
    x_test = x_test.reshape((-1, channels, rows, cols))
    changeInd = int(x_train.shape[0] * (1 - p))

    return Data(x_train[:changeInd], y_train[:changeInd], x_train[changeInd:], y_train[changeInd:], x_test, y_test,
                normalize)


class Data:
    """
    The data-class to hold the different data splits. The data class is initialized with all the splits of the data.
    """

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, normalize=True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        if normalize:
            self.x_train /= 255
            self.x_val /= 255
            self.x_test /= 255

    def subset(self, nTr, nVal, nTe):
        """
        If not all the traning data is needed, one can take out a subset of given sizes
        """
        return Data(self.x_train[:nTr], self.y_train[:nTr],
                    self.x_val[:nVal], self.y_val[:nVal],
                    self.x_test[:nTe], self.y_test[:nTe], normalize=False)

    def size(self):
        return self.x_train.shape[0], self.x_val.shape[0], self.x_test.shape[0]

    def __repr__(self):
        train, val, test = self.size()
        out = "This is a data set of: \n" + str(train) + " training samples, \n"
        out = out + str(val) + " validation samples, and: \n" + str(test)
        out = out + " testing samples."
        return out


# %% Visualization
def showImage(img, label=""):
    """ For being able to plot the handwritten digits. 
        Either one by one, or a matrix of them """
    plt.figure()
    plt.imshow(np.reshape(img, newshape=(28, 28)), vmin=tc.min(img), vmax=tc.max(img), cmap="gray")
    plt.axis('off')
    plt.title(label)
    plt.show()


def showWrong(data, preds, labels):
    """ Shows wrong prediction images with true and guess labels 
        from predictions and true labels """
    wrongs = np.where((preds == labels) is False)[0]
    print(len(wrongs))
    for i in range(np.min((24, len(wrongs)))):
        showImage(data[wrongs[i]], label="True: " + str(labels[wrongs[i]]) + " Guess: " + str(preds[wrongs[i]]))


def plotMany(img_L, B=10, H=10):
    """ B is how many pictures on the x-axis, and H is the y-axis """
    if type(img_L) == tc.Tensor:
        img_L = img_L.numpy()
    plt.figure()
    nr = 0
    canvas = np.zeros((1, 28 * B))
    for i in range(H):
        temp = img_L[nr].reshape((28, 28))
        nr += 1
        for j in range(B - 1):
            temp = np.concatenate((temp, img_L[nr].reshape((28, 28))), axis=1)
            nr += 1
        canvas = np.concatenate((canvas, temp), axis=0)
    plt.imshow(canvas[1:, :], vmin=np.min(img_L), vmax=np.max(img_L), cmap="gray")
    plt.axis('off')
    plt.show()


def plotAccs(train_accs, val_accs, title=None, saveName=None):
    """
    Plots the training and validation accuracies vs. the epoch. Use saveName to save it to a file.
    """
    epochs = np.arange(len(train_accs))
    plt.figure()
    plt.plot(epochs, train_accs, 'r', epochs, val_accs, 'b')
    title = title if title is not None else "Training and Validation Accuracies vs. Epoch"
    plt.title(title)
    plt.legend(['Train accuracy', 'Validation accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if saveName is not None:
        plt.savefig(saveName)
    else:
        plt.show()


# %% CUDA functions
def get_variable(x):
    """
    Converts a tensor to tensor.cuda if cuda is available
    """
    if tc.cuda.is_available():
        return x.cuda()
    return x


def get_data(x):
    """
    Fetch the tensor from the GPU if cuda is available, or simply converting to tensor if not
    """
    if tc.cuda.is_available():
        return x.cpu().data
    return x.data


# %% The training functions
def get_slice(i, size):
    return range(i * size, (i + 1) * size)


criterion = nn.CrossEntropyLoss()


def train_epoch(thisNet, X, y, optimizer, batch_size):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    losses = []
    targs, preds = [], []

    thisNet.train()
    for i in range(num_batches):
        if i % (num_batches // 10) == 0:
            print("--", end='')
        # Sending the batch through the network
        slce = get_slice(i, batch_size)
        X_batch = get_variable(Variable(tc.from_numpy(X[slce])))
        output = thisNet(X_batch)
        # The targets
        y_batch = get_variable(Variable(tc.from_numpy(y[slce]).long()))
        # Computing the error and doing the step
        optimizer.zero_grad()
        batch_loss = criterion(output, y_batch)
        batch_loss.backward()
        optimizer.step()

        losses.append(get_data(batch_loss).numpy())
        predictions = tc.max(output, 1)[1]
        targs += list(y[slce])
        preds += list(get_data(predictions).numpy())
    return np.mean(losses), accuracy_score(targs, preds)


def eval_epoch(thisNet, X, y, batch_size):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    targs, preds = [], []

    thisNet.eval()
    for i in range(num_batches):
        if i % (num_batches // 10) == 0:
            print("--", end='')
        slce = get_slice(i, batch_size)
        X_batch_val = get_variable(Variable(tc.from_numpy(X[slce])))
        output = thisNet(X_batch_val)

        predictions = tc.max(output, 1)[1]
        targs += list(y[slce])
        preds += list(get_data(predictions).numpy())
    return accuracy_score(targs, preds)


def training(net, data, batch_size, num_epochs, optimizer, every=1):
    """
    The traning loop for MNIST
    :param net:
    :param data:
    :param batch_size:
    :param num_epochs:
    :param optimizer:
    :param every:
    :return:
    """

    num_samples_train, num_samples_valid, num_samples_test = data.size()
    num_batches_train = num_samples_train // batch_size
    num_batches_valid = num_samples_valid // batch_size

    # Setting up lists
    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    losses = []

    for epoch in range(num_epochs):

        cur_loss = 0
        net.train()
        for i in range(num_batches_train):
            # Sending the batch through the network
            slce = get_slice(i, batch_size)
            x_batch = get_variable(Variable(tc.from_numpy(data.x_train[slce])))
            output = net(x_batch)

            # Computing gradients and loss
            target_batch = get_variable(Variable(tc.from_numpy(data.y_train[slce]).long()))
            batch_loss = criterion(output, target_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Updating the loss
            cur_loss += batch_loss
        losses.append(cur_loss / batch_size)

        net.eval()

        # Evaluating training data
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            x_batch = get_variable(Variable(tc.from_numpy(data.x_train[slce])))

            output = net(x_batch)
            preds = tc.max(output, 1)[1]

            train_targs += list(data.y_train[slce])
            train_preds += list(get_data(preds).numpy())

        # Evaluating validation data
        valid_preds, valid_targs = [], []
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            x_batch = get_variable(Variable(tc.from_numpy(data.x_val[slce])))

            output = net(x_batch)
            preds = tc.max(output, 1)[1]

            valid_targs += list(data.y_val[slce])
            valid_preds += list(get_data(preds).numpy())
        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(valid_targs, valid_preds)

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)

        if epoch % every == 0:
            print("Epoch %3i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch, losses[-1], train_acc_cur, valid_acc_cur))
    epochs = np.arange(len(train_acc))
    plt.figure()
    plt.plot(epochs, train_acc, 'r', epochs, valid_acc, 'b')
    plt.legend(['Train accuracy', 'Validation accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # The testing accuracy
    test_preds = tc.max(net(get_variable(Variable(tc.from_numpy(data.x_test)))), 1)[1]
    print("---------------|o|----------------\nTesting accuracy on %3i samples: %f" % (
        num_samples_test, accuracy_score(test_preds.numpy(), data.y_test)))


# %% Decomposition functions
def conv_to_tucker2(layer, ranks=None):
    """
    Takes a pretrained convolutional layer and decomposes is using partial
    tucker with the given ranks.
    """
    # Making the decomposition of the weights
    weights = layer.weight.data
    # (Estimating the ranks using VBMF)
    ranks = estimate_ranks(weights, [0, 1]) if ranks is None else ranks
    # Decomposing
    core, [last, first] = partial_tucker(weights, modes=[0, 1], ranks=ranks)

    # Making the layer into 3 sequential layers using the decomposition
    first_layer = Conv2d(in_channels=first.shape[0], out_channels=first.shape[1],
                         kernel_size=1, stride=1, padding=0, bias=False)

    core_layer = Conv2d(in_channels=core.shape[1], out_channels=core.shape[0],
                        kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding, bias=False)

    last_layer = Conv2d(in_channels=last.shape[1], out_channels=last.shape[0],
                        kernel_size=1, stride=1, padding=0, bias=True)

    # The decomposition is chosen as weights in the network (output, input, height, width)
    first_layer.weight.data = tc.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core  # no reshaping needed
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    # The bias from the original layer is added to the last convolution
    last_layer.bias.data = layer.bias.data

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def conv_to_tucker1(layer, rank=None):
    """
    Takes a pretrained convolutional layer and decomposes it using partial tucker with the given rank.
    """
    # Making the decomposition of the weights
    weights = layer.weight.data
    out_ch, in_ch, kernel_size, _ = weights.shape
    # (Estimating the rank)
    rank = estimate_ranks(weights, [0]) if rank is None else [rank]
    core, [last] = partial_tucker(weights, modes=[0], ranks=rank)

    # Turning the convolutional layer into a sequence of two smaller convolutions
    core_layer = Conv2d(in_channels=in_ch, out_channels=rank[0], kernel_size=kernel_size, padding=layer.padding,
                        stride=layer.stride, bias=False)
    last_layer = Conv2d(in_channels=rank[0], out_channels=out_ch, kernel_size=1, padding=0, stride=1, bias=True)

    # Setting the weights:
    core_layer.weight.data = core
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    last_layer.bias.data = layer.bias.data

    new_layers = [core_layer, last_layer]
    return nn.Sequential(*new_layers)


def lin_to_tucker1(layer, rank=None, in_channels=True):
    """
    Takes a linear layer as input, decomposes it using tucker1, and makes it into
    a sequence of two smaller linear layers using the decomposed weights.
    """
    # Making the decomposition of the weights
    weights = layer.weight.data
    nOut, nIn = weights.shape
    if in_channels:
        rank = estimate_ranks(weights, [1]) if rank is None else [rank]
        core, [B] = partial_tucker(weights, modes=[1], ranks=rank)

        # Now we have W = GB^T, we need Wb which means we can seperate into two layers
        BTb = Linear(in_features=nIn, out_features=rank[0], bias=False)
        coreBtb = Linear(in_features=rank[0], out_features=nOut, bias=True)

        # Set up the weights
        BTb.weight.data = tc.transpose(B, 0, 1)
        coreBtb.weight.data = core

        # Bias goes on last layer
        coreBtb.bias.data = layer.bias.data

        new_layers = [BTb, coreBtb]
    else:
        rank = estimate_ranks(weights, [0]) if rank is None else [rank]
        core, [A] = partial_tucker(weights, modes=[0], ranks=rank)

        # Now we have W = AG, we need Wb which means we can do Wb = A (Gb) as two linear layers
        coreb = Linear(in_features=nIn, out_features=rank[0], bias=False)
        Acoreb = Linear(in_features=rank[0], out_features=nOut, bias=True)

        # Let the decomposed weights be the weights of the new
        coreb.weight.data = core
        Acoreb.weight.data = A

        # The bias goes on the second one
        Acoreb.bias.data = layer.bias.data

        new_layers = [coreb, Acoreb]
    return nn.Sequential(*new_layers)


def lin_to_tucker2(layer, ranks=None):
    """
    Takes in a linear layer and decomposes it by tucker-2. Then splits the linear
    map into a sequence of smaller linear maps.
    """
    # Pulling out the weights
    weights = layer.weight.data
    nOut, nIn = weights.shape
    # Estimate (ranks and) weights
    ranks = estimate_ranks(weights, [0, 1]) if ranks is None else ranks
    core, [A, B] = partial_tucker(weights, modes=[0, 1], ranks=ranks)

    # Making the sequence of 3 smaller layers
    BTb = Linear(in_features=nIn, out_features=ranks[1], bias=False)
    coreBTb = Linear(in_features=ranks[1], out_features=ranks[0], bias=False)
    AcoreBTb = Linear(in_features=ranks[0], out_features=nOut, bias=True)

    # Setting the weights
    BTb.weight.data = tc.transpose(B, 0, 1)
    coreBTb.weight.data = core
    AcoreBTb.weight.data = A
    AcoreBTb.bias.data = layer.bias.data

    new_layers = [BTb, coreBTb, AcoreBTb]
    return nn.Sequential(*new_layers)


def numParams(net):
    return sum(np.prod(p.size()) for p in net.parameters())


def estimate_ranks(weight_tensor, dimensions):
    ranks = []
    for dim in dimensions:
        _, diag, _, _ = EVBMF(tl.unfold(weight_tensor, dim))
        ranks.append(diag.shape[dim])
    return ranks


# %% Functions for calculating the theoretical speed-up
# Two matrices of size [N1 x N2] and [N2 x N3] respectively excluding N1 x N3 biases
def linearFLOPs(out_features, in_features):
    """
    Returns the number of FLOPs needed to perform forward push through a linear layer with the given in- and output
    features.
    Based on the paper "Pruning CNNs for resource efficiency" by Molchanov P, Tyree S, Karras T, et al.
    """
    return (2 * in_features - 1) * out_features


def conv_dims(dims, kernels, strides, paddings):
    """
    Computes the resulting output dimensions when performing a convolution with the given kernel, stride and padding.
    INPUT:
            dims - the input dimensions
            kernels - kernel width for each dimension
            strides - strides for each dimension
            paddings - for each dimension
    OUTPUT:
            list of resulting dimensions
    """
    dimensions = len(dims)
    new_dims = tc.empty(dimensions)
    for i in range(dimensions):
        new_dims[i] = int((dims[i] - kernels[i] + 2 * paddings[i]) / strides[i] + 1)
    return new_dims


def convFLOPs(kernel_shape, output_shape):
    """
    The FLOPs needed to perform a convolution.
    INPUT:
            kernel_shape = (out_channels, in_channels, (d_f), d_h, d_w)
            input_shape = ((f), h, w)
    """
    C_out, C_in = kernel_shape[0:2]
    filter_shape = kernel_shape[2:]
    return C_out * tc.prod(output_shape.long()) * (2 * C_in * tc.prod(filter_shape.long()) - 1)


def numFLOPsPerPush(net, input_shape, paddings=None, pooling=None, pool_kernels=None):
    """
    Returns the number of floating point operations needed to make one forward push of each of the layers,
    in a given network. Padding is a list of the layer number that has padding in it (assumed full padding), pooling is
    a list of the layers that have pooling just after them. Pool_kernels are the corresponding pooling kernels for each
    layer that have pooling in them
    """
    FLOPs = []
    layer = 0
    paddings = [] if paddings is None else paddings
    pooling = [] if pooling is None else pooling
    wasConv = False
    output_shape = input_shape
    for weights in list(net.parameters()):
        kernel_shape = tc.tensor(weights.shape)
        if len(kernel_shape) == 2:
            layer += 1
            FLOPs.append(linearFLOPs(kernel_shape[0], kernel_shape[1]))
            wasConv = False
        elif len(kernel_shape) > 2:
            wasConv = True
            layer += 1
            this_padding = kernel_shape[2:] // 2 if layer in paddings else (0, 0)
            output_shape = conv_dims(input_shape, kernel_shape[2:], strides=(1, 1), paddings=this_padding)
            FLOPs.append(convFLOPs(kernel_shape, output_shape))
            if layer in pooling:
                this_kernel = pool_kernels.pop(0)
                input_shape = conv_dims(output_shape, this_kernel, strides=this_kernel, paddings=(0, 0))
            else:
                input_shape = output_shape
        else:
            # Bias term requires number of additions equal to the amount of output values
            if wasConv:
                FLOPs[-1] += tc.prod(output_shape.long())
            else:
                FLOPs[-1] += kernel_shape[0]
    return tc.tensor(FLOPs)


# %% Timing functions
def time_conv(num_obs, input_size, in_ch, out_ch, kernel, padding, bias=True, number=10, num_dim=2):
    """
    Timing a convolutional layer with the given structure.
    INPUT:
            num_obs    : how many observations are to be pushed through
            input_size : the size of the video (frames, height, width)
            in_ch      : input channels
            out_ch     : output channels
            kernel     : the given kernel
            padding    : the given padding
            bias       : if bias is needed
            number     : how many times it should be timed
            num_dim    : number of dimensions (3 for video, 2 for picture)
    OUTPUT:
            the time in seconds
    """
    input_shape = (num_obs, in_ch, *input_size)

    from layer_timing_functions import conv_layer_timing
    from torch.autograd import Variable
    from pic_functions import get_variable
    import torch as tc
    net = conv_layer_timing(in_ch, out_ch, kernel, stride=1, padding=padding,bias=bias, dimensions=num_dim)
    if tc.cuda.is_available():
        net = net.cuda()
    x = get_variable(Variable(tc.rand(input_shape)))
    times = tc.tensor(repeat("net(x)", globals=locals(), number=1, repeat=number))
    return tc.mean(times), tc.std(times), times


def time_lin(num_obs, in_neurons, out_neurons, bias=True, number=10):
    """
    Timing the linear forward push with the given structure:
    INPUT:
            num_obs     : how many observations to be pushed forward
            in_neurons  : how many input neurons in the layer
            out_neurons : how many output neurons
            bias        : if bias is also timed
            number      : how many times it should be timed
    """
    input_shape = (num_obs, in_neurons)
    from layer_timing_functions import lin_layer_timing
    from torch.autograd import Variable
    from pic_functions import get_variable
    import torch as tc
    net = lin_layer_timing(in_neurons, out_neurons, bias)
    if tc.cuda.is_available():
        net = net.cuda()
    x = get_variable(Variable(tc.rand(input_shape)))
    times = tc.tensor(repeat("net(x)", globals=locals(), number=1, repeat=number))
    return tc.mean(times), tc.std(times), times
