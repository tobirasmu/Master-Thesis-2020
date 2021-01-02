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
from tensorly.decomposition import partial_tucker
from mnist import MNIST
from sklearn.metrics import accuracy_score

# For using pytorch nn framework
import torch.nn as nn
from torch.nn import Linear, Conv2d
from torch.autograd import Variable
from VBMF import EVBMF


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


# %% The training loop
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
        X_batch = Variable(tc.from_numpy(X[slce]))
        output = thisNet(X_batch)
        # The targets
        y_batch = Variable(tc.from_numpy(y[slce]).long())
        # Computing the error and doing the step
        optimizer.zero_grad()
        batch_loss = criterion(output, y_batch)
        batch_loss.backward()
        optimizer.step()

        losses.append(batch_loss.data.numpy())
        predictions = tc.max(output, 1)[1]
        targs += list(y[slce])
        preds += list(predictions.data.numpy())
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
        X_batch_val = Variable(tc.from_numpy(X[slce]))
        output = thisNet(X_batch_val)

        predictions = tc.max(output, 1)[1]
        targs += list(y[slce])
        preds += list(predictions.data.numpy())
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
    criterion = nn.CrossEntropyLoss()

    num_samples_train, num_samples_valid, num_samples_test = data.size()
    num_batches_train = num_samples_train // batch_size
    num_batches_valid = num_samples_valid // batch_size

    # Setting up lists
    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    test_acc, test_loss = [], []
    cur_loss = 0
    losses = []

    for epoch in range(num_epochs):

        cur_loss = 0
        net.train()
        for i in range(num_batches_train):
            # Sending the batch through the network
            slce = get_slice(i, batch_size)
            x_batch = Variable(tc.from_numpy(data.x_train[slce]))
            output = net(x_batch)

            # Computing gradients and loss
            target_batch = Variable(tc.from_numpy(data.y_train[slce]).long())
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
            x_batch = Variable(tc.from_numpy(data.x_train[slce]))

            output = net(x_batch)
            preds = tc.max(output, 1)[1]

            train_targs += list(data.y_train[slce])
            train_preds += list(preds.data.numpy())

        # Evaluating validation data
        valid_preds, valid_targs = [], []
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            x_batch = Variable(tc.from_numpy(data.x_val[slce]))

            output = net(x_batch)
            preds = tc.max(output, 1)[1]

            valid_targs += list(data.y_val[slce])
            valid_preds += list(preds.data.numpy())
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
    test_preds = tc.max(net(Variable(tc.from_numpy(data.x_test))), 1)[1]
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
