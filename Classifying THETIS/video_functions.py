import numpy as np
import torch as tc
from timeit import repeat
import torch.nn as nn
from torch.nn import Linear, Conv3d, Conv2d
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import tensorly as tl
from layer_timing_functions import conv_layer_timing, lin_layer_timing

tl.set_backend("pytorch")
from tensorly.decomposition import partial_tucker
from VBMF import EVBMF
import matplotlib.pyplot as plt
import cv2
import os

# %% Writes a tensor to a video
FRAME_RATE = 18  # not true for all videos but okay.


def writeTensor2video(x, name, out_directory=None):
    """
    Writes a tensor of shape (ch, num_frames, height, width) or (num_frames, height, width) (for B/W) to a video at the
    given out_directory. If no directory is given, it will just be placed in the current directory with the name.
    """
    if x.type() != 'torch.ByteTensor':
        x = x.type(dtype=tc.uint8)
    if out_directory is None:
        out_directory = os.getcwd() + '/'
    name = name + '.avi'
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        ch, num_frames, height, width = x.shape
    else:
        ch, num_frames, height, width = x.shape
    writer = cv2.VideoWriter(out_directory + name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FRAME_RATE,
                             (width, height))
    for i in range(num_frames):
        frame = np.moveaxis(x[:, i, :, :].type(dtype=tc.uint8).numpy(), 0, -1)
        if ch == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


# %% Plotting functions
def showFrame(x, title=None, saveName=None):
    """
    Takes in a tensor of shape (ch, height, width) or (height, width) (for B/W) and plots the image using
    matplotlib.pyplot
    """
    x = x * 255 if tc.max(x) <= 1 else x
    if x.type() != "torch.ByteTensor":
        x = x.type(dtype=tc.uint8)
    if len(x.shape) == 2:
        plt.imshow(x.numpy(), cmap="gray")
    else:
        plt.imshow(np.moveaxis(x.numpy(), 0, -1))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    if saveName is not None:
        plt.savefig(saveName)
    else:
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


# %% Training functions
criterion = nn.CrossEntropyLoss()


def get_slice(i, size): return range(i * size, (i + 1) * size)


def train_epoch(this_net, X, y, optimizer, batch_size):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    losses = []
    targs, preds = [], []

    this_net.train()
    for i in range(num_batches):
        # Sending the batch through the network
        slce = get_slice(i, batch_size)
        X_batch = get_variable(Variable(X[slce]))
        output = this_net(X_batch)
        # The targets
        y_batch = get_variable(Variable(y[slce].long()))
        # Computing the error and doing the step
        optimizer.zero_grad()
        batch_loss = criterion(output, y_batch)
        batch_loss.backward()
        optimizer.step()

        losses.append(get_data(batch_loss))
        predictions = tc.max(get_data(output), 1)[1]
        targs += list(y[slce])
        preds += list(predictions)
    return tc.mean(tc.tensor(losses)), accuracy_score(targs, preds)


def eval_epoch(this_net, X, y, output_lists=False):
    # Sending the validation samples through the network
    this_net.eval()
    X_batch = get_variable(Variable(X))
    output = this_net(X_batch)
    preds = tc.max(get_data(output), 1)[1]
    # The targets
    targs = y.long()
    if output_lists:
        return targs, preds
    else:
        return accuracy_score(targs, preds)


# %% The decomposition functions:
def conv_to_tucker2_3d(layer, ranks=None):
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
    first_layer = Conv3d(in_channels=first.shape[0], out_channels=first.shape[1],
                         kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

    core_layer = Conv3d(in_channels=core.shape[1], out_channels=core.shape[0],
                        kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding, bias=False)

    last_layer = Conv3d(in_channels=last.shape[1], out_channels=last.shape[0],
                        kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    # The decomposition is chosen as weights in the network (output, input, height, width)
    first_layer.weight.data = tc.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core  # no reshaping needed
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # The bias from the original layer is added to the last convolution
    last_layer.bias.data = layer.bias.data

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def conv_to_tucker1_3d(layer, rank=None):
    """
    Takes a pretrained convolutional layer and decomposes it using partial tucker with the given rank.
    """
    # Making the decomposition of the weights
    weights = layer.weight.data
    out_ch, in_ch = weights.shape[0:2]
    # (Estimating the rank)
    rank = estimate_ranks(weights, [0]) if rank is None else [rank]
    core, [last] = partial_tucker(weights, modes=[0], ranks=rank)

    # Turning the convolutional layer into a sequence of two smaller convolutions
    core_layer = Conv3d(in_channels=in_ch, out_channels=rank[0], kernel_size=layer.kernel_size, padding=layer.padding,
                        stride=layer.stride, bias=False)
    last_layer = Conv3d(in_channels=rank[0], out_channels=out_ch, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                        stride=(1, 1, 1), bias=True)

    # Setting the weights:
    core_layer.weight.data = core
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    last_layer.bias.data = layer.bias.data

    new_layers = [core_layer, last_layer]
    return nn.Sequential(*new_layers)


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


def lin_to_tucker2(layer, ranks=None):
    """
    Takes in a linear layer and decomposes it by tucker-2. Then splits the linear
    map into a sequence of smaller linear maps.
    """
    # Pulling out the weights
    weights = layer.weight.data
    nOut, nIn = weights.shape
    # Estimate (ranks and) weights
    if ranks is None:
        if nOut < nIn:
            rank = estimate_ranks(weights, [0])
        else:
            rank = estimate_ranks(weights, [1])
    ranks = [rank[0], rank[0]] if ranks is None else ranks
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


def estimate_ranks(weight_tensor, dimensions):
    """
    Estimates the sufficient ranks for a given tensor
    """
    ranks = []
    for dim in dimensions:
        _, diag, _, _ = EVBMF(tl.unfold(weight_tensor, dim))
        ranks.append(diag.shape[dim])
    return ranks


def numParams(net):
    """
    Returns the number of parameters in the entire network.
    """
    return sum(np.prod(p.size()) for p in net.parameters())


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
            this_padding = kernel_shape[2:] // 2 if layer in paddings else (0, 0, 0)
            output_shape = conv_dims(input_shape, kernel_shape[2:], strides=(1, 1, 1), paddings=this_padding)
            FLOPs.append(convFLOPs(kernel_shape, output_shape))
            if layer in pooling:
                this_kernel = pool_kernels.pop(0)
                input_shape = conv_dims(output_shape, this_kernel, strides=this_kernel, paddings=(0, 0, 0))
            else:
                input_shape = output_shape
        else:
            # Bias term requires number of additions equal to the amount of output values
            if wasConv:
                FLOPs[-1] += tc.prod(output_shape.long())
            else:
                FLOPs[-1] += kernel_shape[0]
    return tc.tensor(FLOPs)


# %% Timing a single layer of different types
def time_conv(num_obs, input_size, in_ch, out_ch, kernel, padding, bias=True, sample_size=10, num_dim=3):
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
    burn_in = sample_size // 10
    input_shape = (num_obs, in_ch, *input_size)
    net = conv_layer_timing(in_ch, out_ch, kernel, stride=(1, 1, 1), padding=padding, bias=bias, dimensions=num_dim)
    if tc.cuda.is_available():
        net = net.cuda()
    x = get_variable(Variable(tc.rand(input_shape)))
    times = tc.tensor(repeat("net(x)", globals=locals(), number=1, repeat=(sample_size + burn_in))[burn_in:])
    return tc.mean(times), tc.std(times), times


def time_lin(num_obs, in_neurons, out_neurons, bias=True, sample_size=10):
    """
        Timing the linear forward push with the given structure. Repeats number times and reports the mean, standard
        deviation, and the list of times.
    INPUT:
            num_obs     : how many observations to be pushed forward
            in_neurons  : how many input neurons in the layer
            out_neurons : how many output neurons
            bias        : if bias is also timed
            number      : how many times it should be timed
    OUTPUT:
            mean times, std time, list times
    """
    burn_in = sample_size // 10
    input_shape = (num_obs, in_neurons)
    net = lin_layer_timing(in_neurons, out_neurons, bias)
    if tc.cuda.is_available():
        net = net.cuda()
    x = get_variable(Variable(tc.rand(input_shape)))
    times = tc.tensor(repeat("net(x)", globals=locals(), number=1, repeat=(sample_size + burn_in))[burn_in:])
    return tc.mean(times), tc.std(times), times
