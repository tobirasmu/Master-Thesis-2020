import tensorly as tl
from tensorly.decomposition import partial_tucker
import torch as tc
import torch.nn as nn
from torch.nn import Conv3d, Conv2d, Linear
from tools.VBMF import EVBMF
from copy import deepcopy

tl.set_backend("pytorch")


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
    core, [last, first] = partial_tucker(weights, modes=[0, 1], rank=ranks)

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
    core, [last] = partial_tucker(weights, modes=[0], rank=rank)

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
    core, [last, first] = partial_tucker(weights, modes=[0, 1], rank=ranks)

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
    core, [last] = partial_tucker(weights, modes=[0], rank=rank)

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
    core, [A, B] = partial_tucker(weights, modes=[0, 1], rank=ranks)

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
        core, [B] = partial_tucker(weights, modes=[1], rank=rank)

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
        core, [A] = partial_tucker(weights, modes=[0], rank=rank)

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
        _, diag, _, _ = EVBMF(tl.unfold(weight_tensor, dim).numpy())
        ranks.append(diag.shape[dim])
    return ranks


def compressNet(net):
    """
        Function that compresses the network given above
    """
    net_dec = deepcopy(net)

    net_dec.c1 = conv_to_tucker2_3d(net.c1)
    net_dec.c2 = conv_to_tucker2_3d(net.c2)
    net_dec.l1 = conv_to_tucker2_3d(net.l1)
    net_dec.l2 = lin_to_tucker1(net.l2)
    return net_dec
