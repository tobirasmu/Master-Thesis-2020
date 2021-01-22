HPC = True

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/"
os.chdir(path)

from timeit import repeat
import torch as tc
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear
from torch.nn.functional import relu, softmax
from video_functions import get_variable

HIDDEN_NEURONS = 100
DELTA = 4 * 28 * 120 * 160  # channels, frames, height and width

# Loading the data
directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/Users/Tobias/Desktop/Data/"
X, Y = tc.load(directory + "data.pt")

N = X.shape[0]
nTrain = int(0.85 * N)


# Function that calculates the number of parameters and the number of FLOPs for a forward push
def THETIS_input_numbers(rank, hidden):
    delta = 4 * 28 * 120 * 160
    parms = rank * delta + rank * (hidden + 1) + hidden * 3
    FLOPs = rank * (2 * delta - 1) + hidden * (2 * rank - 1) + 10 * (2 * hidden - 1)
    return parms, FLOPs


# %% The simple network that we are working with
_, channels, frames, height, width = X.shape


class Net(nn.Module):
    """
        The number of input neurons is an input to the initializer
    """

    def __init__(self, in_neurons):
        super(Net, self).__init__()

        self.l1 = Linear(in_features=in_neurons, out_features=HIDDEN_NEURONS)
        self.l_out = Linear(in_features=HIDDEN_NEURONS, out_features=2)

    def forward(self, x):
        x = tc.flatten(x, 1)

        x = relu(self.l1(x))

        return softmax(self.l_out(x), dim=1)


# %% Running for every rank the timing functions
ranks = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200]
test_input = get_variable(Variable(tc.rand((1, DELTA))))
SAMPLE_SIZE = 20
BURN_IN = SAMPLE_SIZE // 10

full_times, approx_times, network_times = [], [], []
# First doing the original network
net = Net(DELTA)
full_times.append(repeat('net(test_input)', globals=globals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[BURN_IN:])
# Now timing each rank
for rank in ranks:
    G1 = tc.rand((DELTA, rank))
    net = Net(rank)
    if tc.cuda.is_available():
        net = net.cuda()
    full_times.append(repeat('net(tc.matmul(test_input, G1))', globals=globals(), number=1, repeat=(SAMPLE_SIZE +
                                                                                                    BURN_IN))[BURN_IN:])
    approx_times.append(repeat('tc.matmul(test_input, G1)', globals=globals(), number=1, repeat=(SAMPLE_SIZE +
                                                                                                 BURN_IN))[BURN_IN:])
    this_test_input = tc.matmul(test_input, G1)
    network_times.append(repeat('net(this_test_input)', globals=globals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[
                         BURN_IN:])
    parms, FLOPs = THETIS_input_numbers(rank, HIDDEN_NEURONS)
    print("For the rank of {} the number of parameters is  {}  and the number of FLOPs is  {}".format(rank, parms,
                                                                                                      FLOPs))

full_times = tc.tensor(full_times)
approx_times = tc.tensor(approx_times)
network_times = tc.tensor(network_times)
full_mean, full_sd = tc.mean(full_times, dim=1), tc.std(full_times, dim=1)
approx_mean, approx_sd = tc.mean(approx_times, dim=1), tc.std(approx_times, dim=1)
network_mean, network_sd = tc.mean(network_times, dim=1), tc.std(network_times, dim=1)

print("Using a sample size of {}\n".format(SAMPLE_SIZE))
for i, rank in enumerate(ranks):
    print("{:-^84}".format(" Rank {:3d} ".format(rank)))
    print("{: ^14s}{: ^14s}{: ^14s}{: ^14s}{: ^14s}{: ^14s}".format("Full", "Std", "Approx", "Std", "Network", "Std"))
    print("{: ^14f}{: ^14f}{: ^14f}{: ^14f}{: ^14f}{: ^14f}".format(full_mean[i + 1], full_sd[i + 1], approx_mean[i],
                                                                    approx_sd[i], network_mean[i], network_sd[i]))
print("{:-^84}".format(" Original network "))
print("{: ^14s}{: ^14s}".format("Full", "Std"))
print("{: ^14f}{: ^14f}".format(full_mean[0], full_sd[0]))