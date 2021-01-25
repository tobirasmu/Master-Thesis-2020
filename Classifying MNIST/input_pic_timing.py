HPC = True
import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying MNIST/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)

import torch as tc
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear
from torch.nn.functional import relu, softmax
from timeit import repeat
from pic_functions import get_variable

HIDDEN_NEURONS = 20
DELTA = 28 * 28


# Function that calculates parameters and FLOPs for MNIST
def MNIST_input_numbers(r, hidden):
    delta = 28 * 28
    parameters = r * delta + r * (hidden + 1) + hidden * 11
    FLOPs = r * (2 * delta - 1) + hidden * (2 * r - 1) + 10 * (2 * hidden - 1)
    return parameters, FLOPs


# %% The simple network that we are working with


class Net(nn.Module):

    def __init__(self, input_neurons):
        super(Net, self).__init__()

        self.l1 = Linear(in_features=input_neurons, out_features=HIDDEN_NEURONS, bias=True)
        self.l_out = Linear(in_features=HIDDEN_NEURONS, out_features=10, bias=True)

    def forward(self, x):
        x = tc.flatten(x, 1)
        x = relu(self.l1(x))

        # x = self.dropout(relu(self.l2(x)))

        return softmax(self.l_out(x), dim=1)


# %% Running for every rank
ranks = [2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 100, 150, 300]
SAMPLE_SIZE = 100000
NUM_PUSHES = 1000
BURN_IN = SAMPLE_SIZE // 10

full_times, approx_times, network_times = [], [], []
# First doing the original network
net = Net(DELTA)
test_input = get_variable(Variable(tc.rand((NUM_PUSHES, 28, 28))))
if tc.cuda.is_available():
    net = net.cuda()
    print("CUDA enabled\n")
full_times.append(repeat('net(test_input)', globals=globals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[BURN_IN:])
orig_parms = DELTA * (HIDDEN_NEURONS + 1) + HIDDEN_NEURONS * 11
orig_FLOPs = HIDDEN_NEURONS * (2 * DELTA - 1) + 10 * (2 * HIDDEN_NEURONS - 1)
print("For the original network the number of parameters is  {}  and the number of FLOPs is  {}".format(orig_parms,
                                                                                                        orig_FLOPs))
# Now timing each rank
test_input = get_variable(Variable(tc.rand((NUM_PUSHES, DELTA))))
for rank in ranks:
    G1 = get_variable(Variable(tc.rand((DELTA, rank))))
    net = Net(rank)
    if tc.cuda.is_available():
        net = net.cuda()
    full_times.append(repeat('net(get_variable(Variable(tc.matmul(test_input, G1))))', globals=globals(), number=1,
                             repeat=(SAMPLE_SIZE + BURN_IN))[BURN_IN:])
    approx_times.append(repeat('tc.matmul(test_input, G1)', globals=globals(), number=1, repeat=(SAMPLE_SIZE +
                                                                                                 BURN_IN))[BURN_IN:])
    this_test_input = get_variable(Variable(tc.matmul(test_input, G1)))
    network_times.append(repeat('net(this_test_input)', globals=globals(), number=1, repeat=(SAMPLE_SIZE + BURN_IN))[
                         BURN_IN:])
    parms, num_FLOPs = MNIST_input_numbers(rank, HIDDEN_NEURONS)
    print("For the rank of {} the number of parameters is  {}  and the number of FLOPs is  {}".format(rank, parms,
                                                                                                      num_FLOPs))

full_times = tc.tensor(full_times) * 1000
approx_times = tc.tensor(approx_times) * 1000
network_times = tc.tensor(network_times) * 1000
full_mean, full_sd = tc.mean(full_times, dim=1), tc.std(full_times, dim=1)
approx_mean, approx_sd = tc.mean(approx_times, dim=1), tc.std(approx_times, dim=1)
network_mean, network_sd = tc.mean(network_times, dim=1), tc.std(network_times, dim=1)

print("Using a sample size of {}\n".format(SAMPLE_SIZE))
for i, rank in enumerate(ranks):
    print("{:-^84}".format(" Rank {:3d} ".format(rank)))
    print("{: ^14s}{: ^14s}{: ^14s}{: ^14s}{: ^14s}{: ^14s}".format("Full", "Std", "Approx", "Std", "Network", "Std"))
    print("{: ^14f}{: ^14f}{: ^14f}{: ^14f}{: ^14f}{: ^14f}".format(full_mean[i + 1], full_sd[i + 1], approx_mean[i],
                                                                    approx_sd[i], network_mean[i], network_sd[i]),
          end='')
    print("   ${:.3f} \\pm {:.3f}$ $(= {:.3f} + {:.3f} )$".format(full_mean[i + 1], full_sd[i + 1], approx_mean[i],
                                                                  network_mean[i]))
print("{:-^84}".format(" Original network "))
print("{: ^14s}{: ^14s}".format("Full", "Std"))
print("{: ^14f}{: ^14f}".format(full_mean[0], full_sd[0]))
