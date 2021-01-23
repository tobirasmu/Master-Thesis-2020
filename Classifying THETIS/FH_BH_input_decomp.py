# %% Trying to decompose the training tensor via Tucker
"""
Rank estimation does not work on a tensor that big since the required memory is 538 TiB (tebibyte), hence the ranks will
be chosen intuitively (obs, frame, height, width). We are interested in the temporal information hence, the frame dimension
will be given full rank (not decomposed). Since the frames are rather simple (BW depth), the spatial dimensions will not
be given full rank
"""
HPC = True

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
import torch.nn as nn
from torch import optim
from numpy.linalg import pinv
from torch.nn import Linear
from torch.nn.functional import relu, softmax
import tensorly as tl
from tensorly.tenalg import mode_dot
from tensorly.decomposition import partial_tucker
import matplotlib.pyplot as plt
from timeit import timeit
from video_functions import writeTensor2video, train_epoch, eval_epoch, plotAccs, showFrame
from sklearn.model_selection import KFold

tl.set_backend('pytorch')

# %% Loading the data


directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/Users/Tobias/Desktop/Data/"
X, Y = tc.load(directory + "data.pt")

N = X.shape[0]
nTrain = int(0.85 * N)

# %% Doing the decomposition
modes = [0]
ranks = [100]

core, [A] = partial_tucker(X[:nTrain], modes=modes, ranks=ranks)


# %% Output a video approximation
# approx_loadings = [-0.060, 0, 0.10, 0.1]
# appr = mode_dot(core, tc.tensor(approx_loadings), mode=modes[0]) * 255
# writeTensor2video(appr[0:3], 'test', "/Users/Tobias/Desktop/")


# %% Scatter plot of the first 2 loading vectors colored with the different classes
plt.figure()
A3s = A[tc.where(Y[:nTrain] == 0)[0]]
A4s = A[tc.where(Y[:nTrain] == 1)[0]]
plt.scatter(A3s[:, 0], A3s[:, 1], facecolor='None', edgecolor='red', s=10)
plt.scatter(A4s[:, 0], A4s[:, 1], facecolor='Blue', edgecolor='blue', marker="x", s=10)
"""
    It does not makes sense to plot the means, since they are very equal, what does however make 2 different clusters
    is the two different locations at which the videos have been shot.
"""
means_loc_1 = tc.mean(A[tc.where(A[:, 1] > 0.075)], dim=0)
means_loc_2 = tc.mean(A[tc.where(A[:, 1] < 0.075)], dim=0)
plt.scatter(means_loc_1[0], means_loc_1[1], facecolor="Black", s=50, marker="s")
plt.scatter(means_loc_2[0], means_loc_2[1], facecolor="Black", s=50, marker="^")
plt.xlabel('1. loading in A')
plt.ylabel('2. loading in A')
plt.legend(labels=('Forehand flat', 'Backhand', 'Mean location 1', 'Mean location 2'), loc=0)
plt.title('Loadings of A for all the training examples')
plt.show()

if not HPC:
    approx_1 = mode_dot(core, means_loc_1, mode=modes[0]) * 255
    showFrame(approx_1[0:3, 14], saveName="/Users/Tobias/Desktop/loc1.png")
    approx_2 = mode_dot(core, means_loc_2, mode=modes[0]) * 255
    showFrame(approx_2[0:3, 14], saveName="/Users/Tobias/Desktop/loc2.png")

# %% Histograms of the loadings
plt.subplot(2, 2, 1)
plt.hist(A[:, 0])
plt.subplot(2, 2, 2)
plt.hist(A[:, 1])

# %% Simple neural networks to be trained
_, channels, frames, height, width = X.shape
hidden_neurons = 100


class Net(nn.Module):

    def __init__(self, in_neurons):
        super(Net, self).__init__()

        self.l1 = Linear(in_features=in_neurons, out_features=hidden_neurons)
        self.l_out = Linear(in_features=hidden_neurons, out_features=2)

    def forward(self, x):
        x = tc.flatten(x, 1)

        x = relu(self.l1(x))

        return softmax(self.l_out(x), dim=1)


net_orig = Net(channels * frames * height * width)
if tc.cuda.is_available():
    net_orig = net_orig.cuda()

# %% Training function
LEARNING_RATE = 0.01  # 0.0001 for original
NUM_EPOCHS = 1000
NUM_FOLDS = 5
BATCH_SIZE = 10


def train(this_net, X_train, y_train, X_test, y_test, saveAt=None):
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
    if saveAt is None:
        saveAt = "/zhome/2a/c/108156/Outputs/accuracies_decomp.png" if HPC else "/Users/Tobias/Desktop" \
                                                                                "/accuracies_input_decomp.png "
    plotAccs(train_accs, val_accs, saveName=saveAt)
    print("{:-^60}\nFinished".format(""))


# %% Training the original network
X_train, X_test = X[:nTrain], X[nTrain:]
Y_train, Y_test = Y[:nTrain], Y[nTrain:]

# train(net_orig, X_train, Y_train, X_test, Y_test, "/zhome/2a/c/108156/Outputs/accuracies_input_dcmp_orig.png")

# %% Approximating the As for the testing set and defining the new network

A_new = tl.unfold(X_test, mode=0) @ pinv(tl.unfold(core, mode=0))

net_dcmp = Net(ranks[0])
if tc.cuda.is_available():
    net_dcmp = net_dcmp.cuda()
print("\n\nTraining the decomposed version with rank {}\n\n".format(ranks[0]))
train(net_dcmp, A, Y_train, A_new, Y_test,
      "/zhome/2a/c/108156/Outputs/accuracies_input_dcmp_decomp_" + str(ranks[0]) + ".png")
