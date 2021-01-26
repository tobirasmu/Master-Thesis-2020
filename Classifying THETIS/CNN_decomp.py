# True if using the GPU
HPC = False

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
import tensorly as tl
from video_functions import numParams, train_epoch, eval_epoch, plotAccs
import torch.optim as optim
from sklearn.model_selection import KFold
from time import process_time

tl.set_backend('pytorch')

# %% Loading the data
t = process_time()
directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/Users/Tobias/Desktop/Data/"
X, Y = tc.load(directory + "data.pt")

print("Took {:.2f} seconds to load the data".format(process_time() - t))

# %% The network that we are working with:
from video_networks import Net, compressNet
_, channels, frames, height, width = X.shape
# Initializing the CNN
net = Net(channels, frames, height, width)

# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/THETIS_network_94.pt"))
else:
    net.load_state_dict(
        tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Trained networks/THETIS_network_94.pt"))

# %% The decomposition functions:
netDec = compressNet(net)

print("The decomposed network has the following architecture\n", netDec)
print("Parameters:\nOriginal: {}  Decomposed: {}  Ratio: {:.3f}\n".format(numParams(net), numParams(netDec),
                                                                          numParams(netDec) / numParams(net)))

if tc.cuda.is_available():
    net = net.cuda()
    print("-- USING GPU --")
    netDec = netDec.cuda()

# %% Training the decomposed network
BATCH_SIZE = 10
NUM_FOLDS = 5
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
nTrain = int(0.85 * X.shape[0])


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
    saveAt = "/zhome/2a/c/108156/Outputs/accuracies_decomp.png" if HPC else "/Users/Tobias/Desktop/accuracies_decomp" \
                                                                            ".png "
    plotAccs(train_accs, val_accs, saveName=saveAt)
    print("{:-^60}\nFinished".format(""))


print("{:-^60s}".format(" Training details "))
print("{: ^20}{: ^20}{: ^20}".format("Learning rate:", "Batch size:", "Number of folds"))
print("{: ^20.4f}{: ^20d}{: ^20d}\n{:-^60}".format(LEARNING_RATE, BATCH_SIZE, NUM_FOLDS, ''))
train(netDec, X[:nTrain], Y[:nTrain], X[nTrain:], Y[nTrain:])
