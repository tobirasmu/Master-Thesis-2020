# True if using the GPU
HPC = True

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/home/tenra/PycharmProjects/Master-Thesis-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
import tensorly as tl
from tools.trainer import train_epoch, eval_epoch
from tools.visualizer import plotAccs
from tools.models import Net, Net2, numParams
from tools.decomp import compressNet
import torch.optim as optim
from sklearn.model_selection import KFold
from time import process_time

tl.set_backend('pytorch')

# %% Loading the data
t = process_time()
directory = "/zhome/2a/c/108156/Data_MSc/" if HPC else "/home/tenra/PycharmProjects/Data Master/"
X, Y = tc.load(directory + "data.pt")
print("Took {:.2f} seconds to load the data".format(process_time() - t))

# %% The network that we are working with:
N, channels, frames, height, width = X.shape
# Initializing the CNN
net = Net2(channels, frames, height, width)

# Loading the parameters of the pretrained network (needs to be after converting the network back to cpu)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/THETIS_new.pt"))
else:
    net.load_state_dict(
        tc.load("/home/tenra/PycharmProjects/Master-Thesis-2020/Trained networks/THETIS_new.pt"))

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

BATCH_SIZE = 20
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.7
WEIGHT_DECAY = 0.01
nTrain = int(0.90 * N)


def train(X_train, y_train):
    train_loss, train_accs = tc.empty(NUM_EPOCHS), tc.empty(NUM_EPOCHS)
    
    optimizer = optim.SGD(netDec.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    epoch, interrupted = 0, False
    while epoch < NUM_EPOCHS:
        print("{:-^40s}".format(" EPOCH {:3d} ".format(epoch + 1)))
        print("{: ^20}{: ^20}".format("Train Loss:", "Train acc.:"))
        try:
            train_loss[epoch], train_accs[epoch] = train_epoch(netDec, X_train, y_train, optimizer=optimizer, 
                                                               batch_size=BATCH_SIZE)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            interrupted = True
            break

        print("{: ^20.4f}{: ^20.4f}".format(train_loss[epoch], train_accs[epoch]))
        epoch += 1
        if interrupted:
            break
    saveAt = "/zhome/2a/c/108156/Outputs/accuracies_finetune.png" if HPC else \
        "/home/tenra/PycharmProjects/Results/accuracies_finetune.png"
    plotAccs(train_accs, saveName=saveAt)
    print("{:-^40}\n".format(""))
    print(f"{'Testing accuracy:':-^40}\n{eval_epoch(net, X[nTrain:], Y[nTrain:]): ^40.4f}")


if __name__=="__main__":
    train(X[:nTrain], Y[:nTrain])
    if HPC:
        tc.save(netDec.cpu().state_dict(), "/zhome/2a/c/108156/Outputs/trained_network_dcmp.pt")