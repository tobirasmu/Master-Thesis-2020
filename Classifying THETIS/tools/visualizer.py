import matplotlib.pyplot as plt
import torch as tc
import numpy as np


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

