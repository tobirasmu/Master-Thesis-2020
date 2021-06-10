import matplotlib.pyplot as plt
import torch as tc
import numpy as np
from matplotlib.lines import Line2D



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


def plotFoldAccs(train_accs: np.array, val_accs: np.array, title=None, saveName=None):
    """
    Plots multiple training curves
    """
    num_folds, num_epochs = train_accs.shape
    epochs = np.arange(num_epochs)
    fig, ax = plt.subplots(1, 1)
    for i in range(num_folds):
        ax.plot(epochs, train_accs[i, :], ls="--", color=f"C{i}")
        ax.plot(epochs, val_accs[i, :], ls="-.", color=f"C{i}")
    handles = [Line2D([0], [0], ls="--", color="black", label="Training acc."), Line2D([0], [0], ls="-.", color="black", label="Validation acc.")]
    handles.append(ax.plot(epochs, np.mean(train_accs, axis=0), ls="solid", linewidth=2, color="blue", label="Mean train acc.")[0])
    handles.append(ax.plot(epochs, np.mean(val_accs, axis=0), ls="solid", linewidth=2, color="red", label="Mean val acc.")[0])
    ax.set(title=title if title is not None else "Training and Validation Accuracies vs. Epoch", xlabel="Epoch", ylabel="Accuracy")
    ax.legend(handles=handles)
    if saveName is not None:
        fig.savefig(saveName)
    else:
        fig.show()