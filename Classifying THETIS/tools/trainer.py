import torch as tc
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


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
    return tc.mean(tc.tensor(losses)), tc.tensor(accuracy_score(targs, preds))


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
        return tc.tensor(accuracy_score(targs, preds))
