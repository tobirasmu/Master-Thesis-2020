HPC = False
import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying MNIST/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)

import torch as tc
from pic_functions import numFLOPsPerPush, numParams
from pic_networks import get_VGG16, compressNetwork, Net
from timeit import repeat
from torch.autograd import Variable
from pic_functions import get_variable, loadMNIST, showImage
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.functional import relu, softmax
import torch.nn as nn
import numpy as np
from time import process_time

NUM_OBS = 100
SAMPLE_SIZE = 1000
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((NUM_OBS, 1, 28, 28))))

data = loadMNIST()

net = Net(1, 28)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/MNIST_network_9866_acc.pt"))
else:
    net.load_state_dict(tc.load("/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Trained "
                                "networks/MNIST_network_9866_acc.pt"))

# %%
i = 61
showImage(tc.tensor(data.x_test[i]))
print(net(tc.tensor(data.x_test[i]).unsqueeze(0)))
data.y_test[i]
# %%
import lime
from lime import lime_image

def batch_pred(images):
    net.eval()
    batch = tc.stack(tuple([tc.Tensor(img[:,:,2]).unsqueeze(0) for img in images]), dim=0)
    return net(batch).detach().numpy()

batch_pred([data.x_test[0],data.x_test[1]])

# %%
explainer = lime_image.LimeImageExplainer()
exp = explainer.explain_instance(data.x_test[i].squeeze(0).astype(float),
                                 batch_pred,
                                 top_labels=5,
                                 hide_color=0,
                                 num_samples=100
                                 )

# %%
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp, mask)
plt.imshow(img_boundry1); plt.show()

# %%
temp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry2 = mark_boundaries(temp, mask)
plt.imshow(img_boundry2); plt.show()