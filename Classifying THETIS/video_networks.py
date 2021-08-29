from copy import deepcopy

import torch as tc
import torch.nn as nn
from torch.nn import Conv3d, MaxPool3d, Linear
from torch.nn.functional import relu, softmax

from video_functions import lin_to_tucker2, lin_to_tucker1, conv_to_tucker2_3d, conv_to_tucker1_3d




