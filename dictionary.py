import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Union


activation_dict: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'relu': F.relu,
    'gelu': F.gelu,
    'sigmoid': F.sigmoid,
    'softmax': F.softmax,
    'selu': F.selu,
    'leaky_relu': F.leaky_relu,
    'tanh': F.tanh
}

optimizer_dict: dict[str, optim.Optimizer] = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'rms_prop' or "rmsprop": optim.RMSprop,
    'adagrad' or 'ada_grad': optim.Adagrad,
    'ada_delta' or 'adadelta': optim.Adadelta
}