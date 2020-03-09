import numpy as np


# sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLu激活函数
def ReLU(x):
    return x if x > 0 else 0