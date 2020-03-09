import numpy as np
import pickle
import config
from activate_func import sigmoid


params_path = config.params_path  # 导入参数


# 加载训练好的参数
def loadParms(path):
    """
    :param path: 参数文件路径
    :return: 参数W,b
    """
    with open(path, 'rb') as f:
        params = pickle.load(f)

    W = params["W"]
    b = params["b"]

    return W, b


class Logistic:
    """
    描述：利用Logistic回归判断图像是否为R
    优化：尝试在不降低识别准确率的前提下降低矩阵大小
    """
    def __init__(self):
        self.W, self.b = loadParms(params_path)

    # 判断R
    def predict(self, x):
        y = sigmoid(np.dot(self.W.T, x) + self.b)
        res = 1 if y >= 0.5 else 0
        return res

