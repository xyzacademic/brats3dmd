import numpy as np


class SCD(object):

    def __init__(self, selected_index, w, bias):
        self.selected_index = selected_index
        self.w = w
        self.bias = bias

    def predict(self, x):
        yp = np.zeros((x.shape[0], self.w.shape[0]), dtype=np.float32)
        for i in range(self.w.shape[0]):
            yp[:, i] = (np.sign(
                x[:, self.selected_index[i]].dot(self.w[i])
                + self.bias[i]) + 1) // 2
        yp = yp.mean(axis=1).round().astype(np.int8)

        return yp