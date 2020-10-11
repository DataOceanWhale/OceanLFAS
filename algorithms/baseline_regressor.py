import numpy as np

class BaselineRegressor:
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        return np.zeros([x_test.shape[0]])
