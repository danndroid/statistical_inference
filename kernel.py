import numpy as np
from .validation import is_fitted


class Estimator:
    '''
    Estimator class
    '''

    def __init__(self):
        print("Kernel Estimator initialized")


    def validate_estimator(self):
        is_fitted(self)


    def fit(self): # to be overriden
        print("fit method")


    def SUPER_fit(self):
        print("fit method")


    def predict(self, X):
        self.validate_estimator()

        if X.shape[1] != self.weights_.shape[0] :
            print('Intercept added')
            if len(X.shape) == 1:
                X = np.vstack((np.ones(len(X)), X)).T
            else:
                X = np.insert(X, 0, 1, axis=1)

        y_hat = X@self.weights_

        return y_hat

