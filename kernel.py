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

        y_hat = X@self.weights_

        return y_hat

