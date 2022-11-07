from symbol import try_stmt
import numpy as np
import pandas as pd

from ..core import Estimator
from .stats import standard_errors, t_statistic, p_value

__all__ = ['OLS'] # for not impotring np and pd to the context

class OLS(Estimator):
    '''
    Ordinary Least Square

    Attributes
    weights_: estimated model parameters
    '''

    name = 'Estimator' # class attribute

    def __init__(
        self,
        ):

        print('>> OLS constructor invoked')
        #self.estimator = 'Ordinary Least Squares' # Instance atribute
        self.__estimator = 'instance attribute' # Instance atribute
        self._weights = 'w1' # this one shown as method
        self.__weights = 'w2' # this one is hidded


    def setname(self, name):
        self.__name = name
    
    def getname(self):
        
        return self.__name

    estimator = property(setname, getname)


    # TODO is_fitted function
    def fit(self, X, y):

        self.optimal(X, y)


    def optimal(self, X, y, intercept=True): # Class method
        
        if intercept:
            print('Intercept added')
            if len(X.shape) == 1:
                X = np.vstack((np.ones(len(X)), X)).T
            else:
                X = np.insert(X, 0, 1, axis=1)
        
        self.X = X
        self.y = y
        self.weights_ = np.linalg.inv(X.T@X)@X.T@y # Optimization
        self.intercept_ = self.weights_[:1][0]

    # TODO: move to partent
    def predict(self, X):

        y_hat = X@self.weights_

        return y_hat


    def moments(self):

        x_mean = np.mean(self.X, axis=0)
        x_var = np.sum((self.X - x_mean)**2, axis=0)

        return x_mean, x_var


    def rss(self, y, y_hat):

        err = y-y_hat
        rss = np.sum(err**2)

        return rss



    def statistics(self):
        print('statistics')

        n = len(self.X)
        d = len(self.weights_)
        y_hat = self.predict(self.X)
        y = self.y


        se = standard_errors(n, d, y, y_hat, self.X)
        self.std_err_ = se
        t, low, upp = t_statistic(self.weights_, se)
        self.t_stats_ = t
        self.interval_low_ = low
        self.interval_upp_ = upp 

        p_values = p_value(t_values=t, df=n-d)
        self.p_values_ = p_values


    def summary_table(self):

        summary_df = pd.DataFrame({'coef': self.weights_,
                    'std_err':self.std_err_,
                    't_stat':self.t_stats_,
                    'low':self.interval_low_,
                    'upp':self.interval_upp_,
                    'p_val':self.p_values_,
                    })

        print(summary_df.round(4))




    def summary(self):

        summary_df = pd.DataFrame({'coef': self.weights_,
                    'p_val':self.p_values_,
                    })

        #print(summary_df.round(4))

        return summary_df.round(4)



