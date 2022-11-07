import numpy as np
from scipy.stats import t


def standard_errors(n, d, y, y_hat, X):

    s_2 = np.dot((y-y_hat),(y-y_hat)) / (n-d)
    var = s_2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(var))

    return se


def t_statistic(parameters, std_err):

    t = parameters / std_err
    low, upp = parameters-2.05*std_err, parameters+2.05*std_err 

    intervals = np.array(list(map(lambda x, y:(x,y), low, upp)))    

    return t, low, upp


def p_value(t_values, df):

    probabilities = []
    for t_val in t_values:
        p = 1 - t.cdf(t_val, df)
        probabilities.append(p)

    return np.array(probabilities)












