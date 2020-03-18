'''
Metrics to measure forecast error 
These are measures currently not found in sklearn or statsmodels
'''
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    '''
    MAPE

    Parameters:
    --------
    y_true -- np.array actual observations from time series
    y_pred -- the predictions to evaluate

    Returns:
    -------
    float, scalar value representing the MAPE (0-100)
    '''
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def coverage(y_true, pred_intervals):
    '''
    Calculates the proportion of the true 
    values are that are covered by the lower
    and upper bounds of the prediction intervals

    Parameters:
    -------
    y_true -- arraylike, actual observations
    pred_intervals -- np.array, matrix (hx2)
    '''
    y_true = np.asarray(y_true)
    lower = np.asarray(pred_intervals.T[0])
    upper = np.asarray(pred_intervals.T[1])
    
    cover = len(np.where((y_true > lower) & (y_true < upper))[0])
    return cover / len(y_true)