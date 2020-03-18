#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:43:40 2020

@author: lucy
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import seaborn as sns  #seaborn is a nice plotting library that sits on top of matplotlib
import matplotlib.style as style
style.use('ggplot')
from statsmodels.tsa.seasonal import seasonal_decompose
from forecast.baseline import Naive1, SNaive
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.eval_measures import rmse


def load_data():
    '''Loads csv file '''
    
    data = pd.read_csv('data/ed_daily_ts_train.csv', index_col='date', parse_dates=True) 
    #parse_dates=True lets pandas know the index is a date field
    data.index.freq='D' #frequency of data is daily. have to set this manually.
    return data


data = load_data()
month_mean_data = data.resample(rule='M').mean()


START_DATE='2009-04-30'
month_mean_data.index= pd.date_range(START_DATE, periods=len(month_mean_data), freq='MS')

#plot
fig, axes= plt.subplots(2,1,sharex=True, figsize=(12,8))
axes[0].figsize=(12,10)
axes[0].plot(data)
axes[0].set(ylabel = 'No. Arrivals in ED per day')

axes[1].plot(month_mean_data)
axes[1].set(xlabel = 'Date', ylabel= 'Average No. daily arrivals in ED per month')
fig.suptitle('Comparison of daily and average daily per month')


def ts_train_test_split(data, split_date):
    '''
    split time series into training and test data

    Parameters
    ----------
    data : pd.DataFrame - time series data.  Index expected as datatimeindex
    split_data :  the date on which to split the time series

    Returns
    -------
    tuple (len=2) 
    0. pandas.DataFrame - training dataset
    1. pandas.DataFrame - test dataset
    '''
    
    train = data.loc[data.index < split_date]
    test = data.loc[data.index >= split_date]
    return train, test

train, test = ts_train_test_split(month_mean_data, '2016-06-30')

# plot train data
bx = train.plot(figsize=(12,6))
bx.set(xlabel='Date', ylabel='Av. daily arrivals train data')
bx.legend(['Training data'])



# SEASONAL DECOMPOSITION
sd_result = seasonal_decompose(train, model='multiplicative')

sd_fig, x = plt.subplots(3,1, sharex=True, figsize=(18,12))
x[0].figsize= (12,10)
x[0].plot(sd_result.trend)
x[0].set(ylabel = 'No. Arrivals - Trend')


x[1].plot(sd_result.seasonal)
x[1].set(ylabel = 'No. Arrivals - Seasonality')

x[2].plot(sd_result.resid)
x[2].set(xlabel = 'Date', ylabel= 'No. Arrivals - Residual')
sd_fig.suptitle('Seasonal Decomposition of ED arrivals train data')


# Simple forecasting baseline -- Seasonal Naive Method 
HORIZON = 12
PERIOD = 12
snf = SNaive(period=PERIOD)
snf.fit(train)
insample_predictions = snf.fittedvalues

ax = train['arrivals'].plot(figsize=(12,4))
print(type(insample_predictions))
insample_predictions.plot(ax=ax)
ax.set_title('Train data with predictions')
ax.set(xlabel='Date', ylabel='ED arrivals');
ax.legend(['Train', 'Predictions'])


# In-sample diagnostics - modelling residuals 
fig1, ax1 = plt.subplots(1, 1, sharex=True, figsize=(12, 8))
ax1.figsize=(12,10)
ax1.plot(snf.resid)
ax1.set_title('Modelling Residuals')
ax1.set_xlabel('Date')
ax1.set_ylabel('actual minus predictions')


# CAN'T PLOT
#sns.distplot(snf.resid.dropna())

