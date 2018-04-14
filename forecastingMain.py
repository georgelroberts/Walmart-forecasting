# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:12:07 2018

@author: George

Identify the forecasted Walmart sales from time-series data

TODO: Merge the features for each store
TODO: Create a submission
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# %% First load the data

os.chdir('C:\\Users\\George\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\Walmart forecasting')

train = pd.read_csv('Data\\train.csv')
features = pd.read_csv('Data\\features.csv')

# %% Describe elements of the data

def dataDescription(train):

    print(train.describe())
    # No NaNs
    print(train.head())
    # 5 columns: store number, department number, date, weekly sales, isHoliday

    # Lets view the weekly sales, grouped by date
    fig, ax = plt.subplots()
    salesByDate = train.groupby('Date').mean().Weekly_Sales
    salesByDate.plot()

    # Lets plot all store's sales by date on the same graph
    salesByStore = train.groupby(by=['Store', 'Date']).mean().Weekly_Sales
    salesByStore.unstack(level=0).plot()

    # Lets plot all department's sales by date on the same graph
    salesByStore = train.groupby(by=['Dept', 'Date']).mean().Weekly_Sales
    salesByStore.unstack(level=0).plot()

    # It looks like all stores have peaks at the same times, however just a
    # single department. Could this be the holiday department?
    holidayDates = train.groupby('Date').mean().IsHoliday
    holidayDates.plot()
    # Very few holiday days...
    # What are the average sales on holidays vs not holidays
    holidaySales = train[train['IsHoliday'] == True].Weekly_Sales.mean()
    nonHolidaySales = train[train['IsHoliday']==False].Weekly_Sales.mean()
    print("The mean weekly sales on a holiday is {:.2f} ".format(holidaySales) +
          "and then {:.2f} for non-holidays.".format(nonHolidaySales))


dataDescription(train)

# %% Add more features

# Marge extra features
train = pd.merge(train,features, on=['Store','Date'])

# Extract features from the Date
train['Date'] = pd.to_datetime(train['Date'])
train['WeekOfYear'] = train['Date'].dt.weekofyear
train['Year'] = train['Date'].dt.year

# %% Clean and fit the data

# Separate the data into train and CV by taking progressive splits from the end
# ie split it into 6 sections. Train on 1, CV 2; train on
# 1,2, CV 3... until train on 1,2,3,4,5 CV 6
n_splits = 8

def findErrorOnFit(n_splits, xgb_n_estimators, train):
    tsCV = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
            ('scal', StandardScaler()),
            ('clf', xgb.XGBRegressor(learning_rate=0.07, max_depth=6,
                                     n_estimators=xgb_n_estimators))])

    predCols = ['Store', 'Dept', 'WeekOfYear','Year', 'IsHoliday_x',
                'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    CVerror = 0
    for train_ind, CV_ind in tsCV.split(train):
        trainMod = train.iloc[train_ind]
        CVMod = train.iloc[CV_ind]

        trainX = trainMod[predCols]
        trainY = trainMod['Weekly_Sales']
        CVX = CVMod[predCols]
        CVY = CVMod['Weekly_Sales']
        pipe.fit(trainX, trainY)
        prediction = pipe.predict(CVX)
        CVerror += mean_absolute_error(CVY, prediction)

    # Print the mean absolute error of the regressor
    CVerror /= n_splits
    print("The mean cross-validated error is ${:.2f}".format(CVerror))
    return CVerror

n_estimators = [10,100,500]
CVerrors=[]

for n_estimator in n_estimators:
    CVerrors.append(findErrorOnFit(5, n_estimator, train))

