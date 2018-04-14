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
from sklearn.linear_model import SGDRegressor
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %% First load the data

os.chdir('C:\\Users\\George\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\Walmart forecasting')

train = pd.read_csv('Data\\train.csv')
features = pd.read_csv('Data\\features.csv')

# %% Describe elements of the data


def dataDescription(train):
    """ Initial look at the data. 3 Plots are generated. """
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
    nonHolidaySales = train[train['IsHoliday'] == False].Weekly_Sales.mean()
    print("The mean weekly sales on holidays is {:.2f} ".format(holidaySales) +
          "and then {:.2f} for non-holidays.".format(nonHolidaySales))


dataDescription(train)

# %% Add more features


def extractFeatures(sample, features):
    """ Not many features are extracted, but this is useful to functionalise.
    More features can be added as necessary later"S"""
    sample = pd.merge(sample, features, on=['Store', 'Date'])

    # Extract features from the Date
    sample['Date'] = pd.to_datetime(sample['Date'])
    sample['WeekOfYear'] = sample['Date'].dt.weekofyear
    sample['Year'] = sample['Date'].dt.year
    return sample


train = extractFeatures(train, features)

# %% Fit the data

# Separate the data into train and CV by taking progressive splits from the end
# ie split it into 6 sections. Train on 1, CV 2; train on
# 1,2, CV 3... until train on 1,2,3,4,5 CV 6

# Only use features if they have no NaNs for the moment

n_splits = 4

predCols = ['Store', 'Dept', 'WeekOfYear', 'Year', 'IsHoliday_x',
            'Temperature', 'Fuel_Price']

# %% XGBoost

def findXGBErrorOnFit(n_splits, train, predCols):
    """ Use the time series split to create cross validation sets and measure
    the average error across all of them after fitting with xgboost"""
    tsCV = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
            ('scal', StandardScaler()),
            ('clf', xgb.XGBRegressor(learning_rate=0.07, max_depth=6,
                                     n_estimators=100))])

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
        sampleWeights = CVX.IsHoliday_x * 4 + 1
        CVerror += mean_absolute_error(CVY, prediction,
                                       sample_weight = sampleWeights)

    # Print the mean absolute error of the regressor
    CVerror /= n_splits
    print("The mean cross-validated error is ${:.2f}".format(CVerror))
    return CVerror

CVerrorXGB = findXGBErrorOnFit(n_splits, train, predCols)

# %% SVM

def findSGDErrorOnFit(n_splits, train, predCols):
    """ Use the time series split to create cross validation sets and measure
    the average error across all of them after fitting with SGDRegressor"""

    tsCV = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
            ('scal', StandardScaler()),
            ('clf', SGDRegressor())])

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
        sampleWeights = CVX.IsHoliday_x * 4 + 1
        CVerror += mean_absolute_error(CVY, prediction,
                                       sample_weight = sampleWeights)

    # Print the mean absolute error of the regressor
    CVerror /= n_splits
    print("The mean cross-validated error is ${:.2f}".format(CVerror))
    return CVerror

CVerrorSGD = findSGDErrorOnFit(n_splits, train, predCols)

# %% Test prediction and submission

def predictAndSubmit(train, features, predCols):
    """ Reformat the data for submission and save the generated predictions"""
    realTest = pd.read_csv('Data\\test.csv')
    sub = pd.DataFrame()
    sub['Id'] = (realTest['Store'].map(str) + '_' +
                 realTest['Dept'].map(str) + '_' +
                 realTest['Date'].map(str))

    realTest = extractFeatures(realTest, features)
    realTestX = realTest[predCols]
    pipe = Pipeline([('scal', StandardScaler()),
                     ('clf', xgb.XGBRegressor(learning_rate=0.07, max_depth=6,
                                              n_estimators=100))])
    trainX = train[predCols]
    trainY = train['Weekly_Sales']
    pipe.fit(trainX, trainY)
    prediction = pipe.predict(realTestX)
    sub['Weekly_Sales'] = prediction
    sub.to_csv('Output\\XGBSubmission.csv', index=False)


