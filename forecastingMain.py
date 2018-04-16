# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:12:07 2018

@author: George

Identify the forecasted Walmart sales from time-series data

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
from sklearn.svm import SVR
from pandas.tools.plotting import autocorrelation_plot
from fbprophet import Prophet
from collections import defaultdict
import xgboost as xgb
import warnings
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)
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
    plt.show()
    fig, ax = plt.subplots()
    autocorrelation_plot(salesByDate)
    plt.show()
    # Lets plot all store's sales by date on the same graph
    salesByStore = train.groupby(by=['Store', 'Date']).mean().Weekly_Sales
    salesByStore.unstack(level=0).plot()
    # Lets plot all department's sales by date on the same graph
    salesByStore = train.groupby(by=['Dept', 'Date']).mean().Weekly_Sales
    salesByStore.unstack(level=0).plot()
    # It looks like all stores have peaks at the same times, however just a
    # single department. Could this be the holiday department?
    fig, ax = plt.subplots()
    holidayDates = train.groupby('Date').mean().IsHoliday
    holidayDates.plot()
    plt.show()
    # Very few holiday days...
    # What are the average sales on holidays vs not holidays
    holidaySales = train[train['IsHoliday'] == True].Weekly_Sales.mean()
    nonHolidaySales = train[train['IsHoliday'] == False].Weekly_Sales.mean()
    print("The mean weekly sales on holidays is {:.2f} ".format(holidaySales) +
          "and then {:.2f} for non-holidays.".format(nonHolidaySales))

    # Lets looks at how complete the data is. Create a series with all possible
    # dates in. Compare with each store and department.
    trainDates = pd.to_datetime(train.Date)
    trainDates = pd.DatetimeIndex(trainDates.unique())
#    First confirm there are no missing dates in the whole range.
#    trainDatesTest = pd.date_range(trainDates.min(),
#                                   trainDates.max(), freq='7D')
#    (trainDates == trainDatesTest).all()
    stores = np.unique(train['Store'])
    depts = np.unique(train['Dept'])
    missingDates = defaultdict(int)
    for store in stores:
        for dept in depts:
            trainThis = train[train['Store'] == store]
            trainThis = trainThis[trainThis['Dept'] == dept]
            missing = len(trainDates) - len(pd.DatetimeIndex(trainThis.Date))
            missingDates[missing] += 1

    # The majority miss nothing. 314 stores don't have certain departments
    # - as can be expected!
    fig, ax = plt.subplots()
    ax.bar(list(missingDates.keys()), missingDates.values())

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

n_splits = 2

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
        train.sort_values(by='Date', inplace=True)
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
                                       sample_weight=sampleWeights)

    # Print the mean absolute error of the regressor
    CVerror /= n_splits
    print("The mean cross-validated error is ${:.2f}".format(CVerror))
    return CVerror


CVerrorXGB = findXGBErrorOnFit(n_splits, train, predCols)

# %% SGD fit


def findSGDErrorOnFit(n_splits, train, predCols):
    """ Use the time series split to create cross validation sets and measure
    the average error across all of them after fitting with SGDRegressor"""

    tsCV = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline([
            ('scal', StandardScaler()),
            ('clf', SGDRegressor(learning_rate='optimal', alpha=0.001))])
            #('clf', SVR(kernel='linear'))])

    CVerror = 0
    for train_ind, CV_ind in tsCV.split(train):
        train.sort_values(by='Date', inplace=True)
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
                                       sample_weight=sampleWeights)

    # Print the mean absolute error of the regressor
    CVerror /= n_splits
    print("The mean cross-validated error is ${:.2f}".format(CVerror))
    return CVerror


CVerrorSGD = findSGDErrorOnFit(n_splits, train, predCols)

# %% FBProphet


def extractStoreDeptCombos(df):
    """ Not all stores have certain departments, so use only those that
    exist"""
    storeDepts = df[['Store', 'Dept']]
    storeDepts = storeDepts.drop_duplicates(subset=['Store', 'Dept'],
                                            keep='first')
    return storeDepts


def findFBProphertErrorOnFit(n_splits, train, predCols):
    """ Use the time series split to create cross validation sets and measure
    the average error across all of them after fitting with FBProphet"""
    tsCV = TimeSeriesSplit(n_splits=2)

    CVerror = 0
    storeDepts = extractStoreDeptCombos(train)
    n = 0
    for store, dept in storeDepts.itertuples(index=False):
        trainThis = train[train['Store'] == store]
        trainThis = trainThis[trainThis['Dept'] == dept]
        if len(trainThis.index) > 142:
            # Only fit if all dates available
            for train_ind, CV_ind in tsCV.split(trainThis):
                trainMod = trainThis.iloc[train_ind]
                CVMod = trainThis.iloc[CV_ind]
                trainFB = pd.DataFrame()
                trainFB['ds'] = trainMod['Date'].astype(str)
                trainFB['y'] = trainMod['Weekly_Sales']
                m = Prophet()
                m.fit(trainFB)
                CVx = pd.DataFrame()
                CVx['ds'] = CVMod['Date'].astype(str)
                CVy = CVMod['Weekly_Sales']
                prediction = m.predict(CVx)
                n += 1
                sampleWeights = CVMod.IsHoliday_x * 4 + 1
                CVerror += mean_absolute_error(CVy, prediction.yhat,
                                               sample_weight=sampleWeights)
            print("Store: {} Dept: {}".format(store, dept))

    # Print the mean absolute error of the regressor
    CVerror /= (n_splits + n)
    print("The mean cross-validated error is ${:.2f}".format(CVerror))
    return CVerror


CVerrorFB = findFBProphertErrorOnFit(n_splits, train, predCols)

# %% Test prediction and submission


def predictAndSubmit(train, features, predCols):
    """ Reformat the data for submission and save the generated predictions"""
    realTest = pd.read_csv('Data\\test.csv')
    realTest['Id'] = (realTest['Store'].map(str) + '_' +
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
    realTest['Weekly_Sales'] = prediction
    realTest[['Id', 'Weekly_Sales']].to_csv('Output\\XGBSubmission.csv',
                                            index=False)

    pipe = Pipeline([('scal', StandardScaler()),
                     ('clf', SGDRegressor())])
    pipe.fit(trainX, trainY)
    prediction = pipe.predict(realTestX)
    realTest['Weekly_Sales'] = prediction
    realTest[['Id', 'Weekly_Sales']].to_csv('Output\\SGDSubmission.csv',
                                            index=False)

    testDates = pd.to_datetime(realTest.Date)
    testDates = pd.DatetimeIndex(testDates.unique())
    storeDepts = extractStoreDeptCombos(realTest)
    noNotFit = 0
    allPred = pd.DataFrame()
    for store, dept in storeDepts.itertuples(index=False):
        trainThis = train[train['Store'] == store]
        trainThis = trainThis[trainThis['Dept'] == dept]
        if len(trainThis.index) > 142:
            # Only fit if all dates available
            trainFB = pd.DataFrame()
            trainFB['ds'] = trainThis['Date'].astype(str)
            trainFB['y'] = trainThis['Weekly_Sales']
            m = Prophet()
            m.fit(trainFB)
            realTestFBx = pd.DataFrame()
            realTestFBx['ds'] = testDates
            prediction = m.predict(realTestFBx)
            predRows = pd.DataFrame({'Store': store, 'Dept': dept,
                                     'Date': testDates, 'y': prediction.yhat})
        else:
            print("Not enough Data")
            noNotFit += 1
        print("Store: {} Dept: {}".format(store, dept))
        allPred = allPred.append(predRows, ignore_index=True)

    print("{} store-date combos not fit".format(noNotFit))

    allPred.drop_duplicates(inplace=True)
    realSub = pd.merge(realTest[['Store', 'Date', 'Dept', 'Id']], allPred,
                       on=['Store', 'Date', 'Dept'], how='left')

    # Fill all NaNs wit the total mean. This could be more advanced obviously,
    # but this will do for the time being
    realSub['y'].fillna((realSub['y'].mean()), inplace=True)
    realSub = realSub[['Id', 'y']]
    realSub.columns = ['Id', 'Weekly_Sales']
    realSub.to_csv('Output\\FBProphetSubmission.csv', index=False)

    # XGBScore = 7972.37008
    # FBProphet score = 5357.68674

    FBSub = pd.read_csv('Output\\FBProphetSubmission.csv')
    XGSub = pd.read_csv('Output\\XGBSubmission.csv')
    ensembleSub = pd.DataFrame((FBSub['Weekly_Sales']*3/4
                                + XGSub['Weekly_Sales']*1/4))
    ensembleSub['Id'] = FBSub['Id']
    ensembleSub[['Id', 'Weekly_Sales']].to_csv(
            'Output\\EnsembleSubmission.csv', index=False)

predictAndSubmit(train, features, predCols)
