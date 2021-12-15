#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, f1_score, accuracy_score

print('This script trains a sample classifier (using simple RandomForestClassifier) '
      'on CERT dataset. By default it takes CERT r5.2 extracted day data '
      'downloaded from https://web.cs.dal.ca/~lcd/data/CERTr5.2/'
      ', train on data of 400 users in first half of the dataset, '
      'then output classification report (instance-based)')

print('For more details, see this paper: Analyzing Data Granularity Levels for'
      ' Insider Threat Detection using Machine Learning. Le, D. C.; Zincir-Heywood, '
      'A. N.; and Heywood, M. I. IEEE Transactions on Network and Service Management,'
      ' 17(1): 30–44. March 2020.')

data = pd.read_csv('day-r5.2.csv.gz')
removed_cols = ['user','day','week','starttime','endtime','sessionid','insider']
x_cols = [i for i in data.columns if i not in removed_cols]

run = 1
np.random.seed(run)

data1stHalf = data[data.week <= max(data.week)/2]
dataTest = data[data.week > max(data.week)/2]

selectedTrainUsers =  set(data1stHalf[data1stHalf.insider > 0]['user'])
nUsers = np.random.permutation(list(set(data1stHalf.user) - selectedTrainUsers))
trainUsers = np.concatenate((list(selectedTrainUsers), nUsers[:400-len(selectedTrainUsers)]))

unKnownTestUsers = list(set(dataTest.user) - selectedTrainUsers)

xTrain = data1stHalf[data1stHalf.user.isin(trainUsers)][x_cols].values
yTrain = data1stHalf[data1stHalf.user.isin(trainUsers)]['insider'].values
yTrainBin = yTrain > 0

xTest = dataTest[dataTest.user.isin(unKnownTestUsers)][x_cols].values
yTest = dataTest[dataTest.user.isin(unKnownTestUsers)]['insider'].values
yTestBin = yTest > 0

rf = RandomForestClassifier(n_jobs=-1)

rf.fit(xTrain, yTrainBin)

print(classification_report(yTestBin, rf.predict(xTest)))

