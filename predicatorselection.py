#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code demonstrates predictive models using python.
 Often, building a predictor involves algorithm search and parameter search.
 
In this exercise, we will
  - use the sample dataset called iris and create a predictor column of 0,1
  - assess dimension reduction with principal components and correlation
  - then loop over 
          alternative proportions of score=1 / score=0
          alternative predictive algorithms 

The outer loop is necessary especially with rare event prediction, amplifying the event signal
by reducing the number of non-event observations.

The xgboost algorithm deserves parameter search to manage under/over fitting
and performance.  That exercise can be found in hyperparametersearch.py
in this repository.
"""

import pandas as pd
import numpy as np

bestshare = 1.0

#
# Data Load
#
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = np.where(iris.target == 0, 1, 0) # predict the target setosa species

#
# Data Preprocessing
# is excluded here but commonly includes:
#    casting categorical columns as 0,1 columns using pandas.get_dummies function
#    standardizing/normalizing numeric columns
#    replacing NA values with 0 or median of other obs values

#
# Dimension Reduction
# Variable Selection
# is excluded here but commomly includes:
#    Principal Component Analysis
#    Univariate correlation 
#    Multivariate correlation

#
# Alternative model results
#
# This section assumes an X and y
#

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

rates = [rate for rate in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0] \
         if rate > (sum(y==1)/sum(y==0)) ]

# a generalized routine for fitting, predicting and returning metrics
def predictions(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    resultsdict = {}
    resultsdict["model"] = model.__module__
    resultsdict["confusion"] = metrics.confusion_matrix(y_test, y_pred)
    resultsdict["accuracy"] = metrics.accuracy_score(y_test, y_pred)
    resultsdict["precision"] = metrics.precision_score(y_test, y_pred)
    resultsdict["F1"] = metrics.f1_score(y_test, y_pred)
    resultsdict["recall"] = metrics.recall_score(y_test, y_pred)
    return(resultsdict)

# some models deserve parameter tuning
def find_best_parameters(model, parameters, X, y, cv=10, verbose=1, n_jobs=-1):
    grid_object = GridSearchCV(model, parameters, scoring=metrics.make_scorer(metrics.accuracy_score), cv=cv, verbose=verbose, n_jobs=n_jobs)
    grid_object = grid_object.fit(X, y)
    return grid_object.best_estimator_

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Support Vector Machine
from sklearn.svm import SVC
# xgboost
import xgboost as xgb
from xgboost import XGBClassifier
# Decision tree
from sklearn.tree import DecisionTreeClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Neural Network
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
def nn_classifier(activation_func='relu'):
    model = Sequential()
    model.add(Dense(15, activation=activation_func))
    model.add(Dense(7, activation=activation_func))
    model.add(Dense(3, activation=activation_func))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

models = [LogisticRegression(), 
          GaussianNB(), 
          find_best_parameters(RandomForestClassifier(), 
              {'n_estimators': [20,40,80,160], 'max_features': ['log2', 'sqrt','auto']}, X, y),
          find_best_parameters(SVC(kernel='linear'), 
              {'C': [1.0, 2.0, 4.0], 'gamma': [0.001,0.1,1.,10.]}, X, y),
          find_best_parameters(XGBClassifier(),
              {'max_depth': [3,6,10],'learning_rate': [0.01,0.05,0.1],'n_estimators': [100,500,1000],'colsample_bytree': [0.3,0.7]}, X, y),
          DecisionTreeClassifier(),
          KNeighborsClassifier(),
          KerasClassifier(build_fn=nn_classifier, epochs=10, batch_size=32)
         ]


# Try amplifying the import, outer for loop over score=1 as a share of score=0

print()
resultslist = []

for bestshare in rates:
    print("Fitting models for",bestshare)
    X_best =  X[y == 1]
    y_best =  pd.Series(y[y == 1])  # trivial
    X_worst = X[y == 0].sample(n=round(X_best.shape[0]/bestshare), random_state=1)
    y_worst = pd.Series(y)[y == 0].sample(n=round(X_best.shape[0]/bestshare), random_state=1)
    X_train, X_test, y_train, y_test = \
        train_test_split(pd.concat([X_best,X_worst], axis=0), 
                         pd.concat([y_best,y_worst], axis=0),
                         test_size = 0.33,random_state = 0)
    
    for m in models:
        print("   fitting", m.__module__)
        results = predictions(m, X_train, X_test, y_train, y_test)
        results["share"] = bestshare
        resultslist.append(results)

print(pd.DataFrame(resultslist).to_string())

#
#  Visualize
#
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cnf_matrix, annot=True, cmap='Greens')
ax.set_title("Classifier");
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()
