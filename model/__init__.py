#!/usr/bin/env python
# coding: utf-8

from wrappers import *
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor

methods_dict = {
    'ridge': Ridge,
    'pf': PolynomialFeatures,
    'scaler': StandardScaler,
    'PLS':PLSRegressionWrapper,
    'savgol':SavgolWrapper,
    'SVC':SVC,
    'PCA':PCA
}
