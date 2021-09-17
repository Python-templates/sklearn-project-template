#!/usr/bin/env python
# coding: utf-8

from .data_transformations import *
from .wrappers import *

from wrappers import *
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import TSNE

methods_dict = {
    'tsne':TSNE,
    'pf': PolynomialFeatures,
    'scaler': StandardScaler,
    'PLS':PLSRegressionWrapper,
    'savgol':SavgolWrapper,
    'SVC':SVC,
    'PCA':PCA
}
