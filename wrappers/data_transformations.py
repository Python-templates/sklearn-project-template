import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


# Example of savitzky golay filter implementation
class SavgolWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, win_length=7, polyorder=2, deriv=0):
        self.win_length = win_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_sav = []
        sp = [self.win_length, self.polyorder, self.deriv]
        for signal in X:
            if self.win_length != 0:
                signal = savgol_filter(signal, sp[0], sp[1], sp[2])
            signatures_sav.append(signal)
        return np.array(signatures_sav)
