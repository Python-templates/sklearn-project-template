from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


# Example of savitzky golay filter implementation
class SavgolWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, win_length=10):
        self.win_length = win_length
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_sav = []
        sp = [self.win_length, 2, 2]
        for signal in X:
            signal = savgol_filter(signal, sp[0], sp[1], sp[2])
            signatures_sav.append(signal)
        return signatures_sav
