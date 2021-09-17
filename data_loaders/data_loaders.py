from base import BaseDataLoader
from data_loaders import data_handler
from sklearn.datasets import load_iris, load_boston


class Classification(BaseDataLoader):
    def __init__(self, data_path, shuffle, test_split, random_state, stratify, training):
        '''set data_path in configs if data localy stored'''

        X, y = load_iris(return_X_y=True)
        data_handler.X_data = X
        data_handler.y_data = y

        super().__init__(data_handler, shuffle, test_split, random_state, stratify, training)



class Regression(BaseDataLoader):
    def __init__(self, data_path, shuffle, test_split, random_state, stratify, training):
        '''set data_path in configs if data localy stored'''

        X, y = load_boston(return_X_y=True)
        data_handler.X_data = X
        data_handler.y_data = y

        super().__init__(data_handler, shuffle, test_split, random_state, stratify, training)
