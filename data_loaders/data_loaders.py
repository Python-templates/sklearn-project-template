from base import BaseDataLoader
from data_loaders import data_handler
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split


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
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=1,
                                                            shuffle=True)

        data_handler.X_data = X_train
        data_handler.y_data = y_train
        data_handler.X_data_test = X_test
        data_handler.y_data_test = y_test

        super().__init__(data_handler, shuffle, test_split, random_state, stratify, training)
