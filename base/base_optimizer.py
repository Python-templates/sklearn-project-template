import os
import pickle
from sklearn.utils import shuffle
from abc import abstractmethod


class BaseOptimizer():
    def __init__(self, data_loader, search_method, config):
        self.X_train, self.y_train = data_loader.get_data()
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=1)
        self.search_method = search_method
        self.save_dir = config.save_dir
        self.config = config

    def _perform_grid_search(self):
        # sorted(sklearn.metrics.SCORERS.keys()) -> get available metrics
        self.search_method.fit(self.X_train, self.y_train)
        return self.search_method

    def _save_model(self, model):
        save_path = os.path.join(self.save_dir, "model.pkl")
        with open(save_path,'wb') as f:
            pickle.dump(model, f)

    def load_model(self):
        load_path = os.path.join(self.save_dir, "model.pkl")
        with open(load_path, 'rb') as f:
            model = pickle.load(f)
            print(model)
        return model

    def save_report(self, report, name_txt):
        save_path = os.path.join(self.save_dir, name_txt)
        with open(save_path, "w") as text_file:
            text_file.write(report)

    def optimize(self):
        gs = self._perform_grid_search()
        model = self.get_model(gs)
        train_report = self.create_train_report(gs)
        self._save_model(model)
        self.save_report(train_report, "report_train.txt")

    def create_train_report(self, cor):
        '''Should return report from training'''
        return "Train report not configured."

    def create_test_report(self, y_test, y_pred):
        '''Should return report from testing'''
        return "Test report not configured."

    @abstractmethod
    def get_model(self, cor):
        '''Should return fitted model and report'''
        raise NotImplementedError


