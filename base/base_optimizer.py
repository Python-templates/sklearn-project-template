import os
import pickle
from sklearn.utils import shuffle
import numpy as np
from abc import abstractmethod


class BaseOptimizer():
    def __init__(self, model, data_loader, search_method, config):
        self.X_train, self.y_train = data_loader.get_data()
        self.model = model
        self.search_method = search_method
        self.save_dir = config.save_dir
        self.debug = config.debug
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
        if self.debug:
            self._debug_true()
        else:
            self._debug_false()

    def _debug_false(self):
        gs = self._perform_grid_search()
        model = self.fitted_model(gs)
        train_report = self.create_train_report(gs)
        self._save_model(model)
        self.save_report(train_report, "report_train.txt")

    def _debug_true(self):
        x = self.X_train
        y = self.y_train
        # for each parameter take just the first element from param_grid
        if hasattr(self.search_method, "param_grid"):
            param_grid = self.search_method.param_grid[0].copy()
            for param in param_grid.keys():
                param_grid[param] = param_grid[param][0]

            self.model.set_params(**param_grid)

            print("-----------------------------------------------------------------")
            print("Model architecture:")
            print("input: {}".format(x.shape))
            for layer in self.model:
                if hasattr(layer, "fit_transform"):
                    x = layer.fit_transform(x,y)
                elif hasattr(layer, "fit") and hasattr(layer, "predict"):
                    layer.fit(x,y)
                    x = layer.predict(x)
                else:
                    x = np.array([])
                    print(f"Warning: {layer} layer dimensions wrong!")

                print("layer {}: {}".format(layer, x.shape))
            print("-----------------------------------------------------------------")
        else:
            print("\n Error: Debug option only available for GridSearch")
        quit()

    def create_train_report(self, cor):
        '''Should return report from training'''
        return "Train report not configured."

    def create_test_report(self, y_test, y_pred):
        '''Should return report from testing'''
        return "Test report not configured."

    @abstractmethod
    def fitted_model(self, cor):
        '''Should return fitted model'''
        raise NotImplementedError


