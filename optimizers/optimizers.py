import numpy as np
from base import BaseOptimizer
from sklearn.metrics import classification_report, mean_absolute_error


class OptimizerClassification(BaseOptimizer):
    def __init__(self, model, data_loader, search_method, scoring, mnt, config):
        self.scoring = scoring
        self.mnt = mnt
        super().__init__(model, data_loader, search_method, config)

    def fitted_model(self, cor):
        clf_results = cor.cv_results_
        params = np.array(clf_results["params"])
        means = clf_results["mean_test_score"]

        if self.mnt == 'min':
            sort_idx = np.argsort(means)
        if self.mnt == 'max':
            sort_idx = np.argsort(means)[::-1]

        params_sorted = params[sort_idx]
        self.model.set_params(**params_sorted[0]) # define the best model
        self.model.fit(self.X_train, self.y_train)

        return self.model

    def create_train_report(self, cor):
        print("Optimizing for: ", self.scoring)
        print("_________________")

        clf_results = cor.cv_results_
        params = np.array(clf_results["params"])
        means = clf_results["mean_test_score"]
        stds = clf_results["std_test_score"]

        if self.mnt == 'min':
            sort_idx = np.argsort(means)
        if self.mnt == 'max':
            sort_idx = np.argsort(means)[::-1]

        indexes = np.arange(len(means))

        indexes_sorted = indexes[sort_idx]
        means_sorted = means[sort_idx]
        stds_sorted = stds[sort_idx]
        params_sorted = params[sort_idx]

        train_report = f"###   Optimizing for {self.scoring}   ###\n\n"
        for idx, mean, std, params_ in zip(indexes_sorted, means_sorted, stds_sorted, params_sorted):
            print("%d   %0.3f (+/-%0.03f) for %r"
                % (idx, mean, std * 2, params_))
            train_report += f"{mean:.3f}  +/-{std*2:.3f}  for  {params_}\n"
        train_report += f"\n###   Best model:   ###\n\n {str(self.model)}"
        train_report += f"\n Number of samples used for training: {len(self.y_train)}"

        return train_report

    def create_test_report(self, y_test, y_pred):
        test_report = str(classification_report(y_test, y_pred))
        print("\n Report on test data:")
        print(test_report)
        test_report += f"\n\n True Values:\n {y_test}"
        test_report += f"\n Pred Values:\n {y_pred}"
        return test_report



class OptimizerRegression(BaseOptimizer):
    def __init__(self, model, data_loader, search_method, scoring, mnt, config):
        self.scoring = scoring
        self.mnt = mnt
        super().__init__(model, data_loader, search_method, config)

    def fitted_model(self, cor):
        clf_results = cor.cv_results_
        params = np.array(clf_results["params"])
        means = clf_results["mean_test_score"]

        if self.mnt == 'min':
            sort_idx = np.argsort(means)
        if self.mnt == 'max':
            sort_idx = np.argsort(means)[::-1]

        params_sorted = params[sort_idx]
        self.model.set_params(**params_sorted[0]) # define the best model
        self.model.fit(self.X_train, self.y_train)

        return self.model

    def create_train_report(self, cor):
        print("Optimizing for: ", self.scoring)
        print("_________________")

        clf_results = cor.cv_results_
        params = np.array(clf_results["params"])
        means = clf_results["mean_test_score"]
        stds = clf_results["std_test_score"]

        if self.mnt == 'min':
            sort_idx = np.argsort(means)
        if self.mnt == 'max':
            sort_idx = np.argsort(means)[::-1]

        indexes = np.arange(len(means))

        indexes_sorted = indexes[sort_idx]
        means_sorted = means[sort_idx]
        stds_sorted = stds[sort_idx]
        params_sorted = params[sort_idx]

        train_report = f"###   Optimizing for {self.scoring}   ###\n\n"
        for idx, mean, std, params_ in zip(indexes_sorted, means_sorted, stds_sorted, params_sorted):
            print("%d   %0.3f (+/-%0.03f) for %r"
                % (idx, mean, std * 2, params_))
            train_report += f"{mean:.3f}  +/-{std*2:.3f}  for  {params_}\n"
        train_report += f"\n###   Best model:   ###\n\n {str(self.model)}"
        train_report += f"\n Number of samples used for training: {len(self.y_train)}"

        return train_report

    def create_test_report(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        test_report = f"True Values:\n {y_test}"
        test_report += f"\n Pred Values:\n {y_pred}"
        test_report += f"\n MAE: \n {mae}"
        print(test_report)
        return test_report
