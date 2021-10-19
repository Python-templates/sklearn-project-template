# Scikit-learn-project-template


## About the project
* Folder structure suitable for many machine learning projects. Especially for those with small amount of available training data.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning. It supports grid search, random search and bayesian search.
* Abstract base classes for faster development:
  * `BaseOptimizer` handles execution of grid search, saving and loading of models and formation of test and train reports.
  * `BaseDataLoader` handles splitting of training and testing data. Spilt is performed depending on settings provided in config file.
  * `BaseModel` handles construction of consecutive steps defined in config file.
* Suitable for tunining of machine learning models which follow `scikit-learn` nomenclature.
    For the time being tested open libraries:
  * scikit-learn
  * sktime
  * tsfresh

## Getting Started

To get a local copy up and running follow steps below.
### Requirements
* Python >= `3.7`
* Packages included in `requirements.txt` file
* (Anaconda for easy installation)

### Install dependencies

Create and activate virtual environment:
```sh
conda create -n yourenvname python=3.7
conda activate yourenvname
```

Install packages:
```sh
python -m pip install -r requirements.txt
```

## Folder Structure
  ```
  sklearn-project-template/
  │
  ├── main.py - main script to start training and (optionally) testing
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_optimizer.py
  │
  ├── configs/ - holds configuration for training and testing
  │   ├── config_classification.json
  │   └── config_regression.json
  │
  ├── data/ - default directory for storing input data
  │
  ├── data_loaders/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── models/ - models
  │   ├── __init__.py - defined models by name
  │   └── models.py
  │
  ├── optimizers/ - optimizers
  │   └── optimizers.py
  │
  ├── saved/ - config, model and reports are saved here
  │   ├── Classification
  │   └── Regression
  │
  ├── utils/ - utility functions
  │   ├── parse_config.py - class to handle config file and cli options
  │   ├── parse_params.py
  │   └── utils.py
  │
  ├── wrappers/ - wrappers of modified sklearn models or self defined transforms
  │   ├── data_transformations.py
  │   └── wrappers.py
  ```

## Usage
Models in this repo are trained on two well-known datasets: iris and boston. First is used for classification and second for regression problem.

Run classification:
   ```sh
python main.py -c configs/config_classification.json
   ```
Run regression:
   ```sh
python main.py -c configs/config_regression.json
   ```

### Config file format
Config files are in `.json` format. Example of such config is shown below:
```javascript
{
    "name": "Classification",   // session name

    "model": {
        "type": "Model",    // model name
        "args": {
            "pipeline": ["scaler", "PLS", "pf", "SVC"],     // pipeline of methods
            "unions": {     // unions of methods included in pipeline
            }
        }
    },

    "tuned_parameters":[{   // hyperparameters to be tuned with search method
                        "SVC__kernel": ["rbf"],
                        "SVC__gamma": [1e-5, 1e-6, 1],
                        "SVC__C": [1, 100, 1000],
                        "PLS__n_components": [1,2,3]
                    }],

    "optimizer": "OptimizerClassification",    // name of optimizer

    "search_method":{
        "type": "GridSearchCV",    // method used to search through parameters
        "args": {
            "refit": false,
            "n_jobs": -1,
            "verbose": 2,
            "error_score": 0
        }
    },

    "cross_validation": {
        "type": "RepeatedStratifiedKFold",     // type of cross-validation used
        "args": {
            "n_splits": 5,
            "n_repeats": 10,
            "random_state": 1
        }
    },

    "data_loader": {
        "type": "Classification",      // name of dataloader class
        "args":{
            "data_path": "data/path-to-file",    // path to data
            "shuffle": true,    // if data shuffled before optimization
            "test_split": 0.2,  // use split method for model testing
            "stratify": true,   // if data stratified before optimization
            "random_state":1    // random state for repeaded output
        }
    },

    "score": "max balanced_accuracy",     // mode and metrics used for scoring
    "test_model": true,     // if model is tested after training
    "debug": false,         // debug model architecture
    "save_dir": "saved/"    // directory of saved reports, models and configs
}

```

Additional parameters can be added to config file. See `scikit-learn` documentation for description of tuned parameters, search method and cross validation. Possible metrics for model evaluation could be found [here](https://scikit-learn.org/stable/modules/model_evaluation.html).

### Pipeline
Methods added to config pipeline must be first defined in `models/__init__.py` file. For previous example of config file the following must be added:

  ```python
from wrappers import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

methods_dict = {
    'pf': PolynomialFeatures,
    'scaler': StandardScaler,
    'PLS':PLSRegressionWrapper,
    'SVC':SVC,
}
  ```
Majority of algorithms implemented in `scikit-learn` library can be directly imported and used. Some algorithms need a little modification before usage. Such an example is Partial least squares (PLS). Modification is implemented in `wrappers/wrappers.py`. In case you want to implement your own method it can be done as well. An example wrapper for Savitzky golay filter is shown in `wrappers/data_transformations.py`. Implementation must satisfy standard method calls, eg. fit(), tranform() etc.

### Unions

Unions concatenates results of multiple transformer methods. Those are applied in parallel to the input data. This is useful if you want to combine several feature mechanisms into a single transformer. For example, if you want to merge results from Principal component analysis (PCA) and Partial least squares (PLS) you can do the following:

```javascript
"pipeline": ["scaler", "pca-pls", "SVC"],
"unions": {
    "pca-pls": ["PLS", "PCA"]
}
```

In pipeline you must write self made-up name of a method (in this case `pca-pls`) and then use the same name as a key in unions dictionary. Value to coresponding key must be list of methods (in this case consisting of "PCA" and "PLS"). Hyperparameters which are tuned with a chosen search method must be separated with double underscore (following `scikit-learn` nomenclature). In case you want to tune number of components of both methods you can do the following:

```javascript
"tuned_parameters":[{
    "pca-pls__PLS__n_components": [1,2,3],
    "pca-pls__PCA__n_components": [1,2,3]
}],
```

Please refer to `configs/config_unions.json` for unions example.
### Debug

To debug model architecture set debug flag in config file to `true`. It will print model  by steps with coresponding consecutive outputs produced at each step. Model debugging will only work with `GridSearchCV` search method. In case many parameters are listed to choose from only first ones will be used for evaluation. Debugging is useful in cases when you want to get a sense of what happens at separate step.

## Customization


### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
        CustomArgs(['-cv', '--cross_validation'], type=int, target='cross_validation;args;n_repeats'),
      # options added here can be modified by command line flags.
]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target`
number of repeats in cross validation option is `('cross_validation', 'args', 'n_repeats')` because `config['cross_validation']['args']['n_repeats']` points to number of repeats.


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` handles:
    * Train/test procedure
    * Data shuffling

* **Usage**

    Loaded data must be assigned to data_handler (dh) in appropriate manner. If dh.X_data_test and dh.y_data_test are not assigned in advance, train/test split could be created by base data loader. In case `"test_split":0.0` is set in config file, whole dataset is used for training. Another option is to assign both train and test sets as shown below. In this case train data will be used for optimization and test data will be used for evaluation of a model.

    ```python
    data_handler.X_data = X_train
    data_handler.y_data = y_train
    data_handler.X_data_test = X_test
    data_handler.y_data_test = y_test
    ```
* **Example**

  Please refer to `data_loaders/data_loaders.py` for data loading example.

### Optimizer
* **Writing your own optimizer**

1. **Inherit ```BaseOptimizer```**

    `BaseOptimizer` handles:
    * Optimization procedure
    * Model saving and loading
    * Report saving


2. **Implementing abstract methods**

    You need to implement `fitted_model()` which must return fitted model.
    Optionally you can implement format of train/test reports with `create_train_report()` and `create_test_report()`.

* **Example**

  Please refer to `optimizers/optimizers.py` for optimizer example.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Initialization defined in config pipeline
    * Modification of steps

2. **Implementing abstract methods**

    You need to implement `created_model()` which must return created model.

* **Usage**

    Initialization of pipeline methods is performed with `create_steps()`. Steps can be later modified with the use of `change_step()`. An example on how to change a step is shown bellow where Sequential feature selector is added to the pipeline.

    ```python
    def __init__(self, pipeline):
        steps = self.create_steps(pipeline)

        rf = RandomForestRegressor(random_state=1)
        clf = TransformedTargetRegressor(regressor=rf,
                                        func=np.log1p,
                                        inverse_func=np.expm1)
        sfs = SequentialFeatureSelector(clf, n_features_to_select=2, cv=3)

        steps = self.change_step('sfs', sfs, steps)

        self.model = Pipeline(steps=steps)

    ```

    Beware that in this case 'sfs' needs to be added to pipeline in config file. Otherwise, no step in the pipeline is changed.

* **Example**

  Please refer to `models/models.py` model example.

## Common Questions About Hyperparameter Optimization

### How to Choose Between Random and Grid Search?
* Choose the method based on your needs. I recommend starting with grid and doing a random search if you have the time.
* Grid search is appropriate for small and quick searches of hyperparameter values that are known to perform well generally.
* Random search is appropriate for discovering new hyperparameter values or new combinations of hyperparameters, often resulting in better performance, although it may take more time to complete.

### How to Speed-Up Hyperparameter Optimization?
* Ensure that you set the “n_jobs” argument to the number of cores on your machine.
* Evaluate on a smaller sample of your dataset.
* Explore a smaller search space.
* Use fewer repeats and/or folds for cross-validation.
* Execute the search on a faster machine, such as AWS EC2.
* Use an alternate model that is faster to evaluate.

More on: [machinelearningmastery](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/).

## Roadmap

See [open issues](https://github.com/janezlapajne/sklearn-project-template/issues) to request a feature or report a bug.

## Contribution

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

How to start with contribution:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Feel free to contribute any kind of function or enhancement.

## License
This project is licensed under the MIT License. See  LICENSE for more details.

## Acknowledgements
This project is inspired by the project [pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque). I would like to confess that some functions, architecture and some parts of readme were directly copied from this repo. But to be honest, what should I do - the project is absolutely amazing!

Additionally, special thanks to the creator of Machine learning mastery, [Jason Brownlee, PhD](https://machinelearningmastery.com/about/) for insightful articles published!

## Consider supporting

Do you feel generous today? I am still a student and would make a good use of some extra money :P



[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/janezlapajne)

<!-- Odspodi ni več.

____________

This is a simple python project template for Visual studio code.

Create and activate virtual environment:

   ```sh
   python -m venv .venv
   ```
   ```sh
   "./.venv/Scripts/activate"
   ```

   or

   ```sh
   conda create -n yourenvname python=x.x anaconda
   ```
   ```sh
   conda activate yourenvname
   ```

Clear git cached files and directories:

   ```sh
   git rm --cached -r .vscode
   ```
   ```sh
   git rm --cached .env
   ```

Set path to project root directory in `.env`, e.g.:

   ```sh
   PYTHONPATH=C:\\Users\\janezla\\Documents\\python-project-template
   ```

Set python path in vscode workspace settings, e.g.:
   ```sh
   "python.pythonPath": "C:\\Users\\janezla\\Anaconda3\\envs\\yourenvname\\python"
   ``` -->

