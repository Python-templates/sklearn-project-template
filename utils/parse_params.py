
from sklearn.utils.fixes import loguniform
from scipy.stats import randint
from sklearn.experimental import enable_halving_search_cv  # noqa
import sklearn.model_selection as model_selection_
import skopt as skopt_
from skopt.space import Real, Categorical, Integer

def modify_params(search_method_params, config):
    tuned_parameters = config["tuned_parameters"]
    search_type = config["search_method"]["type"]

    assert search_type == "GridSearchCV" or \
            search_type == "RandomizedSearchCV" or \
            search_type == "HalvingGridSearchCV" or \
            search_type == "HalvingGridSearchCV" or \
            search_type == "BayesSearchCV" , \
            f"Search type {search_type} not supported."

    if "Grid" in search_type:
        search_method_params['param_grid'] = tuned_parameters

    elif search_type == "BayesSearchCV":
        for method_name in tuned_parameters[0]:
            temp = tuned_parameters[0][method_name]
            if len(temp) == 3 and temp[0] == 'RS':
                tuned_parameters[0][method_name] = Real(temp[1], temp[2], prior="log-uniform")
            elif len(temp) == 3 and temp[0] == 'RSI':
                tuned_parameters[0][method_name] = Integer(temp[1], temp[2], prior="uniform")
            else:
                tuned_parameters[0][method_name] = Categorical(temp)
        search_method_params['search_spaces'] = tuned_parameters

    else:
        for method_name in tuned_parameters[0]:
            temp = tuned_parameters[0][method_name]
            if len(temp) == 3 and temp[0] == 'RS':
                tuned_parameters[0][method_name] = loguniform(temp[1], temp[2])
            elif len(temp) == 3 and temp[0] == 'RSI':
                tuned_parameters[0][method_name] = randint(temp[1], temp[2])
        search_method_params['param_distributions'] = tuned_parameters

    return search_method_params, search_type


def get_lib(search_type):
    if search_type == "GridSearchCV" or \
            search_type == "RandomizedSearchCV" or \
            search_type == "HalvingGridSearchCV" or \
            search_type == "HalvingRandomSearchCV":
        return model_selection_
    elif search_type == "BayesSearchCV":
        return skopt_

