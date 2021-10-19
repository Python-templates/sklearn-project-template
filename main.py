import os
import glob
import argparse
import collections
from utils import ConfigParser, read_json, modify_params, get_lib
import data_loaders.data_loaders as data_loaders_
import models.models as models_
import optimizers.optimizers as optimizers_
import sklearn.model_selection as model_selection_


def main(config):

    data_loader = config.init_obj('data_loader', data_loaders_, **{'training':True})
    model = config.init_obj('model', models_).created_model()
    cross_val = config.init_obj('cross_validation', model_selection_)
    mnt, scoring = config['score'].split()

    search_method_params = {
        'estimator': model,
        'scoring': scoring,
        'cv': cross_val
    }
    search_method_params, search_type = modify_params(search_method_params, config)
    search_method = config.init_obj('search_method', get_lib(search_type), **search_method_params)

    Optimizer = config.import_module('optimizer', optimizers_)
    optim = Optimizer(model=model,
                      data_loader=data_loader,
                      search_method=search_method,
                      scoring=scoring,
                      mnt=mnt,
                      config=config)

    optim.optimize()


    if config['test_model']:
        data_loader = config.init_obj('data_loader', data_loaders_, **{'training':False})
        X_test, y_test = data_loader.get_data()
        model = optim.load_model()
        y_pred = model.predict(X_test)
        test_report = optim.create_test_report(y_test, y_pred)
        optim.save_report(test_report, 'report_test.txt')


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Sklearn Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-cv', '--cross_validation'], type=int, target='cross_validation;args;n_repeats'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)


    # configs_list = glob.glob(os.path.join("configs", "*.json"))
    # for cfg_fname in configs_list:
    #     config = read_json(cfg_fname)
    #     config = ConfigParser(config)
    #     main(config)
